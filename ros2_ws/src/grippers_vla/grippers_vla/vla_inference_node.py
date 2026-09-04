"""vla_inference_node — VLA 정책으로 파지를 수행한다.

## 흐름

    RunVlaGrasp (FSM 이 부름)
      └ 반복:
          그리퍼캠 프레임 (토픽) + 관절 상태 (arm_driver/get_policy_state)
            → 정책 forward → 관절 청크
            → arm_driver/execute_joint_chunk 로 재생
            → 끝나면 다시

## ⚠️ 이 노드는 서보에 직접 쓰지 않는다

`/dev/soarm` 은 arm_driver 가 독점한다. 다른 프로세스가 driver_sdk 로 같이
붙으면 팔 이동이 통째로 깨진다(GetArmState.srv 주석). 그래서 명령은 전부
ExecuteJointChunk 액션을 거친다 — 관절 한계, 스텝당 이동 상한, E-STOP 취소가
전부 그 한 곳을 지난다.

LeRobot 의 로봇 클래스(SO101Follower)를 여기서 쓰면 안 되는 이유도 같다. 그쪽은
`connect()` 에서 **자기 캘리브레이션의 Homing_Offset 을 서보에 써 버린다.**
그러면 arm_driver 의 교시 자세가 전부 다른 물리 자세를 가리키게 된다
(arm_driver_node._check_taught_calibration 의 2026-08-29 사례).

## 좌표계

관측도 명령도 **LeRobot 정규화 단위**다 — 정책이 낸 값을 변환 없이 그대로
싣고, 하드웨어 좌표계로의 변환은 arm_driver 한 곳에서 한다. 이유는
ExecuteJointChunk.action 의 표 참고.

## task 문자열은 물체를 고르지 않는다

RunVlaGrasp.action 의 경고 참고. 정책은 정면 중앙의 것을 집는다.
"""

import threading
import time

import numpy as np
import rclpy
from grippers_interfaces.action import ExecuteJointChunk, RunVlaGrasp
from grippers_interfaces.srv import GetPolicyState
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

SERVO_COUNT = 6
GRIPPER_CAM_TOPIC_DEFAULT = "gripper_cam/image_raw"
# 학습 fps. 청크를 이 주기로 재생해야 정책이 가정한 시간축과 맞는다.
FPS_DEFAULT = 30.0
# 스텝당 관절 이동 상한(도). rollout_policy.py 의 --max-rel 기본값과 같다.
MAX_STEP_DEG_DEFAULT = 5.0
RUN_TIMEOUT_SEC_DEFAULT = 40.0
# 프레임이 이보다 오래됐으면 쓰지 않는다. 낡은 관측으로 추론하면 정책이 이미
# 지나간 상황에 반응한다 — 그 상태로 청크를 통째로 재생하면 3초를 헛돈다.
MAX_FRAME_AGE_SEC = 1.5
# 액션/서비스 대기 상한. domain/adapters/real/_ros_call.py 의 값과 맞춘다.
SERVICE_TIMEOUT_SEC = 3.0
ACTION_SERVER_TIMEOUT_SEC = 5.0


def _wait(future, timeout_sec: float) -> bool:
    """future 완료를 기다린다 — **추가 executor 를 만들지 않는다.**

    ⚠️ `rclpy.spin_until_future_complete` 를 쓰면 같은 노드를 두 executor 가
    동시에 스핀하게 되고, rclpy 에서 지원되지 않는 조합이다. 2026-08-23 실기에서
    mission_orchestrator 의 구독 콜백이 영영 응답하지 않게 된 원인이 그것이었다
    (domain/adapters/real/_ros_call.py 모듈 주석). 완료는 이미 돌고 있는
    executor 가 처리하므로 여기서는 기다리기만 한다.
    """
    done = threading.Event()
    future.add_done_callback(lambda _f: done.set())
    return done.wait(timeout=timeout_sec)


def _bgr_from_image_msg(msg):
    """Image -> BGR ndarray. cv_bridge 를 쓰지 않는다.

    이 환경의 cv_bridge 확장은 numpy 1.x ABI 로 빌드돼 있어 numpy 2.x 와 안 맞고
    세그폴트를 낸다(perception_node._bgr_from_image_msg 의 2026-08-23 기록).
    torch 가 numpy 2.x 를 끌어오는 이 노드에서는 특히 확실히 터진다.
    """
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc not in ("bgr8", "rgb8"):
        raise ValueError(f"지원하지 않는 인코딩: {msg.encoding}")
    img = buf.reshape(msg.height, msg.width, 3)
    return img[:, :, ::-1] if enc == "rgb8" else img


class VlaInferenceNode(Node):
    def __init__(self):
        super().__init__("vla_inference_node")
        self.declare_parameter("ckpt", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("task", "pick up the object")
        self.declare_parameter("fps", FPS_DEFAULT)
        self.declare_parameter("max_step_deg", MAX_STEP_DEG_DEFAULT)
        self.declare_parameter("timeout_sec", RUN_TIMEOUT_SEC_DEFAULT)
        self.declare_parameter("gripper_cam_topic", GRIPPER_CAM_TOPIC_DEFAULT)
        # 청크에서 실제로 재생할 스텝 수. 0 이면 체크포인트 값을 쓴다.
        #
        # ⚠️ 함부로 줄이지 말 것 — **청크가 곧 시계다.** ACT 는 시간을 안 보므로
        # "언제 그리퍼를 여는가"가 청크 안에만 들어 있다. v5 의 첫 청크는
        # "가만히 있다가 50스텝쯤부터 연다"라, 30 으로 자르면 열기 직전에
        # 잘리고 같은 관측으로 같은 청크가 다시 나와 무한 반복한다
        # (2026-09-02 실기: 25초 동안 관절 폭 최대 2.8도).
        self.declare_parameter("n_action_steps", 0)

        self._frame = None
        self._frame_stamp = 0.0
        self._frame_lock = threading.Lock()
        self._policy = None
        self._pre = None
        self._post = None
        self._policy_error = None
        self._chunk_steps = 0

        # 콜백 그룹을 나눈다. 실행 콜백이 재생 결과를 기다리는 동안에도 취소와
        # 이미지 수신이 처리돼야 한다 — 한 그룹에 몰면 그 사이에 멈춘다.
        server_group = ReentrantCallbackGroup()
        client_group = MutuallyExclusiveCallbackGroup()

        topic = self.get_parameter("gripper_cam_topic").value
        # 최신 프레임 한 장만 쓴다. 밀린 프레임을 붙들면 낡은 관측으로 추론한다.
        self.create_subscription(
            Image, topic, self._on_image,
            QoSProfile(depth=1,
                       history=QoSHistoryPolicy.KEEP_LAST,
                       reliability=QoSReliabilityPolicy.BEST_EFFORT,
                       durability=QoSDurabilityPolicy.VOLATILE),
            callback_group=client_group,
        )
        self._state_client = self.create_client(
            GetPolicyState, "arm_driver/get_policy_state", callback_group=client_group)
        self._chunk_client = ActionClient(
            self, ExecuteJointChunk, "arm_driver/execute_joint_chunk",
            callback_group=client_group)
        self._server = ActionServer(
            self, RunVlaGrasp, "vla/run_grasp",
            execute_callback=self._execute_run,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=server_group,
        )
        self.get_logger().info(f"그리퍼캠 구독: {topic}")

        # 정책 적재는 오래 걸린다(SmolVLA 는 최초 210초, ACT 는 몇 초). 여기서
        # 막으면 FSM 의 wait_for_server 가 먼저 터진다 — 서버는 바로 띄우고
        # 적재는 뒤에서 한다. 준비 전에 들어온 goal 은 이유를 붙여 거절한다.
        threading.Thread(target=self._load_policy, daemon=True).start()

    # ── 적재 ────────────────────────────────────────────────────────────────

    def _load_policy(self) -> None:
        ckpt = str(self.get_parameter("ckpt").value or "").strip()
        if not ckpt:
            self._policy_error = "ckpt 파라미터가 비어 있습니다"
            self.get_logger().warn(f"정책 미적재 — {self._policy_error}")
            return
        try:
            import torch  # noqa: F401  (여기서 처음 끌어온다 — 무겁다)
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors

            device = str(self.get_parameter("device").value)
            started = time.monotonic()
            cfg = PreTrainedConfig.from_pretrained(ckpt)
            cfg.device = device
            requested = int(self.get_parameter("n_action_steps").value)
            if requested > 0:
                cfg.n_action_steps = requested
            policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
            policy.to(device).eval()
            # 학습 때 device 가 전처리기에 박혀 있다. 안 덮으면 텐서가 그쪽으로
            # 가서 "mat1 is on xpu:0" 같은 오류로 죽는다.
            pre, post = make_pre_post_processors(
                cfg, pretrained_path=ckpt,
                preprocessor_overrides={"device_processor": {"device": device}},
            )
            self._chunk_steps = int(cfg.n_action_steps)
            self._pre, self._post, self._policy = pre, post, policy
            self.get_logger().info(
                f"정책 적재 완료 {cfg.type}  chunk {cfg.chunk_size} / 재생 "
                f"{self._chunk_steps}스텝  device {device}  {time.monotonic()-started:.1f}s"
            )
        except Exception as e:  # 적재 실패로 노드를 죽이지 않는다 — 이유를 남기고 거절한다
            self._policy_error = f"{type(e).__name__}: {e}"
            self.get_logger().error(f"정책 적재 실패 — {self._policy_error}")

    # ── 관측 ────────────────────────────────────────────────────────────────

    def _on_image(self, msg) -> None:
        try:
            frame = _bgr_from_image_msg(msg)
        except ValueError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)
            return
        # ⚠️ 여기서 회전하지 않는다. perception_node 가 발행 전에 이미
        # gripper_cam_geometry.orient() 로 180도 돌려 학습 방향으로 맞춰 놓는다.
        # 한 번 더 돌리면 정책이 뒤집힌 화면을 본다.
        with self._frame_lock:
            self._frame = frame.copy()
            self._frame_stamp = time.monotonic()

    def _latest_frame(self):
        with self._frame_lock:
            if self._frame is None:
                return None, 0.0
            return self._frame, time.monotonic() - self._frame_stamp

    def _read_policy_state(self):
        if not self._state_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_SEC):
            raise RuntimeError("arm_driver/get_policy_state 서비스 없음")
        future = self._state_client.call_async(GetPolicyState.Request())
        if not _wait(future, SERVICE_TIMEOUT_SEC) or not future.done():
            future.cancel()
            raise RuntimeError("get_policy_state 응답 없음")
        res = future.result()
        if res is None or not res.ok:
            raise RuntimeError(f"get_policy_state 실패: {getattr(res, 'message', '응답 없음')}")
        return list(res.positions)

    # ── 추론 ────────────────────────────────────────────────────────────────

    def _predict_chunk(self, frame, state, task):
        """관측 하나로 관절 청크를 만든다. 반환은 (steps x 6) 리스트.

        ⚠️ `predict_action_chunk` 가 아니라 `select_action` 을 반복해서 부른다.
        정책은 큐가 빈 스텝에서만 실제로 forward 를 돌고 나머지는 큐에서 꺼내
        주므로(modeling_act.py), forward 는 여전히 한 번이다. 이 경로를 쓰는
        이유는 **실기로 검증된 경로가 이쪽**이기 때문이다 —
        rollout_policy.py 가 같은 전처리·후처리 조합으로 돌던 그 경로다.
        청크 텐서를 직접 후처리기에 넣으면 정규화 해제가 [B,T,D] 에서 어떻게
        방송되는지에 기대게 된다.
        """
        import torch

        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        st = torch.tensor(state, dtype=torch.float32)
        self._policy.reset()  # 큐를 비워 이 관측으로 새 청크를 뽑게 한다
        chunk = []
        with torch.no_grad():
            for _ in range(self._chunk_steps):
                act = self._post(self._policy.select_action(self._pre({
                    "observation.state": st.unsqueeze(0),
                    "observation.images.gripper": img.unsqueeze(0),
                    "task": [task],
                }))).squeeze(0).float()
                if not torch.isfinite(act).all():
                    raise RuntimeError("정책이 NaN/Inf 를 냈습니다")
                chunk.append([float(v) for v in act])
        return chunk

    def _play_chunk(self, chunk, fps, max_step_deg):
        """청크를 arm_driver 에 넘겨 재생시키고 결과를 돌려준다."""
        if not self._chunk_client.wait_for_server(timeout_sec=ACTION_SERVER_TIMEOUT_SEC):
            raise RuntimeError("arm_driver/execute_joint_chunk 액션 서버 없음")
        goal = ExecuteJointChunk.Goal(
            positions=[v for step in chunk for v in step],
            fps=float(fps),
            max_step_deg=float(max_step_deg),
        )
        goal_future = self._chunk_client.send_goal_async(goal)
        if not _wait(goal_future, ACTION_SERVER_TIMEOUT_SEC) or not goal_future.done():
            goal_future.cancel()
            raise RuntimeError("청크 goal 수락 응답 없음")
        handle = goal_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("청크 goal 거부됨")
        # 재생 시간 + 여유. 상한을 재생 시간에 묶어야 서버가 멈췄을 때 알 수 있다.
        budget = len(chunk) / float(fps) + 10.0
        result_future = handle.get_result_async()
        if not _wait(result_future, budget) or not result_future.done():
            # ⚠️ 그냥 빠져나가면 팔은 계속 움직인다.
            handle.cancel_goal_async()
            result_future.cancel()
            raise RuntimeError(f"청크 결과 없음 ({budget:.1f}s) — 취소 요청함")
        wrapped = result_future.result()
        if wrapped is None or not wrapped.result.ok:
            raise RuntimeError(
                f"청크 재생 실패: {getattr(wrapped.result, 'message', '결과 없음')}"
                if wrapped else "청크 결과 메시지 없음")
        return wrapped.result

    # ── 액션 ────────────────────────────────────────────────────────────────

    def _execute_run(self, goal_handle):
        req = goal_handle.request
        result = RunVlaGrasp.Result()
        feedback = RunVlaGrasp.Feedback()
        chunks = 0
        try:
            if self._policy is None:
                raise ValueError(self._policy_error or "정책을 아직 적재하는 중입니다")

            task = (req.task or "").strip() or str(self.get_parameter("task").value)
            fps = float(self.get_parameter("fps").value)
            max_step = (float(req.max_step_deg) if req.max_step_deg > 0
                        else float(self.get_parameter("max_step_deg").value))
            timeout = (float(req.timeout_sec) if req.timeout_sec > 0
                       else float(self.get_parameter("timeout_sec").value))
            self.get_logger().info(
                f"VLA 파지 시작 task=\"{task}\"  {self._chunk_steps}스텝@{fps:.0f}fps  "
                f"max_step {max_step:.1f}도  상한 {timeout:.0f}s")

            started = time.monotonic()
            while time.monotonic() - started < timeout:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.ok = False
                    result.chunks_executed = chunks
                    result.message = f"취소됨 — 청크 {chunks}개 재생"
                    self.get_logger().warn(f"VLA 파지 {result.message}")
                    return result

                frame, age = self._latest_frame()
                if frame is None:
                    raise RuntimeError("그리퍼캠 프레임이 아직 없습니다 — "
                                       "perception_node 의 gripper_cam_publish_hz 확인")
                if age > MAX_FRAME_AGE_SEC:
                    raise RuntimeError(f"그리퍼캠 프레임이 {age:.1f}s 낡았습니다")

                state = self._read_policy_state()
                t_inf = time.monotonic()
                chunk = self._predict_chunk(frame, state, task)
                inference_ms = (time.monotonic() - t_inf) * 1000.0

                # 추론이 청크 수명보다 길면 팔이 청크 사이에서 선다. 막지는
                # 않는다 — 정적 진단용으로 일부러 느린 정책을 걸 수 있다.
                life_ms = len(chunk) / fps * 1000.0
                if inference_ms > life_ms:
                    self.get_logger().warn(
                        f"추론 {inference_ms:.0f}ms > 청크 수명 {life_ms:.0f}ms "
                        f"(듀티 {inference_ms/life_ms*100:.0f}%) — 청크 사이에서 팔이 멈춥니다",
                        throttle_duration_sec=10.0)

                played = self._play_chunk(chunk, fps, max_step)
                chunks += 1
                feedback.chunk = chunks
                feedback.elapsed_sec = float(time.monotonic() - started)
                feedback.inference_ms = float(inference_ms)
                goal_handle.publish_feedback(feedback)
                self.get_logger().info(
                    f"청크 {chunks}: 추론 {inference_ms:.0f}ms, {played.message}")

            # ⚠️ ok=True 는 "오류 없이 돌았다"이지 "물체를 집었다"가 아니다.
            # 이 노드는 파지 성공을 판정할 방법이 없다 — 정책은 언제 끝났는지
            # 알려주지 않는다. 성공 판정은 호출부가 기존 파지 경로와 똑같이
            # 그리퍼 부하(arm_driver/get_load)로 한다.
            result.ok = True
            result.chunks_executed = chunks
            result.message = (f"상한 {timeout:.0f}s 도달 — 청크 {chunks}개 재생 "
                              f"(파지 성공 여부는 부하로 판정할 것)")
            goal_handle.succeed()
        except ValueError as e:
            self.get_logger().warn(f"VLA 파지 거부: {e}")
            result.ok = False
            result.message = str(e)
            result.chunks_executed = chunks
            goal_handle.abort()
        except Exception as e:
            self.get_logger().error(f"VLA 파지 오류: {e}")
            result.ok = False
            result.message = str(e)
            result.chunks_executed = chunks
            goal_handle.abort()
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VlaInferenceNode()
    # 실행 콜백이 재생 결과를 기다리는 동안 취소·이미지·서비스 응답이 처리돼야
    # 한다 — 단일 스레드 executor 로는 그 자리에서 교착한다.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
