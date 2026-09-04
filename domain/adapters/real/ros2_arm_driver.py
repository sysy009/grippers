"""Ros2ArmDriver — mission_orchestrator가 쓰는 ArmDriver 포트 구현.
arm_driver_node에 액션/서비스로 말을 건다. domain.values.Point3 인스턴스를
geometry_msgs/Point 생성자 자리에 그대로 넘기면 rclpy가 필드 타입을
assert로 검사해서 런타임 AssertionError가 난다 — 여기서 필드별로 옮긴다."""

from geometry_msgs.msg import Point
from grippers_interfaces.action import (
    MoveToCartesian,
    MoveToFloorPose,
    ReorientArm,
    RunVlaGrasp,
)
from grippers_interfaces.srv import GetLoad, OffsetBaseYaw, SetGripper
from rclpy.action import ActionClient
from std_srvs.srv import Trigger

from domain.adapters.real._ros_call import ESTOP_TIMEOUT_SEC, call_action, call_service
from domain.ports.arm_driver import ArmDriver
from domain.values import Point3

# 부하를 읽지 못했을 때의 값. 0.0 = '빈 채'이므로 GRASP는 파지 실패로 판정해
# 재시도한다 — 부하를 모르는 채로 성공 판정을 내려 물체를 든 줄 알고 이송하는
# 것보다 안전하다 (arm_driver_node._read_load의 None 처리와 같은 논리).
LOAD_UNKNOWN = 0.0

# ── 느린 팔 서비스의 대기 시간 ─────────────────────────────────────────────
#
# 기본 SERVICE_TIMEOUT_SEC(3.0s)은 이 둘에게 너무 짧다. 서버가 일부러
# **동작이 끝날 때까지 붙들고 있다가** 응답하기 때문이다. 짧게 잡으면 팔은
# 제대로 움직였는데 호출자만 실패로 받는다 — 2026-08-28 실기에서
# set_gripper 와 offset_base_yaw 가 매번 그렇게 나왔고, GRASP 가 "servo 1이
# 거부했다"로 잘못 보고됐다. 조용한 오보라 로그만 봐서는 팔 고장처럼 보인다.
#
# 값은 서버 쪽 상한에서 거꾸로 잡는다(arm_driver_node 상수):
#   set_gripper       = GRIPPER_MOTION_TIMEOUT_SEC(4.0) + GRASP_SETTLE_SEC(1.5)
#                       = 5.5s 가 최악 -> 여유를 둬 8.0s
#   offset_base_yaw   = 글라이드 + 도달 대기. 보정은 단계 수를 거리에 맞춰
#                       0.5s 안팎이지만 정착이 느릴 수 있어 8.0s
GRIPPER_TIMEOUT_SEC = 8.0
BASE_YAW_TIMEOUT_SEC = 8.0

# VLA 파지 결과 상한. 기본 ACTION_RESULT_TIMEOUT_SEC(60초)로는 부족할 수
# 있다 - 이 액션은 자기 timeout_sec 까지 청크를 계속 재생하고, 노드
# 기본값이 40초다. 서버 상한보다 넉넉해야 정상 동작을 실패로 끊지 않는다.
VLA_RESULT_TIMEOUT_SEC = 180.0


class Ros2ArmDriver(ArmDriver):
    def __init__(self, node):
        self._node = node
        self._move_client = ActionClient(node, MoveToCartesian, "arm_driver/move_to_cartesian")
        self._floor_pose_client = ActionClient(
            node, MoveToFloorPose, "arm_driver/move_to_floor_pose"
        )
        self._reorient_client = ActionClient(node, ReorientArm, "arm_driver/reorient")
        # ⚠️ arm_driver 가 아니라 vla_inference_node 가 서버다. 도메인 입장에서는
        # "팔에게 파지를 시킨다"라 ArmDriver 포트에 두지만, 실제 상대는 다른
        # 노드다 - 그 노드가 안 떠 있으면 여기서 False 로 떨어진다.
        self._vla_client = ActionClient(node, RunVlaGrasp, "vla/run_grasp")
        self._gripper_client = node.create_client(SetGripper, "arm_driver/set_gripper")
        self._load_client = node.create_client(GetLoad, "arm_driver/get_load")
        self._fold_client = node.create_client(Trigger, "arm_driver/fold_to_cradle")
        self._hold_client = node.create_client(Trigger, "arm_driver/hold_position")
        self._yaw_client = node.create_client(OffsetBaseYaw, "arm_driver/offset_base_yaw")

    def move_to_cartesian(self, xyz_m: Point3, down: bool = False) -> bool:
        """도달하면 True. 액션 서버가 없거나 결과가 오지 않으면 **False**
        (계약 상세는 `ArmDriver.move_to_cartesian` 참고 — 현재 FSM은 안 부른다)."""
        goal = MoveToCartesian.Goal(
            target=Point(x=xyz_m.x, y=xyz_m.y, z=xyz_m.z),
            down=down,
        )
        result = call_action(self._node, self._move_client, goal, label="move_to_cartesian")
        if result is None:
            return False
        return result.reached

    def move_to_floor_pose(self, profile: str, stage: str) -> bool:
        goal = MoveToFloorPose.Goal(profile=profile, stage=stage)
        result = call_action(
            self._node,
            self._floor_pose_client,
            goal,
            label="move_to_floor_pose",
        )
        return result is not None and result.reached

    def set_gripper(self, width_mm: float) -> None:
        """포트가 값을 돌려주지 않으므로 실패는 경고 로그로만 남는다
        (`call_service` 안에서 남긴다). 호출자는 실패를 직접 알 수 없지만,
        그리퍼가 닫히지 않았으면 뒤이은 `get_load()` 가 빈 채 부하를 읽어
        `GRASP` 가 파지 실패로 판정한다 — 실패가 조용히 삼켜지지는 않는다.

        ⚠️ 단위 변경: 예전엔 deg(각도)를 받아 SetGripper.srv의 bool closed로
        이진화했다. 이제 width_mm(mm)을 그대로 실어 보낸다 — 각도 변환은
        arm_driver_node의 캘리브레이션 테이블 몫이지 여기서 하지 않는다."""
        req = SetGripper.Request(width_mm=width_mm)
        call_service(self._node, self._gripper_client, req, label="set_gripper",
                     timeout_sec=GRIPPER_TIMEOUT_SEC)

    def get_load(self) -> float:
        """그리퍼 부하 비율(0~1). 서비스가 없거나 응답이 없으면
        **`LOAD_UNKNOWN`(0.0)** — 파지 실패로 판정된다."""
        res = call_service(self._node, self._load_client, GetLoad.Request(), label="get_load")
        if res is None:
            return LOAD_UNKNOWN
        return res.load_ratio

    def reorient(self, phi_rad: float) -> bool:
        """정착하면 True. 액션 서버가 없거나 결과가 오지 않으면 **False**
        (계약 상세는 `ArmDriver.reorient` 참고 — 서버가 아직 스텁이고 현재
        FSM도 안 부른다)."""
        goal = ReorientArm.Goal(phi=phi_rad)
        result = call_action(self._node, self._reorient_client, goal, label="reorient")
        if result is None:
            return False
        return result.settled

    def fold_to_cradle(self) -> bool:
        """접었으면 True. 서비스가 없거나 응답이 없으면 **False**."""
        res = call_service(self._node, self._fold_client, Trigger.Request(), label="fold_to_cradle")
        if res is None:
            return False
        return res.success

    def offset_base_yaw(self, offset_rad: float) -> bool:
        """servo 1 좌우 보정. 서비스가 없거나 노드가 거부하면 **False** —
        그 경우 호출자가 Host에 다시 세워 달라고 넘긴다."""
        req = OffsetBaseYaw.Request(offset_rad=float(offset_rad))
        res = call_service(self._node, self._yaw_client, req, label="offset_base_yaw",
                           timeout_sec=BASE_YAW_TIMEOUT_SEC)
        if res is None:
            return False
        if not res.ok:
            self._node.get_logger().warn(f"offset_base_yaw 거부: {res.message}")
        return res.ok

    def run_vla_grasp(self, task: str, timeout_sec: float = 0.0,
                      max_step_deg: float = 0.0) -> bool:
        """정책이 오류 없이 돌았으면 True. 서버 부재·오류·취소는 **False**.

        ⚠️ True 가 파지 성공이 아니다 - 포트 docstring 참고. 호출부가 부하와
        confirm_grasp 로 따로 판정한다."""
        goal = RunVlaGrasp.Goal(
            task=task,
            timeout_sec=float(timeout_sec),
            max_step_deg=float(max_step_deg),
        )
        result = call_action(
            self._node, self._vla_client, goal, label="run_vla_grasp",
            result_timeout_sec=VLA_RESULT_TIMEOUT_SEC,
        )
        if result is None:
            return False
        if not result.ok:
            self._node.get_logger().warn(f"run_vla_grasp 실패: {result.message}")
        return result.ok

    def hold_position(self) -> None:
        # stop()과 같은 이유로 응답을 기다리지 않는다 — E-STOP 경로에서 호출되므로
        # (states.py EstopState) 늦어지면 안 된다. 같은 이유로 wait_for_service()도
        # 인자 없이 부르면 안 된다 — 서비스가 안 떠 있을 때 무기한 블록돼
        # "기다리지 않는다"는 의도가 정반대로 뒤집힌다.
        if not self._hold_client.wait_for_service(timeout_sec=ESTOP_TIMEOUT_SEC):
            self._node.get_logger().error("hold_position: 서비스 없음 — 정지 실패")
            return
        self._hold_client.call_async(Trigger.Request())
