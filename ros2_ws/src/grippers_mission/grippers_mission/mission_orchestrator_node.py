"""mission_orchestrator_node — Pi 미션 FSM을 ROS2로 감싼다 (팀 확정, 2026-08-26).

FSM 자체는 별도 스레드에서 순차 실행하고, rclpy는 MultiThreadedExecutor로
스핀해서 E-STOP이 FSM 블로킹 도중에도 즉시 들어올 수 있게 한다.

## 무엇이 바뀌었나

예전 오케스트레이터는 `/command`(음성 명령 문자열)를 받아 `MissionTask`를
한 번 돌리는 구조였다. 목표 선정과 경로 계산이 Pi에 있던 시절의 모양이다.

이제 **Host가 미션을 주도한다.** 명령은 UDP로 오고, 그 안에 상태와 속도만
들어 있다. FSM은 시작해서 끝나는 것이 아니라 **계속 도는 루프**이고,
Host가 보내는 state가 진행을 정한다. 그래서 명령 큐도, 재실행 루프도 없다.

`/mission/state`는 그대로 발행한다 — 아레나 관중 오버레이와 디버깅이
이 토픽을 본다. Host 보고와 별개의 경로다.
"""

import sys
import threading
import time
import traceback

sys.path.insert(0, "/grippers")  # PYTHONPATH 미설정 환경 대비 안전장치

import rclpy
from grippers_interfaces.msg import MissionState as MissionStateMsg
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty

from domain.adapters.fake.fake_arm import FakeArm
from domain.adapters.fake.fake_base import FakeBase
from domain.adapters.fake.fake_host_link import FakeHostLink, FakeLidar
from domain.adapters.fake.scripted_perception import ScriptedPerception
from domain.adapters.logged_port import LoggedPort
from domain.adapters.real.ros2_arm_driver import Ros2ArmDriver
from domain.adapters.real.ros2_lidar import Ros2Lidar
from domain.adapters.real.ros2_mecanum_base import Ros2MecanumBase
from domain.adapters.real.ros2_perception import Ros2Perception
from domain.adapters.real.udp_host_link import UdpHostLink
from domain.task.baseline_mission import BaselineMission, BaselinePorts

# FSM 한 사이클의 목표 주기. Host가 보내는 명령 주기와 맞춘다.
CYCLE_PERIOD_S = 0.1


class MissionOrchestratorNode(Node):
    def __init__(self):
        super().__init__("mission_orchestrator")
        cb_group = ReentrantCallbackGroup()

        self._state_pub = self.create_publisher(
            MissionStateMsg,
            "/mission/state",
            qos_profile=QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            ),
        )
        self.create_subscription(
            Empty, "/mission/emergency_stop", self._on_estop, 10,
            callback_group=cb_group)

        self.declare_parameter("use_fake_base", False)
        self.declare_parameter("use_fake_arm", False)
        self.declare_parameter("use_fake_perception", False)
        self.declare_parameter("use_fake_host", False)
        self.declare_parameter("host_ip", "192.168.0.10")
        # 파지 방식. "classic" = 실측 프로필 시퀀스, "vla" = 학습 정책.
        #
        # 기본값이 classic 인 이유: 시연에서 검증된 경로가 그쪽이고, vla 는
        # vla_inference_node 와 정책 체크포인트가 같이 떠 있어야 동작한다.
        # 알 수 없는 값이 오면 classic 으로 떨어뜨린다 - 오타 하나로 파지
        # 경로가 조용히 바뀌는 것보다 낫다.
        self.declare_parameter("grasp_backend", "classic")

        use_fake_base = self.get_parameter("use_fake_base").value
        use_fake_arm = self.get_parameter("use_fake_arm").value
        use_fake_perception = self.get_parameter("use_fake_perception").value
        use_fake_host = self.get_parameter("use_fake_host").value
        grasp_backend = str(self.get_parameter("grasp_backend").value or "classic").strip()
        if grasp_backend not in ("classic", "vla"):
            self.get_logger().warn(
                f'grasp_backend="{grasp_backend}" 는 모르는 값입니다 - classic 으로 진행합니다')
            grasp_backend = "classic"
        self.get_logger().info(f"파지 백엔드: {grasp_backend}")

        self._estop = threading.Event()
        self._host = (FakeHostLink() if use_fake_host
                      else UdpHostLink(self.get_parameter("host_ip").value,
                                       logger=self.get_logger()))
        # ⚠️ 2026-08-27 발견: LoggedPort.__init__(self, name, delegate, logger)인데
        # 여기 세 호출이 전부 (delegate, "이름", logger) 순서로 뒤바뀌어 있었다 —
        # self._name에 실제 드라이버 객체가, self._delegate에 문자열이 들어가
        # ports.base.stop() 같은 모든 실호출이 `'str' object has no attribute
        # 'stop'`으로 죽었다(mission_orchestrator FSM 크래시의 원인). 도메인
        # pytest 스위트는 이 ROS 전용 파일을 안 건드리므로 여태 안 걸렸다.
        self._ports = BaselinePorts(
            # quiet: `liveness`는 미션 루프가 매 사이클 부르는 폴링이라
            # 로그에 남기면 초당 10줄씩 쌓여 정작 봐야 할 포트 호출을 묻는다.
            # 판정 **결과**는 상태가 바뀔 때 BASE_UNRESPONSIVE 보고로 남는다.
            base=LoggedPort("base", FakeBase() if use_fake_base else Ros2MecanumBase(self),
                            self.get_logger(), quiet={"liveness"}),
            arm=LoggedPort("arm", FakeArm() if use_fake_arm else Ros2ArmDriver(self),
                           self.get_logger()),
            perception=LoggedPort(
                "perception",
                ScriptedPerception() if use_fake_perception else Ros2Perception(self),
                self.get_logger()),
            host=self._host,
            lidar=(FakeLidar() if use_fake_perception else Ros2Lidar(self)),
            estop=self._estop,
            grasp_backend=grasp_backend,
        )

        self._started = 0.0
        self._contacts = 0
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()

    def _on_estop(self, _msg):
        self.get_logger().warn("E-STOP 수신 — FSM을 즉시 중단시킵니다")
        self._estop.set()

    def _publish_state(self, name):
        msg = MissionStateMsg()
        msg.state = name
        msg.contact_count = self._contacts
        msg.elapsed_s = time.monotonic() - self._started
        self._state_pub.publish(msg)

    def _run_forever(self):
        """FSM을 계속 돌린다.

        `BaselineMission.run()`은 DONE에서 끝난다 — Host가 다음 미션을
        시작할 수 있어야 하므로 끝나면 새로 만들어 다시 돈다. 미션의
        시작과 끝을 정하는 것도 Host다."""
        while rclpy.ok():
            # elapsed_s는 **이번 미션**의 경과 시간이다. 예전에는 이 대입이
            # while 바깥에 있어서 노드 기동 이후의 총 시간이 실렸는데, 그러면
            # 미션을 여러 번 도는 지금 구조에서 값이 계속 커지기만 해서
            # "한 사이클에 몇 초 걸렸나"(이 프로젝트의 성공 지표)를 못 읽는다.
            self._started = time.monotonic()
            try:
                for state in BaselineMission(self._ports).run():
                    self._publish_state(state.name)
                    time.sleep(CYCLE_PERIOD_S)
                    if not rclpy.ok():
                        return
            except Exception:                    # noqa: BLE001 -- 실기 루프
                self.get_logger().error(f"FSM 예외:\n{traceback.format_exc()}")
                self._ports.base.stop()
                self._ports.arm.hold_position()
                time.sleep(1.0)


def main(args=None):
    rclpy.init(args=args)
    node = MissionOrchestratorNode()
    executor = MultiThreadedExecutor()
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
