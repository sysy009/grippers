"""FakeArm — ArmDriver 포트의 테스트 더블. 하드웨어·ROS2 없이 도메인 FSM을 검증한다.
기본값은 전부 '성공'이며, 생성자 인자로 실패 시나리오를 주입한다."""

from domain.ports.arm_driver import ArmDriver
from domain.values import Point3

# 실측 기반 기본값 (2026-08-18, n=25, 정착 후 · 원시값 절대값 / 1023).
# get_load()는 포트 계약상 0~1 정규화 비율이다(domain/ports/arm_driver.py).
# 테스트가 1.0 같은 도달 불가능한 값을 쓰면 임계값이 실기와 어긋나 있어도
# 초록불이 나므로, Fake도 실측 분포 안의 값을 쓴다.
LOAD_EMPTY = 0.03  # 빈 채 / 파지 실패(놓침) 0.027~0.031
LOAD_HOLDING = 0.14  # 가베(정육면체) 0.137 — 5/5 일관


class FakeArm(ArmDriver):
    def __init__(
        self,
        move_ok: bool = True,
        yaw_offset_ok: bool = True,
        reorient_ok: bool = True,
        fold_ok: bool = True,
        vla_ok: bool = True,
        load_ratio: float | list[float] = LOAD_HOLDING,
    ):
        self._move_ok = move_ok
        self._yaw_offset_ok = yaw_offset_ok
        self._reorient_ok = reorient_ok
        self._fold_ok = fold_ok
        self._vla_ok = vla_ok
        # get_load()는 GRASP(높을수록 성공)과 HANDOVER(낮을수록 성공)가 정반대
        # 의미로 같이 쓴다 — 상수 하나로는 두 상태를 동시에 성공시킬 수 없어
        # ScriptedPerception.script처럼 호출 순서대로 값을 반환하고, 소진되면
        # 마지막 값을 반복한다. 스칼라를 주면 항상 그 값(기존 동작과 동일)이다.
        self._load_ratios = (
            [load_ratio] if isinstance(load_ratio, (int, float)) else list(load_ratio)
        )
        self._load_call_count = 0
        self.move_calls = []
        self.floor_pose_calls = []
        self.gripper_widths = []
        self.yaw_offsets = []
        # 붙잡기는 안전 경로다 — 복구가 실패했을 때 최소한 이건 불렸는지
        # 테스트가 확인할 수 있어야 한다(2026-08-29).
        self.hold_calls = 0
        # VLA 백엔드가 실제로 불렸는지 / 어떤 지시문으로 불렸는지.
        # classic 과 vla 를 가르는 유일한 관측 지점이라 테스트가 본다.
        self.vla_calls = []

    def run_vla_grasp(self, task: str, timeout_sec: float = 0.0,
                      max_step_deg: float = 0.0) -> bool:
        self.vla_calls.append((task, timeout_sec, max_step_deg))
        return self._vla_ok

    def move_to_floor_pose(self, profile: str, stage: str) -> bool:
        self.floor_pose_calls.append((profile, stage))
        return self._move_ok

    def move_to_cartesian(self, xyz_m: Point3, down: bool = False) -> bool:
        self.move_calls.append((xyz_m, down))
        return self._move_ok

    def set_gripper(self, width_mm: float) -> None:
        self.gripper_widths.append(width_mm)

    def get_load(self) -> float:
        idx = min(self._load_call_count, len(self._load_ratios) - 1)
        self._load_call_count += 1
        return self._load_ratios[idx]

    def reorient(self, phi_rad: float) -> bool:
        return self._reorient_ok

    def fold_to_cradle(self) -> bool:
        return self._fold_ok

    def offset_base_yaw(self, offset_rad: float) -> bool:
        """`yaw_offset_ok=False`로 한계각 초과·관절 범위 밖 거부를 주입한다."""
        self.yaw_offsets.append(offset_rad)
        return self._yaw_offset_ok

    def hold_position(self) -> None:
        self.hold_calls += 1
