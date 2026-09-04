"""ArmDriver 포트 — ROS2를 전혀 모르는 순수 ABC."""

from abc import ABC, abstractmethod

from domain.values import Point3


class ArmDriver(ABC):
    @abstractmethod
    def move_to_floor_pose(self, profile: str, stage: str) -> bool:
        """실측된 수평 바닥 파지 자세로 단계 이동한다.

        ``stage`` 는 ``idle``, ``safe``(145 mm), ``grasp``, ``midpoint`` 또는
        바구니 투하용 ``drop``(195 mm)이다.
        프로필/단계를 지원하지 않거나 하드웨어 이동에 실패하면 ``False``다.
        """

    @abstractmethod
    def move_to_cartesian(self, xyz_m: Point3, down: bool = False) -> bool:
        """손끝을 xyz_m(m)로 이동한다. 도달 불가하면 False.
        그리퍼 개폐는 이 메서드가 하지 않는다 — `set_gripper()` 를 별도로 호출한다.

        **실패(도달 불가 · 서버 부재 · 응답 없음)는 예외가 아니라 `False`.**

        ⚠️ 2026-08-28: 현재 실기 FSM(`baseline_mission.BaselineGraspState`)은
        이 메서드를 안 부른다 — 좌표 기반 파지였던 예전 설계의 흔적이고,
        지금은 실측 프로필/단계 기반인 `move_to_floor_pose()`로 대체됐다.
        real/fake 어댑터와 액션 서버는 계속 살아 있고 테스트도 돌지만,
        미션 경로에서는 죽은 코드다 — 지우지 않은 이유는 손끝 좌표 이동이
        나중에 다시 필요해질 수 있어서다(예: 임의 물체 위치 대응)."""

    @abstractmethod
    def set_gripper(self, width_mm: float) -> None:
        """그리퍼 개구 폭을 width_mm(mm)로 맞춘다.
        ⚠️ 단위가 deg(각도)에서 mm(개구 폭)로 바뀌었다. 서보 각도 변환은
        어댑터(FeetechArm) 내부 캘리브레이션 테이블이 담당한다 (미결 #4 결과 반영).

        **돌려줄 값이 없으므로 실패는 로그로만 남는다.** 다만 조용히 삼켜지지는
        않는다 — 그리퍼가 닫히지 않았으면 뒤이은 `get_load()` 가 빈 채 부하를
        읽어 `GRASP` 가 파지 실패로 판정한다."""

    @abstractmethod
    def get_load(self) -> float:
        """그리퍼(id6) 부하 비율 — **0.0~1.0 으로 정규화된 값**이다.

        ⚠️ 서보 원시값(STS3215 PRESENT_LOAD 는 0~1023)을 그대로 돌려주면 안 된다.
        정규화는 어댑터 뒤편(arm_driver_node)의 몫이다 — 도메인은 서보 레지스터
        범위를 알지 못한다. Fake 는 정규화된 값을, real 은 원시값을 주는 식으로
        계약이 갈라지면 CI는 통과하는데 실기에서만 파지 판정이 항상 실패한다.

        **실패(서비스 부재 · 응답 없음 · 서보 읽기 실패)는 `0.0`** — '빈 채'로
        보아 파지 실패로 판정된다. 부하를 모르는 채로 성공 판정을 내려 물체를
        든 줄 알고 이송하는 것보다 안전하다."""

    @abstractmethod
    def reorient(self, phi_rad: float) -> bool:
        """손목을 장축-수평면 각 φ(rad)로 재조정한다. 정착에 실패하면 False.

        **서버 부재 · 응답 없음도 `False`.**

        ⚠️ 2026-08-28: 실기 서버(`arm_driver_node._execute_reorient`)가 아직
        스텁이다 — 실제 손목 재조정 없이 `settled=True`만 돌려준다. 현재
        실기 FSM도 이 메서드를 안 부른다. move_to_cartesian과 같은 이유로
        남겨 둔, 아직 완성되지 않은 미래용 훅이다."""

    @abstractmethod
    def fold_to_cradle(self) -> bool:
        """팔을 이동용 거치 자세로 접는다. **실패는 `False`.**

        구현(`arm_driver_node._on_fold_to_cradle`)은 서보 부하를 접기 전후로
        확인하는 완성된 로직이고 테스트도 있다(`test_arm_hardware_contract.py`).
        다만 2026-08-28 기준 `baseline_mission`의 실기 FSM은 이 메서드를 안
        부른다 — 수동 도구·향후 이동 단계용으로 남겨 둔 상태다."""

    @abstractmethod
    def offset_base_yaw(self, offset_rad: float) -> bool:
        """servo 1(팔 베이스 요)을 현재 위치에서 offset_rad만큼 돌린다.

        GRASP 하강 **전에** 부르는 좌우 보정이다 — 물체가 턱이 쓸고 갈 영역
        안에 있지만 가운데가 아닐 때 Pi가 스스로 고치는 수단이다(사용자 지시
        2026-08-26). 메카넘 옆걸음이 아니라 이 관절을 쓰는 이유는 베이스의
        속도 데드밴드 때문에 최소 옆걸음이 15mm로 고치려는 오차보다 커서다.

        **한계각을 넘거나 관절 범위를 벗어나면 움직이지 않고 `False`.**
        무리하게 돌리는 것보다 Host에 다시 세워 달라고 하는 편이 싸다."""

    @abstractmethod
    def run_vla_grasp(self, task: str, timeout_sec: float = 0.0,
                      max_step_deg: float = 0.0) -> bool:
        """VLA 정책으로 파지를 수행한다. **실패(서버 부재·오류·취소)는 `False`.**

        ⚠️ `True` 는 "오류 없이 돌았다"이지 **"물체를 집었다"가 아니다.** 정책은
        자기가 끝났는지 알려주지 않으므로 이 호출만으로는 성공을 알 수 없다.
        파지 성공 판정은 호출부가 기존 경로와 똑같이 `get_load()` 와
        `perception.confirm_grasp()` 두 신호로 한다.

        ⚠️ `task` 는 **어느 물체를 집을지 고르지 않는다.** 정책은 그리퍼캠 정면
        중앙에 있는 것을 집는다 — 학습 데이터에서 목표가 항상 중앙이었기
        때문이다(2026-09-04 확인: ACT 는 task 를 입력으로 받지도 않고,
        SmolVLA 는 받지만 효과가 프레임 효과의 10%). 물체 선택은 이 호출
        **전에** 주행부가 목표를 정면에 놓는 것으로 이뤄진다.

        0 이하의 인자는 노드 기본값을 쓰라는 뜻이다."""

    @abstractmethod
    def hold_position(self) -> None:
        """현재 관절 자세를 그대로 유지한다 (E-STOP 시 파지물 낙하 방지용).

        E-STOP 경로다 — **응답을 기다리지 않는다.** 실패해도 돌려줄 값이 없으므로
        로그만 남긴다."""
