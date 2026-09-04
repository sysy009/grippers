"""Fake 어댑터의 실패 표현이 포트 계약과 일치하는지 고정한다.

**Fake와 real이 같은 상황을 다르게 표현하면 "CI 테스트가 실기 동작을 보장한다"는
Fake 어댑터의 존재 이유가 무너진다.** 이 프로젝트에서 이미 두 번 났던 사고다:

- `ScriptedInterpreter.parse()` 는 `ValueError`, `Ros2CommandInterpreter.parse()` 는
  `None` (PR #9 리뷰 B항)
- `FakeArm.get_load()` 는 0~1 정규화, `Ros2ArmDriver.get_load()` 는 서보 원시값 (PR #136)

둘 다 CI는 초록불인데 실기에서만 깨지는 종류다 — 도메인 테스트는 Fake의 표현만 보기
때문이다. 아래 표가 포트별 실패값의 단일 기준이고, 계약이 다시 갈라지면 여기서 잡힌다.

real 쪽은 `rclpy` 가 있어야 import 되므로 여기서 함께 호출해 비교할 수 없다 —
real 어댑터가 이 표와 같은 값을 돌려주는지는
`tests/test_real_adapter_timeouts.py` 가 AST 정적 검사로 본다.

⚠️ 2026-08-26 팀 확정으로 표가 줄고 늘었다. `BaseDriver`의 좌표 메서드 셋과
`Perception`의 탐색 메서드 셋이 사라졌고(Host가 가져갔다), Host 링크와
라이다 포트가 새로 들어왔다."""

import math

import pytest

from domain.adapters.fake.fake_arm import FakeArm
from domain.adapters.fake.fake_base import FakeBase
from domain.adapters.fake.fake_host_link import FakeHostLink, FakeLidar
from domain.adapters.fake.scripted_interpreter import ScriptedInterpreter
from domain.adapters.fake.scripted_perception import ScriptedPerception
from domain.ports.arm_driver import ArmDriver
from domain.ports.base_driver import BaseDriver
from domain.ports.baseline_ports import HostLink, Lidar
from domain.ports.command_interpreter import CommandInterpreter
from domain.ports.perception import Perception
from domain.values import Point3

_POINT = Point3(x=0.2, y=0.0, z=0.0)

# (포트, 메서드, 실패를 주입한 Fake 호출, 기대 실패값, 포트 docstring에 있어야 할 문구)
FAILURE_CONTRACTS = [
    (
        BaseDriver,
        "creep_forward",
        lambda: FakeBase(creep_ok=False).creep_forward(0.10),
        False,
        "False",
    ),
    (ArmDriver, "move_to_floor_pose",
     lambda: FakeArm(move_ok=False).move_to_floor_pose("chess_rook", "grasp"), False, "False"),
    (ArmDriver, "move_to_cartesian",
     lambda: FakeArm(move_ok=False).move_to_cartesian(_POINT), False, None),
    (ArmDriver, "get_load", lambda: FakeArm(load_ratio=0.0).get_load(), 0.0, None),
    (ArmDriver, "reorient", lambda: FakeArm(reorient_ok=False).reorient(0.0), False, None),
    (ArmDriver, "fold_to_cradle", lambda: FakeArm(fold_ok=False).fold_to_cradle(), False, None),
    (
        # 정책 노드가 없거나 오류로 끝나면 False — GRASP가 실패로 읽어
        # APPROACH로 되돌아간다. ⚠️ 반대로 True는 파지 성공이 아니다.
        ArmDriver,
        "run_vla_grasp",
        lambda: FakeArm(vla_ok=False).run_vla_grasp("pick up the rook"),
        False,
        "`False`",
    ),
    (
        # servo 1이 한계각을 넘는 보정을 거부하면 False — 호출자가 Host에
        # 다시 세워 달라고 넘긴다.
        ArmDriver,
        "offset_base_yaw",
        lambda: FakeArm(yaw_offset_ok=False).offset_base_yaw(0.5),
        False,
        "`False`",
    ),
    (
        # 자기 뎁스캠이 목표를 못 찾으면 None — GRASP 조건 판정이 그걸
        # 미충족으로 읽어 Host에 되돌려준다.
        Perception,
        "identify_target",
        lambda: ScriptedPerception(label=None).identify_target(),
        None,
        "`None`",
    ),
    (
        # 이 항목만 반환값 전체가 아니라 안전 판정 필드를 본다 — 거리는 시나리오마다
        # 다르지만 "모르면 멈춘다"는 contact_risk 하나로 표현된다.
        Perception,
        "monitor_clearance",
        lambda: ScriptedPerception(contact_risk=True).monitor_clearance().contact_risk,
        True,
        "contact_risk=True",
    ),
    (
        Perception,
        "remember_target",
        lambda: ScriptedPerception(target_remembered=False).remember_target("rook"),
        False,
        "`False`",
    ),
    (
        Perception,
        "confirm_grasp",
        lambda: ScriptedPerception(grasp_confirmed=False).confirm_grasp(),
        False,
        "`False`",
    ),
    (
        # 명령이 아직 안 온 것과 "정지하라"는 전혀 다른 사건이다. 이 포트는
        # 앞의 것을 None으로 말하고, 워치독이 그것을 정지로 옮긴다.
        HostLink,
        "latest_command",
        lambda: FakeHostLink([None]).latest_command(),
        None,
        "**None**",
    ),
    (
        # 라이다는 "모르면 실패"다 — 판정하지 않는 쪽이 INSERT를 막아 안전하다.
        Lidar,
        "basket_face",
        lambda: FakeLidar().basket_face().ok,
        False,
        "`ok=False`",
    ),
    (
        CommandInterpreter,
        "parse",
        lambda: ScriptedInterpreter().parse("등록되지 않은 문형"),
        None,
        "`None`",
    ),
]

# PR #137 이 실패 계약 docstring을 추가하는 메서드들. 그 PR이 머지되면 위 표의
# 마지막 칸을 채우고 이 집합을 비운다.
DOCSTRING_PENDING_IN_PR_137 = {
    "ArmDriver.move_to_cartesian",
    "ArmDriver.get_load",
    "ArmDriver.reorient",
    "ArmDriver.fold_to_cradle",
}


def _row_id(row):
    port, method = row[0], row[1]
    return f"{port.__name__}.{method}"


@pytest.mark.parametrize("row", FAILURE_CONTRACTS, ids=_row_id)
def test_fake_returns_the_contracted_failure_value(row):
    """실패를 주입한 Fake가 포트 계약과 **정확히 같은 값**을 돌려준다."""
    _port, _method, call, expected, _marker = row
    actual = call()

    assert actual == expected, f"{_row_id(row)}: 계약은 {expected!r}인데 Fake는 {actual!r}"
    # bool/float/None은 == 로 서로 통과하는 조합이 있다(False == 0.0, 0.0 == False).
    # 계약이 갈라지는 건 대개 '표현'이 다른 경우라 타입까지 본다.
    assert type(actual) is type(
        expected
    ), f"{_row_id(row)}: 값은 같지만 타입이 다르다 — {type(actual)} vs {type(expected)}"


@pytest.mark.parametrize("row", FAILURE_CONTRACTS, ids=_row_id)
def test_port_docstring_states_the_failure_value(row):
    """계약이 코드 어디에도 적혀 있지 않으면 같은 사고가 반복된다 — 포트
    docstring이 실패값을 실제로 말하는지 본다."""
    port, method, _call, _expected, marker = row
    if marker is None:
        assert _row_id(row) in DOCSTRING_PENDING_IN_PR_137
        return
    doc = getattr(port, method).__doc__ or ""
    assert marker in doc, f"{_row_id(row)}: 포트 docstring에 실패값({marker})이 없다"


def test_undocumented_contracts_are_only_the_ones_pr137_adds():
    """docstring이 비어 있는 항목이 조용히 늘어나지 않게 한다."""
    pending = {_row_id(row) for row in FAILURE_CONTRACTS if row[4] is None}
    assert pending == DOCSTRING_PENDING_IN_PR_137


def test_every_port_method_with_a_failure_value_is_covered():
    """실패값이 있는 포트 메서드가 표에서 빠지면 검사에 구멍이 난다.

    실패를 값으로 표현할 수 없는 메서드(반환 타입이 None이거나 실패 개념이 없는 것)는
    제외 목록에 명시한다 — 목록에 없는 메서드가 새로 생기면 여기서 걸린다."""
    no_failure_value = {
        "BaseDriver.apply_velocity",  # 반환값 없음 — cmd_vel은 fire-and-forget
        "BaseDriver.stop",  # E-STOP 경로 — 반환값 없음, 로그만
        "ArmDriver.set_gripper",  # 반환값 없음 — 뒤이은 get_load()가 실패를 드러냄
        "ArmDriver.hold_position",  # E-STOP 경로 — 반환값 없음
        "HostLink.report",  # 반환값 없음 — 안 닿으면 Host 워치독이 판단
        "CommandInterpreter.confirm_phrase",  # 실패해도 빈 문자열, 미션은 계속
    }
    covered = {_row_id(row) for row in FAILURE_CONTRACTS}
    declared = {
        f"{port.__name__}.{name}"
        for port in (BaseDriver, ArmDriver, Perception, CommandInterpreter,
                     HostLink, Lidar)
        for name in port.__abstractmethods__
    }

    assert declared == covered | no_failure_value


# ── 실패가 실제로 흡수되는지 ──────────────────────────────────────────────
#
# 값만 맞추면 "Fake가 계약대로 말한다"까지고, 그 말을 FSM이 **듣는지**는
# 별개다. 아래는 그 두 번째 절반이다.


def test_라이다_관측_실패는_INSERT를_막는다():
    """`ok=False`를 돌려줘도 거리 필드를 읽고 진행하면 계약이 무의미해진다."""
    import threading

    from domain.adapters.fake.fake_host_link import FakeHostLink as _Host
    from domain.ports.baseline_ports import HostCommand, MissionState, Report
    from domain.task.baseline_mission import (
        BaselineCarryState,
        BaselinePorts,
        LinkWatchdog,
    )

    host = _Host([HostCommand(MissionState.INSERT, stop=True)])
    ports = BaselinePorts(
        base=FakeBase(), arm=FakeArm(load_ratio=0.14),
        perception=ScriptedPerception(), host=host, lidar=FakeLidar(),
        estop=threading.Event(), watchdog=LinkWatchdog(),
    )

    nxt = BaselineCarryState("queen").execute(ports)

    assert Report.INSERT_BLOCKED in host.reported_kinds
    assert isinstance(nxt, BaselineCarryState)


def test_목표_식별_실패는_GRASP를_막는다():
    """`identify_target()`의 None을 무시하고 내려가면 그리퍼가 바닥을 긁는다."""
    import threading

    from domain.adapters.fake.fake_host_link import FakeHostLink as _Host
    from domain.ports.baseline_ports import HostCommand, MissionState, Report
    from domain.task.baseline_mission import (
        BaselineApproachState,
        BaselinePorts,
        LinkWatchdog,
    )

    host = _Host([HostCommand(MissionState.GRASP, stop=True)])
    ports = BaselinePorts(
        base=FakeBase(), arm=FakeArm(load_ratio=0.03),
        perception=ScriptedPerception(label=None), host=host, lidar=FakeLidar(),
        estop=threading.Event(), watchdog=LinkWatchdog(),
    )

    nxt = BaselineApproachState().execute(ports)

    assert Report.GRASP_BLOCKED in host.reported_kinds
    assert isinstance(nxt, BaselineApproachState)


def test_미세_전진_실패는_파지를_중단시킨다():
    """`creep_forward()`의 False를 무시하면 물체가 턱 사이에 없는 채로 닫는다.

    ⚠️ 2026-08-29 순서 변경 전에는 여기서 `arm.floor_pose_calls == []`를
    확인했다 — 전진이 팔보다 먼저였으므로 전진이 실패하면 팔은 한 번도 안
    움직였다. 이제 전진은 **팔이 내려가 그리퍼가 열린 뒤**라, 전진이
    실패하는 시점에 팔은 이미 grasp 자세에 있다. 확인해야 할 것은 "팔이 안
    움직였다"가 아니라 **"닫지 않고 멈췄다"**로 바뀐다."""
    import threading

    from domain.adapters.fake.fake_host_link import FakeHostLink as _Host
    from domain.ports.baseline_ports import Report
    from domain.task.baseline_mission import (
        BaselineApproachState,
        BaselineGraspState,
        BaselinePorts,
        LinkWatchdog,
    )

    host = _Host()
    arm = FakeArm(load_ratio=0.03)
    ports = BaselinePorts(
        base=FakeBase(creep_ok=False), arm=arm,
        perception=ScriptedPerception(), host=host, lidar=FakeLidar(),
        estop=threading.Event(), watchdog=LinkWatchdog(),
    )

    nxt = BaselineGraspState("queen", 0.02).execute(ports)

    assert Report.GRASP_FAILED in host.reported_kinds
    assert isinstance(nxt, BaselineApproachState)

    stages = [stage for _profile, stage in arm.floor_pose_calls]
    assert stages == ["safe", "grasp", "recover_idle"], (
        f"전진 실패 뒤의 경로가 틀렸다: {stages}\n"
        "  · midpoint/carry 로 가면 안 된다(물체가 턱 사이에 없다)\n"
        "  · 팔을 바닥에 둔 채 끝내도 안 된다(Host 가 곧 주행을 지시한다)")
    # 물체가 턱 사이에 안 들어왔으므로 닫으면 안 된다. 여는 폭은 내려가기
    # 전에 이미 나갔으므로 그 한 번만 있어야 한다.
    assert len(arm.gripper_widths) == 1, (
        f"전진이 실패했는데 그리퍼를 또 움직였다: {arm.gripper_widths}")


def test_파지_실패는_팔을_바닥에_두고_끝내지_않는다():
    """실패 뒤 Host 는 곧바로 주행을 지시한다 — 그때 팔이 바닥에 있으면
    그리퍼가 바닥과 물체를 가로질러 쓸린다.

    파지 경로의 실패는 대부분 팔이 **이미 내려간 뒤** 난다(전진·닫기·
    들어올리기). 실기로 검증된 도구들은 전부 recover_idle 로 팔을 올린다
    (tools/grasp_test_console.recover_to_idle) — FSM 만 안 하고 있었다.
    """
    import threading

    from domain.adapters.fake.fake_host_link import FakeHostLink as _Host
    from domain.ports.baseline_ports import Report
    from domain.task.baseline_mission import (
        BaselineApproachState,
        BaselineGraspState,
        BaselinePorts,
        LinkWatchdog,
    )

    host = _Host()
    # 부하가 안 오르는 팔 — 닫았는데 아무것도 안 물린 경우다.
    arm = FakeArm(load_ratio=0.03)
    ports = BaselinePorts(
        base=FakeBase(), arm=arm, perception=ScriptedPerception(),
        host=host, lidar=FakeLidar(), estop=threading.Event(),
        watchdog=LinkWatchdog(),
    )

    nxt = BaselineGraspState("queen", 0.030).execute(ports)

    assert Report.GRASP_FAILED in host.reported_kinds
    assert isinstance(nxt, BaselineApproachState)
    stages = [stage for _profile, stage in arm.floor_pose_calls]
    assert stages[-1] == "recover_idle", (
        f"팔을 바닥에 둔 채 Host 에 돌려줬다: {stages}")


def test_복구도_실패하면_붙잡고_사람에게_알린다():
    """복구 경로가 원래 실패를 덮으면 안 된다 — 진짜 원인이 로그에서 묻힌다."""
    import threading

    from domain.adapters.fake.fake_host_link import FakeHostLink as _Host
    from domain.ports.baseline_ports import Report
    from domain.task.baseline_mission import (
        BaselineGraspState,
        BaselinePorts,
        LinkWatchdog,
    )

    class StuckArm(FakeArm):
        """내려가기는 하는데 복구는 못 하는 팔."""

        def move_to_floor_pose(self, profile: str, stage: str) -> bool:
            ok = super().move_to_floor_pose(profile, stage)
            return False if stage == "recover_idle" else ok

    host = _Host()
    arm = StuckArm(load_ratio=0.03)
    ports = BaselinePorts(
        base=FakeBase(), arm=arm, perception=ScriptedPerception(),
        host=host, lidar=FakeLidar(), estop=threading.Event(),
        watchdog=LinkWatchdog(),
    )

    BaselineGraspState("queen", 0.030).execute(ports)

    assert arm.hold_calls > 0, "복구에 실패했으면 최소한 붙잡아야 한다"
    failed = [detail for kind, _s, detail, _f in host.reports
              if kind == Report.GRASP_FAILED]
    assert failed and "수동 정렬" in failed[-1], (
        f"팔이 어디 있는지 사람에게 안 알렸다: {failed}")
