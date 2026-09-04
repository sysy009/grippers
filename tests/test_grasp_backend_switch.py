"""GRASP 백엔드 전환 계약 — classic 과 vla 가 갈리는 곳과 갈리지 않는 곳.

핵심은 **갈리지 않는 쪽**이다. 손을 대는 방식만 바뀌고 성공 판정은 그대로여야
한다 — 새 경로가 기존 안전 게이트를 우회하면 그게 가장 위험한 회귀다.
"""

import threading

import pytest
from domain.adapters.fake.fake_arm import FakeArm
from domain.adapters.fake.fake_base import FakeBase
from domain.adapters.fake.fake_host_link import FakeHostLink
from domain.adapters.fake.scripted_perception import ScriptedPerception
from domain.ports.baseline_ports import MissionState, Report
from domain.task.baseline_mission import (
    BaselineApproachState,
    BaselineCarryState,
    BaselineGraspState,
    BaselinePorts,
    LinkWatchdog,
)

EMPTY_LOAD = 0.03
HOLDING_LOAD = 0.14
CREEP_M = 0.02


def _ports(backend, *, arm=None, perception=None, host=None):
    return BaselinePorts(
        base=FakeBase(),
        arm=arm or FakeArm(load_ratio=HOLDING_LOAD),
        # confirm_grasp()=True 여야 두 신호 중 뎁스 쪽이 통과한다.
        perception=perception or ScriptedPerception(label="rook", grasp_confirmed=True),
        host=host or FakeHostLink(),
        lidar=None,
        estop=threading.Event(),
        watchdog=LinkWatchdog(),
        grasp_backend=backend,
    )


def _grasp(ports):
    return BaselineGraspState("rook", CREEP_M).execute(ports)


# ── 갈리는 것 ──────────────────────────────────────────────────────────────


def test_classic은_정책을_부르지_않는다():
    arm = FakeArm(load_ratio=HOLDING_LOAD)
    _grasp(_ports("classic", arm=arm))

    assert arm.vla_calls == []


def test_vla는_정책을_한_번_부르고_미세전진을_하지_않는다():
    """정책은 물체 앞 20cm 에서 뻗는 것까지 배웠다 — creep 은 classic 전용이다."""
    arm = FakeArm(load_ratio=HOLDING_LOAD)
    ports = _ports("vla", arm=arm)

    _grasp(ports)

    assert len(arm.vla_calls) == 1
    assert arm.vla_calls[0][0] == "pick up the rook"
    assert ports.base.creep_forward_calls == []


def test_vla는_턱을_직접_열고_닫지_않는다():
    """개폐도 정책 안에 들어 있다. 여기서 또 명령하면 정책 동작을 덮어쓴다."""
    arm = FakeArm(load_ratio=HOLDING_LOAD)

    _grasp(_ports("vla", arm=arm))

    assert arm.gripper_widths == []


def test_모르는_백엔드_이름은_classic으로_떨어진다():
    """오타 하나로 파지 경로가 조용히 바뀌면 안 된다."""
    arm = FakeArm(load_ratio=HOLDING_LOAD)

    _grasp(_ports("VLA_오타", arm=arm))

    assert arm.vla_calls == []


# ── 갈리지 않는 것 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", ["classic", "vla"])
def test_양쪽_다_CARRY로_끝난다(backend):
    """물체를 문 채 IDLE로 접으면 그리퍼가 라이다를 가린다 — 양쪽 공통이다."""
    arm = FakeArm(load_ratio=HOLDING_LOAD)
    ports = _ports(backend, arm=arm)

    nxt = _grasp(ports)

    assert isinstance(nxt, BaselineCarryState)
    assert ("chess_rook", "carry") in arm.floor_pose_calls


@pytest.mark.parametrize("backend", ["classic", "vla"])
def test_양쪽_다_빈손이면_실패로_되돌아간다(backend):
    """부하 게이트를 vla 가 우회하면 안 된다."""
    host = FakeHostLink()
    ports = _ports(backend, arm=FakeArm(load_ratio=EMPTY_LOAD), host=host)

    nxt = _grasp(ports)

    assert Report.GRASP_FAILED in host.reported_kinds
    assert isinstance(nxt, BaselineApproachState)


@pytest.mark.parametrize("backend", ["classic", "vla"])
def test_양쪽_다_목표가_그대로면_실패다(backend):
    """부하만 보면 턱끼리 문 경우도 통과한다 — 뎁스 신호도 양쪽 다 요구한다."""
    host = FakeHostLink()
    ports = _ports(
        backend,
        arm=FakeArm(load_ratio=HOLDING_LOAD),
        perception=ScriptedPerception(label="rook", grasp_confirmed=False),
        host=host,
    )

    nxt = _grasp(ports)

    assert Report.GRASP_FAILED in host.reported_kinds
    assert isinstance(nxt, BaselineApproachState)


@pytest.mark.parametrize("backend", ["classic", "vla"])
def test_양쪽_다_목표를_먼저_기억한다(backend):
    """confirm_grasp() 가 "있던 물체가 사라졌다"를 보려면 먼저 기억해야 한다."""
    perception = ScriptedPerception(label="rook", grasp_confirmed=True)
    _grasp(_ports(backend, perception=perception))

    assert perception.remembered_cls == "rook"
    assert perception.remember_target_calls == 1


def test_정책_실행이_실패하면_GRASP도_실패한다():
    host = FakeHostLink()
    ports = _ports("vla", arm=FakeArm(load_ratio=HOLDING_LOAD, vla_ok=False), host=host)

    nxt = _grasp(ports)

    assert Report.GRASP_FAILED in host.reported_kinds
    assert isinstance(nxt, BaselineApproachState)


def test_vla_지시문은_라벨로_만들어진다():
    arm = FakeArm(load_ratio=HOLDING_LOAD)
    ports = _ports("vla", arm=arm)
    ports.vla_task_template = "grab the {label} now"

    BaselineGraspState("queen", CREEP_M).execute(ports)

    assert arm.vla_calls[0][0] == "grab the queen now"
