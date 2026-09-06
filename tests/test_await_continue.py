"""AWAIT_CONTINUE/AWAIT_COMMAND/IDLE 는 없다 — 원래 RETURN_HOME 버전으로 복귀
(2026-09-04 밤 사용자 지시: "AWAIT 다 없애라고. 원래 RETURN_HOME 있던 버전으로
내놔").

이 기능은 2026-09-02에 "그룹(chess/toy) 하나가 화면에서 다 소진되면 계속할지
정지할지 사람에게 묻는다"는 시연용으로 들어왔다가, 2026-09-04 밤 실기에서
번거롭다는 이유로 통째로 빠졌다. State.AWAIT_CONTINUE/AWAIT_COMMAND/IDLE 과
MissionFSM.on_continue()/on_stop()/submit_next_command() 도 코드에서 같이
지웠으므로, 이 파일은 그 기능이 정말 없어졌는지(그룹이 소진돼도 예외 없이
RETURN_HOME으로 간다)만 확인한다."""

from __future__ import annotations

import pytest

import sys
from pathlib import Path

_HOST = Path(__file__).resolve().parent.parent / "host"
sys.path.insert(0, str(_HOST))
sys.path.insert(0, str(_HOST / "aruco"))

import mission_config as mcfg               # noqa: E402
from mission import MissionFSM, State        # noqa: E402

from conftest import PiSim                    # noqa: E402


# ⚠️ 2026-09-06 사용자 지시로 홈 복귀 기본값이 꺼졌다
# (mission_config.RETURN_HOME_ENABLED). 이 파일의 시험들은 **켜졌을 때의
# 동작**을 못 박는 것이라, 여기서만 명시적으로 켠다 — 기본값이 바뀌었다고
# 검증된 동작의 시험을 지우면, 나중에 다시 켰을 때 아무도 그게 맞는지
# 모른다.
@pytest.fixture(autouse=True)
def _return_home_on(monkeypatch):
    monkeypatch.setattr(mcfg, "RETURN_HOME_ENABLED", True)


MAX_STEPS = 900


def _run_until(fsm: MissionFSM, sim: PiSim, piece_map, states, max_steps=MAX_STEPS):
    """`states` 중 하나에 도달할 때까지 돌린다. 실패하면 예외."""
    for _ in range(max_steps):
        fsm.step(sim.pose(), piece_map, sim)
        if fsm.state in states:
            return fsm
    raise AssertionError(
        f"{max_steps} 사이클 안에 {[s.name for s in states]} 에 못 갔다 — "
        f"현재 {fsm.state.name}")


def test_State에_AWAIT나_IDLE이_더는_없다():
    names = {s.name for s in State}
    assert "AWAIT_CONTINUE" not in names
    assert "AWAIT_COMMAND" not in names
    assert "IDLE" not in names


def test_MissionFSM에_on_continue_on_stop_submit_next_command이_더는_없다():
    fsm = MissionFSM()
    assert not hasattr(fsm, "on_continue")
    assert not hasattr(fsm, "on_stop")
    assert not hasattr(fsm, "submit_next_command")


def test_그룹이_소진돼도_묻지_않고_곧장_RETURN_HOME으로_간다():
    """rook(chess)을 넣었는데 화면에 다른 기물이 하나도 없다 — 예전 같으면
    AWAIT_CONTINUE 로 갔을 상황이지만, 지금은 예외 없이 RETURN_HOME 이다."""
    sim = PiSim()
    fsm = MissionFSM()
    assert fsm.begin_carrying("rook")
    _run_until(fsm, sim, {}, {State.RETURN_HOME})


def test_같은_그룹이_남아있어도_RETURN_HOME으로_간다():
    sim = PiSim()
    fsm = MissionFSM()
    assert fsm.begin_carrying("rook")
    _run_until(fsm, sim, {"knight": [(0.9, 0.9)]}, {State.RETURN_HOME})


def test_RETURN_HOME은_기본_위치에_도착하면_곧장_SEARCH_TARGET으로_간다():
    sim = PiSim()
    fsm = MissionFSM()
    assert fsm.begin_carrying("rook")
    _run_until(fsm, sim, {}, {State.RETURN_HOME})
    _run_until(fsm, sim, {}, {State.SEARCH_TARGET})
