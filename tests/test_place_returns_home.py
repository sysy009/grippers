"""PLACE 완료 후 SEARCH_TARGET이 아니라 RETURN_HOME으로 간다 (시연용, 2026-09-02).

## 왜

바구니 바로 앞은 매번 접근 각도·거리가 조금씩 다른 자리다. PLACE가 끝나자마자
그 자리에서 SEARCH_TARGET을 시작하면 매 라운드가 다른 자리에서 시작돼
시연이 매번 다르게 보인다. `_skip_target`(기물을 포기했을 때)이 이미 쓰던
"RETURN_HOME을 한 번 거쳐 항상 같은 자리에서 다음 라운드를 시작한다"는
원칙을 PLACE 완료 경로에도 그대로 적용한다.
"""

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

# 2026-09-02~09-04 사이 한동안 AWAIT_CONTINUE(그룹이 화면에서 다 소진되면
# RETURN_HOME 대신 사람에게 "계속할까요?"를 묻는 기능)가 있어서, 이 파일의
# 테스트는 그 판정을 피하려고 knight(같은 chess 그룹)가 남아 있는 피지도를
# 줬었다. 2026-09-04 밤 사용자 지시로 AWAIT_CONTINUE 가 통째로 없어져
# PLACE 완료는 언제나 RETURN_HOME 이므로 지금은 이 값이 굳이 필요하지
# 않지만, 있어도 결과가 달라지지 않으므로(다른 그룹 기물이 화면에 있는
# 상태를 그대로 재현) 그대로 둔다.
_OTHER_CHESS_PIECE_REMAINS = {"knight": [(0.9, 0.9)]}


def test_PLACE_완료는_SEARCH_TARGET이_아니라_RETURN_HOME으로_간다():
    sim = PiSim()
    fsm = MissionFSM()
    assert fsm.begin_carrying("rook")

    # PLACE -> NUDGE_BOX 보정 왕복(정상 동작)과 진짜 완료를 구분해야 한다 —
    # "직전이 PLACE였고 지금이 RETURN_HOME"인 순간만 완료로 본다
    # (tests/test_basket_close_loop.py 의 _run_to_place_done 과 같은 이유).
    was_place = False
    for _ in range(MAX_STEPS):
        was_place = fsm.state == State.PLACE
        fsm.step(sim.pose(), _OTHER_CHESS_PIECE_REMAINS, sim)
        if was_place and fsm.state == State.RETURN_HOME:
            break
        if was_place and fsm.state not in (State.PLACE, State.NUDGE_BOX):
            raise AssertionError(
                f"PLACE에서 예상 밖의 상태({fsm.state.name})로 넘어갔다")
    else:
        raise AssertionError("PLACE가 RETURN_HOME으로 끝나지 않았다")


def test_RETURN_HOME을_거쳐_결국_SEARCH_TARGET에_도착한다():
    """중간에 한 번 쉬어 가는 것뿐, 다음 라운드로 계속 이어져야 한다."""
    sim = PiSim()
    fsm = MissionFSM()
    assert fsm.begin_carrying("rook")

    for n in range(1, MAX_STEPS + 1):
        fsm.step(sim.pose(), _OTHER_CHESS_PIECE_REMAINS, sim)
        if fsm.state == State.SEARCH_TARGET:
            break
    else:
        raise AssertionError(f"{MAX_STEPS} 사이클 안에 SEARCH_TARGET에 못 갔다 — "
                              f"상태 {fsm.state.name}")


def test_RETURN_HOME_도착지는_DEFAULT_HOME_XY다():
    """다음 라운드가 항상 같은 자리에서 시작된다는 것을 좌표로 고정한다."""
    sim = PiSim()
    fsm = MissionFSM()
    assert fsm.begin_carrying("rook")

    for _ in range(MAX_STEPS):
        fsm.step(sim.pose(), _OTHER_CHESS_PIECE_REMAINS, sim)
        if fsm.state == State.SEARCH_TARGET:
            break
    else:
        raise AssertionError("SEARCH_TARGET에 못 갔다")

    dist = ((sim.x - mcfg.DEFAULT_HOME_XY[0]) ** 2
            + (sim.y - mcfg.DEFAULT_HOME_XY[1]) ** 2) ** 0.5
    assert dist <= mcfg.HOME_ARRIVE_TOL_M
