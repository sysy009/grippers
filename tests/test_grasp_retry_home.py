"""파지 재시도 상한(GRASP_FAIL_MAX_RETRIES)과 포기 후 기본 위치 복귀
(RETURN_HOME) — 사용자 지시, 2026-09-01.

## 배경

정렬 문제(GRASP_BLOCKED)는 _align_tries/GRASP_FORCE_* 가 이미 상한을
관리한다(test_grasp_force.py). 그런데 "조건은 다 맞았는데 팔을 내려도
매번 놓치는" 순수 물리적 실패(poll_status()=="FAILED", 강제 아님)는
예전에 상한이 아예 없었다 — mission.py 가 그 값을 읽고도 아무 것도
안 해서, Host 는 다음 사이클에도 그대로 "GRASP" 를 다시 보내고 Pi 는
파지 시퀀스 전체를 무한 재시도할 수 있었다.

또한 포기(_skip_target) 뒤에는 SEARCH_TARGET 으로 곧장 돌아가지 않고
mcfg.DEFAULT_HOME_XY 로 먼저 복귀한다 — 실패한 자리(기물 코앞이거나
이상한 각도)에 그대로 남지 않고 매번 같은 예측 가능한 자리에서 다음
탐색을 시작하기 위함이다.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_HOST = Path(__file__).resolve().parent.parent / "host"
sys.path.insert(0, str(_HOST))
sys.path.insert(0, str(_HOST / "aruco"))

import mission_config as mcfg                       # noqa: E402
from mission import MissionFSM, State                # noqa: E402

from conftest import PiSim                            # noqa: E402


# ⚠️ 2026-09-07 사용자 지시로 mission_config.VLA_GRASP_ONLY 기본값이 True 가
# 됐다. 그러면 Host 의 파지 재시도 사다리(GRASP_ALIGN·GRASP_REPLAN·
# GRASP_FORCE·재시도 상한)가 통째로 꺼진다 — 실패하면 상태를 리셋하고
# SEARCH_TARGET 부터 다시 할 뿐이다.
#
# 이 파일의 시험들은 **그 사다리가 켜졌을 때의 계약**을 못 박는 것이라
# 여기서만 명시적으로 되살린다. 기본값이 바뀌었다고 검증된 동작의 시험을
# 지우면, 나중에 다시 켰을 때 아무도 그게 맞는지 모른다(같은 저장소의
# test_grasp_retry_home.py 가 RETURN_HOME_ENABLED 에 쓴 것과 같은 방식).
@pytest.fixture(autouse=True)
def _grasp_ladder_on(monkeypatch):
    monkeypatch.setattr(mcfg, "VLA_GRASP_ONLY", False)


# ⚠️ 2026-09-06 사용자 지시로 홈 복귀 기본값이 꺼졌다
# (mission_config.RETURN_HOME_ENABLED). 이 파일의 시험들은 **켜졌을 때의
# 동작**을 못 박는 것이라, 여기서만 명시적으로 켠다 — 기본값이 바뀌었다고
# 검증된 동작의 시험을 지우면, 나중에 다시 켰을 때 아무도 그게 맞는지
# 모른다.
@pytest.fixture(autouse=True)
def _return_home_on(monkeypatch):
    monkeypatch.setattr(mcfg, "RETURN_HOME_ENABLED", True)


MAX_STEPS = 500


class AlwaysFailPi(PiSim):
    """GRASP(강제 아님) 요청마다 정렬 문제 없이 순수 물리적 실패로만
    답하는 Pi. GRASP_FORCE 로는 절대 안 넘어간다 — 이 시늉의 목적은
    GRASP_FAIL_MAX_RETRIES 경로만 켜는 것이라, force 로 새는지는
    test_grasp_force.py 쪽 책임이다."""

    def poll_status(self) -> str:
        last_status = self.sent[-1][1] if self.sent else None
        if last_status in ("GRASP", "GRASP_FORCE"):
            self.last_correction = None   # 정렬 문제가 아니다 — 보정 없음
            return "FAILED"
        return "IDLE"


def _begin_grasp(fsm: MissionFSM) -> None:
    fsm.state = State.GRASP
    fsm.target_label = "rook"
    fsm._target_xy = (1.0, 0.6)
    fsm.dest_xy = (1.271, 1.30)


#: 파지 실패 재시도는 상태를 리셋하고 SEARCH_TARGET 부터 다시 한다
#: (2026-09-06). 그러니 재시도가 진행되려면 그 기물이 화면에 **계속 보여야**
#: 한다 — 빈 지도({})를 주면 리셋 뒤 찾을 대상이 없어 SEARCH_TARGET 에서
#: 그냥 대기한다(그게 "기물이 없을 때"의 정상 동작이다).
_ROOK_VISIBLE = {"rook": [(1.0, 0.6)]}


def _run_until(fsm, link, predicate, max_steps=MAX_STEPS, piece_map=_ROOK_VISIBLE):
    for n in range(1, max_steps + 1):
        fsm.step(link.pose(), piece_map, link)
        if predicate(fsm):
            return n
    pytest.fail(f"{max_steps} 사이클 안에 조건에 도달하지 못했다 — 상태 {fsm.state.name}")


# ── GRASP_FAIL_MAX_RETRIES ──────────────────────────────────────────────


def test_상한_전_실패는_상태를_리셋하고_처음부터_다시_찾는다():
    """⚠️ 2026-09-06 밤에 바뀌었다 — 예전에는 GRASP 에 머물러 그 자리에서
    다시 시도했다.

    사용자 지시가 두 번에 걸쳐 좁혀졌다. 먼저 "재시도 사이에 물체 위치를
    다시 찾게 approach 상태로", 그다음 "approach 로 가는 것보다 상태를
    아예 reset 하는 게 더 좋겠다".

    APPROACH_PIECE 로만 되돌리면 대상 라벨과 정렬·강제파지 카운터가 실패한
    시도의 상태를 그대로 물고 간다. SEARCH_TARGET 부터 다시 하면 기물
    지도에서 지금 보이는 것으로 대상을 새로 고르고 카운터가 0 에서
    시작한다."""
    fsm = MissionFSM()
    _begin_grasp(fsm)
    link = AlwaysFailPi()

    fsm.step(link.pose(), {}, link)

    assert fsm.state == State.SEARCH_TARGET
    # ⚠️ 실패 횟수만은 살아남아야 한다 — 지우면 실패-리셋-실패로 영원히 돈다.
    assert fsm._grasp_fail_tries == 1
    # 나머지는 초기화된다.
    assert fsm.target_label is None
    assert fsm._grasp_yaw_latched is None
    assert fsm._forcing_grasp is False
    # 아직 포기가 아니므로 skipped 에 넣지 않는다 — 넣으면 SEARCH_TARGET 이
    # 같은 기물을 다시 못 고른다.
    assert fsm.skipped == []


def test_반복적_물리_실패는_상한에서_포기하고_기본_위치로_향한다():
    fsm = MissionFSM()
    _begin_grasp(fsm)
    link = AlwaysFailPi()

    _run_until(fsm, link, lambda f: f.state == State.RETURN_HOME)

    # ⚠️ 상수는 "재시도 횟수"이므로 포기까지의 실패 횟수는 그보다 하나 많다
    # (첫 시도 + 재시도 N회). 2026-09-06 이전에는 비교가 `>=` 라 1 일 때
    # 재시도가 0회였다 — 이름·주석과 어긋나 있었다.
    assert fsm._grasp_fail_tries == mcfg.GRASP_FAIL_MAX_RETRIES + 1
    assert fsm.target_label is None
    assert len(fsm.skipped) == 1


def test_강제_시도_중_실패는_일반_상한을_안_건드린다():
    """GRASP_FORCE 실패는 그 나름의 상한(GRASP_FORCE_MAX_ATTEMPTS)이 있다
    — 여기 새로 넣은 GRASP_FAIL_MAX_RETRIES 카운터와 겹쳐 세면 안 된다."""
    from vehicle_link import RE_AIM, GraspCorrection

    class NearMissThenForceFail(PiSim):
        def poll_status(self) -> str:
            last_status = self.sent[-1][1] if self.sent else None
            if last_status == "GRASP_FORCE":
                self.last_correction = None
                return "FAILED"
            self.last_correction = GraspCorrection(RE_AIM, "턱 폭 밖", lateral_mm=95.0)
            return "BUSY"

    fsm = MissionFSM()
    _begin_grasp(fsm)
    link = NearMissThenForceFail()

    for _ in range(4000):
        fsm.step(link.pose(), {}, link)
        if fsm.state == State.RETURN_HOME:
            break
    else:
        pytest.fail("RETURN_HOME 에 도달하지 못했다")

    # 강제 실패 경로로 포기했다 — 일반 재시도 카운터는 안 쌓였어야 한다.
    assert fsm._grasp_fail_tries == 0


# ── RETURN_HOME ──────────────────────────────────────────────────────────


def test_RETURN_HOME_은_기본_위치까지_주행한_뒤_SEARCH_TARGET_으로_돌아간다():
    fsm = MissionFSM()
    fsm.state = State.RETURN_HOME
    link = PiSim(x=1.0, y=0.6, yaw_deg=90.0)

    _run_until(fsm, link, lambda f: f.state == State.SEARCH_TARGET, max_steps=2000)

    dist = math.hypot(link.x - mcfg.DEFAULT_HOME_XY[0],
                      link.y - mcfg.DEFAULT_HOME_XY[1])
    assert dist <= mcfg.HOME_ARRIVE_TOL_M


def test_기본_위치는_주행_격자_안에_있다():
    """DRIVE_AREA_Y 밖(예: 델타의 DELIVER_HERE_XY=0.200)을 목표로 주면
    GridPathPlanner 가 격자 경계에서 멈추고, 그 잔여 거리가 HOME_ARRIVE_
    TOL_M 언저리에 우연히 걸치는 조합에서만 "도착"으로 잘못 판정된다 —
    다른 조합에서는 영영 RETURN_HOME 에 갇힌다(2026-09-01 실측으로 확인).
    이 값 자체가 항상 격자 안에 있어야 그 위험이 구조적으로 없다."""
    y0, y1 = mcfg.DRIVE_AREA_Y
    assert y0 <= mcfg.DEFAULT_HOME_XY[1] <= y1


def test_기본_위치_복귀는_여러_시작_지점에서_모두_도착한다():
    """격자 경계 근처라 시작 각도에 따라 갇힐 수 있다 — 여러 방향에서
    확인한다."""
    starts = [(1.0, 0.6, 90.0), (0.2, 1.0, 0.0), (1.6, 1.2, 180.0),
             (0.35, 0.35, -45.0)]
    for x, y, yaw in starts:
        fsm = MissionFSM()
        fsm.state = State.RETURN_HOME
        link = PiSim(x=x, y=y, yaw_deg=yaw)
        _run_until(fsm, link, lambda f: f.state == State.SEARCH_TARGET,
                  max_steps=2000)
