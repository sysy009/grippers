"""파지 게이트의 **하한** — 너무 가까우면 물러난다 (2026-09-08 사용자 지시).

## 왜 생겼나

게이트가 단방향이었다:

    if dist <= mcfg.GRASP_TRIGGER_DIST_M:   # 이 한 줄이 전부
        ...
        self.state = State.GRASP

하한이 없어서 20cm 에 서 있어도 그대로 파지로 넘어갔다. 좌우에는 ±41mm 창이
있는데(grasp_alignment.VLA_PAN_LIMIT_DEG, 학습 분포 1.5σ) 전후에는 아무 창도
없었다.

되돌릴 경로도 없었다 — 물러나는 유일한 길인 GRASP_REPLAN 의 BACKOFF(0.15)는
VLA_GRASP_ONLY=True 가 끊어 놓았다. 한 번 가까워지면 실패 -> 전체 리셋 ->
같은 자리에서 즉시 재파지로 무한히 돌 수 있었다.

## 상한이 왜 같이 필요한가

하한(0.35)과 게이트(0.36) 사이가 **1cm** 뿐이다. 정지 오버슈트가 그보다 크면
물러났다 다시 접근할 때마다 또 걸린다 — 앞뒤로만 오가며 시연이 멎는다. 이
저장소는 그 종류의 멎음을 이미 두 번 겪었다(파지 37회 재시도, NUDGE<->PLACE
154회). 그래서 상한을 다 쓰면 가까운 채로라도 집는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HOST = Path(__file__).resolve().parent.parent / "host"
sys.path.insert(0, str(_HOST))
sys.path.insert(0, str(_HOST / "aruco"))

import mission_config as mcfg                                    # noqa: E402
from mission import MissionFSM, State                            # noqa: E402

from conftest import PiSim                                       # noqa: E402

_TARGET_XY = (1.0, 0.6)


def _approaching(fsm: MissionFSM) -> None:
    fsm.state = State.APPROACH_PIECE
    fsm.target_label = "queen"
    fsm._target_xy = _TARGET_XY
    fsm.dest_xy = (1.271, 1.30)


def _pi_at(dist_m: float) -> PiSim:
    """목표 정면으로 dist_m 떨어진 자리에 세운다 (요는 목표를 향한다)."""
    return PiSim(x=_TARGET_XY[0], y=_TARGET_XY[1] - dist_m, yaw_deg=90.0)


def test_하한보다_가까우면_파지로_안_넘어가고_물러난다():
    fsm = MissionFSM()
    _approaching(fsm)
    link = _pi_at(mcfg.GRASP_MIN_DIST_M - 0.05)

    fsm.step(link.pose(), {}, link)

    assert fsm.state == State.APPROACH_PIECE, "가까운데 파지로 넘어갔다"
    assert link.sent[-1][0] == "back", f"물러나지 않았다: {link.sent[-1][0]}"


def test_하한과_게이트_사이면_평소대로_파지로_넘어간다():
    fsm = MissionFSM()
    _approaching(fsm)
    # 창 한가운데
    link = _pi_at((mcfg.GRASP_MIN_DIST_M + mcfg.GRASP_TRIGGER_DIST_M) / 2.0)

    fsm.step(link.pose(), {}, link)

    assert fsm.state == State.GRASP
    assert link.sent[-1][0] == "stop"


def test_후진_상한을_다_쓰면_가까운_채로라도_집는다():
    """⚠️ 이게 없으면 앞뒤로만 오가다 시연이 멎는다."""
    fsm = MissionFSM()
    _approaching(fsm)
    link = _pi_at(mcfg.GRASP_MIN_DIST_M - 0.05)

    for i in range(mcfg.GRASP_TOO_CLOSE_MAX_BACKOFFS):
        fsm.step(link.pose(), {}, link)
        assert fsm.state == State.APPROACH_PIECE, f"{i + 1}번째에서 넘어가 버렸다"
        assert link.sent[-1][0] == "back"

    # 상한을 다 썼다 — 이제는 가까워도 진행한다.
    fsm.step(link.pose(), {}, link)
    assert fsm.state == State.GRASP, "상한을 다 썼는데도 안 넘어간다 — 무한 후진"


def test_상한은_대상이_바뀔_때만_풀린다():
    """같은 기물을 다시 고르는 경로에서 풀면 상한이 영영 안 찬다 —
    _grasp_fail_tries 가 같은 이유로 같은 자리에서만 풀린다."""
    fsm = MissionFSM()
    _approaching(fsm)
    fsm._too_close_backoffs = mcfg.GRASP_TOO_CLOSE_MAX_BACKOFFS
    link = _pi_at(mcfg.GRASP_MIN_DIST_M - 0.05)

    fsm.step(link.pose(), {}, link)

    assert fsm.state == State.GRASP
    assert fsm._too_close_backoffs == mcfg.GRASP_TOO_CLOSE_MAX_BACKOFFS


def test_하한이_게이트보다_작다():
    """짝이 뒤집히면 게이트가 영영 안 열린다."""
    assert mcfg.GRASP_MIN_DIST_M < mcfg.GRASP_TRIGGER_DIST_M
