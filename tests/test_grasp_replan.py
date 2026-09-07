"""GRASP_REPLAN — GRASP_ALIGN을 반복해도 안 풀리면 오버헤드 카메라로 크게
다시 세운다 (사용자 지시, 2026-09-02).

## 왜 이 기능이 생겼나

07:12 rook 실기에서 APPROACH_PIECE -> GRASP 전이가 거리만 보고 yaw 는
전혀 확인하지 않아, 회전이 덜 정리된 채(그날은 52도) GRASP 에 들어갔다.
그 결과 Pi 뎁스캠 화각을 완전히 벗어났고, BACK_OFF(3cm 후진)+스윕을 30번
(GRASP_FORCE_AFTER_TRIES) 반복해도 다시 정면에 세우지 못한 채 강제 파지도
실패해 33초간 락업됐다(같은 커밋의 vehicle_link 락업 수정 참고 — 이건
그 위에 쌓인 별개의, "찾았어도 못 고치는" 문제다).

이 파일은 GRASP_ALIGN을 GRASP_REPLAN_AFTER_TRIES 번 반복하면 Host가 목표
에서 물러났다 yaw 를 타이트하게 맞춰 다시 접근하는지, 그리고 그 재접근이
GRASP_FORCE 보다 먼저·더 적은 시도로 걸리는지 확인한다."""

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
from vehicle_link import RE_AIM, GraspCorrection      # noqa: E402

from conftest import PiSim                            # noqa: E402
from test_grasp_force import NearMissPi, MAX_STEPS    # noqa: E402


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

_TARGET_XY = (1.0, 0.6)


def _begin_grasp(fsm: MissionFSM, *, robot_xy=(1.0, 0.6 - mcfg.GRASP_TRIGGER_DIST_M),
                 yaw_deg: float = 90.0) -> PiSim:
    fsm.state = State.GRASP
    fsm.target_label = "rook"
    fsm._target_xy = _TARGET_XY
    fsm.dest_xy = (1.271, 1.30)
    link = NearMissPi(force_outcome="succeed", x=robot_xy[0], y=robot_xy[1],
                      yaw_deg=yaw_deg)
    return link


def _run_until(fsm, link, predicate, max_steps=MAX_STEPS):
    for n in range(1, max_steps + 1):
        fsm.step(link.pose(), {}, link)
        if predicate(fsm):
            return n
    pytest.fail(f"{max_steps} 사이클 안에 조건에 도달하지 못했다 — 상태 {fsm.state.name}")


def test_재계획_문턱_전에는_GRASP_REPLAN을_안_보낸다():
    fsm = MissionFSM()
    link = _begin_grasp(fsm)

    for _ in range(MAX_STEPS):
        fsm.step(link.pose(), {}, link)
        if fsm._align_tries >= mcfg.GRASP_REPLAN_AFTER_TRIES:
            break
        assert fsm.state != State.GRASP_REPLAN
    assert fsm._align_tries >= mcfg.GRASP_REPLAN_AFTER_TRIES


def test_재정렬을_다발로_반복하면_물러났다_타이트한_yaw로_다시_접근한다():
    fsm = MissionFSM()
    link = _begin_grasp(fsm)

    _run_until(fsm, link, lambda f: f.state == State.GRASP_REPLAN)

    assert fsm._replan_tries == 1
    assert fsm._replan_backoff_xy is not None
    # 물러나는 지점은 목표에서 GRASP_TRIGGER_DIST_M 보다 더 멀어야 한다 —
    # 그래야 재접근이 트리거 거리 안쪽에서 다시 yaw 를 잴 여유가 생긴다.
    bx, by = fsm._replan_backoff_xy
    dist_from_target = math.hypot(bx - _TARGET_XY[0], by - _TARGET_XY[1])
    assert dist_from_target > mcfg.GRASP_TRIGGER_DIST_M

    # 물러난 지점에 실제로 도착하면(PiSim 이 명령을 실제로 적분한다)
    # 타이트한 yaw 게이트를 켜고 APPROACH_PIECE 로 돌아간다.
    _run_until(fsm, link, lambda f: f.state == State.APPROACH_PIECE)
    assert fsm._tight_yaw_gate is True


def test_재계획_뒤_재접근은_yaw가_안_맞으면_바로_GRASP로_안_들어간다():
    """트리거 거리 안이라도 _tight_yaw_gate 가 켜져 있으면, yaw 가 맞을
    때까지 GRASP 로 넘어가지 않고 제자리에서 겨눈다."""
    fsm = MissionFSM()
    fsm.state = State.APPROACH_PIECE
    fsm.target_label = "rook"
    fsm._target_xy = _TARGET_XY
    fsm._tight_yaw_gate = True
    # 트리거 거리 안이지만 목표를 정면으로 보고 있지 않다(옆을 보고 있다).
    link = NearMissPi(x=_TARGET_XY[0], y=_TARGET_XY[1] - mcfg.GRASP_TRIGGER_DIST_M / 2,
                      yaw_deg=90.0 + 45.0)

    fsm.step(link.pose(), {}, link)
    assert fsm.state == State.APPROACH_PIECE
    assert not fsm.ready_to_advance
    assert link.sent and link.sent[-1][0] in ("yaw+", "yaw-")
    assert link.sent[-1][1] == "APPROACH_PIECE"

    # 계속 겨누면 결국 허용치 안에 들어와 GRASP 로 넘어간다.
    _run_until(fsm, link, lambda f: f.state == State.GRASP, max_steps=2000)


def test_재계획은_예산만큼만_반복하고_그_뒤엔_강제_파지로_넘어간다():
    fsm = MissionFSM()
    link = _begin_grasp(fsm)

    _run_until(fsm, link, lambda f: f.state == State.CARRY_TO_DEST, max_steps=MAX_STEPS * 3)

    assert fsm._replan_tries <= mcfg.GRASP_REPLAN_MAX_ATTEMPTS
    # NearMissPi 는 force_outcome="succeed" 이므로 결국 강제 파지로 성공한다
    # — 재계획이 문제를 못 풀어도(이 시늉은 rotate만 반복하는 정지 오차라
    # 재계획으로도 안 풀린다) 기존 강제 파지 경로로 안전하게 이어진다는
    # 뜻이다.
    assert link.force_attempts_seen >= 1
