"""GRASP_FORCE — 재정렬을 충분히 반복해도 안 맞으면 한 번 강제로 시도한다
(사용자 지시, 2026-08-31).

grippers 저장소(Pi)의 MissionState.GRASP_FORCE 짝. 여기서는 Host 쪽만 —
_align_tries 가 mission_config.GRASP_FORCE_AFTER_TRIES 를 넘으면 "GRASP"
대신 "GRASP_FORCE" 를 보내는지, 실패하면 재정렬 한 번을 더 거쳐 재시도하고
그마저 실패하면 포기하는지 확인한다. Pi 가 실제로 정렬 창을 건너뛰는지는
grippers 저장소의 test_baseline_mission.py 쪽에서 검증한다 — 여기서는
"Pi 가 그렇게 응답한다면 Host 가 옳게 반응하는가"만 본다.
"""

from __future__ import annotations

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

# 실기 재현과 무관한 순수 정렬 반복이라 임의의 값을 쓴다 — 핵심은 "매번
# 같은 재회전 요구가 온다"는 것뿐이다.
_STUCK_CORRECTION = GraspCorrection(RE_AIM, "턱 폭 밖", lateral_mm=95.0)

# align_tries 1회는 GRASP(1사이클) + GRASP_ALIGN(회전 완료까지 여러 사이클)
# 왕복이라 GRASP_FORCE_AFTER_TRIES(30) 를 채우는 데 수백 사이클이 걸린다.
MAX_STEPS = 4000


class NearMissPi(PiSim):
    """정렬 창 바로 밖에 걸려서 GRASP_ALIGN 을 아무리 반복해도 안 맞는 Pi.

    "GRASP_FORCE" 로 보낸 시도에만 `force_outcome` 을 적용한다 — 그 전의
    평소 "GRASP" 요청에는 항상 같은 재정렬 요구만 돌려준다."""

    def __init__(self, *args, force_outcome: str = "fail", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.force_outcome = force_outcome
        self.force_attempts_seen = 0

    def poll_status(self) -> str:
        last_status = self.sent[-1][1] if self.sent else None
        if last_status == "GRASP_FORCE":
            self.force_attempts_seen += 1
            self.last_correction = None
            return "GRASP_DONE" if self.force_outcome == "succeed" else "FAILED"
        self.last_correction = _STUCK_CORRECTION
        return "BUSY"


def _begin_grasp(fsm: MissionFSM) -> None:
    fsm.state = State.GRASP
    fsm.target_label = "rook"
    fsm._target_xy = (1.0, 0.6)
    fsm.dest_xy = (1.271, 1.30)


def _run_until(fsm, link, predicate, max_steps=MAX_STEPS):
    for n in range(1, max_steps + 1):
        fsm.step(link.pose(), {}, link)
        if predicate(fsm):
            return n
    pytest.fail(f"{max_steps} 사이클 안에 조건에 도달하지 못했다 — 상태 {fsm.state.name}")


def test_강제_문턱_전에는_평소처럼_재정렬만_한다():
    """GRASP_FORCE_AFTER_TRIES 를 채우기 전엔 절대 GRASP_FORCE 를 안 보낸다."""
    fsm = MissionFSM()
    _begin_grasp(fsm)
    link = NearMissPi(force_outcome="succeed")

    for _ in range(MAX_STEPS):
        fsm.step(link.pose(), {}, link)
        if fsm._align_tries >= mcfg.GRASP_FORCE_AFTER_TRIES:
            break
        assert not link.sent or link.sent[-1][1] != "GRASP_FORCE"
    assert fsm._align_tries >= mcfg.GRASP_FORCE_AFTER_TRIES


def test_재정렬을_충분히_반복하면_강제로_시도해서_성공한다():
    fsm = MissionFSM()
    _begin_grasp(fsm)
    link = NearMissPi(force_outcome="succeed")

    _run_until(fsm, link, lambda f: f.state == State.CARRY_TO_DEST)

    assert link.force_attempts_seen == 1
    assert fsm._align_tries >= mcfg.GRASP_FORCE_AFTER_TRIES
    assert fsm.target_label == "rook"   # 성공했으니 그대로 들고 CARRY 로


def test_강제가_실패하면_재정렬_한번_거쳐_다시_강제하고_그래도_안되면_포기한다():
    fsm = MissionFSM()
    _begin_grasp(fsm)
    link = NearMissPi(force_outcome="fail")

    _run_until(fsm, link, lambda f: f.state == State.SEARCH_TARGET)

    assert link.force_attempts_seen == mcfg.GRASP_FORCE_MAX_ATTEMPTS
    assert fsm.target_label is None
    assert len(fsm.skipped) == 1


def test_강제_시도_사이에_재정렬이_최소_한번은_낀다():
    """강제 실패 직후 align_tries 가 그대로면 곧장 또 강제하지 않는다 —
    실패했다고 바로 재도전하면 예산(GRASP_FORCE_MAX_ATTEMPTS)을 순식간에
    다 써서 "재정렬 후 재시도"라는 사용자 지시가 무의미해진다."""
    fsm = MissionFSM()
    _begin_grasp(fsm)
    link = NearMissPi(force_outcome="fail")

    align_tries_at_force: list[int] = []
    seen = 0
    for _ in range(MAX_STEPS):
        fsm.step(link.pose(), {}, link)
        if link.sent and link.sent[-1][1] == "GRASP_FORCE" and link.force_attempts_seen > seen:
            seen = link.force_attempts_seen
            align_tries_at_force.append(fsm._align_tries)
        if fsm.state == State.SEARCH_TARGET:
            break

    assert len(align_tries_at_force) == mcfg.GRASP_FORCE_MAX_ATTEMPTS
    # 두 번째 강제 시점의 align_tries 가 첫 번째보다 커야(=그 사이에 실제
    # 재정렬이 있었어야) 한다.
    assert align_tries_at_force[1] > align_tries_at_force[0]
