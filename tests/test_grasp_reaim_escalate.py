"""GRASP_ALIGN 의 ROTATE(RE_AIM) 미수렴 escalation (사용자 지시, 2026-09-01).

2026-09-01 실기: 좌우 오차가 -69mm -> +25mm -> -63mm 로 165초 동안 수렴하지
않고 부호만 뒤집혔다 — GRASP_READY 에 한 번도 못 이르렀다. 같은 자리에서
제자리 회전(GRASP_ALIGN_YAW_STEP_DEG, 고정 3도)만 반복하면 그리퍼 끝 좌우
위치가 거리에 반비례해 민감하게 튀어 오버슈트하기 쉽다는 것이 원인으로
지목됐다.

여기서 확인하는 것은 두 가지다.

  * 같은 방향(RE_AIM)이 GRASP_REAIM_ESCALATE_AFTER_TRIES 회 넘게 연속되면
    Host 가 더 돌리지 않고 BACK_OFF(후진 한 걸음)로 바꾸는가.
  * 그 후진은 **진짜** BACK_OFF(뎁스캠이 목표를 못 봄)가 아니므로, 후진
    뒤 좌우 스윕(GRASP_SWEEP_*)을 돌지 않고 곧장 GRASP 로 돌아가는가.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HOST = Path(__file__).resolve().parent.parent / "host"
sys.path.insert(0, str(_HOST))
sys.path.insert(0, str(_HOST / "aruco"))

from mission import MissionFSM, State                       # noqa: E402
from vehicle_link import BACK_OFF, RE_AIM, GraspCorrection   # noqa: E402

from conftest import PiSim                                   # noqa: E402

import pytest                                                # noqa: E402
import mission_config as mcfg                                # noqa: E402


# ⚠️ 2026-09-07 사용자 지시로 mission_config.VLA_GRASP_ONLY 기본값이 True 가
# 됐다. 그러면 Host 의 파지 재시도 사다리(GRASP_ALIGN·GRASP_REPLAN·
# GRASP_FORCE·재시도 상한)가 통째로 꺼진다 — 실패하면 상태를 리셋하고
# SEARCH_TARGET 부터 다시 할 뿐이다.
#
# 이 파일의 시험들은 **그 사다리가 켜졌을 때의 계약**을 못 박는 것이라
# 여기서만 명시적으로 되살린다.
@pytest.fixture(autouse=True)
def _grasp_ladder_on(monkeypatch):
    monkeypatch.setattr(mcfg, "VLA_GRASP_ONLY", False)

# 실기 재현과 무관한 순수 반복이라 임의의 값을 쓴다 — 핵심은 "매번 같은
# 재회전 요구가 온다"는 것뿐이다(test_grasp_force.py 의 _STUCK_CORRECTION
# 과 같은 이유).
_STUCK_CORRECTION = GraspCorrection(RE_AIM, "턱 폭 밖", lateral_mm=95.0)

MAX_STEPS = 4000


class StuckRotatePi(PiSim):
    """언제나 같은 재회전 요구만 돌려주는 Pi."""

    def poll_status(self) -> str:
        self.last_correction = _STUCK_CORRECTION
        return "BUSY"


def _begin_grasp(fsm: MissionFSM) -> None:
    fsm.state = State.GRASP
    fsm.target_label = "rook"
    fsm._target_xy = (1.0, 0.6)
    fsm.dest_xy = (1.271, 1.30)


def test_회전_보정이_문턱_넘게_연속되면_후진으로_전환한다():
    fsm = MissionFSM()
    _begin_grasp(fsm)
    link = StuckRotatePi()

    for _ in range(MAX_STEPS):
        fsm.step(link.pose(), {}, link)
        if (fsm.state == State.GRASP_ALIGN and fsm._align is not None
                and fsm._align.kind == BACK_OFF):
            break
    else:
        raise AssertionError("연속 회전 보정이 후진으로 전환되지 않았다")

    assert fsm._align_reaim_backoff is True


def test_회전_미수렴으로_전환된_후진은_스윕을_돌지_않는다():
    fsm = MissionFSM()
    _begin_grasp(fsm)
    link = StuckRotatePi()

    reached_backoff = False
    back_to_grasp = False
    for _ in range(MAX_STEPS):
        fsm.step(link.pose(), {}, link)
        if (fsm.state == State.GRASP_ALIGN and fsm._align is not None
                and fsm._align.kind == BACK_OFF):
            reached_backoff = True
        if reached_backoff:
            # 스윕이 돌면 이 필드가 "left"/"right" 로 잡힌다 — 회전
            # 미수렴으로 빌려 쓴 후진은 목표를 여전히 보고 있으므로
            # 절대 스윕에 들어가면 안 된다(진짜 BACK_OFF 와의 차이).
            assert fsm._align_sweep_stage is None, (
                "회전 미수렴 후진에서 좌우 스윕이 돌았다 — "
                "_align_reaim_backoff 가드가 빠졌다")
            if fsm.state == State.GRASP:
                back_to_grasp = True
                break

    assert reached_backoff, "연속 회전 보정이 후진으로 전환되지 않았다"
    assert back_to_grasp, "후진 뒤 GRASP 로 돌아오지 못했다"
