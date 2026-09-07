"""파지 시도 이력이 있으면 BACK_OFF에도 물러나지 않는다 (2026-09-02 실기).

## 왜 이 기능이 생겼나

09-02 실기: rook이 8회 연속 "들어 올리지 못함"(부하 미달)으로 실패한 뒤,
Pi가 "뎁스 카메라가 정면에서 목표를 찾지 못했다"(BACK_OFF)를 반복 보고했다.
GRASP_ALIGN을 3회 반복하자 GRASP_REPLAN이 발동해 목표에서 크게 물러났다
회전했는데, 사용자가 실기로 직접 확인한 바로는 **그 시점에 rook이 이미
그리퍼 사이에 들어와 파지돼 있었다** — 물체가 사라진 게 아니라 뎁스캠
화각/거리 판독이 잡힌 물체 때문에 어긋난 것이었다. 큰 후진+회전이 헐겁게
물린 rook을 떨어뜨려 미션이 통째로 어긋났다.

그래서 실제 파지 시도(`_grasp_fail_tries > 0`)가 이미 있었던 뒤에 오는
BACK_OFF는, "물러나면 나아진다"는 전제(corrections.from_grasp_precondition
문서 참고 — 원래는 시도 전 상태를 전제한 설계다) 자체가 깨졌을 수 있다고
보고 차체를 움직이지 않는다."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HOST = Path(__file__).resolve().parent.parent / "host"
sys.path.insert(0, str(_HOST))
sys.path.insert(0, str(_HOST / "aruco"))

import mission_config as mcfg                                  # noqa: E402
from mission import MissionFSM, State                            # noqa: E402
from vehicle_link import BACK_OFF, RE_AIM, GraspCorrection        # noqa: E402

from conftest import PiSim                                       # noqa: E402


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


class FixedCorrectionPi(PiSim):
    """항상 같은 보정을 돌려주는 Pi — 물러나는지 아닌지만 본다."""

    def __init__(self, *args, correction: GraspCorrection, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._correction = correction

    def poll_status(self) -> str:
        self.last_correction = self._correction
        return "BUSY"


def _begin_grasp(fsm: MissionFSM, *, grasp_fail_tries: int) -> None:
    fsm.state = State.GRASP
    fsm.target_label = "rook"
    fsm._target_xy = _TARGET_XY
    fsm.dest_xy = (1.271, 1.30)
    fsm._grasp_fail_tries = grasp_fail_tries


def test_파지_시도_전이면_BACK_OFF에_평소대로_물러난다():
    fsm = MissionFSM()
    _begin_grasp(fsm, grasp_fail_tries=0)
    link = FixedCorrectionPi(
        correction=GraspCorrection(BACK_OFF, "정면에서 못 찾음"),
        x=_TARGET_XY[0], y=_TARGET_XY[1] - mcfg.GRASP_TRIGGER_DIST_M)

    fsm.step(link.pose(), {}, link)   # GRASP -> GRASP_ALIGN 전이
    assert fsm.state == State.GRASP_ALIGN
    fsm.step(link.pose(), {}, link)   # GRASP_ALIGN 이 실제로 움직임을 낸다
    assert link.sent and link.sent[-1][0] == "back"


def test_파지_시도_이후엔_BACK_OFF에도_물러나지_않는다():
    fsm = MissionFSM()
    _begin_grasp(fsm, grasp_fail_tries=1)
    link = FixedCorrectionPi(
        correction=GraspCorrection(BACK_OFF, "정면에서 못 찾음"),
        x=_TARGET_XY[0], y=_TARGET_XY[1] - mcfg.GRASP_TRIGGER_DIST_M)

    for _ in range(10):
        before_xy = (link.x, link.y, link.yaw_deg)
        fsm.step(link.pose(), {}, link)
        # GRASP에 머무르고, 차체는 한 치도 안 움직인다 — "stop" 말고는
        # 아무 것도 보내지 않는다.
        assert fsm.state == State.GRASP
        assert link.sent[-1][0] == "stop"
        assert (link.x, link.y, link.yaw_deg) == before_xy

    # 그래도 align_tries는 세어져서 강제 파지/포기 사다리는 진행된다.
    assert fsm._align_tries >= 10


def test_파지_시도_이후엔_GRASP_REPLAN도_발동하지_않는다():
    fsm = MissionFSM()
    _begin_grasp(fsm, grasp_fail_tries=1)
    link = FixedCorrectionPi(
        correction=GraspCorrection(BACK_OFF, "정면에서 못 찾음"),
        x=_TARGET_XY[0], y=_TARGET_XY[1] - mcfg.GRASP_TRIGGER_DIST_M)

    for _ in range(mcfg.GRASP_REPLAN_AFTER_TRIES * 5):
        fsm.step(link.pose(), {}, link)
        assert fsm.state != State.GRASP_REPLAN
    assert fsm._replan_tries == 0


def test_RE_AIM은_파지_시도_이후에도_평소대로_동작한다():
    """이 안전장치는 BACK_OFF(뎁스캠이 못 찾음/너무 가까움)에만 건다 —
    RE_AIM(턱 폭 밖)은 물체가 여전히 보이는 경우라 대상이 아니다."""
    fsm = MissionFSM()
    _begin_grasp(fsm, grasp_fail_tries=1)
    link = FixedCorrectionPi(
        correction=GraspCorrection(RE_AIM, "턱 폭 밖", lateral_mm=95.0),
        x=_TARGET_XY[0], y=_TARGET_XY[1] - mcfg.GRASP_TRIGGER_DIST_M)

    fsm.step(link.pose(), {}, link)   # GRASP -> GRASP_ALIGN 전이
    assert fsm.state == State.GRASP_ALIGN
    fsm.step(link.pose(), {}, link)   # GRASP_ALIGN 이 실제로 움직임을 낸다
    assert link.sent and link.sent[-1][0] in ("yaw+", "yaw-")
