"""Pi 가 보내는 **구조화된 보정**을 Host 가 읽는다 (2026-08-28).

## 왜 이 파일이 생겼나

`vehicle_link.py` 는 Pi 의 한글 사유를 정규식으로 뜯어 보정 종류를 추측하고
있었고, 그 주석은 스스로 "임시 다리"라고 적어 두었다 — 제대로 된 해법은 Pi 가
보정 코드를 함께 보내는 것이라고.

확인해 보니 **Pi 쪽은 그 일을 이미 끝냈다.** `domain/task/corrections.py` 가
`Correction(action, lateral_m, forward_m, yaw_rad)` 을 만들고
`udp_host_link.report()` 가 그것을 보고 JSON 의 `fix` 필드로 실어 보낸다.
Host 만 그 필드를 안 읽고 있었다.

그 결과가 2026-08-28 실기의 두 자리다.

  * run6 — `뎁스 카메라가 정면에서 목표를 찾지 못했다` 를 문장 파서가 못
    알아들어 UNFIXABLE 로 떨어뜨렸고, Host 가 `rook 보류: 고칠 수 없음` 으로
    **기물을 통째로 포기했다.** 물러나면 되는 상황이었다.
  * 바구니 — 문장 파서가 "멀다" 만 잡고 "하한보다 가깝다" 는 일부러 안 잡는다.
    너무 붙어 서면 아무 보정도 못 받고 그 자리에 영원히 선다.

여기서 고정하는 성질: **구조화된 값이 오면 그것을 쓰고, 안 오면(옛 Pi 빌드)
예전처럼 문장을 파싱한다.** 둘 중 하나만 되면 내일 실기가 막힌다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HOST = Path(__file__).resolve().parent.parent / "host"
sys.path.insert(0, str(_HOST))
sys.path.insert(0, str(_HOST / "aruco"))

import mission_config as mcfg                                    # noqa: E402
from mission import MissionFSM, State                            # noqa: E402
from vehicle_link import (BACK_OFF, CREEP_IN, RE_AIM, SHIFT,      # noqa: E402
                          UNFIXABLE, WAIT, basket_fix_from_fix,
                          classify_correction, correction_from_fix)

from conftest import PiSim                                        # noqa: E402


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


# ── 동작 이름 대응 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("action, kind", [
    ("retreat",   BACK_OFF),
    ("advance",   CREEP_IN),
    ("rotate",    RE_AIM),
    ("reacquire", UNFIXABLE),
    ("wait",      WAIT),
])
def test_Pi_동작이_Host_어휘로_옮겨진다(action, kind):
    fix = correction_from_fix({"action": action, "lateral_m": 0.02,
                                "forward_m": 0.0, "yaw_rad": 0.0},
                               insert=False)
    assert fix is not None and fix.kind == kind


def test_모르는_동작은_None_이라_문장_파싱으로_내려간다():
    """UNFIXABLE 로 굳히면 옛 Pi 빌드와 붙었을 때 기물을 포기한다."""
    assert correction_from_fix({"action": "teleport"}, insert=False) is None
    assert correction_from_fix(None, insert=False) is None
    assert correction_from_fix("retreat", insert=False) is None   # dict 가 아니다


def test_숫자를_mm_로_옮긴다():
    fix = correction_from_fix({"action": "rotate", "lateral_m": -0.095,
                                "forward_m": 0.012}, insert=False)
    assert fix.lateral_mm == pytest.approx(-95.0)
    assert fix.forward_mm == pytest.approx(12.0)


def test_바구니_앞_좌우_어긋남은_횡이동으로_없앤다():
    """기물 앞에서는 회전이 맞다 — 턱을 물체 쪽으로 돌리는 것이다. 바구니
    앞에서는 회전하면 거리와 yaw 가 같이 틀어져 여섯 조건을 동시에 흔든다."""
    piece = correction_from_fix({"action": "rotate", "lateral_m": -0.079},
                                 insert=False)
    basket = correction_from_fix({"action": "rotate", "lateral_m": -0.079},
                                  insert=True)

    assert piece.kind == RE_AIM
    assert basket.kind == SHIFT


def test_yaw_가_실려_오면_바구니_앞에서도_회전이다():
    """면의 법선이 틀어진 것은 옆으로 밀어서 못 고친다."""
    fix = correction_from_fix({"action": "rotate", "yaw_rad": 0.09},
                               insert=True)
    assert fix.kind == RE_AIM


# ── 우선순위: 구조화 > 문장 ────────────────────────────────────────────────


def test_구조화된_값이_문장보다_우선이다():
    """문장은 사람용이다. 기계는 기계용 값을 읽어야 한다."""
    detail = "물체가 전진 거리 밖이다 — 재직진 필요"
    fix = (correction_from_fix({"action": "retreat", "forward_m": -0.01},
                                insert=False)
           or classify_correction(detail))
    assert fix.kind == BACK_OFF          # 문장(CREEP_IN)이 아니라 fix 를 따랐다


def test_fix_가_없으면_예전처럼_문장을_읽는다():
    """현장의 Pi 가 옛 빌드일 수 있다 — 그것 때문에 못 움직이면 안 된다."""
    fix = classify_correction("물체가 턱 선보다 가깝다 (12mm < 30mm) — 후진 필요")
    assert fix.kind == BACK_OFF


def test_둘_다_모르면_고칠_수_없는_것으로_본다():
    assert classify_correction("배터리가 부족하다").kind == UNFIXABLE


# ── 움직여도 되는가 ────────────────────────────────────────────────────────


def test_Pi_가_스스로_고치는_중이면_차를_움직이지_않는다():
    """WAIT 는 servo 1 보정 중이다 — 이때 차가 움직이면 보정과 겹쳐 어긋난다."""
    fix = correction_from_fix({"action": "wait", "lateral_m": 0.027},
                               insert=False)
    assert fix.actionable is False


def test_다시_보기는_방향이_없으므로_움직이지_않는다():
    """REACQUIRE 에는 방향이 없다 — 여기서 움직이면 찍는 것이다.

    방향을 아는 경우에 Pi 는 REACQUIRE 가 아니라 RETREAT 를 보낸다
    (2026-08-29, corrections.from_grasp_precondition)."""
    fix = correction_from_fix({"action": "reacquire"}, insert=False)
    assert fix.kind == UNFIXABLE and fix.actionable is False


# ── 바구니 보정 ────────────────────────────────────────────────────────────


def test_너무_멀면_전진_오차가_그대로_온다():
    fix = basket_fix_from_fix({"action": "advance", "forward_m": 0.211})
    assert fix.forward_m == pytest.approx(0.211)


def test_너무_가까우면_음수_오차가_온다():
    """문장 파서가 일부러 안 잡는 경우다 — 여기서만 살아난다."""
    fix = basket_fix_from_fix({"action": "retreat", "forward_m": -0.015})
    assert fix.forward_m == pytest.approx(-0.015)


def test_좌우_보정은_lateral_로_온다():
    fix = basket_fix_from_fix({"action": "rotate", "lateral_m": -0.079})
    assert fix.lateral_m == pytest.approx(-0.079)
    assert fix.forward_m is None


# ── yaw 보정 (10:18 실기, 2026-09-02) ────────────────────────────────────
#
# corrections.from_insert()는 거리 다음으로 yaw(face_yaw_error_rad)를
# 본다 — 거리는 맞는데 라이다 평면 자체가 정면이 아닌 경우다. 그런데
# basket_fix_from_fix()는 여태 lateral_m만 읽고 있어서, 이 경우 Host가
# 아무 계획도 못 세우고 PLACE에서 INSERT_BLOCKED("정렬이 틀어졌다")만
# 반복하는 락업이 됐다.


def test_yaw_보정은_yaw_rad로_온다():
    fix = basket_fix_from_fix({"action": "rotate", "yaw_rad": 0.17,
                               "lateral_m": 0.0})
    assert fix.yaw_rad == pytest.approx(0.17)
    assert fix.lateral_m is None


def test_yaw가_0이면_평소대로_좌우로_읽는다():
    """Correction은 안 쓰는 축도 0.0 기본값으로 다 채워 보낸다 — yaw_rad가
    진짜 0이면 그건 좌우 보정이라는 뜻이지 "값이 있다"는 뜻이 아니다."""
    fix = basket_fix_from_fix({"action": "rotate", "yaw_rad": 0.0,
                               "lateral_m": -0.079})
    assert fix.lateral_m == pytest.approx(-0.079)
    assert fix.yaw_rad is None


@pytest.mark.parametrize("yaw_rad, axis", [(0.17, "rotate_left"),
                                            (-0.17, "rotate_right")])
def test_yaw_보정은_방향에_맞는_axis로_계획된다(yaw_rad, axis):
    """예전에는 여기서 아무 계획도 못 내 PLACE에 영원히 갇혔다(10:18 실기)."""
    fsm = MissionFSM()
    fix = basket_fix_from_fix({"action": "rotate", "yaw_rad": yaw_rad,
                               "lateral_m": 0.0})

    plan = fsm._plan_basket_fix(fix)

    assert plan is not None
    amount, plan_axis = plan
    assert plan_axis == axis
    assert amount == pytest.approx(abs(yaw_rad))


def test_yaw_보정은_거리보다_먼저_예산이_바닥나도_그대로_나온다():
    """회전(rad)과 병진(m) 예산은 단위가 달라 따로 관리한다 — 한쪽이
    바닥나도 다른 쪽 보정은 여전히 낼 수 있어야 한다."""
    fsm = MissionFSM()
    fsm._basket_creep_used = mcfg.BASKET_CREEP_BUDGET_M   # 병진 예산 소진
    fix = basket_fix_from_fix({"action": "rotate", "yaw_rad": 0.17,
                               "lateral_m": 0.0})

    plan = fsm._plan_basket_fix(fix)

    assert plan is not None and plan[1] == "rotate_left"


def test_yaw_회전_예산도_넘겨서까지_돌지_않는다():
    fsm = MissionFSM()
    fsm._basket_yaw_used = mcfg.BASKET_YAW_BUDGET_RAD - 0.05
    fix = basket_fix_from_fix({"action": "rotate", "yaw_rad": 0.17,
                               "lateral_m": 0.0})

    plan = fsm._plan_basket_fix(fix)

    assert plan is not None
    assert plan[0] == pytest.approx(0.05)


def test_바구니에_너무_붙으면_후진을_계획한다():
    """예전에는 여기서 아무 계획도 못 내 그 자리에 영원히 섰다."""
    fsm = MissionFSM()
    fix = basket_fix_from_fix({"action": "retreat", "forward_m": -0.030})

    plan = fsm._plan_basket_fix(fix)

    assert plan is not None
    distance_m, axis = plan
    assert axis == "back"
    assert distance_m == pytest.approx(0.030)


def test_옛_Pi_빌드의_거리값도_그대로_계획된다():
    """`forward_m` 이 없으면 라이다 판독에서 Host 목표를 뺀다."""
    from vehicle_link import BasketFix
    fsm = MissionFSM()

    plan = fsm._plan_basket_fix(BasketFix(distance_m=0.351))

    assert plan is not None and plan[1] == "forward"
    assert plan[0] == pytest.approx(0.351 - mcfg.BASKET_TARGET_LIDAR_M)


# ── run6 회귀: 목표를 못 봤을 때 포기하지 않는다 ────────────────────────────


class BlindPi(PiSim):
    """뎁스캠이 목표를 못 보는 Pi.

    물체가 너무 가까워 화각 아래로 빠진 상황이다. 실기 Pi 는 이때
    `corrections.from_grasp_precondition` 이 만든 **`retreat`** 를 함께 보낸다
    — REACQUIRE 가 아니다. REACQUIRE 에는 방향이 없어서 Host 가 움직일 수
    없는데, 이 경우는 Pi 가 방향을 안다(더 붙으면 어느 경우에도 나빠진다).
    """

    def poll_status(self) -> str:
        self.last_correction = correction_from_fix(
            {"action": "retreat", "lateral_m": 0.0,
             "forward_m": 0.0, "yaw_rad": 0.0}, insert=False)
        return "BUSY"


def test_목표를_못_봐도_기물을_포기하지_않는다():
    """2026-08-28 run6 이 여기서 죽었다.

    로그의 `[mission] rook 보류: 고칠 수 없음 — 뎁스 카메라가 정면에서 목표를
    찾지 못했다` 가 그 순간이다. 그 뒤 Host 는 SEARCH_TARGET 으로 돌아가
    아무것도 하지 않았다."""
    fsm = MissionFSM()
    fsm.state = State.GRASP
    fsm.target_label = "rook"
    fsm._target_xy = (1.0, 0.6)
    fsm.dest_xy = (1.271, 1.30)
    link = BlindPi()

    fsm.step(link.pose(), {}, link)

    assert fsm.state == State.GRASP_ALIGN, "포기하지 말고 재정렬로 가야 한다"
    assert fsm.target_label == "rook", "기물을 놓지 않았어야 한다"


def test_목표를_못_봤을_때는_후진으로_실행된다():
    """앞으로 가면 더 안 보인다 — 방향이 반대면 고쳐지지 않는다."""
    fsm = MissionFSM()
    fsm.state = State.GRASP
    fsm.target_label = "rook"
    fsm._target_xy = (1.0, 0.6)
    fsm.dest_xy = (1.271, 1.30)
    link = BlindPi()

    fsm.step(link.pose(), {}, link)          # GRASP -> GRASP_ALIGN
    fsm.step(link.pose(), {}, link)          # 첫 걸음

    commands = [cmd for cmd, status in link.sent if status == "GRASP_ALIGN"]
    assert commands and commands[0] == "back"


@pytest.mark.skip(
    reason="2026-08-31 임시 변경(반복 테스트용, mission.py GRASP 블록 주석 참고) — "
           "재정렬 예산을 다 써도 지금은 포기하지 않고 계속 시도한다. "
           "그 변경을 되돌리면 이 테스트도 같이 켤 것.")
def test_재정렬_예산을_다_쓰면_그때는_보류한다():
    """무한히 물러나지는 않는다 — 한계는 그대로 지킨다."""
    fsm = MissionFSM()
    fsm.state = State.GRASP
    fsm.target_label = "rook"
    fsm._target_xy = (1.0, 0.6)
    fsm.dest_xy = (1.271, 1.30)
    fsm._align_tries = mcfg.GRASP_ALIGN_MAX_TRIES
    link = BlindPi()

    fsm.step(link.pose(), {}, link)

    assert fsm.state == State.SEARCH_TARGET
    assert fsm.target_label is None
