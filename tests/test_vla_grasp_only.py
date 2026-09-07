"""파지는 정책이 통째로 한다 — 재시도 사다리를 끈다 (2026-09-07).

## 지시

"재시도 과정을 빼줘 — 상태를 초기화하고 전체 상황을 보고 search target ->
 approach piece 로 진행하는 루트를 다시 진행하게 해줘. 물체 파지 조건에
 대한 인식 시퀀스를 vla 기반으로 바꿔줘. kica927 에서 가져온 내용 중에
 파지 관련 부분은 모두 배제해줘."

## 무엇을 끄는가

    GRASP_ALIGN    Pi 의 뎁스 정렬 판정(GRASP_BLOCKED)으로 재정렬
    GRASP_REPLAN   그게 반복되면 오버헤드로 크게 다시 세우기
    GRASP_FORCE    그래도 안 되면 정렬 판정을 건너뛰고 강제 파지
    재시도 상한    GRASP_FAIL_MAX_RETRIES 를 세다 skipped 에 넣고 포기

넷 다 팀원 브랜치에서 왔고 전부 **뎁스캠 정렬 판정**을 전제로 한다. 그런데
지금 구성(use_depth_gate:=false)에서는 Pi 가 그 판정을 아예 안 한다 —
"뎁스 관문 꺼짐 — 정렬 판정 없이 진행"을 보내고 바로 파지로 간다. 그래서 위
셋은 이미 트리거될 수가 없었다(2026-09-07 실기 로그에 한 번도 안 나온다).
이 스위치는 그 사실을 명시하고, 남아 있던 재시도 상한까지 같이 끈다.

## 실패하면 하는 일은 하나뿐이다

상태를 통째로 리셋하고 SEARCH_TARGET 부터 다시. 대상도 그때 보이는 것으로
새로 고른다. ⚠️ 상한이 없다 — 못 집는 물체 하나만 남으면 영원히 돈다.
사람이 보다가 멈추는 전제다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HOST = Path(__file__).resolve().parent.parent / "host"
sys.path.insert(0, str(_HOST))
sys.path.insert(0, str(_HOST / "aruco"))

import mission_config as mcfg                                # noqa: E402
from mission import MissionFSM, State                         # noqa: E402
from vehicle_link import RE_AIM, GraspCorrection              # noqa: E402

from conftest import PiSim                                    # noqa: E402

_ROOK_VISIBLE = {"rook": [(1.0, 0.6)]}


def _fsm_in_grasp():
    fsm = MissionFSM()
    fsm.state = State.GRASP
    fsm.target_label = "rook"
    fsm._target_xy = (1.0, 0.6)
    fsm.dest_xy = (1.271, 1.30)
    return fsm


class AlwaysFailPi(PiSim):
    """GRASP 요청마다 순수 물리적 실패로만 답한다."""

    def poll_status(self) -> str:
        last = self.sent[-1][1] if self.sent else None
        if last in ("GRASP", "GRASP_FORCE"):
            self.last_correction = None
            return "FAILED"
        return "IDLE"


class AlwaysCorrectPi(PiSim):
    """정렬이 안 맞는다고 계속 보정을 보내는 Pi — 예전 사다리의 입구."""

    def poll_status(self) -> str:
        self.last_correction = GraspCorrection(RE_AIM, "턱 폭 밖", lateral_mm=95.0)
        return "BUSY"


# ── 기본값 ─────────────────────────────────────────────────────────────────


def test_기본값이_켜져_있다():
    """이 브랜치는 VLA 파지 경로다 — 기본이 꺼져 있으면 지시와 어긋난다."""
    assert mcfg.VLA_GRASP_ONLY is True


# ── 실패 처리 ──────────────────────────────────────────────────────────────


def test_실패하면_상태를_리셋하고_SEARCH_TARGET_부터_다시():
    fsm = _fsm_in_grasp()

    fsm.step(PiSim().pose(), {}, AlwaysFailPi())

    assert fsm.state == State.SEARCH_TARGET
    assert fsm.target_label is None
    assert fsm._target_xy is None
    assert fsm._grasp_yaw_latched is None
    assert fsm._forcing_grasp is False


def test_재시도_횟수를_안_센다():
    """⚠️ "재시도"라는 개념 자체를 뺐다. 카운터가 남으면 상한이 되살아난다."""
    fsm = _fsm_in_grasp()

    fsm.step(PiSim().pose(), {}, AlwaysFailPi())

    assert fsm._grasp_fail_tries == 0
    assert fsm._grasp_fail_for is None


def test_실패해도_기물을_포기하지_않는다():
    """skipped 에 넣으면 SEARCH_TARGET 이 그 기물을 다시 못 고른다."""
    fsm = _fsm_in_grasp()
    link = AlwaysFailPi()

    for _ in range(12):
        fsm.step(link.pose(), _ROOK_VISIBLE, link)

    assert fsm.skipped == [], "포기하면 안 된다 — 상한이 없는 것이 의도다"


def test_반복_실패해도_RETURN_HOME_으로_안_빠진다():
    """예전에는 상한을 넘기면 포기하고 기본 위치로 갔다."""
    fsm = _fsm_in_grasp()
    link = AlwaysFailPi()

    for _ in range(30):
        fsm.step(link.pose(), _ROOK_VISIBLE, link)
        assert fsm.state != State.RETURN_HOME


# ── 사다리 차단 ────────────────────────────────────────────────────────────


def test_보정이_와도_재정렬로_안_넘어간다():
    """⚠️ 이것이 kica927 파지 사다리의 입구다 — GRASP_ALIGN."""
    fsm = _fsm_in_grasp()
    link = AlwaysCorrectPi()

    for _ in range(20):
        fsm.step(link.pose(), _ROOK_VISIBLE, link)
        assert fsm.state not in (State.GRASP_ALIGN, State.GRASP_REPLAN)


def test_강제_파지로_안_넘어간다():
    fsm = _fsm_in_grasp()
    link = AlwaysCorrectPi()

    for _ in range(20):
        fsm.step(link.pose(), _ROOK_VISIBLE, link)

    assert fsm._forcing_grasp is False
    assert fsm._forced_grasp_tries == 0
    assert all(status != "GRASP_FORCE" for _cmd, status in link.sent)


# ── 껐을 때는 예전 동작 ────────────────────────────────────────────────────


def test_끄면_예전_사다리가_그대로_살아난다(monkeypatch):
    """되돌릴 수 있어야 한다 — 상한이 필요해지는 날이 올 수 있다."""
    monkeypatch.setattr(mcfg, "VLA_GRASP_ONLY", False)
    fsm = _fsm_in_grasp()
    link = AlwaysFailPi()

    fsm.step(link.pose(), {}, link)

    assert fsm._grasp_fail_tries == 1, "꺼져 있으면 카운터가 살아 있어야 한다"


def test_어떤_재시도_카운터도_안_움직인다():
    """⚠️ "재시도 시퀀스가 확실히 없어졌나"에 대한 답을 기계가 지키게 한다.

    mission.py 에서 카운터를 올리거나 강제를 켜는 자리는 일곱 군데다
    (_grasp_fail_tries / _replan_tries / _forcing_grasp / _forced_grasp_tries
    / _align_tries / _reaim_tries). 하나라도 움직이면 사다리의 어느 갈래가
    아직 살아 있다는 뜻이다.

    보정을 계속 보내는 Pi 와 계속 실패하는 Pi 양쪽으로 돌려 본다 — 사다리의
    입구가 그 둘이다."""
    for link_cls in (AlwaysCorrectPi, AlwaysFailPi):
        fsm = _fsm_in_grasp()
        link = link_cls()
        for _ in range(40):
            fsm.step(link.pose(), _ROOK_VISIBLE, link)
            if fsm.state == State.GRASP:
                continue
            fsm.state = State.GRASP        # 리셋돼도 다시 밀어 넣어 계속 두드린다
            fsm.target_label = "rook"
            fsm._target_xy = (1.0, 0.6)
        counters = {
            "_grasp_fail_tries": fsm._grasp_fail_tries,
            "_replan_tries": fsm._replan_tries,
            "_forced_grasp_tries": fsm._forced_grasp_tries,
            "_align_tries": fsm._align_tries,
            "_reaim_tries": fsm._reaim_tries,
        }
        assert all(v == 0 for v in counters.values()), f"{link_cls.__name__}: {counters}"
        assert fsm._forcing_grasp is False
        assert fsm.skipped == []


# ── 실패하면 처음 켠 상태로 (2026-09-07) ──────────────────────────────────
#
# "집기 이후의 실패 시에 물건 찾기로 돌아간다기보다 상황이 전체적으로 아예
#  초기화되어서 처음부터 다시 진행하는 방식으로 하는 게 좋겠다."
#
# _reset_for_next_target 은 대상·조준만 지운다. 주행 시퀀서, 경로 계획기,
# 바구니 접근 상태, 보류 목록, 투하 목적지는 실패한 시도의 흔적을 그대로
# 물고 갔다 — 그러면 "처음부터"가 아니라 "이어서"다.


def test_실패하면_새로_만든_FSM_과_구별되지_않는다():
    """⚠️ 필드를 하나씩 확인하지 않는다 — 목록을 적어 두면 새 상태가
    추가될 때 조용히 빠진다. 갓 만든 FSM 과 통째로 비교한다."""
    fsm = _fsm_in_grasp()
    link = AlwaysFailPi()

    # 실패한 시도의 흔적을 여기저기 남겨 둔다
    fsm.dest_xy = (1.271, 1.30)
    fsm.dest_box_name = "chess"
    fsm.skipped = [((0.5, 0.5), 123.0)]
    fsm._reaim_tries = 2
    fsm._align_tries = 4
    fsm.last_cmd = "yaw+"
    fsm.nav_goal = (0.9, 0.9)

    fsm.step(PiSim().pose(), {}, link)

    fresh = MissionFSM()
    dirty = {k: (v, vars(fresh).get(k)) for k, v in vars(fsm).items()
             if k not in ("_drive", "_path_planner", "_drive_stall")
             and repr(v) != repr(vars(fresh).get(k))}
    assert not dirty, f"초기화 안 된 필드: {sorted(dirty)}"


def test_주행_시퀀서와_경로_계획기도_새것이_된다():
    """이것들은 값 비교가 안 되는 객체라 위 시험에서 제외했다 — 대신
    **다른 객체로 갈렸는지**를 본다. 같은 객체가 남아 있으면 지난 시도의
    회전 이력·경로가 그대로 이어진다."""
    fsm = _fsm_in_grasp()
    before = (id(fsm._drive), id(fsm._path_planner), id(fsm._drive_stall))

    fsm.step(PiSim().pose(), {}, AlwaysFailPi())

    after = (id(fsm._drive), id(fsm._path_planner), id(fsm._drive_stall))
    assert before != after, "같은 객체를 계속 쓰면 지난 시도의 상태가 남는다"


def test_실행_내내_유지되어야_할_것은_안_지운다():
    """category 는 `--category chess` 처럼 실행 내내 붙는 필터다. 이것까지
    지우면 실패 한 번에 사용자가 준 조건이 사라진다."""
    fsm = MissionFSM(manual_mode=True, category="toy")
    fsm.state = State.GRASP
    fsm.target_label = "soccer"
    fsm._target_xy = (1.0, 0.6)

    fsm.step(PiSim().pose(), {}, AlwaysFailPi())

    assert fsm.category == "toy"
    assert fsm.manual_mode is True


def test_예전_경로는_전체_초기화를_안_한다(monkeypatch):
    """VLA_GRASP_ONLY 를 끄면 상한을 세야 하므로 카운터가 살아 있어야 한다."""
    monkeypatch.setattr(mcfg, "VLA_GRASP_ONLY", False)
    fsm = _fsm_in_grasp()
    fsm.dest_box_name = "chess"

    fsm.step(PiSim().pose(), {}, AlwaysFailPi())

    assert fsm._grasp_fail_tries == 1
