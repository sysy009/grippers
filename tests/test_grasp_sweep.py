"""GRASP_ALIGN BACK_OFF 후 좌우 스윕 (사용자 지시, 2026-09-01).

2026-09-01 실기에서 BACK_OFF(뎁스캠이 정면에서 목표를 못 봄)를 20번 반복해
직선으로 약 0.9m 물러나다 탑뷰 추적 범위 밖으로 나갔다 — 후진만으로는
반복해도 안 나아지는 사례였다. 그래서 한 걸음 후진한 뒤 좌(1.0s)→우(2.0s)로
훑으며 다시 찾고, 찾는 순간(=Pi 가 더 이상 "못 찾음" 이 아닌 다른 응답을
주는 순간) 바로 스윕을 멈추고 정상 GRASP 판정으로 돌아간다.

Pi 의 관측(observe_target 다중 프레임 합의)은 정지 상태를 전제하므로 회전
"중"에는 물을 수 없다 — 그래서 짧게(GRASP_SWEEP_BURST_SEC) 끊어 돌리고
멈춰서 GRASP 로 확인하는 것을 반복한다. 여기서는 그 왕복이 실제로 좌→우
순서로, 예산 안에서 일어나는지, 찾으면 바로 멈추는지, 못 찾으면 다음
BACK_OFF 사이클로 넘어가 align_tries·GRASP_FORCE 안전망이 그대로
살아있는지를 본다."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HOST = Path(__file__).resolve().parent.parent / "host"
sys.path.insert(0, str(_HOST))
sys.path.insert(0, str(_HOST / "aruco"))

import mission                                        # noqa: E402
import mission_config as mcfg                          # noqa: E402
from localizer import Pose                              # noqa: E402
from mission import MissionFSM, State                  # noqa: E402
from vehicle_link import BACK_OFF, GraspCorrection      # noqa: E402

from conftest import PiSim                              # noqa: E402


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

_NOT_FOUND = GraspCorrection(BACK_OFF, "뎁스 카메라가 정면에서 목표를 찾지 못했다")

# Host 실측 루프 주기(conftest.PiSim.dt 와 같은 값) 만큼 매 스텝 가짜 시계를
# 흘려보낸다 — 실제 시간을 기다리지 않고도 버스트/구간 예산을 자연스럽게
# 넘긴다.
_TICK = 1.0 / 14.0
MAX_STEPS = 20000


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


class SweepPi(PiSim):
    """BACK_OFF(못 찾음)만 반복하다가, GRASP 확인이 `found_after` 번째에
    도달하면 그 뒤로는 찾은 것으로 바뀌는 Pi. `checks`는 GRASP/GRASP_FORCE
    상태로 넘어가 실제로 확인(=poll_status 가 그 상태로 불림)한 횟수다 —
    스윕 버스트마다 GRASP_ALIGN→GRASP 왕복이 하나씩 생기므로, 이 횟수가
    곧 "몇 번째 버스트에서 찾았는가"다."""

    def __init__(self, *args, found_after: "int | None" = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.found_after = found_after
        self.checks = 0
        self.sweep_cmds_seen: list[str] = []

    def send(self, cmd) -> None:
        super().send(cmd)
        if cmd.status == "GRASP_ALIGN" and cmd.cmd in ("yaw+", "yaw-"):
            self.sweep_cmds_seen.append(cmd.cmd)

    def poll_status(self) -> str:
        last_cmd, last_status = (self.sent[-1] if self.sent else (None, None))
        if last_status in ("GRASP", "GRASP_FORCE"):
            self.checks += 1
            if self.found_after is not None and self.checks >= self.found_after:
                self.last_correction = None
                return "GRASP_DONE"
            self.last_correction = _NOT_FOUND
            return "BUSY"
        return "IDLE"


class PoseLossSweepPi(SweepPi):
    """스윕 도중 포즈를 잃는 시늉 — 2026-09-01 코드 리뷰에서 나온 걱정을
    검증한다: `_grasp_sweep_step`의 예산이 벽시계 기준이라, 포즈를 잃은
    사이클(step()이 맨 앞에서 그냥 돌아가는 사이클)에 흐른 시간까지
    스윕이 회전한 시간으로 잘못 셀 수 있다. `pose_lost`를 True로 두는
    동안은 `pose().ok`가 False다 — 이 스윕을 만든 계기 자체가 BACK_OFF
    반복 중 포즈를 잃은 사고였으니, 스윕 도중 포즈를 또 잃는 것도
    현실적인 시나리오다."""

    pose_lost: bool = False

    def pose(self) -> Pose:
        p = super().pose()
        if self.pose_lost:
            return Pose(x=p.x, y=p.y, yaw_deg=p.yaw_deg,
                        ok=False, n_cams=0, fresh=False)
        return p


def _begin_grasp(fsm: MissionFSM) -> None:
    fsm.state = State.GRASP
    fsm.target_label = "rook"
    fsm._target_xy = (1.0, 0.6)
    fsm.dest_xy = (1.271, 1.30)


def _run_ticking(fsm, link, clock, predicate, max_steps=MAX_STEPS):
    for n in range(1, max_steps + 1):
        clock.advance(_TICK)
        fsm.step(link.pose(), {}, link)
        if predicate(fsm):
            return n
    pytest.fail(f"{max_steps} 사이클 안에 조건에 도달하지 못했다 — 상태 {fsm.state.name}")


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(mission.time, "monotonic", clock)
    return clock


def test_후진_한걸음_뒤_좌회전부터_스윕을_시작한다(fake_clock):
    fsm = MissionFSM()
    _begin_grasp(fsm)
    link = SweepPi(found_after=None)

    # GRASP → (BACK_OFF 보고) → GRASP_ALIGN 진입 + 후진 한 걸음 완료까지.
    _run_ticking(fsm, link, fake_clock,
                 lambda f: f._align_sweep_stage == "left")
    assert fsm.last_cmd == "yaw+"   # 좌회전부터


def test_좌우_다_훑어도_못_찾으면_스윕을_접고_다음_BACK_OFF로_넘어간다(fake_clock):
    fsm = MissionFSM()
    _begin_grasp(fsm)
    link = SweepPi(found_after=None)

    _run_ticking(fsm, link, fake_clock,
                 lambda f: f._align_sweep_stage == "left")
    # 좌(1.0s)+우(2.0s) 예산을 다 쓸 때까지 스윕이 이어지다가 결국 접힌다.
    _run_ticking(fsm, link, fake_clock,
                 lambda f: f._align_sweep_stage is None and f.state == State.GRASP_ALIGN)
    # 다음 BACK_OFF 사이클은 스윕 없이 다시 후진부터(align_tries 는 계속
    # 누적돼 GRASP_FORCE 안전망까지 그대로 이어진다).
    assert link.checks >= 1
    assert fsm._align_tries >= 1


def test_스윕_도중_찾으면_바로_멈추고_CARRY로_간다(fake_clock):
    fsm = MissionFSM()
    _begin_grasp(fsm)
    # 두 번째 확인(=두 번째 버스트) 때부터 찾은 것으로 바뀐다 — 좌회전
    # 스윕이 끝나기 전, 즉 우회전까지 안 가고 멈추는지 본다.
    link = SweepPi(found_after=2)

    _run_ticking(fsm, link, fake_clock, lambda f: f.state == State.CARRY_TO_DEST)

    assert link.checks == 2
    # 우회전(yaw-)까지 갈 필요가 없었다 — 좌회전만으로 찾았다.
    assert "yaw-" not in link.sweep_cmds_seen
    assert fsm.target_label == "rook"


def test_스윕은_좌회전_다음에_우회전_순서를_지킨다(fake_clock):
    fsm = MissionFSM()
    _begin_grasp(fsm)
    # 좌(1.0s) 예산을 확실히 다 쓰고 우회전으로 넘어간 뒤에나 찾도록,
    # 넉넉히 뒤(10번째 확인)에서 찾은 것으로 한다.
    link = SweepPi(found_after=10)

    _run_ticking(fsm, link, fake_clock, lambda f: f.state == State.CARRY_TO_DEST)

    assert "yaw+" in link.sweep_cmds_seen
    assert "yaw-" in link.sweep_cmds_seen
    # 좌회전이 우회전보다 먼저 나와야 한다.
    first_left = link.sweep_cmds_seen.index("yaw+")
    first_right = link.sweep_cmds_seen.index("yaw-")
    assert first_left < first_right


def test_스윕_도중_포즈를_잃어도_예산을_까먹지_않는다(fake_clock):
    """2026-09-01 코드 리뷰: `_grasp_sweep_step`의 좌/우 예산은 벽시계
    기준(now - phase_start)이다. 포즈를 잃은 사이클엔 `step()`이 맨 앞
    `if not pose.ok: return self.state`에서 끝나 `_grasp_sweep_step`
    자체가 안 불리고 회전 명령도 안 나간다 — 그런데 예산 시계가 그동안도
    그냥 흘렀다면, 포즈가 돌아왔을 때 "한 번도 안 돈 채로 좌 예산을 다
    썼다"고 착각해 바로 우로 넘어가거나 스윕을 접어버릴 수 있다. 이
    스윕을 만든 계기 자체가 BACK_OFF 반복 중 포즈를 잃은 사고였다는 걸
    생각하면, 스윕 도중 또 포즈를 잃는 것도 현실적인 시나리오다.

    좌 예산(1.0s)보다 긴 1.5초 동안 포즈를 잃었다가 돌려놔도 여전히
    "left" 단계에 남아 있는지, 그리고 회전 명령이 실제로 이어지는지를
    본다 — 고치기 전이었다면 이 지점에서 이미 "right"로 넘어가 있어야
    맞다(1.5s > 1.0s 예산)."""
    fsm = MissionFSM()
    _begin_grasp(fsm)
    link = PoseLossSweepPi(found_after=None)

    _run_ticking(fsm, link, fake_clock,
                 lambda f: f._align_sweep_stage == "left")

    link.pose_lost = True
    lost_until = fake_clock.now + 1.5
    cmds_before = len(link.sweep_cmds_seen)
    while fake_clock.now < lost_until:
        fake_clock.advance(_TICK)
        fsm.step(link.pose(), {}, link)
    # 포즈를 잃은 동안은 회전 명령이 하나도 안 나가야 한다 — step()이
    # 맨 앞에서 그냥 돌아갔다는 뜻이다.
    assert len(link.sweep_cmds_seen) == cmds_before

    link.pose_lost = False
    assert fsm._align_sweep_stage == "left"
    fake_clock.advance(_TICK)
    fsm.step(link.pose(), {}, link)
    # 예산을 까먹지 않았다면 포즈가 돌아온 직후에도 여전히 좌회전이어야
    # 한다 — 까먹었다면(고치기 전) 이미 "right"(yaw-)로 넘어가 있었다.
    assert fsm.last_cmd == "yaw+"
    assert fsm._align_sweep_stage == "left"
