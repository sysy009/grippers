"""바구니 앞 정렬 폐루프 — 2026-08-28 실기에서 막혔던 지점.

그날 GRASP 는 성공했는데 INSERT 가 95회 연속 거부됐다.

    INSERT_BLOCKED 바구니가 멀다 (라이다 0.351m > 0.155m)
                   좌우로 밀려 있다 (-79mm > ±70mm)

Host 는 그 자리에 서서 `PLACE_DONE` 만 기다렸다 — Pi 가 보내 주는 숫자를
읽는 코드가 없었다. 이 파일은 그 자리에서 출발해 INSERT 까지 가는지를
로봇 없이 확인한다. 출발 조건(0.351m / -79mm)은 시늉이 아니라 그날의
실측값이고 `PiSim` 이 그 값을 그대로 낸다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import (PI_LATERAL_TOLERANCE_M, PI_STOP_LIDAR_M,  # noqa: E402
                      PI_STOP_TOLERANCE_M, PiSim)

_HOST = Path(__file__).resolve().parent.parent / "host"
sys.path.insert(0, str(_HOST))

import config as cfg                       # noqa: E402
import mission_config as mcfg              # noqa: E402
from mission import MissionFSM, State      # noqa: E402
from vehicle_link import parse_basket_fix  # noqa: E402


@pytest.fixture(autouse=True)
def _freeze_chess_box_at_2026_08_28_position(monkeypatch):
    """이 파일은 2026-08-28 그날의 실측(라이다 0.351m, 좌우 -79mm)을 그대로
    재현한다 — `PiSim`이 그 숫자를 내려면 그날 기준이던 상자 좌표
    (chess 중심 1.350, 1.625)를 그대로 써야 한다. 라이브 config.BOXES는
    그 사이 한동안 1.450으로 바뀌었다가(2026-09-04, 착오로 밝혀짐)
    2026-09-05에 x는 1.350으로 정정됐고, 2026-09-07에 y를 1.575로 줄였다가
    2026-09-08에 1.625로 되돌렸다(정지 지점은 BOX_APPROACH_MARGIN_M 이 맡는다)
    — 그래서 지금은 우연히 같다. 이 파일은 라이브 값의 향방과 무관하게 그날 재현 좌표를
    명시적으로 고정해 둔다 — 나중에 config.BOXES가 또 바뀌어도 이 파일의
    재현 시나리오는 깨지지 않아야 하기 때문이다."""
    monkeypatch.setitem(cfg.BOXES, "chess", (1.350, 1.625, 180.0))

# 무한 루프가 이 프로젝트의 최대 리스크다. "느리게라도 끝났다"가 아니라
# 상한 안에 못 끝나면 그 자체를 실패로 본다(Pi 저장소 conftest 와 같은 원칙).
# 0.196m 를 0.06 m/s 로 가면 3.3초 = 14Hz 에서 46 사이클이고, 좌우 79mm 가
# 1.3초 = 19 사이클이다. 판정 왕복을 넉넉히 얹어 400 으로 둔다.
#
# ⚠️ 2026-09-02, 시연용으로 PLACE 완료 뒤 SEARCH_TARGET 으로 곧장 가지 않고
# RETURN_HOME(기본 위치, 바구니에서 대략 1m 대각선)을 한 번 거치도록
# 바꿨다 — 그만큼 사이클이 늘어난다(실측: 이 파일의 기본 시나리오가
# 442사이클에 끝남). 800으로 두 배 가까운 여유를 준다.
MAX_STEPS = 800

# 2026-09-02~09-04 사이 한동안 있던 AWAIT_CONTINUE(그룹이 화면에서 다
# 소진되면 RETURN_HOME 대신 사람에게 물어보는 기능) 때문에, PLACE 완료 시
# 빈 피지도 `{}` 를 주면 RETURN_HOME 이 아니라 그쪽으로 새 버릴 수 있어
# knight(같은 chess 그룹)가 하나 더 남아 있는 피지도를 줬었다.
# 2026-09-04 밤 사용자 지시로 그 기능이 통째로 없어져 지금은 굳이 필요
# 없지만, 있어도 결과가 같으므로(이 파일은 어차피 RETURN_HOME 이후가
# 아니라 INSERT 정렬 폐루프를 본다) 그대로 둔다.
_OTHER_CHESS_PIECE_REMAINS = {"knight": [(0.9, 0.9)]}


def _run(sim: PiSim, max_steps: int = MAX_STEPS) -> tuple[MissionFSM, int]:
    """룩을 든 상태로 시작해 PLACE 가 끝날 때까지 돌린다."""
    fsm = MissionFSM()
    assert fsm.begin_carrying("rook")
    for n in range(1, max_steps + 1):
        fsm.step(sim.pose(), _OTHER_CHESS_PIECE_REMAINS, sim)
        if fsm.state == State.SEARCH_TARGET:
            return fsm, n          # PLACE 를 끝내고 다음 대상을 찾으러 갔다
    pytest.fail(
        f"{max_steps} 사이클 안에 INSERT 를 못 끝냈다 — "
        f"상태 {fsm.state.name}, 라이다 {sim.lidar_m:.3f}m, "
        f"좌우 {sim.lateral_m * 1000:+.0f}mm")


def _run_to_place_done(sim: PiSim, max_steps: int = MAX_STEPS) -> tuple[MissionFSM, int]:
    """룩을 든 상태로 시작해 PLACE 가 **막 끝나는 그 순간**까지 돌린다.

    2026-09-02부터 PLACE 완료는 SEARCH_TARGET이 아니라 RETURN_HOME으로
    이어진다(시연용, _skip_target과 같은 이유로 항상 같은 자리에서 다음
    탐색을 시작하게 함). "바구니 앞에 어떻게 섰는가"를 보는 테스트는 그
    뒤 RETURN_HOME으로 주행해 버린 좌표가 아니라 이 전이 순간의 좌표를
    봐야 한다 — PLACE에서 NUDGE_BOX로 되돌아가는 보정 왕복(정상 동작)과
    구분하려고, "직전이 PLACE였고 지금이 그 다음 라운드"인 순간만 완료로
    본다.

    ⚠️ 2026-09-06: 다음 라운드가 RETURN_HOME 인지 SEARCH_TARGET 인지는
    mission_config.RETURN_HOME_ENABLED 가 정한다(사용자 지시로 기본 꺼짐).
    이 파일은 **바구니 앞에 어떻게 섰는가**를 보는 것이지 그 뒤 어디로
    가는지를 보는 게 아니므로, 둘 다 완료로 받는다."""
    fsm = MissionFSM()
    assert fsm.begin_carrying("rook")
    was_place = False
    for n in range(1, max_steps + 1):
        was_place = fsm.state == State.PLACE
        fsm.step(sim.pose(), _OTHER_CHESS_PIECE_REMAINS, sim)
        if was_place and fsm.state in (State.RETURN_HOME, State.SEARCH_TARGET):
            return fsm, n
    pytest.fail(
        f"{max_steps} 사이클 안에 INSERT 를 못 끝냈다 — "
        f"상태 {fsm.state.name}, 라이다 {sim.lidar_m:.3f}m, "
        f"좌우 {sim.lateral_m * 1000:+.0f}mm")


def test_실기_출발조건이_그날_관측과_같다():
    """이 시늉이 딴 것을 시험하고 있지 않은지부터 고정한다."""
    sim = PiSim()
    assert round(sim.lidar_m, 3) == 0.351
    assert round(sim.lateral_m * 1000) == -79


# 2026-09-05, 사용자 지시("라이다 뺀 상황으로 전제하고 다시 수정해")로
# CARRY_TO_DEST가 부채꼴(사선) 진입 + 동적 정렬로 바뀐 뒤 아래 3개가 깨졌다.
# 원인은 반경 숫자가 아니라 이 저장소가 아예 모르는 값이다 — ArUco 마커와
# 로봇의 실제 물리적 앞부분(그리퍼/라이다)이 얼마나 떨어져 있는지
# (tests/conftest.py의 LIDAR_AHEAD_M≈0.099m, 2026-08-28 사고 한 건에서
# 역산한 값, 실제 코드엔 이 오프셋을 아는 상수가 없다). 사선으로 더 가까운
# 부채꼴(반경 0.15m, ArUco 기준)에서 진입하다 보니, 시뮬레이션상 그 오프셋
# 만큼 로봇의 "실제" 앞부분이 이미 목표를 지나쳐 있는 경우가 나온다
# (아래 test_그날_막힌_자리에서_INSERT까지_간다: 라이다 -0.05m, 물리적으로
# 벽을 지나친 값).
#
# hard_stop(BASKET_HARD_STOP_MARGIN_M=0.05, mission.py NUDGE_BOX)의 여유를
# 늘려서 덮는 방법도 검토했지만, 그 상수는 "너무 크면 정상 INSERT까지
# 막는다"는 경고가 이미 코드에 있는 값이고(2026-09-03), 실측 못한 오프셋을
# 또 다른 실측 못한 숫자로 덮는 것이라 사용자가 거부했다 — 지금 상태
# 그대로 두고 **다음 실기(2026-09-08 전)에서 직접 확인**하기로 했다.
#
# 그래서 이 3개는 지운 게 아니라 skip으로 남긴다 — 지금 코드가 이 역사적
# 시나리오(2026-08-28)를 여전히 안전하게 재현하는지는 실기 전까지 모른다는
# 뜻을, 조용히 통과하는 초록불 대신 눈에 보이게 남겨 두는 것이다.
_SECTOR_APPROACH_OVERSHOOT_SKIP = (
    "2026-09-05 사선 부채꼴 접근 도입 후 ArUco-그리퍼 오프셋 미검증으로 "
    "실패 — 위 모듈 주석 참고. 다음 실기에서 직접 확인하기로 함(사용자 지시).")


@pytest.mark.skip(reason=_SECTOR_APPROACH_OVERSHOOT_SKIP)
def test_그날_막힌_자리에서_INSERT까지_간다():
    sim = PiSim()
    fsm, steps = _run_to_place_done(sim)

    # 도착 조건은 Pi 가 받아 주는 범위 안이어야 한다 — 그것도 **가장자리가
    # 아니라 여유를 두고**. 상한에 딱 붙어 서면 판독이 1mm 만 튀어도 다시
    # 거부당한다.
    assert PI_STOP_LIDAR_M - PI_STOP_TOLERANCE_M <= sim.lidar_m
    assert sim.lidar_m <= PI_STOP_LIDAR_M + PI_STOP_TOLERANCE_M
    margin = min(abs(sim.lidar_m - (PI_STOP_LIDAR_M - PI_STOP_TOLERANCE_M)),
                 abs((PI_STOP_LIDAR_M + PI_STOP_TOLERANCE_M) - sim.lidar_m))
    assert margin >= 0.005, f"수용 창 가장자리에 너무 붙었다 (여유 {margin*1000:.1f}mm)"
    assert abs(sim.lateral_m) <= PI_LATERAL_TOLERANCE_M
    # 그리고 예산 안에서 끝나야 한다 — 0.196m 전진 + 0.079m 횡이동.
    assert fsm._basket_creep_used <= mcfg.BASKET_CREEP_BUDGET_M
    assert steps < MAX_STEPS


@pytest.mark.skip(reason=_SECTOR_APPROACH_OVERSHOOT_SKIP)
def test_횡이동_명령을_실제로_쓴다():
    """좌우 79mm 를 회전으로 고치려 들면 거리와 yaw 가 같이 틀어진다."""
    sim = PiSim()
    _run(sim)
    cmds = {c for c, status in sim.sent if status == "NUDGE_BOX"}
    assert "go" in cmds
    assert cmds & {"left", "right"}, f"횡이동을 안 썼다 — 보낸 명령 {cmds}"


@pytest.mark.skip(reason=_SECTOR_APPROACH_OVERSHOOT_SKIP)
def test_거리를_먼저_맞추고_좌우를_나중에_본다():
    """Pi 의 좌우 추정은 가까이 붙어야 정확하다(basket_lidar_align 주석)."""
    sim = PiSim()
    fsm = MissionFSM()
    fsm.begin_carrying("rook")
    first_lateral_step = None
    for n in range(1, MAX_STEPS + 1):
        fsm.step(sim.pose(), {}, sim)
        if sim.sent and sim.sent[-1][0] in ("left", "right"):
            first_lateral_step = n
            break
    assert first_lateral_step is not None
    # 좌우를 처음 건드리는 시점엔 거리가 이미 허용 범위 안이어야 한다.
    assert sim.lidar_m <= PI_STOP_LIDAR_M + PI_STOP_TOLERANCE_M


def test_좌우를_모르면_거리만_맞추고_멈춘다():
    """`lateral_known=False` 는 0이 아니라 **모른다**는 뜻이다.

    바구니가 방위각 창을 양쪽 다 채우면 가장자리가 안 보여 중심을 못
    낸다. 그때 0으로 읽고 "가운데"라고 판단하면 물체가 바구니 밖에
    떨어진다 — Host 도 지어내지 말아야 한다."""
    sim = PiSim(lateral_known=False)
    fsm, _ = _run_to_place_done(sim)
    assert sim.lidar_m <= PI_STOP_LIDAR_M + PI_STOP_TOLERANCE_M


def test_예산을_넘겨서까지_밀지_않는다():
    """판독이 이상해 같은 보정이 계속 나와도 바구니를 밀고 들어가면 안 된다."""
    sim = PiSim(freeze_lateral=True)     # 옆으로 가라고 해도 안 움직인다
    fsm = MissionFSM()
    fsm.begin_carrying("rook")
    for _ in range(MAX_STEPS):
        fsm.step(sim.pose(), {}, sim)
        if fsm.state == State.SEARCH_TARGET:
            break
    assert fsm._basket_creep_used <= mcfg.BASKET_CREEP_BUDGET_M + 1e-9


@pytest.mark.parametrize("detail, distance_m, lateral_m", [
    ("바구니가 멀다 (라이다 0.351m > 0.155m) / 좌우로 밀려 있다 (-79mm > ±70mm)",
     0.351, -0.079),
    ("바구니가 멀다 (라이다 0.240m > 0.155m)", 0.240, None),
    ("좌우로 밀려 있다 (+85mm > ±70mm)", None, 0.085),
])
def test_Pi_문장에서_숫자를_되꺼낸다(detail, distance_m, lateral_m):
    fix = parse_basket_fix(detail)
    assert fix is not None
    assert fix.distance_m == distance_m
    assert fix.lateral_m == lateral_m


def test_고칠_수_없는_사유는_보정으로_읽지_않는다():
    """숫자가 없으면 지어내지 않는다 — 엉뚱하게 움직이는 것보다 낫다."""
    assert parse_basket_fix("그리퍼가 비어 있다 (부하 0.0300 < 0.0469)") is None
    # 하한보다 가까운 경우는 판독 자체를 믿을 수 없어 일부러 안 잡는다.
    assert parse_basket_fix("라이다 판독이 하한보다 가깝다 (0.120m < 0.100m)") is None


def test_겨냥점이_정지_오버슈트를_흡수할_만큼_창_안쪽이다():
    """Pi 의 `domain/task/motion.py` 가 계산해 둔 오버슈트를 그대로 쓴다.

    Host 가 "멈춰"라고 판단한 순간부터 바퀴가 실제로 설 때까지 지연이
    쌓인다(그 파일 기준 Host 125ms + UDP/수신 10ms + Pi 100ms = 235ms).
    바구니 접근 속도 0.06 m/s 에서 그동안 **14mm** 를 더 간다. 그런데 Pi 의
    수용 반폭은 15mm 다 — 오버슈트가 창과 거의 같은 크기다.

    그래서 겨냥점을 창 가장자리에 두면 우연히 들어갈 수는 있어도 제어되는
    것이 아니다. 겨냥점은 오버슈트를 흡수할 만큼 안쪽이어야 한다. 2026-08-28
    에 겨냥점을 상한(0.155m)에 뒀다가 폐루프가 거기 수렴해 놓고 영원히
    거부당했다.
    """
    stop_latency_s = 0.125 + 0.010 + 0.100     # motion.py 의 표와 같은 값
    overshoot_m = 0.06 * stop_latency_s

    lower = PI_STOP_LIDAR_M - PI_STOP_TOLERANCE_M
    margin = mcfg.BASKET_TARGET_LIDAR_M - lower

    assert margin >= overshoot_m, (
        f"겨냥점 여유 {margin * 1000:.0f}mm 가 정지 오버슈트 "
        f"{overshoot_m * 1000:.0f}mm 보다 작다 — 멈추다 창을 지나친다")


def test_보정_데드밴드가_수용_반폭보다_좁다():
    """데드밴드가 수용 반폭만큼 넓으면 수렴점이 창 가장자리가 된다."""
    assert mcfg.BASKET_DISTANCE_DEADBAND_M < PI_STOP_TOLERANCE_M
    assert mcfg.BASKET_LATERAL_DEADBAND_M < PI_LATERAL_TOLERANCE_M


def test_Host_겨냥점이_Pi_수용창_안이다():
    """두 저장소의 상수가 갈라지면 Host 는 다 왔다고 믿고 Pi 는 계속
    거부하는 교착이 된다."""
    assert (PI_STOP_LIDAR_M - PI_STOP_TOLERANCE_M
            <= mcfg.BASKET_TARGET_LIDAR_M
            <= PI_STOP_LIDAR_M + PI_STOP_TOLERANCE_M)
