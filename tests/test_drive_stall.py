"""명령은 나가는데 차가 안 움직이는 것을 탑뷰로 잡는다 (2026-09-07).

이 차량에는 바퀴가 실제로 도는지 아는 수단이 없다 — `/odom_raw` 는
`/cmd_vel` 을 적분해 되돌려줄 뿐이고 바퀴 엔코더 피드백이 없다. 그래서
모터가 죽어도 오도메트리는 멀쩡해 보인다. `base_liveness` 가 이미 그 한계를
적어 두고 "진짜 확인은 오버헤드 ArUco(Host 소유)나 사람 눈"이라고 했다.

실기에서 두 번 물렸다:

    2026-08-28  NUDGE_BOX 에서 go 155회 동안 1mm 도 안 움직였다
    2026-09-07  APPROACH_PIECE 에서 yaw- 34.7초 동안 yaw 폭 1.33도

둘 다 소프트웨어는 끝까지 정상이었고, 사람이 눈으로 보기 전까지 몰랐다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HOST = Path(__file__).resolve().parent.parent / "host"
sys.path.insert(0, str(_HOST))

from drive_stall import DriveStallWatch, STALL_SEC          # noqa: E402


def _hold(w, seconds, cmd="yaw-", x=0.41, y=1.00, yaw=70.8, step=0.1):
    """그 명령을 유지한 채 제자리에 서 있는 상황."""
    out = None
    t = 0.0
    while t <= seconds:
        out = w.update(t, cmd, True, x, y, yaw)
        t += step
    return out


# ── 잡는다 ─────────────────────────────────────────────────────────────────


def test_명령하는데_안_움직이면_잡는다():
    """⚠️ 2026-09-07 실기 재현 — yaw- 를 계속 보내는데 제자리."""
    w = DriveStallWatch()
    assert _hold(w, STALL_SEC + 1.0) is not None


def test_문턱_전에는_안_잡는다():
    """짧게 멈칫하는 것까지 경보로 만들면 아무도 안 본다."""
    w = DriveStallWatch()
    assert _hold(w, STALL_SEC - 2.0) is None


def test_실측_잡음은_정체로_친다():
    """정지 상태 탑뷰 잡음(2026-09-07): x·y 폭 7mm, yaw 폭 1.33도.
    이만큼 흔들리는 것은 "움직였다"가 아니다."""
    w = DriveStallWatch()
    out = None
    t = 0.0
    while t <= STALL_SEC + 1.0:
        # 잡음 범위 안에서 흔들리게 한다
        out = w.update(t, "yaw-", True, 0.41 + 0.003 * ((int(t * 10) % 3) - 1),
                       1.00, 70.8 + 0.5 * ((int(t * 10) % 3) - 1))
        t += 0.1
    assert out is not None


# ── 안 잡는다 ──────────────────────────────────────────────────────────────


def test_실제로_돌면_안_잡는다():
    """0.6 rad/s 면 8초에 275도다 — 정상 주행은 근처도 안 간다."""
    w = DriveStallWatch()
    out = None
    t = 0.0
    while t <= STALL_SEC + 5.0:
        out = w.update(t, "yaw-", True, 0.41, 1.00, (70.8 - 34.0 * t) % 360.0)
        t += 0.1
    assert out is None


def test_직진해도_안_잡는다():
    w = DriveStallWatch()
    out = None
    t = 0.0
    while t <= STALL_SEC + 5.0:
        out = w.update(t, "go", True, 0.41 + 0.15 * t, 1.00, 70.8)
        t += 0.1
    assert out is None


def test_세워_둔_것은_정체가_아니다():
    """⚠️ GRASP·PLACE 는 차를 일부러 세운다. 그걸 경보로 만들면 안 된다."""
    w = DriveStallWatch()
    assert _hold(w, STALL_SEC + 5.0, cmd="stop") is None
    assert _hold(w, STALL_SEC + 5.0, cmd=None) is None


def test_포즈를_못_보면_판정하지_않는다():
    """안 움직인 것과 못 본 것은 다르다."""
    w = DriveStallWatch()
    out = None
    t = 0.0
    while t <= STALL_SEC + 2.0:
        out = w.update(t, "yaw-", False, 0.41, 1.00, 70.8)
        t += 0.1
    assert out is None


# ── 각도 경계 ──────────────────────────────────────────────────────────────


def test_각도가_0도를_넘어가도_회전으로_센다():
    """359도 -> 1도는 2도 회전이지 358도가 아니다."""
    w = DriveStallWatch()
    w.update(0.0, "yaw-", True, 0.41, 1.00, 359.0)
    out = w.update(1.0, "yaw-", True, 0.41, 1.00, 1.0)   # 2도만 돌았다
    assert out is None          # 아직 문턱 전
    # 2도는 STALL_TURN_DEG(3도) 아래라 "안 움직임"으로 남아야 한다
    assert _hold(w, STALL_SEC + 1.0, yaw=1.0) is not None


def test_움직이면_시계가_다시_시작한다():
    """찔끔씩이라도 진짜로 움직이는 동안은 경보가 안 떠야 한다."""
    w = DriveStallWatch()
    out = None
    t = 0.0
    yaw = 70.8
    while t <= STALL_SEC * 3:
        if int(t * 10) % 40 == 0:      # 4초마다 5도씩 실제로 돈다
            yaw += 5.0
        out = w.update(t, "yaw-", True, 0.41, 1.00, yaw)
        t += 0.1
    assert out is None
