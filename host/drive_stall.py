"""명령은 나가는데 차가 안 움직이는 것을 탑뷰로 잡는다 (2026-09-07).

## 왜 필요한가

이 차량에는 **바퀴가 실제로 도는지 아는 수단이 없다.** `/odom_raw` 는
`/cmd_vel` 을 적분해 되돌려줄 뿐이고(odom_publisher 가 구독하는 것이
cmd_vel·set_odom 뿐이다), 바퀴에 엔코더 피드백이 없다. 그래서 모터가
죽어도 오도메트리는 멀쩡해 보인다.

`domain/task/base_liveness.py` 가 이미 그 한계를 적어 뒀다 — "바퀴가 실제로
도는지는 모른다... **진짜 확인은 오버헤드 ArUco(Host 소유)나 사람 눈이다**".
이 모듈이 그 "오버헤드 ArUco" 쪽이다.

## 무엇을 잡는가

    2026-08-28  NUDGE_BOX 에서 go 가 155회 나가는 동안 1mm 도 안 움직였다
    2026-09-07  APPROACH_PIECE 에서 yaw- 를 34.7초 보내는 동안 yaw 폭 1.33도

둘 다 소프트웨어는 끝까지 정상이었다 — 명령이 바퀴 보드까지 정확히 갔다.
사람이 눈으로 보기 전까지 아무도 몰랐고, 그 사이 시간을 태웠다.

## 문턱의 근거 (2026-09-07 실측)

정지 상태에서 탑뷰 잡음은 이 정도다:

    x, y   폭 6~7mm      표준편차 0.2~1.3mm
    yaw    폭 1.3~3.4도  표준편차 0.23~0.53도

그래서 이동 15mm·회전 3도를 "안 움직였다"의 상한으로 잡는다 — 잡음보다
크고, 정상 주행이 8초에 내는 값보다는 한참 작다(0.6rad/s 면 8초에 275도).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

#: 이 시간 동안 계속 명령했는데도 안 움직이면 정체로 본다.
STALL_SEC = 8.0
#: 그 사이 움직인 거리(m)·회전(도)이 **둘 다** 이 아래면 정체다.
STALL_MOVE_M = 0.015
STALL_TURN_DEG = 3.0

#: 차를 실제로 움직이는 명령. "stop"·None 은 정체 판정에서 뺀다 —
#: 세워 둔 것을 안 움직인다고 하면 안 된다.
MOVING_CMDS = frozenset({"go", "back", "left", "right", "yaw+", "yaw-"})


def _turn_deg(a: float, b: float) -> float:
    """두 각(도) 사이의 최단 회전량. 359도와 1도는 2도 차이다."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass
class _Anchor:
    t: float
    x: float
    y: float
    yaw: float


class DriveStallWatch:
    """명령이 나가는 동안의 실제 이동을 지켜본다.

    ⚠️ 판정만 한다 — 멈추거나 상태를 바꾸지 않는다. 명령이 안 닿는 상황이라
    소프트웨어로 세울 수 없고(base_liveness 의 2026-08-28 기록), 자동으로
    미션을 접어도 차가 서지 않는다. 할 수 있는 일은 **사람을 부르는 것**뿐이다.
    """

    def __init__(self, stall_sec: float = STALL_SEC,
                 move_m: float = STALL_MOVE_M, turn_deg: float = STALL_TURN_DEG):
        self.stall_sec = stall_sec
        self.move_m = move_m
        self.turn_deg = turn_deg
        self._anchor: Optional[_Anchor] = None
        self.stalled_for: Optional[float] = None

    def reset(self) -> None:
        """명령이 끊겼거나 차가 움직였다 — 처음부터 다시 센다."""
        self._anchor = None
        self.stalled_for = None

    def update(self, now: float, cmd: Optional[str],
               pose_ok: bool, x: float, y: float, yaw_deg: float) -> Optional[float]:
        """이번 사이클을 반영하고, 정체 중이면 **정체 지속 시간(초)** 을 준다.

        포즈를 못 보면(`pose_ok=False`) 판정하지 않는다 — 안 움직인 것과
        못 본 것은 다르다."""
        if cmd not in MOVING_CMDS or not pose_ok:
            self.reset()
            return None

        if self._anchor is None:
            self._anchor = _Anchor(now, x, y, yaw_deg)
            return None

        a = self._anchor
        moved = math.hypot(x - a.x, y - a.y)
        turned = _turn_deg(yaw_deg, a.yaw)
        if moved >= self.move_m or turned >= self.turn_deg:
            # 움직였다 — 기준점을 여기로 옮기고 시계를 다시 시작한다.
            self._anchor = _Anchor(now, x, y, yaw_deg)
            self.stalled_for = None
            return None

        held = now - a.t
        self.stalled_for = held if held >= self.stall_sec else None
        return self.stalled_for
