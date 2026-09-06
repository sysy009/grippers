"""GRASP 진입 전 겨눔 — 왼쪽 편향 보정과 게이트 상시 적용 (2026-09-02).

## 왜 이 기능이 생겼나

09:45 실기: 파지 자체는 잘 됐지만, GRASP 에 들어갈 때마다 기물이 그리퍼
화면에서 계속 왼쪽에 남았다(사용자 확인 — 시도마다 정도가 비슷하고 매번
왼쪽, 3~5cm, GRASP_TRIGGER_DIST_M=0.40m 기준 각도로 환산하면 약 4~7도).

Host 의 겨냥 계산(`_yaw_error_to_target_deg`)은 오버헤드 카메라 좌표와
ArUco pose 만 쓰는 순수 기하라 임의의 방향으로 쏠릴 이유가 없다 — 매번
비슷한 크기로 한쪽에 쏠린다는 것은 ArUco 마커/그리퍼캠 광축이 차체
정면축과 어긋난 고정 장착 오차 쪽이 유력하다. 그래서 `mission_config.
PIECE_AIM_YAW_TRIM_DEG` 만큼을 계산에 더해 보정한다.

⚠️ 07:12 rock 실기(52도 미스얼라인, GRASP_REPLAN 기능의 계기) 대응 때는
이 정밀 겨눔 게이트(`_facing_target`)를 GRASP_REPLAN 이 보낸 재접근에만
켰다(`_tight_yaw_gate`) — "평소 APPROACH_PIECE 는 거리만 본다"고 명시
했었다. 그런데 그 "평소" 경로가 경로 추종(DriveSequencer)이 남긴 헤딩을
그대로 쓰다 보니, 큰 사고는 없어도 매번 몇 도씩 쏠린 채로 GRASP 에
들어가고 있었다. 트림을 더해도 게이트가 평소엔 안 걸리면 트림 자체가
적용될 자리가 없으므로, 이 게이트를 모든 APPROACH_PIECE 접근에 상시
적용하도록 넓혔다.
"""

from __future__ import annotations

import pytest

import math
import sys
from pathlib import Path

_HOST = Path(__file__).resolve().parent.parent / "host"
sys.path.insert(0, str(_HOST))
sys.path.insert(0, str(_HOST / "aruco"))

import mission_config as mcfg                       # noqa: E402
from mission import MissionFSM, State                 # noqa: E402

from conftest import PiSim                             # noqa: E402

_TARGET_XY = (1.0, 1.0)


def test_트림_상수가_계산된_오차에_그대로_더해진다():
    fsm = MissionFSM()
    fsm._target_xy = _TARGET_XY
    robot_xy = (1.0, 0.6)   # 목표는 정북(90도) 방향
    from mission import Pose
    pose = Pose(x=robot_xy[0], y=robot_xy[1], yaw_deg=90.0,
                ok=True, n_cams=2, fresh=True)

    err = fsm._yaw_error_to_target_deg(pose, robot_xy)

    # 기하만 보면 오차는 0(정확히 정북을 보고 있다) — 거기에 트림이
    # 그대로 얹힌다.
    # ⚠️ 2026-09-06 밤에 뒤집혔다. 트림은 이제 **차체를 안 돌린다** —
    # 겨눔과 정면 게이트가 쓰는 이 값에는 안 들어가고, GRASP 가 Pi 로 보내는
    # yaw_correction_deg 에만 더해져 servo 1 이 흡수한다.
    #
    # 왜: 트림을 여기 더했더니 정면 게이트(허용 6도)가 영영 안 닫혀 GRASP
    # 진입 자체가 막혔다(실기 626번 주행, 팔 호출 0건). 차체 회전이
    # bang-bang 이라 한 번에 약 10도씩 넘어가서 12도 창에 못 앉는다.
    assert err == pytest.approx(0.0, abs=1e-9), (
        "겨눔 오차에 트림이 들어가면 정면 게이트가 안 닫힌다")


def test_기하로는_허용치_안이어도_트림_때문에_더_돌아야_할_수_있다():
    """왜 게이트를 상시로 넓혔는지 보여 주는 핵심 사례.

    순수 기하 오차가 GRASP_REPLAN_YAW_TOLERANCE_DEG 안(옛 기준으로는
    "이미 정면")이어도, 트림을 더한 실제 오차는 허용치를 넘을 수 있다 —
    이 경우 트림 없이 판단했다면 놓쳤을 왼쪽 편향이다."""
    fsm = MissionFSM()
    fsm._target_xy = _TARGET_XY
    from mission import Pose
    # 목표까지 기하 오차가 정확히 2도 나도록 로봇을 살짝 옆에 둔다.
    raw_err_deg = 2.0
    dist = 0.6
    bearing_to_target = 90.0
    yaw_deg = bearing_to_target - raw_err_deg
    robot_xy = (1.0 - dist * math.sin(math.radians(bearing_to_target)),
                1.0 - dist * math.cos(math.radians(bearing_to_target)))
    pose = Pose(x=robot_xy[0], y=robot_xy[1], yaw_deg=yaw_deg,
                ok=True, n_cams=2, fresh=True)

    # 트림 없는 기하 오차(2도)만으로는 허용치(6도) 안이라 "정면"으로
    # 봤을 상황.
    assert abs(raw_err_deg) <= mcfg.GRASP_REPLAN_YAW_TOLERANCE_DEG
    # 트림(5도)을 더하면 넘는다 — 이제는 "정면"이 아니다.
    assert not fsm._facing_target(pose, robot_xy)


def test_tight_yaw_gate_없이도_평소_접근이_안_맞으면_바로_GRASP로_안_들어간다():
    """2026-09-02까지는 `_tight_yaw_gate`(GRASP_REPLAN 재접근)가 꺼져 있으면
    거리만 보고 바로 GRASP 로 넘어갔다 — 이제는 평소 접근도 겨눔부터
    본다."""
    fsm = MissionFSM()
    fsm.state = State.APPROACH_PIECE
    fsm.target_label = "rook"
    fsm._target_xy = _TARGET_XY
    assert fsm._tight_yaw_gate is False   # 재계획을 거치지 않은 평소 접근

    # 트리거 거리 안이지만 목표를 정면으로 보고 있지 않다(옆을 보고 있다).
    link = PiSim(x=_TARGET_XY[0],
                y=_TARGET_XY[1] - mcfg.GRASP_TRIGGER_DIST_M / 2,
                yaw_deg=90.0 + 45.0)

    fsm.step(link.pose(), {}, link)

    assert fsm.state == State.APPROACH_PIECE
    assert not fsm.ready_to_advance
    assert link.sent and link.sent[-1][0] in ("yaw+", "yaw-")

    # 계속 겨누면 결국 (트림이 반영된) 허용치 안에 들어와 GRASP 로 넘어간다.
    for _ in range(2000):
        fsm.step(link.pose(), {}, link)
        if fsm.state == State.GRASP:
            break
    assert fsm.state == State.GRASP
