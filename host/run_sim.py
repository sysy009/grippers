"""카메라·geti 모델·차량 없이 LiveMap 과 미션 로직만 돌려보는 리허설용 진입점.

시연 전에 화면 구성과 상태 전이를 확인하려는 용도다. run_mission.py 와 비교하면
바깥 껍데기(입력을 어디서 얻는가)만 다르고 안쪽은 전부 같은 코드가 돈다:

    run_mission.py :  카메라 -> ArUco(localizer) -> pose
                      카메라 -> geti(PieceTracker) -> 기물 지도
    run_sim.py     :  가상 로봇 적분              -> pose
                      고정 배치 + 집기/놓기 반영  -> 기물 지도

MissionFSM · navigator(GridPathPlanner/DriveSequencer) · LiveMap 은 실물과
똑같은 것을 그대로 쓴다. 그래서 여기서 보이는 ㄱ자 경로 · 회피 · 상태 전이 ·
Next 버튼 표시등은 실제 시연 때와 같은 로직의 결과다.

반대로 여기서 검증되지 '않는' 것: 카메라 개방/USB 대역폭, ArUco 검출과
캘리브레이션 오차, geti 추론 속도와 오검출, 실제 차량의 주행 오차. 그건
카메라와 모델을 붙이고 run_mission.py 로 확인해야 한다.

사용법
    python run_sim.py                  # 자동 진행
    python run_sim.py --step           # 수동 — LiveMap 의 Next 버튼으로 한 단계씩
    python run_sim.py --speed 0.5      # 가상 로봇을 2배 빠르게 (기본 0.25 m/s)
    python run_sim.py --noise 0.005    # pose 에 5 mm 지터를 섞어 ArUco 흔들림 흉내

q 또는 Ctrl+C, 또는 창을 닫으면 종료된다.
"""

from __future__ import annotations

import argparse
import math
import random
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# config.py/localizer.py 는 aruco/ 하위폴더에 있다(팀원이 동기화하는 파일이라
# 건드리지 않는다) — 그 폴더를 경로에 추가해서 기존처럼 bare import 로 쓴다.
sys.path.insert(0, str(Path(__file__).parent / "aruco"))

import config as cfg
from localizer import Pose

import mission_config as mcfg
from live_map import LiveMap
from mission import MissionFSM, State
from vehicle_link import MissionCommand, VehicleLink

XY = tuple[float, float]
PieceMap = dict[str, list[XY]]

# 가상 로봇 — 실물 속도를 재서 맞춘 값이 아니라 "눈으로 따라가기 좋은" 값이다.
SIM_SPEED_MPS = 0.25        # cmd="go" 일 때 전진 속도
SIM_YAW_RATE_DPS = 90.0     # cmd="yaw+"/"yaw-" 일 때 회전 속도
                            # 이 둘을 너무 올리면 한 사이클 이동량이 판정
                            # 허용오차를 넘어 목표 주위를 영영 맴돈다. 기본값
                            # 기준 회전 90/20 = 4.5도 < DRIVE_YAW_TOLERANCE_DEG(5.0),
                            # 전진 0.25/20 = 12.5 mm < AXIS_LEG_TOLERANCE_M(30 mm).
SIM_ACTION_SEC = 1.2        # GRASP/PLACE 를 차량이 수행하는 데 걸린다고 치는 시간
SIM_HZ = 20                 # 사이클 주기 (mission_config.WAYPOINT_HZ 는 10)

# 시작 기물 배치 — 전부 작업 영역(WORKSPACE_X x WORKSPACE_Y) 안에 둔다.
# 라벨은 geti 모델이 실제로 내는 6종(live_map.KNOWN_LABELS)과 같아야 LiveMap
# 이 전용 아이콘으로 그린다.
DEFAULT_PIECES: PieceMap = {
    "soccer": [(0.55, 0.70)],
    "star":   [(1.30, 0.62)],
    "box":    [(0.90, 1.05)],
    "queen":  [(1.45, 1.15)],
    "knight": [(0.35, 1.20)],
    "rook":   [(1.05, 0.48)],
}

# 로봇 시작 자세 — 작업 영역 앞(y < WORKSPACE_Y[0])에서 +y 를 보고 선다.
START_XY: XY = (0.90, 0.25)
START_YAW_DEG = 90.0

_stop = False


def _on_sigint(signum, frame):
    global _stop
    _stop = True


class SimVehicleLink(VehicleLink):
    """차량이 붙어 있는 척하는 링크.

    ConsoleVehicleLink 는 GRASP/PLACE 를 보내는 즉시 완료로 치는데, 그러면
    집고 놓는 순간이 한 사이클에 지나가 버려서 화면으로 확인할 수가 없다.
    여기서는 SIM_ACTION_SEC 만큼 "BUSY" 를 돌려주다가 완료를 보고한다.

    마지막으로 받은 명령(last)은 가상 로봇을 움직이는 데 쓴다 — 실제 차량이
    이 명령을 받아 모터를 돌리는 자리에 해당한다.
    """

    def __init__(self, action_sec: float = SIM_ACTION_SEC, quiet: bool = False) -> None:
        self.action_sec = action_sec
        self.quiet = quiet
        self.last: Optional[MissionCommand] = None
        self._done_at: Optional[float] = None
        self._action: Optional[str] = None

    def send(self, cmd: MissionCommand) -> None:
        self.last = cmd
        # 집기/놓기 판정은 cmd 가 아니라 status 로 한다 - 새 규격에서 그 두
        # 단계의 cmd 는 항상 "stop" 이라 cmd 만으로는 구분이 안 된다.
        if cmd.status in ("GRASP", "PLACE"):
            if self._action != cmd.status:        # 이 동작의 첫 사이클에만 타이머를 건다
                self._action = cmd.status
                self._done_at = time.monotonic() + self.action_sec
        else:
            self._action, self._done_at = None, None

        if not self.quiet:
            extra = f"target={cmd.target_label}" if cmd.target_label else ""
            print(f"\r[sim] {cmd.cmd:5s} [{cmd.status:14s}] "
                  f"robot=({cmd.robot_x:6.3f},{cmd.robot_y:6.3f},{cmd.robot_yaw_deg:6.1f}°) "
                  f"{extra:30s}", end="", flush=True)

    def poll_status(self) -> str:
        if self._done_at is not None and time.monotonic() >= self._done_at:
            done, self._action, self._done_at = f"{self._action}_DONE", None, None
            return done
        return "BUSY" if self._action else "IDLE"


class SimRobot:
    """DriveCommand 대로 움직이는 가상 차량.

    실제 차량과 같은 제약을 건다 — 사선으로 안 가고, "go" 는 지금 바라보는
    방향으로만 전진한다. 그래서 DriveSequencer 가 정렬을 제대로 못 시키면
    여기서도 목표를 못 맞춘다(= 로직 버그가 화면에 그대로 드러난다).

    새 규격에서는 명령에 목표 좌표도 목표 방위각도 안 들어온다("go"/"stop"/
    "yaw+"/"yaw-" 뿐). 그래서 여기서도 목표 앞에서 멈추도록 깎아 주지
    않는다 — 실제 Pi 도 그럴 수 없기 때문이다. 제때 "stop" 을 보내는 건
    전적으로 Host 몫이고, 못 보내면 지나치는 모습이 화면에 그대로 나온다.
    """

    def __init__(self, xy: XY = START_XY, yaw_deg: float = START_YAW_DEG,
                 speed: float = SIM_SPEED_MPS, yaw_rate: float = SIM_YAW_RATE_DPS) -> None:
        self.speed, self.yaw_rate = speed, yaw_rate
        self.reset()

    def reset(self) -> None:
        self.x, self.y = START_XY
        self.yaw_deg = START_YAW_DEG

    def apply(self, cmd: Optional[MissionCommand], dt: float) -> None:
        if cmd is None or cmd.cmd == "stop":
            return

        if cmd.cmd == "yaw+":                     # 반시계 (config 의 yaw 정의)
            self.yaw_deg = (self.yaw_deg + self.yaw_rate * dt + 180.0) % 360.0 - 180.0

        elif cmd.cmd == "yaw-":                   # 시계
            self.yaw_deg = (self.yaw_deg - self.yaw_rate * dt + 180.0) % 360.0 - 180.0

        elif cmd.cmd == "go":
            step = self.speed * dt
            self.x += step * math.cos(math.radians(self.yaw_deg))
            self.y += step * math.sin(math.radians(self.yaw_deg))

    def pose(self, noise_m: float = 0.0) -> Pose:
        jx = random.gauss(0.0, noise_m) if noise_m else 0.0
        jy = random.gauss(0.0, noise_m) if noise_m else 0.0
        jyaw = random.gauss(0.0, noise_m * 100.0) if noise_m else 0.0
        return Pose(x=self.x + jx, y=self.y + jy, yaw_deg=self.yaw_deg + jyaw,
                    ok=True, n_cams=2, age_s=0.0, fresh=True)


def _copy_pieces() -> PieceMap:
    return {label: list(pts) for label, pts in DEFAULT_PIECES.items()}


def _take_piece(pmap: PieceMap, label: str, near: XY) -> None:
    """label 기물 중 near 에 가장 가까운 하나를 지도에서 뺀다 (= 집었다)."""
    pts = pmap.get(label)
    if not pts:
        return
    idx = min(range(len(pts)),
              key=lambda i: math.hypot(pts[i][0] - near[0], pts[i][1] - near[1]))
    pts.pop(idx)
    if not pts:
        pmap.pop(label)


def _drop_piece(pmap: PieceMap, label: str) -> None:
    """상자 "중심"에 내려놓는다.

    로봇이 멈추는 자리(_box_front_xy)가 아니라 상자 중심에 넣는 게 맞다 —
    실제로도 팔이 상자 안까지 뻗는다. 상자 중심은 y 가 WORKSPACE_Y 밖이라
    mission._nearest_piece 의 후보에서 자동으로 빠진다(방금 넣은 걸 다시
    집으러 가지 않는 이유).
    """
    box_name = mcfg.PIECE_DEST_BOX.get(label)
    if box_name is None:
        return
    bx, by, _ = cfg.BOXES[box_name]
    pmap.setdefault(label, []).append((bx, by))


def _headless(args) -> int:
    """LiveMap 없이 통주행하고 요약만 낸다.

    **왜 따로 있는가** — `main()` 은 창이 닫힐 때까지 도는 구조라 자동 검증에
    쓸 수 없다. FSM 을 고칠 때마다 성공 경로가 그대로인지 봐야 하는데, 사람이
    창을 보고 판단하는 방법밖에 없었다.

    ⚠️ `SimVehicleLink` 의 GRASP/PLACE 완료 판정이 **벽시계**라, 사이클을 빨리
    돌리면 타이머가 안 끝나 GRASP 에서 멈춘 것처럼 보인다. 그래서 여기서는
    `action_sec=0` 으로 즉시 완료시킨다.

    ⚠️ 기물 총 개수로 판정하면 안 된다 — `_drop_piece` 가 상자 중심에 다시
    넣으므로 총합은 안 줄어든다. **작업 영역 안**에 남은 개수를 본다.
    """
    def in_ws(pmap) -> int:
        return sum(1 for pts in pmap.values() for x, y in pts
                   if cfg.in_workspace(x, y))

    robot = SimRobot(speed=args.speed, yaw_rate=args.yaw_rate)
    pieces = _copy_pieces()
    fsm = MissionFSM(manual_mode=args.step)
    link = SimVehicleLink(action_sec=0.0, quiet=True)

    total = in_ws(pieces)
    dt = 1.0 / SIM_HZ
    carried: Optional[str] = None
    prev = fsm.state
    transitions = 0
    seen = set()

    for i in range(args.max_cycles):
        pose = robot.pose(noise_m=args.noise)
        fsm.step(pose, pieces, link)
        if prev == State.GRASP and fsm.state == State.CARRY_TO_DEST:
            carried = fsm.target_label
            if carried:
                _take_piece(pieces, carried, (pose.x, pose.y))
        elif prev == State.PLACE and fsm.state == State.SEARCH_TARGET:
            if carried:
                _drop_piece(pieces, carried)
            carried = None
        if fsm.state != prev:
            transitions += 1
            seen.add(fsm.state.name)
        prev = fsm.state
        robot.apply(link.last if fsm.state not in
                    (State.SEARCH_TARGET, State.GRASP, State.PLACE) else None, dt)
        if in_ws(pieces) == 0:
            break

    left = in_ws(pieces)
    print(f"작업영역 기물 {total} -> {left} · 전이 {transitions}회 · {i + 1} 사이클")
    print("밟은 상태:", " ".join(sorted(seen)))
    print("최종 pose: (%.3f, %.3f, %.1f)" % (robot.pose().x, robot.pose().y,
                                             robot.pose().yaw_deg))
    if fsm.state is State.HALTED:
        print(f"⛔ HALTED — {fsm.halt_reason}")
    print("결과:", "OK" if left == 0 else "FAIL — 다 못 옮겼다")
    return 0 if left == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="카메라·모델·차량 없이 LiveMap 과 미션 로직만 돌리는 리허설")
    ap.add_argument("--step", action="store_true",
                    help="수동 모드로 시작 — LiveMap 의 Next 버튼으로 한 단계씩 진행")
    ap.add_argument("--speed", type=float, default=SIM_SPEED_MPS,
                    help=f"가상 로봇 전진 속도 m/s (기본 {SIM_SPEED_MPS})")
    ap.add_argument("--yaw-rate", type=float, default=SIM_YAW_RATE_DPS,
                    help=f"가상 로봇 회전 속도 deg/s (기본 {SIM_YAW_RATE_DPS})")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="pose 에 섞을 표준편차 m (예: 0.005 = 5 mm). 기본 0(무잡음)")
    ap.add_argument("--quiet", action="store_true", help="명령 로그를 찍지 않는다")
    ap.add_argument("--headless", action="store_true",
                    help="LiveMap 없이 끝까지 돌리고 결과만 낸다 (회귀 검증용). "
                         "기물을 다 옮기면 0, 못 옮기면 1 로 끝난다")
    ap.add_argument("--max-cycles", type=int, default=20000,
                    help="--headless 에서 이 사이클 안에 못 끝내면 실패로 본다")
    args = ap.parse_args()

    if args.headless:
        return _headless(args)

    signal.signal(signal.SIGINT, _on_sigint)

    robot = SimRobot(speed=args.speed, yaw_rate=args.yaw_rate)
    pieces = _copy_pieces()
    fsm = MissionFSM(manual_mode=args.step)
    link = SimVehicleLink(quiet=args.quiet)

    def _reset_all() -> None:
        robot.reset()
        pieces.clear()
        pieces.update(_copy_pieces())
        fsm.reset()
        print("\n[sim] 리셋 — 로봇/기물/미션 상태 초기화\n")

    def _toggle_mode() -> None:
        fsm.set_manual_mode(not fsm.manual_mode)
        _reset_all()
        print(f"[sim] 모드 -> {'MANUAL' if fsm.manual_mode else 'AUTO'}\n")

    live_map = LiveMap(on_reset=_reset_all, on_next=fsm.request_advance,
                       on_back=fsm.request_back, on_toggle_mode=_toggle_mode)

    print("\n시뮬레이션 시작 — 카메라·모델·차량 없이 미션 로직만 돕니다.")
    print(f"기물 {sum(len(v) for v in pieces.values())}개, "
          f"{'수동(Next 버튼으로 진행)' if args.step else '자동'} 모드")
    print("창을 닫거나 Ctrl+C 로 종료\n")

    dt = 1.0 / SIM_HZ
    carried: Optional[str] = None
    prev_state = fsm.state

    try:
        while not _stop:
            cycle_start = time.monotonic()
            pose = robot.pose(noise_m=args.noise)

            fsm.step(pose, pieces, link)

            # 집기/놓기는 상태가 실제로 넘어간 순간에만 반영한다 — GRASP 상태에
            # 머무는 동안은 아직 안 집은 것이다.
            if prev_state == State.GRASP and fsm.state == State.CARRY_TO_DEST:
                carried = fsm.target_label     # PLACE 뒤에는 지워지므로 지금 붙잡아 둔다
                if carried:
                    _take_piece(pieces, carried, (pose.x, pose.y))
                    print(f"\n[sim] 집음: {carried}\n")
            elif prev_state == State.PLACE and fsm.state == State.SEARCH_TARGET:
                if carried:
                    _drop_piece(pieces, carried)
                    print(f"\n[sim] 내려놓음: {carried} -> "
                          f"{mcfg.PIECE_DEST_BOX.get(carried)} 상자\n")
                carried = None
            prev_state = fsm.state

            robot.apply(link.last if fsm.state not in
                        (State.SEARCH_TARGET, State.GRASP, State.PLACE) else None, dt)

            live_map.update(pose, pieces, goal=fsm.nav_goal, nav=fsm.last_nav,
                            corner=fsm.nav_corner, path=fsm.nav_path,
                            state_name=fsm.state.name,
                            target_label=fsm.target_label,
                            ready=fsm.ready_to_advance, manual_mode=fsm.manual_mode,
                            cmd=fsm.last_cmd)
            if live_map.closed():
                break

            slept = time.monotonic() - cycle_start
            if slept < dt:
                time.sleep(dt - slept)
    finally:
        live_map.close()

    print(f"\n\n종료 — 마지막 상태: {fsm.state.name}, "
          f"작업 영역에 남은 기물 "
          f"{sum(1 for pts in pieces.values() for p in pts if cfg.in_workspace(*p))}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
