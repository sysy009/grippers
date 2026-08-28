"""탑뷰+ArUco 픽업 → 이동 → 내려놓기 미션 상태머신.

Host PC 가 하는 일은 여기까지다: 로봇 pose 와 기물 지도를 보고 "지금 뭘
해야 하는지"(mode)와 "다음 목표 좌표"를 계산해서 VehicleLink 로 넘긴다.

실제로 집고 내려놓는 동작(SmolVLA, 그리퍼캠+차량 RGB캠)은 전부 차량이 자기
카메라로 알아서 한다 — Host PC 는 그 계산을 하지 않는다. Host PC 가 하는
판단은 "충분히 가까워졌다"(GRASP_TRIGGER_DIST_M / PLACE_TRIGGER_DIST_M) 뿐이고,
GRASP·PLACE 상태에서는 차량이 "끝났다"고 보고할 때까지 좌표 계산 없이
기다리기만 한다.

라벨을 고정하지 않고, 매 SEARCH_TARGET 사이클마다 "지금 로봇 위치에서 가장
가까운 기물"을 라벨 무관하게 고른다. 목적지는 그 기물의 라벨로
mission_config.PIECE_DEST_BOX 에서 자동으로 찾는다(체스말 → chess 상자,
나머지 → toy 상자). 하나 내려놓으면 끝나지 않고 다시 SEARCH_TARGET 으로
돌아가 반복한다 — 화면에 더 이상 기물이 안 보일 때까지(작업 영역 밖, 즉
상자 자리에 내려놓은 기물은 다음 후보에서 자동으로 빠진다. config.WORKSPACE_Y
참고) 계속 돈다.
"""

from __future__ import annotations

import math
import sys
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import mission_config as mcfg

# config.py/localizer.py 는 aruco/ 하위폴더에 있다(팀원이 동기화하는 파일이라
# 건드리지 않는다) — 그 폴더를 경로에 추가해서 기존처럼 bare import 로 쓴다.
sys.path.insert(0, str(Path(__file__).parent / "aruco"))

import config as cfg
from localizer import Pose, box_pose
from navigator import GridPathPlanner, DriveCommand, DriveMode, DriveSequencer
from vehicle_link import (BACK_OFF, CREEP_IN, RE_AIM, SHIFT, MissionCommand,
                          VehicleLink)

XY = tuple[float, float]
PieceMap = dict[str, list[XY]]


class State(Enum):
    SEARCH_TARGET = auto()     # 기물이 지도에 보일 때까지 대기
    APPROACH_PIECE = auto()    # 목표 기물 앞까지 접근(회피 포함)
    GRASP = auto()             # 차량이 SmolVLA 로 집는 동안 대기
    GRASP_ALIGN = auto()       # Pi 가 "영역 밖이다, 다시 세워 달라"(GRASP_BLOCKED) 해서 재정렬 중
    CARRY_TO_DEST = auto()     # 목적지까지 이동(회피 포함)
    FACE_BOX = auto()          # 상자 앞 도착 후 정해진 방향(BOX_FACE_YAW_DEG)으로 제자리 회전
    NUDGE_BOX = auto()         # 그 방향으로 BOX_NUDGE_M 만큼만 더 전진하고 정지
    PLACE = auto()             # 차량이 SmolVLA 로 내려놓는 동안 대기
    INSERT_ALIGN = auto()      # Pi 가 "투하 조건이 안 맞는다"(INSERT_BLOCKED) 해서 재정렬 중
    HALTED = auto()            # 사람이 개입해야 한다. 물체를 든 채 갈 곳이 없을 때
    DONE = auto()


def _other_pieces(piece_map: PieceMap, exclude_xy: Optional[XY] = None,
                   tol: float = 0.05) -> list[XY]:
    """지도에 있는 모든 기물 좌표. exclude_xy 를 주면 그 근처(tol 이내) 한 점만 뺀다.

    라벨 전체를 빼지 않고 좌표로 빼는 이유: 같은 라벨의 다른 기물(예: 폰이
    여러 개)은 여전히 진짜 장애물이기 때문이다.
    """
    pts = [xy for pts in piece_map.values() for xy in pts]
    if exclude_xy is None:
        return pts
    return [p for p in pts if math.hypot(p[0] - exclude_xy[0], p[1] - exclude_xy[1]) > tol]


def _nearest_piece(piece_map: PieceMap, robot_xy: XY,
                   skip: Optional[list[XY]] = None) -> Optional[tuple[str, XY]]:
    """작업 영역(WORKSPACE_X x WORKSPACE_Y) 안에 있는 기물 중 로봇과 가장
    가까운 (라벨, 좌표).

    y 가 WORKSPACE_Y 밖(상자 자리)이면 이미 옮겨놓은 것으로 보고 후보에서
    뺀다 — 안 그러면 방금 내려놓은 기물을 바로 또 집으러 간다. x 가
    WORKSPACE_X(=방 전체 폭 0~1.8m) 밖이면 물리적으로 있을 수 없는 자리라
    오검출로 보고 뺀다.
    """
    wx0, wx1 = cfg.WORKSPACE_X
    wy0, wy1 = cfg.WORKSPACE_Y
    best: Optional[tuple[str, XY]] = None
    best_d = math.inf
    for label, pts in piece_map.items():
        for p in pts:
            if not (wx0 <= p[0] <= wx1 and wy0 <= p[1] <= wy1):
                continue
            # 재정렬을 다 써도 못 집은 기물은 후보에서 뺀다 — 안 그러면 같은
            # 기물 앞에서 영원히 재정렬만 반복한다. 라벨이 아니라 좌표로 빼는
            # 이유는 _other_pieces 와 같다(같은 라벨의 다른 개체는 살려둔다).
            if skip and any(math.hypot(p[0] - s[0], p[1] - s[1]) <= mcfg.SKIP_RADIUS_M
                            for s in skip):
                continue
            d = (p[0] - robot_xy[0]) ** 2 + (p[1] - robot_xy[1]) ** 2
            if d < best_d:
                best, best_d = (label, p), d
    return best


def _box_front_xy(box_name: str) -> XY:
    """상자 "중심"이 아니라 상자 앞(작업영역 쪽)에서 멈출 좌표.

    config.BOXES 의 상자들은 전부 뒤쪽 벽에 붙어 있고(y 큰 쪽) 작업영역
    (y 작은 쪽)을 향해 열려 있다 — 그래서 상자 절반 길이(BOX_L/2)만큼
    안쪽으로 빼고, 거기에 mission_config.BOX_APPROACH_MARGIN_M 만큼 더
    뗀 지점을 목적지로 준다. 상자 중심을 그대로 목적지로 주면 로봇이
    상자 안쪽까지 들어가려고 한다.
    """
    bx, by, _byaw = box_pose(box_name)
    offset = cfg.BOX_L / 2.0 + mcfg.BOX_APPROACH_MARGIN_M
    return (bx, by - offset)


def _send_drive(link: VehicleLink, pose: Pose, status: str, nav: DriveCommand,
                 target_label: Optional[str] = None) -> str:
    """DriveSequencer/FACE_BOX 가 낸 명령을 MissionCommand 로 옮겨 보내고,
    실제로 보낸 cmd 문자열을 돌려준다(LiveMap 표시용 — fsm.last_cmd).

    ROTATE 는 그대로 안 보내고 "yaw+"/"yaw-" 로 방향까지 정해서 보낸다 —
    yaw_error_deg(목표-현재, CCW가 양수) 부호가 그대로 회전 방향이다. 차량은
    그 방향으로 계속 돌고 있다가 "stop" 이 오면 멈추기만 하면 되고, 각도
    계산은 전혀 할 필요가 없다.
    """
    if nav.mode == DriveMode.FORWARD:
        cmd = "go"
    elif nav.mode == DriveMode.STOP:
        cmd = "stop"
    else:   # ROTATE
        cmd = "yaw+" if nav.yaw_error_deg >= 0 else "yaw-"
    link.send(MissionCommand(
        cmd, status, pose.x, pose.y, pose.yaw_deg, target_label=target_label,
    ))
    return cmd


class MissionFSM:
    def __init__(self, manual_mode: bool = False) -> None:
        """manual_mode=True 면 조건이 충족돼도 상태를 자동으로 안 넘기고,
        request_advance() 가 불릴 때까지 기다린다 — LiveMap 의 Next 버튼용.
        조건 충족 여부는 매 사이클 self.ready_to_advance 에 반영된다(수동
        모드가 아니어도 참고용으로 계속 갱신됨)."""
        self.manual_mode = manual_mode
        self.ready_to_advance = False
        self._advance_requested = False
        self._back_requested = False
        self._insert_align = None
        self._insert_align_from = None
        self._insert_align_tries = 0
        self._insert_tries = 0
        self.halt_reason = None
        self._path_planner = GridPathPlanner()
        self._drive = DriveSequencer()
        self.reset()

    def request_advance(self) -> None:
        """수동 모드에서 "다음" 버튼을 눌렀을 때 부른다. 조건이 아직 안
        맞았으면(ready_to_advance=False) 아무 효과 없다 — 조건이 맞는
        순간 바로 다음 상태로 넘어간다."""
        self._advance_requested = True

    def request_back(self) -> None:
        """"이전" 버튼 — 한 단계 전 상태로 되돌아간다. ready_to_advance
        조건과 무관하게 항상 즉시 적용된다(자동 모드에서도 동작 — 뒤로가기는
        "조건 충족"이 아니라 사용자 판단이라서). 실제 차량이 붙어 있다면
        GRASP/PLACE 를 넘어 되돌아가는 건 Host PC 쪽 목표만 되돌릴 뿐 —
        차량이 이미 물리적으로 집었거나/내려놨다면 그 동작 자체가 취소되진
        않는다(지금은 차량 없이 시험하는 용도)."""
        self._back_requested = True

    def _go_back(self) -> None:
        prev = {
            State.APPROACH_PIECE: State.SEARCH_TARGET,
            State.GRASP: State.APPROACH_PIECE,
            State.GRASP_ALIGN: State.GRASP,
            State.CARRY_TO_DEST: State.GRASP,
            State.FACE_BOX: State.CARRY_TO_DEST,
            State.NUDGE_BOX: State.FACE_BOX,
            State.PLACE: State.NUDGE_BOX,
            State.INSERT_ALIGN: State.PLACE,
            # HALTED 에서 뒤로가기는 **사람이 복구했다는 뜻**이다. 다시 세우는
            # 것부터 하도록 FACE_BOX 로 돌린다.
            State.HALTED: State.FACE_BOX,
        }.get(self.state)
        if prev is None:
            return   # SEARCH_TARGET 은 맨 앞이라 더 되돌아갈 데가 없다
        if prev == State.SEARCH_TARGET:
            self.target_label = None
            self._target_xy = None
            self.dest_xy = None
        self._path_planner.reset()
        self._drive.reset()
        self.nav_goal = None
        self.nav_corner = None
        self.nav_path = None
        self.last_nav = None
        self.last_cmd = None
        self._nudge_from = None
        self.ready_to_advance = False
        self.state = prev

    def set_manual_mode(self, manual: bool) -> None:
        """자동/수동 모드를 바꾸고 처음부터 다시 시작한다 — LiveMap 의 Mode
        버튼용. 도중에 모드만 바꾸면 "지금 이 상태로 계속 자동 진행할지"가
        애매해지므로, 모드 전환은 항상 reset() 과 같이 묶는다."""
        self.manual_mode = manual
        self.reset()

    def _should_advance(self) -> bool:
        if not self.manual_mode:
            return True
        if self._advance_requested:
            self._advance_requested = False
            return True
        return False

    def reset(self) -> None:
        """모든 상태를 처음(SEARCH_TARGET)으로 되돌린다 — LiveMap 리셋 버튼용.

        차량이 지금 뭘 하고 있었는지는 모르므로, 실제 차량이 붙어 있는
        상태에서 이 버튼을 누르면 Host PC 쪽 목표만 잊고 처음부터 다시
        찾는다는 뜻이다(차량 동작 자체를 멈추라는 명령은 아님) — 지금은
        차량 없이 시험하는 용도로만 쓴다.
        """
        self.state = State.SEARCH_TARGET
        self.target_label: Optional[str] = None
        self._target_xy: Optional[XY] = None
        self.dest_xy: Optional[XY] = None
        self.ready_to_advance = False
        self._advance_requested = False
        self._back_requested = False

        # LiveMap 이 "지금 어디로 가는 중인지" 그릴 수 있게 마지막 계산을 남겨둔다.
        # goal 은 이번 단계의 최종 목적지(기물 또는 상자), corner 는 축정렬
        # 경로가 꺾이는 모서리(축 하나가 이미 끝났으면 None), nav 는
        # DriveSequencer 가 낸 이번 사이클 명령이다. 이동 중이 아니면 다 None.
        self.nav_goal: Optional[XY] = None
        self.nav_corner: Optional[XY] = None
        self.nav_path: Optional[list[XY]] = None   # 계획기가 낸 전체 경로(화면용)
        self.last_nav: Optional[DriveCommand] = None
        self.last_cmd: Optional[str] = None   # 실제로 보낸 "go"/"stop"/"yaw+"/"yaw-"
        self._nudge_from: Optional[XY] = None   # NUDGE_BOX 진입 시점의 위치

        # GRASP_ALIGN 용. _align 은 지금 수행 중인 보정, _align_from 은 그
        # 보정을 시작한 시점의 pose(얼마나 움직였는지 재는 기준),
        # _align_tries 는 이 기물에 대해 재정렬을 몇 번 했는지다.
        self._align = None
        self._align_from: Optional[tuple[float, float, float]] = None
        self._align_tries = 0
        # INSERT_ALIGN 용. 위 셋과 같은 역할이되 바구니 쪽이다. **따로 두는
        # 이유**는 GRASP 보정과 INSERT 보정이 서로 새는 것을 막기 위해서다
        # (vehicle_link 의 슬롯 분리와 같은 이유).
        self._insert_align = None
        self._insert_align_from: Optional[tuple[float, float, float]] = None
        self._insert_align_tries = 0
        self._insert_tries = 0        # INSERT_FAILED 재시도 횟수

        # HALTED 로 멈춘 이유. 화면과 콘솔에 그대로 보여준다.
        self.halt_reason: Optional[str] = None

        # 재정렬을 다 쓰고도 못 집은 기물 좌표. SEARCH_TARGET 후보에서 뺀다.
        self.skipped: list[XY] = []

        self._path_planner.reset()
        self._drive.reset()

    def _approach(self, pose: Pose, robot_xy: XY, target_xy: XY,
                  obstacles: list[XY], target_label: Optional[str],
                  link: VehicleLink) -> float:
        """target_xy 로 축정렬 경로를 한 사이클 진행시키고, 실제 목표까지의
        (사선) 직선거리를 돌려준다 — GRASP/PLACE 전이 판정은 부분목표가
        아니라 이 값으로 해야 한다(부분목표는 중간 지점일 뿐이라, 그걸로
        판정하면 아직 한 축 남았는데 도착했다고 착각한다)."""
        self.nav_goal = target_xy
        # 회피는 GridPathPlanner 가 격자 탐색으로 전부 처리한다 —
        # DriveSequencer/next_waypoint 쪽엔 장애물을 안 넘긴다. 거기 회피
        # 로직은 가장 가까운 장애물 하나만 보고 우회점을 잡아서, 서로 밀어내는
        # 장애물 두 개 사이에서 영원히 왕복하는 버그가 있었다.
        sub_goal, corner, blocked_by = self._path_planner.update(
            robot_xy, pose.yaw_deg, target_xy, obstacles)
        self.nav_corner = corner
        self.nav_path = self._path_planner.last_path
        nav = self._drive.update(robot_xy, pose.yaw_deg, sub_goal, [])
        nav.blocked_by = blocked_by
        self.last_nav = nav
        self.last_cmd = _send_drive(link, pose, self.state.name, nav, target_label=target_label)
        return math.hypot(target_xy[0] - robot_xy[0], target_xy[1] - robot_xy[1])

    def _skip_target(self, why: str) -> None:
        """지금 대상을 보류하고 SEARCH_TARGET 으로 돌아간다.

        좌표를 `skipped` 에 남기는 것이 핵심이다 — 안 남기면 SEARCH_TARGET 이
        같은 기물을 또 "가장 가까운 것"으로 골라 무한 반복한다."""
        if self._target_xy is not None:
            self.skipped.append(self._target_xy)
        print(f"[mission] {self.target_label} 보류: {why}")
        self._align = None
        self._align_from = None
        self._insert_align = None
        self._insert_align_from = None
        self._insert_align_tries = 0
        self._insert_tries = 0
        self.target_label = None
        self._target_xy = None
        self.dest_xy = None
        self.ready_to_advance = False
        self.last_cmd = None
        self._path_planner.reset()
        self._drive.reset()
        self.state = State.SEARCH_TARGET

    def _halt(self, why: str) -> None:
        """물체를 든 채 갈 곳이 없을 때 멈춘다. **사람이 개입해야 풀린다.**

        `_skip_target` 과 나누는 기준은 **손이 비었는가** 하나다. 손이 비었으면
        다음 기물로 가면 되지만, 물체를 든 채로는 SEARCH_TARGET 으로 돌아가도
        새 기물을 집을 수 없다 — 그리퍼가 차 있으니 Pi 가 GRASP 를 거부한다.
        그런 상태로 계속 도는 것보다 멈춰서 보이는 쪽이 낫다.

        수동 모드에서 Prev 를 누르면 FACE_BOX 로 돌아가 다시 시도한다."""
        self.halt_reason = why
        print(f"[mission] ⛔ 멈춤: {why}")
        self._insert_align = None
        self._insert_align_from = None
        self.ready_to_advance = False
        self.last_cmd = "stop"
        self._path_planner.reset()
        self._drive.reset()
        self.state = State.HALTED

    def step(self, pose: Pose, piece_map: PieceMap, link: VehicleLink) -> State:
        if not pose.ok:
            # 로봇을 잃으면 이번 사이클은 그냥 넘어간다 — 명령을 안 보내면
            # 차량 쪽 워치독이 알아서 멈춘다(마지막 좌표로 계속 가면 안 됨).
            return self.state

        if self._back_requested:
            self._back_requested = False
            self._go_back()
            return self.state

        robot_xy = (pose.x, pose.y)

        if self.state == State.SEARCH_TARGET:
            self.nav_goal = None
            self.nav_corner = None
            self.nav_path = None
            self.last_nav = None
            self.last_cmd = None
            found = _nearest_piece(piece_map, robot_xy, self.skipped)
            self.ready_to_advance = found is not None
            if found is not None and self._should_advance():
                label, xy = found
                dest_box = mcfg.PIECE_DEST_BOX.get(label)
                if dest_box is None:
                    # 목적지 매핑이 없는 라벨 — 건드리지 않고 다음 후보를 기다린다.
                    self.ready_to_advance = False
                else:
                    self.target_label, self._target_xy = label, xy
                    self.dest_xy = _box_front_xy(dest_box)
                    # 재정렬 예산은 **대상 1개** 스코프다. 미션 누적으로 두면
                    # 첫 기물이 예산을 다 쓴 뒤 나머지가 전부 첫 시도에서
                    # 보류된다. 되돌리는 자리는 대상이 바뀌는 여기 하나뿐이다.
                    self._align_tries = 0
                    self._path_planner.reset()   # 새 구간 시작
                    self._drive.reset()
                    self.ready_to_advance = False
                    self.state = State.APPROACH_PIECE

        elif self.state == State.APPROACH_PIECE:
            assert self._target_xy is not None
            obstacles = _other_pieces(piece_map, exclude_xy=self._target_xy)
            dist = self._approach(pose, robot_xy, self._target_xy, obstacles,
                                   self.target_label, link)
            self.ready_to_advance = dist <= mcfg.GRASP_TRIGGER_DIST_M
            if self.ready_to_advance and self._should_advance():
                self.ready_to_advance = False
                self.state = State.GRASP

        elif self.state == State.GRASP:
            self.nav_goal = None
            self.nav_corner = None
            self.nav_path = None
            self.last_nav = None
            self.last_cmd = "stop"
            link.send(MissionCommand("stop", "GRASP", pose.x, pose.y, pose.yaw_deg,
                                      target_label=self.target_label))
            # poll_status() 는 한 번 물으면 그 응답을 소비한다(다시 물으면
            # IDLE) — 그래서 GRASP_DONE 을 본 뒤로는 다시 안 묻고 그 사실을
            # ready_to_advance 에 붙들어 둔다(수동 모드에서 버튼 누를 때까지
            # 여러 사이클 걸릴 수 있어서, 매번 새로 물으면 신호를 놓친다).
            if not self.ready_to_advance:
                status = link.poll_status()
                if status == "GRASP_DONE":
                    self.ready_to_advance = True
                elif status == "FAILED":
                    # Pi 는 두 독립 신호(그리퍼 부하 + 뎁스에서 물체가 사라짐)를
                    # 다 확인하고서야 실패를 낸다. 같은 자리에서 다시 시켜도
                    # 같은 결과일 가능성이 높으므로 **재시도하지 않고** 보류한다.
                    # 보류가 가능한 이유는 아직 손이 비어 있어서다 — 투하 실패
                    # (State.PLACE)는 물체를 든 채라 정책이 다르다.
                    self._skip_target("Pi 파지 실패(GRASP_FAILED)")
                    return self.state

            # Pi 가 "조건이 안 맞는다, 수정된 명령을 달라"고 했으면 재정렬로
            # 넘어간다. 여기서 아무것도 안 하면 Pi 는 계속 기다리고 Host 는
            # 계속 GRASP 를 보내서 영원히 멈춰 있다 — Pi 의 계약이 "스스로
            # 고쳐서 진행하지 않는다"이므로 움직이는 쪽은 Host 뿐이다.
            correction = link.take_correction()
            if correction is not None and not self.ready_to_advance:
                if not correction.actionable:
                    # E-STOP·미실측 상수·그리퍼가 안 비었음 등. 차를 움직여도
                    # 안 풀리므로 이 기물은 보류하고 다음으로 간다.
                    self._skip_target(f"고칠 수 없음 — {correction.detail}")
                elif self._align_tries >= mcfg.GRASP_ALIGN_MAX_TRIES:
                    self._skip_target(
                        f"재정렬 {self._align_tries}회 소진 — {correction.detail}")
                else:
                    self._align = correction
                    self._align_from = (pose.x, pose.y, pose.yaw_deg)
                    self._align_tries += 1
                    self.ready_to_advance = False
                    self.state = State.GRASP_ALIGN
                return self.state

            if self.ready_to_advance and self._should_advance():
                self.ready_to_advance = False
                self._path_planner.reset()   # 새 구간 시작
                self._drive.reset()
                self.state = State.CARRY_TO_DEST

        elif self.state == State.GRASP_ALIGN:
            # 한 걸음만 움직이고 GRASP 로 돌아간다. Pi 가 다시 관측해서
            # 여전히 안 맞으면 또 BLOCKED 를 보내고, 그때 한 걸음 더 간다.
            self.nav_goal = None
            self.nav_corner = None
            self.nav_path = None
            self.last_nav = None
            assert self._align is not None and self._align_from is not None
            fx, fy, fyaw = self._align_from

            if self._align.kind is RE_AIM or self._align.kind == RE_AIM:
                # lateral 은 + 가 왼쪽이다. 물체가 왼쪽에 있으면 턱을 왼쪽으로
                # 돌려야 하므로 반시계(yaw+)다.
                want = mcfg.GRASP_ALIGN_YAW_STEP_DEG
                sign = 1.0 if (self._align.lateral_mm or 0.0) >= 0 else -1.0
                turned = abs((pose.yaw_deg - fyaw + 180.0) % 360.0 - 180.0)
                done = turned >= want
                cmd = "stop" if done else ("yaw+" if sign > 0 else "yaw-")
            else:
                moved = math.hypot(pose.x - fx, pose.y - fy)
                done = moved >= mcfg.GRASP_ALIGN_STEP_M
                cmd = "stop" if done else ("back" if self._align.kind == BACK_OFF
                                           else "go")

            link.send(MissionCommand(cmd, "GRASP_ALIGN", pose.x, pose.y,
                                      pose.yaw_deg, target_label=self.target_label))
            self.last_cmd = cmd
            self.ready_to_advance = done
            if done and self._should_advance():
                self._align = None
                self._align_from = None
                self.ready_to_advance = False
                self.state = State.GRASP

        elif self.state == State.CARRY_TO_DEST:
            assert self.dest_xy is not None
            obstacles = _other_pieces(piece_map)
            dist = self._approach(pose, robot_xy, self.dest_xy, obstacles, None, link)
            self.ready_to_advance = dist <= mcfg.PLACE_TRIGGER_DIST_M
            if self.ready_to_advance and self._should_advance():
                self.ready_to_advance = False
                self.state = State.FACE_BOX

        elif self.state == State.FACE_BOX:
            # 상자 앞엔 도착했지만 아직 방향이 안 맞을 수 있다(어느 축으로
            # 마지막에 들어왔는지에 따라 다름) — PLACE 로 넘어가기 전에
            # 항상 정해진 방향(BOX_FACE_YAW_DEG, map 기준 "12시"=+y)을 보고
            # 서게 만든다. next_waypoint 를 또 쓸 필요 없이(목적지에 이미
            # 도착했으므로 이동은 없고 회전만 필요) 방위각 오차만 직접 계산한다.
            self.nav_goal = None
            self.nav_corner = None
            self.nav_path = None
            yaw_err = (mcfg.BOX_FACE_YAW_DEG - pose.yaw_deg + 180.0) % 360.0 - 180.0
            aligned = abs(yaw_err) <= mcfg.DRIVE_YAW_TOLERANCE_DEG
            self.ready_to_advance = aligned
            nav = DriveCommand(
                mode=DriveMode.STOP if aligned else DriveMode.ROTATE,
                waypoint=robot_xy, target_yaw_deg=mcfg.BOX_FACE_YAW_DEG,
                yaw_error_deg=yaw_err, dist_to_target=0.0, blocked_by=None,
            )
            self.last_nav = nav
            self.last_cmd = _send_drive(link, pose, "FACE_BOX", nav)
            if aligned and self._should_advance():
                self.ready_to_advance = False
                self._nudge_from = None
                self.state = State.NUDGE_BOX

        elif self.state == State.NUDGE_BOX:
            # 도착 판정은 목적지에서 PLACE_TRIGGER_DIST_M 떨어진 자리에서
            # 걸린다 — 거기서 바로 내려놓으면 상자와 멀다. 방향을 맞춘 뒤
            # (FACE_BOX) 그 방향으로 BOX_NUDGE_M 만큼만 붙여 준다.
            #
            # 계획기를 안 쓴다. 5 cm 직진이라 경로랄 게 없고, 이 구간은
            # 목적지가 이미 기물 회피구역과 겹칠 수 있어서(상자 앞에 기물이
            # 서 있는 경우) 계획기에 맡기면 "길 없음"이 나온다.
            self.nav_goal = None
            self.nav_corner = None
            self.nav_path = None
            if self._nudge_from is None:
                self._nudge_from = robot_xy
            heading = math.radians(mcfg.BOX_FACE_YAW_DEG)
            goal = (self._nudge_from[0] + mcfg.BOX_NUDGE_M * math.cos(heading),
                    self._nudge_from[1] + mcfg.BOX_NUDGE_M * math.sin(heading))
            moved = math.hypot(robot_xy[0] - self._nudge_from[0],
                               robot_xy[1] - self._nudge_from[1])
            yaw_err = (mcfg.BOX_FACE_YAW_DEG - pose.yaw_deg + 180.0) % 360.0 - 180.0
            aligned = abs(yaw_err) <= mcfg.DRIVE_YAW_TOLERANCE_DEG
            done = moved >= mcfg.BOX_NUDGE_M
            # 전진 중에 방위가 틀어지면 다시 맞춘다 — 5 cm 라도 비스듬히
            # 들어가면 상자 정면에 안 선다.
            if done:
                mode = DriveMode.STOP
            elif not aligned:
                mode = DriveMode.ROTATE
            else:
                mode = DriveMode.FORWARD
            self.ready_to_advance = done
            nav = DriveCommand(
                mode=mode, waypoint=goal, target_yaw_deg=mcfg.BOX_FACE_YAW_DEG,
                yaw_error_deg=yaw_err,
                dist_to_target=max(mcfg.BOX_NUDGE_M - moved, 0.0), blocked_by=None,
            )
            self.last_nav = nav
            self.last_cmd = _send_drive(link, pose, "NUDGE_BOX", nav)
            if done and self._should_advance():
                self.ready_to_advance = False
                self._nudge_from = None
                self.state = State.PLACE

        elif self.state == State.PLACE:
            self.nav_goal = None
            self.nav_corner = None
            self.nav_path = None
            self.last_nav = None
            self.last_cmd = "stop"
            link.send(MissionCommand("stop", "PLACE", pose.x, pose.y, pose.yaw_deg))
            if not self.ready_to_advance:
                status = link.poll_status()
                if status == "PLACE_DONE":
                    self.ready_to_advance = True
                elif status == "FAILED":
                    # 파지 실패와 달리 **보류할 수 없다** — 물체를 들고 있어서
                    # 다음 기물로 갈 수가 없다. 상자 앞에서 다시 세우고
                    # 재시도하되, 예산을 다 쓰면 멈춰서 사람을 부른다.
                    self._insert_tries += 1
                    if self._insert_tries > mcfg.INSERT_RETRY_MAX:
                        self._halt(f"투하 {self._insert_tries - 1}회 재시도 실패 — "
                                   "물체를 든 채로 멈춘다")
                        return self.state
                    print(f"[mission] 투하 실패 {self._insert_tries}/"
                          f"{mcfg.INSERT_RETRY_MAX} — 상자 앞에서 다시 세운다")
                    self._insert_align_tries = 0
                    self._path_planner.reset()
                    self._drive.reset()
                    self.state = State.FACE_BOX
                    return self.state

            # Pi 가 "투하 조건이 안 맞는다"(INSERT_BLOCKED)고 했으면 대응한다.
            # 여기서 아무것도 안 하면 Pi 는 계속 기다리고 Host 는 계속 PLACE 를
            # 보내서 영원히 멈춰 있다 — GRASP 쪽과 같은 구조다.
            correction = link.take_insert_correction()
            if correction is not None and not self.ready_to_advance:
                if correction.transient:
                    # "차체가 아직 정지하지 않았다" 같은 것. 가만히 있으면
                    # 풀린다 — 여기서 움직이면 판독이 또 흔들려서 조건이
                    # 영영 안 맞는다. 명령을 바꾸지 않고 다음 사이클을 기다린다.
                    pass
                elif not correction.actionable:
                    self._halt(f"투하 조건을 Host 가 못 고친다 — {correction.detail}")
                    return self.state
                elif self._insert_align_tries >= mcfg.INSERT_ALIGN_MAX_TRIES:
                    self._halt(f"투하 재정렬 {self._insert_align_tries}회 소진 — "
                               f"{correction.detail}")
                    return self.state
                else:
                    self._insert_align = correction
                    self._insert_align_from = (pose.x, pose.y, pose.yaw_deg)
                    self._insert_align_tries += 1
                    self.state = State.INSERT_ALIGN
                    return self.state
            if self.ready_to_advance and self._should_advance():
                # 하나 끝났다고 멈추지 않는다 — 다음 기물을 다시 찾는다.
                # 화면에 기물이 더 없으면 SEARCH_TARGET 에서 계속 대기한다.
                self.ready_to_advance = False
                self.target_label = None
                self._target_xy = None
                self.dest_xy = None
                self._insert_tries = 0
                self._insert_align_tries = 0
                self.state = State.SEARCH_TARGET

        elif self.state == State.INSERT_ALIGN:
            # GRASP_ALIGN 과 같은 구조: 한 걸음만 움직이고 PLACE 로 돌아간다.
            # 다른 점은 걸음이 잘고(창이 ±15mm 라 30mm 면 건너뛴다) 좌우
            # 이동(SHIFT)이 있다는 것이다.
            self.nav_goal = None
            self.nav_corner = None
            self.nav_path = None
            self.last_nav = None
            assert self._insert_align is not None
            assert self._insert_align_from is not None
            fx, fy, fyaw = self._insert_align_from
            kind = self._insert_align.kind
            sign = 1.0 if (self._insert_align.lateral_mm or 0.0) >= 0 else -1.0

            if kind == RE_AIM:
                # yaw_error 가 양수면 바구니 면의 법선이 왼쪽을 향한다 =
                # 반시계로 돌아야 맞춰진다 (basket_lidar_align.py 의
                # yaw_error = atan2(ny, nx), 정면 +x · 왼쪽 +y).
                turned = abs((pose.yaw_deg - fyaw + 180.0) % 360.0 - 180.0)
                done = turned >= mcfg.INSERT_ALIGN_YAW_STEP_DEG
                cmd = "stop" if done else ("yaw+" if sign > 0 else "yaw-")
            elif kind == SHIFT:
                # 좌우 오차를 회전으로 고치면 거리와 yaw 가 같이 틀어져서 여섯
                # 조건을 동시에 흔든다. 메카넘 횡이동이면 다른 조건을 안 건드린다.
                moved = math.hypot(pose.x - fx, pose.y - fy)
                done = moved >= mcfg.INSERT_ALIGN_LATERAL_STEP_M
                cmd = "stop" if done else ("left" if sign > 0 else "right")
            else:
                moved = math.hypot(pose.x - fx, pose.y - fy)
                done = moved >= mcfg.INSERT_ALIGN_STEP_M
                cmd = "stop" if done else ("back" if kind == BACK_OFF else "go")

            link.send(MissionCommand(cmd, "INSERT_ALIGN", pose.x, pose.y,
                                      pose.yaw_deg))
            self.last_cmd = cmd
            self.ready_to_advance = done
            if done and self._should_advance():
                self._insert_align = None
                self._insert_align_from = None
                self.ready_to_advance = False
                self.state = State.PLACE

        elif self.state == State.HALTED:
            # 물체를 든 채 서 있는다. 워치독에 걸려 Pi 가 멋대로 멈추지 않도록
            # 명령은 계속 보낸다 — 보내는 것은 stop 이라 차는 안 움직인다.
            self.nav_goal = None
            self.nav_corner = None
            self.nav_path = None
            self.last_nav = None
            self.last_cmd = "stop"
            link.send(MissionCommand("stop", "HALTED", pose.x, pose.y, pose.yaw_deg))

        return self.state
