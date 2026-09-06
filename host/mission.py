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
import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import mission_config as mcfg

# config.py/localizer.py 는 aruco/ 하위폴더에 있다(팀원이 동기화하는 파일이라
# 건드리지 않는다) — 그 폴더를 경로에 추가해서 기존처럼 bare import 로 쓴다.
sys.path.insert(0, str(Path(__file__).parent / "aruco"))

import config as cfg
from localizer import Pose, box_pose
import basket_target
from navigator import GridPathPlanner, DriveCommand, DriveMode, DriveSequencer
from vehicle_link import (BACK_OFF, CREEP_IN, RE_AIM, GraspCorrection,
                          MissionCommand, VehicleLink)

XY = tuple[float, float]
PieceMap = dict[str, list[XY]]


class State(Enum):
    SEARCH_TARGET = auto()     # 기물이 지도에 보일 때까지 대기
    APPROACH_PIECE = auto()    # 목표 기물 앞까지 접근(회피 포함)
    GRASP = auto()             # 차량이 SmolVLA 로 집는 동안 대기
    GRASP_ALIGN = auto()       # Pi 가 "영역 밖이다, 다시 세워 달라"(GRASP_BLOCKED) 해서 재정렬 중
    GRASP_REPLAN = auto()      # GRASP_ALIGN이 반복되면 오버헤드 카메라로 크게 다시 세운다 (2026-09-02)
    CARRY_TO_DEST = auto()     # 목적지까지 이동(회피 포함)
    FACE_BOX = auto()          # 상자 접근 부채꼴 진입 후 목표중심 방향(동적, 2026-09-05)으로 제자리 회전
    NUDGE_BOX = auto()         # 그 방향으로 BOX_NUDGE_M 만큼만 더 전진하고 정지
    PLACE = auto()             # 차량이 SmolVLA 로 내려놓는 동안 대기
    RETURN_HOME = auto()       # 기물을 포기했거나 하나를 다 옮긴 뒤 mcfg.DEFAULT_HOME_XY 로 복귀 중
    DONE = auto()


def _in_workspace(p: XY) -> bool:
    """작업 영역(WORKSPACE_X x WORKSPACE_Y) 안인가.

    같은 판정이 _nearest_piece · _find_label · visible_labels 에 인라인으로
    들어 있다. 그 셋은 건드리지 않았다 — 여기서는 SEARCH_TARGET 이 "왜 후보가
    없는지" 설명할 때와 host/diag_pieces.py 가 쓴다.
    """
    wx0, wx1 = cfg.WORKSPACE_X
    wy0, wy1 = cfg.WORKSPACE_Y
    return wx0 <= p[0] <= wx1 and wy0 <= p[1] <= wy1


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
                   skip: Optional[list[XY]] = None,
                   category: Optional[str] = None) -> Optional[tuple[str, XY]]:
    """작업 영역(WORKSPACE_X x WORKSPACE_Y) 안에 있는 기물 중 로봇과 가장
    가까운 (라벨, 좌표).

    y 가 WORKSPACE_Y 밖(상자 자리)이면 이미 옮겨놓은 것으로 보고 후보에서
    뺀다 — 안 그러면 방금 내려놓은 기물을 바로 또 집으러 간다. x 가
    WORKSPACE_X(=방 전체 폭 0~1.8m) 밖이면 물리적으로 있을 수 없는 자리라
    오검출로 보고 뺀다.

    skip(재정렬/파지를 다 써도 못 집은 기물, MissionFSM.skipped)은 1순위
    후보에서 뺀다 — 안 그러면 같은 기물 앞에서 영원히 재정렬만 반복한다.
    라벨이 아니라 좌표로 빼는 이유는 _other_pieces 와 같다(같은 라벨의
    다른 개체는 살려둔다).

    category(2026-09-05, `--category` CLI 옵션 — mission_config.PIECE_DEST_BOX
    가 "chess"/"toy"로 정하는 그 값)를 주면, 목적지 상자가 그 카테고리인
    라벨만 후보로 본다 — 다른 카테고리 기물은 화면에 있어도 통째로
    무시한다("체스말만 정리해줘"). None(기본값)이면 예전처럼 라벨 무관.

    ⚠️ 다만 **그것 말고 후보가 하나도 없을 때만** skip 을 무시하고 다시
    찾는다(사용자 지시, 2026-09-01) — 다른 기물이 남아 있는 동안은 실패
    직후 같은 기물을 또 들이미는 낭비를 막고, 그것만 남았을 때는 영원히
    손 놓지 않고 다시 시도한다. 두 번째 시도도 실패하면 skip 에 다시
    쌓이므로(_skip_target), 다음 라운드도 "다른 후보가 없을 때만" 조건을
    똑같이 거쳐 재시도된다 — 여러 기물이 있는 정상 상황에서는 계속
    미뤄지고, 그것 하나만 남은 상황에서만 반복 시도가 벌어진다. category
    는 이 두 단계 모두에 똑같이 걸린다 — 다른 카테고리로는 아예 안 넘어간다.
    """
    def _search(use_skip: bool) -> Optional[tuple[str, XY]]:
        wx0, wx1 = cfg.WORKSPACE_X
        wy0, wy1 = cfg.WORKSPACE_Y
        best: Optional[tuple[str, XY]] = None
        best_d = math.inf
        for label, pts in piece_map.items():
            if category is not None and mcfg.PIECE_DEST_BOX.get(label) != category:
                continue
            for p in pts:
                if not (wx0 <= p[0] <= wx1 and wy0 <= p[1] <= wy1):
                    continue
                if use_skip and skip and any(
                        math.hypot(p[0] - s[0], p[1] - s[1]) <= mcfg.SKIP_RADIUS_M
                        for s in skip):
                    continue
                d = (p[0] - robot_xy[0]) ** 2 + (p[1] - robot_xy[1]) ** 2
                if d < best_d:
                    best, best_d = (label, p), d
        return best

    found = _search(use_skip=True)
    if found is None and skip:
        found = _search(use_skip=False)
    return found


def _find_label(piece_map: PieceMap, label: str, robot_xy: XY,
                skip: Optional[list[XY]] = None) -> Optional[tuple[str, XY]]:
    """특정 라벨만 대상으로 로봇과 가장 가까운 것 하나를 고른다 — 사용자
    지시(instruction_resolver.py 가 해석한 라벨)로 라벨이 정해졌을 때
    쓴다. 같은 라벨이 여러 개일 수 있어(예: 폰) 그중 가까운 걸 고른다.
    그 라벨이 지금 안 보이면 None.

    `_nearest_piece`와 조건(작업영역·skip)을 그대로 맞춘다 — 이 라벨의
    다른(skip 안 된) 개체가 남아 있는 동안은 포기한 것을 다시 들이밀지
    않되, **이 라벨에 그것 말고 후보가 하나도 없으면** skip 을 무시하고
    다시 찾는다(사용자 지시, 2026-09-01 — _nearest_piece 와 같은 이유).
    안 그러면 지시받은 라벨의 유일한 개체가 한 번 포기됐을 때 그 라벨이
    다시 보일 때까지(=영원히) 기다리게 된다."""
    def _search(use_skip: bool) -> Optional[XY]:
        wx0, wx1 = cfg.WORKSPACE_X
        wy0, wy1 = cfg.WORKSPACE_Y
        best_xy: Optional[XY] = None
        best_d = math.inf
        for p in piece_map.get(label, []):
            if not (wx0 <= p[0] <= wx1 and wy0 <= p[1] <= wy1):
                continue
            if use_skip and skip and any(
                    math.hypot(p[0] - s[0], p[1] - s[1]) <= mcfg.SKIP_RADIUS_M
                    for s in skip):
                continue
            d = (p[0] - robot_xy[0]) ** 2 + (p[1] - robot_xy[1]) ** 2
            if d < best_d:
                best_xy, best_d = p, d
        return best_xy

    best_xy = _search(use_skip=True)
    if best_xy is None and skip:
        best_xy = _search(use_skip=False)
    return (label, best_xy) if best_xy is not None else None


def visible_labels(piece_map: PieceMap) -> list[str]:
    """작업 영역 안에 실제로 보이는 라벨만 정렬해서 돌려준다.

    run_mission.py 가 사용자 지시를 Claude 로 해석시킬 때, 화면에 없는
    라벨을 후보로 주면 안 보이는 걸 골라버릴 수 있어서 이 목록을 같이
    넘긴다(instruction_resolver.InstructionResolver.submit() 참고)."""
    wx0, wx1 = cfg.WORKSPACE_X
    wy0, wy1 = cfg.WORKSPACE_Y
    return sorted({
        label for label, pts in piece_map.items()
        if any(wx0 <= p[0] <= wx1 and wy0 <= p[1] <= wy1 for p in pts)
    })


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
    if box_name == "toy":
        # 2026-09-03 실기 — mission_config.TOY_DEST_X_SHIFT_LEFT_M 참고.
        bx -= mcfg.TOY_DEST_X_SHIFT_LEFT_M
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
    elif nav.mode == DriveMode.ESCAPE:
        # yaw+/yaw- 헌팅 워치독(navigator.DriveSequencer._enter_rotate) 이
        # 끼워 넣은, 정렬 무시한 짧은 전진 — cmd 상으로는 FORWARD와 같은
        # "go"지만 목표를 정면으로 보고 있다는 보장이 없다.
        cmd = "go"
    else:   # ROTATE
        cmd = "yaw+" if nav.yaw_error_deg >= 0 else "yaw-"
    link.send(MissionCommand(
        cmd, status, pose.x, pose.y, pose.yaw_deg, target_label=target_label,
    ))
    return cmd


class MissionFSM:
    def __init__(self, manual_mode: bool = False,
                category: Optional[str] = None) -> None:
        """manual_mode=True 면 조건이 충족돼도 상태를 자동으로 안 넘기고,
        request_advance() 가 불릴 때까지 기다린다 — LiveMap 의 Next 버튼용.
        조건 충족 여부는 매 사이클 self.ready_to_advance 에 반영된다(수동
        모드가 아니어도 참고용으로 계속 갱신됨).

        category(2026-09-05, run_mission.py의 `--category chess`/`--category
        toy`)를 주면, SEARCH_TARGET의 기본 "가장 가까운 것" 규칙이 그
        카테고리(mission_config.PIECE_DEST_BOX 값) 기물만 후보로 본다 — 실행
        내내 유지되는 필터다(자연어 지시 한 번짜리 override와 다르다).
        사용자가 터미널에 직접 친 지시(set_instruction)는 이 필터를
        무시하고 원하는 라벨로 바로 간다 — 명시적으로 지목한 것까지 막을
        이유는 없어서다."""
        self.manual_mode = manual_mode
        self.category = category
        self.ready_to_advance = False
        self._advance_requested = False
        self._back_requested = False
        self._path_planner = GridPathPlanner()
        self._drive = DriveSequencer()
        self.reset()

    def request_advance(self) -> None:
        """수동 모드에서 "다음" 버튼을 눌렀을 때 부른다. 조건이 아직 안
        맞았으면(ready_to_advance=False) 아무 효과 없다 — 조건이 맞는
        순간 바로 다음 상태로 넘어간다."""
        self._advance_requested = True

    def set_instruction(self, target_label: str, dest_xy: Optional[XY] = None) -> bool:
        """사용자 지시(instruction_resolver.py 가 해석한 라벨)를 처리
        대상으로 삼는다 — "가장 가까운 기물" 규칙을 한 번 덮어쓴다
        (사용자 지시, 2026-09-01 — 팀원의 2026-08-31 handoff 델타의
        set_instruction()을 이 저장소의 현재 구조에 맞춰 이식).

        `dest_xy`를 주면(자연어 지시가 "fetch" 의도일 때, 예: "퀸 가져와")
        그 좌표로 옮긴다 — instruction_resolver.py 의 intent 판단에 따라
        run_mission.py 가 mission_config.DELIVER_HERE_XY 를 넘겨준다.
        안 주면(기본값, "organize" 의도나 라벨만 말한 경우) 기존처럼
        PIECE_DEST_BOX 로 정해지는 상자로 옮긴다.

        손이 비어있으면(아직 안 집었으면, SEARCH_TARGET/APPROACH_PIECE)
        그 즉시 지금 하던 걸 버리고 이 라벨로 전환한다. 이미 뭔가 집어서
        옮기는 중(GRASP 이후 — GRASP_ALIGN 포함)이면 무리해서 끼어들지
        않고, 지금 들고 있는 걸 상자에 넣는 것까지 마친 뒤(PLACE 완료 ->
        SEARCH_TARGET 복귀 시점에) 자동으로 적용되도록 큐에 쌓아둔다 —
        들고 있던 걸 그냥 놓아버리는 안전하지 않은 동작을 피하기 위함.

        반환값: 손이 비어서 즉시 반영됐으면 True, 지금 하던 일을 마치고
        나중에 적용되도록 큐에 쌓였으면 False (run_mission.py 가 이 값으로
        LiveMap 피드백 문구를 다르게 보여준다)."""
        if self.state in (State.SEARCH_TARGET, State.APPROACH_PIECE):
            self._instructed_label = target_label
            self._instructed_dest_xy = dest_xy
            if self.state == State.APPROACH_PIECE:
                # 지금 쫓던 기물을 버리고 새 지시로 즉시 재탐색.
                self.state = State.SEARCH_TARGET
                self.target_label = None
                self._target_xy = None
                self.dest_xy = None
                self.dest_box_name = None
                self._face_target_yaw_deg = None
                self.ready_to_advance = False
                self._path_planner.reset()
                self._drive.reset()
            return True
        self._queued_instruction_label = target_label
        self._queued_instruction_dest_xy = dest_xy
        return False

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
        }.get(self.state)
        if prev is None:
            return   # SEARCH_TARGET 은 맨 앞이라 더 되돌아갈 데가 없다
        if prev == State.SEARCH_TARGET:
            self.target_label = None
            self._target_xy = None
            self.dest_xy = None
            self.dest_box_name = None
            self._face_target_yaw_deg = None
            self._instructed_label = None   # 뒤로가기는 사용자 개입이라 지시도 같이 취소
            self._instructed_dest_xy = None
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
        # 지금 dest_xy가 어느 상자를 향한 것인지(config.BOXES의 키) — 상자가
        # 아니라 사용자에게 직접 가져다주는 경우("가져와")는 None이다.
        # CARRY_TO_DEST가 이 값의 유무로 부채꼴 게이트(basket_target.
        # check_approach_sector)를 쓸지, 기존 고정 지점 도착 판정을 쓸지
        # 가른다(2026-09-05, 사용자 지시).
        self.dest_box_name: Optional[str] = None
        # FACE_BOX/NUDGE_BOX가 정렬할 목표 방위각 — 상자를 향할 때는
        # CARRY_TO_DEST가 check_approach_sector()로 진입 지점마다 다르게
        # 계산해 넣는다(고정 mcfg.BOX_FACE_YAW_DEG가 아니다). None이면
        # 두 상태 모두 mcfg.BOX_FACE_YAW_DEG로 대체한다("가져와" 등 상자가
        # 없는 경우).
        self._face_target_yaw_deg: Optional[float] = None
        self.ready_to_advance = False
        self._advance_requested = False
        self._back_requested = False
        # 사용자 지시(instruction_resolver.py, 2026-09-01)로 정해진 다음
        # 목표 라벨과 목적지 오버라이드. set_instruction() 참고.
        self._instructed_label: Optional[str] = None
        self._instructed_dest_xy: Optional[XY] = None
        # 지금 든 것을 내려놓은 뒤 적용할 지시 — 손이 안 비어 있을 때
        # set_instruction() 이 여기 쌓아 둔다.
        self._queued_instruction_label: Optional[str] = None
        self._queued_instruction_dest_xy: Optional[XY] = None

        # 1회성 이벤트 — run_mission.py 가 매 사이클 읽고 소비한다
        # (읽었으면 None 으로 되돌려야 한다). "방금 파지/투입에 성공한
        # 그 개체"만 piece_map.PieceTracker.suppress_at() 으로 숨기려는
        # 것이다(2026-09-05, 사용자 지시) — 라벨 전체가 아니라 정확히 그
        # 좌표의 그 트랙 하나만 지도에서 빼야, 같은 라벨의 다른 개체(예:
        # rook 두 개 중 하나만 잡았을 때)가 계속 후보로 남는다.
        self.last_grasp_event: Optional[tuple[str, XY]] = None
        self.last_place_event: Optional[tuple[str, XY]] = None

        # LiveMap 이 "지금 어디로 가는 중인지" 그릴 수 있게 마지막 계산을 남겨둔다.
        # goal 은 이번 단계의 최종 목적지(기물 또는 상자), corner 는 축정렬
        # 경로가 꺾이는 모서리(축 하나가 이미 끝났으면 None), nav 는
        # DriveSequencer 가 낸 이번 사이클 명령이다. 이동 중이 아니면 다 None.
        self.nav_goal: Optional[XY] = None
        self.nav_corner: Optional[XY] = None
        self.nav_path: Optional[list[XY]] = None   # 계획기가 낸 전체 경로(화면용)
        self.last_nav: Optional[DriveCommand] = None
        self.last_cmd: Optional[str] = None   # 실제로 보낸 "go"/"stop"/"yaw+"/"yaw-"
        # 포즈를 잃었을 때 즉시 "stop"을 보내기 위한 표시용 좌표(2026-09-02,
        # step() 의 pose.ok 분기 참고) — 차량 제어에는 안 쓰인다.
        self._last_good_pose: Optional[Pose] = None
        self._nudge_from: Optional[XY] = None   # NUDGE_BOX 진입 시점의 위치
        # NUDGE_BOX 진입 시점의 방위(도) — axis가 rotate_left/rotate_right일
        # 때만 쓴다(제자리 회전이라 위치가 아니라 각도로 진행량을 잰다).
        self._nudge_yaw_from: Optional[float] = None
        # NUDGE_BOX 가 이번에 갈 (거리 m, 방향). PLACE 가 Pi 의 라이다 판독을
        # 보고 채운다. None 이면 첫 진입이라 기존 BOX_NUDGE_M 만큼만 붙인다.
        self._nudge_plan: Optional[tuple] = None
        # gate_ok/facing_ok(Host 자체 ArUco 판정)만으로 PLACE 전환을
        # 확정하기 전에 몇 사이클 연속으로 같은 결과가 나오는지 센다
        # (2026-09-05, 사용자 지시 — "정지 명령을 보낸 그 순간의 좌표"가
        # 아니라 "실제로 멈춘 뒤"의 좌표로 최종 판정하려는 것). 라이브
        # 안전장치(live_too_close/hard_stop/ready_early/Pi 라이다 fix)로
        # 끝난 경우는 이미 실측이라 debounce가 필요 없다 — 아래 step()
        # NUDGE_BOX 참고.
        self._nudge_gate_streak = 0
        # 진전 감시 — 마지막으로 인정한 이동량과 그 시각의 마감.
        self._nudge_best = 0.0
        # Pi 라이다 기준 오차(축에 맞게 forward_m/yaw_rad/lateral_m)로 본
        # 최선값 — 2026-09-03 실기: ArUco 자세 추정 노이즈만으로 moved 가
        # 흔들려서, 차가 눈으로는 안 움직였는데도 정지 감시가 한 번도 안
        # 걸린 사고가 있었다. Pi 값이 있으면 그쪽을 우선한다(아래 step()
        # NUDGE_BOX 참고).
        self._nudge_best_pi_error: Optional[float] = None
        self._nudge_stall_at = 0.0
        self._nudge_stall_warned = False
        # 바구니 앞 폐루프가 지금까지 쓴 총 이동량 — 예산 한계선용.
        self._basket_creep_used = 0.0
        # 같은 폐루프의 회전판 예산(rad) — 미터 예산과 단위가 달라 따로 둔다.
        self._basket_yaw_used = 0.0
        # PLACE에서 Pi가 REACQUIRE(방향 없음)를 연속으로 보낸 횟수 —
        # BASKET_LOST_REPLAN_AFTER_TRIES 넘으면 CARRY_TO_DEST로 크게
        # 다시 접근한다.
        self._basket_lost_tries = 0

        # GRASP_ALIGN 용. _align 은 지금 수행 중인 보정, _align_from 은 그
        # 보정을 시작한 시점의 pose(얼마나 움직였는지 재는 기준),
        # _align_tries 는 이 기물에 대해 재정렬을 몇 번 했는지다.
        self._align = None
        self._align_from: Optional[tuple[float, float, float]] = None
        self._align_tries = 0
        # ROTATE(RE_AIM) 가 연속으로 몇 번째인지(2026-09-01,
        # GRASP_REAIM_ESCALATE_AFTER_TRIES 용) — RE_AIM 이 아닌 보정이
        # 오거나 이 기물을 새로 잡을 때 0으로 되돌린다.
        self._reaim_tries = 0
        # 지금 GRASP_ALIGN 이 회전 미수렴 때문에 만든 "가짜" BACK_OFF 인지
        # (2026-09-01). 진짜 BACK_OFF(뎁스캠이 목표를 못 봄)와 달리, 이
        # 경우는 목표를 여전히 보고 있으므로 후진 한 걸음 뒤 좌우 스윕을
        # 돌 필요가 없다 — 곧장 GRASP 로 돌아가 다시 재본다.
        self._align_reaim_backoff = False
        # BACK_OFF 후진 뒤 좌우로 훑는 재탐색용(2026-09-01). None 이면 아직
        # 후진 단계거나 스윕이 필요 없는 보정 중이고, "left"/"right" 면 그
        # 방향으로 훑는 중이다. _sweep_phase_start 는 지금 방향(좌 또는 우)
        # 을 시작한 시각, _sweep_burst_until 은 이번 짧은 버스트가 끝나는
        # 시각 — 버스트가 끝날 때마다 멈추고 GRASP 로 넘겨 확인한다.
        self._align_sweep_stage: Optional[str] = None
        self._align_sweep_phase_start: Optional[float] = None
        self._align_sweep_burst_until: Optional[float] = None
        # 강제 파지(GRASP_FORCE, 2026-08-31) 용. _forcing_grasp 는 지금
        # Pi 에 강제 시도를 보내 놓고 결과를 기다리는 중인지, _forced_grasp_tries
        # 는 이 기물에 몇 번 강제했는지, _align_tries_at_last_force 는
        # 마지막으로 강제했을 때의 _align_tries 값이다(그 뒤로 재정렬이
        # 최소 한 번은 더 있어야 다시 강제한다 — 안 그러면 실패 직후 바로
        # 또 강제해서 GRASP_FORCE_MAX_ATTEMPTS 를 순식간에 다 쓴다).
        self._forcing_grasp = False
        self._forced_grasp_tries = 0
        self._align_tries_at_last_force: Optional[int] = None
        # 오버헤드 재계획(GRASP_REPLAN, 2026-09-02) 용 — GRASP_ALIGN 을
        # GRASP_REPLAN_AFTER_TRIES 번 반복해도 안 풀리면, Pi 의 좁은 정면
        # 뎁스캠 대신 오버헤드 카메라로 크게 물러났다 다시 세운다(07:12
        # rook 실기: yaw 52도로 어긋난 채 GRASP 에 들어가 정면 시야를
        # 완전히 벗어났고, 그 뒤 3cm 후진+스윕을 30번 반복해도 못 고쳤다).
        # _replan_tries/_align_tries_at_last_replan 은 강제 파지 카운터와
        # 같은 이유로 나뉜다 — 재계획 한 번 뒤 최소 한 다발(3회)은 더
        # 재정렬해 봐야 다음 재계획을 허용한다. _tight_yaw_gate 는 재계획
        # 뒤 재접근에서만 켜진다 — 평소 APPROACH_PIECE 는 거리만 본다.
        self._replan_tries = 0
        self._align_tries_at_last_replan: Optional[int] = None
        self._tight_yaw_gate = False
        self._replan_backoff_xy: Optional[XY] = None
        # 정렬 문제가 아니라 "순수 물리적"으로 파지가 반복 실패한 횟수
        # (mcfg.GRASP_FAIL_MAX_RETRIES, 2026-09-01 사용자 지시). GRASP_BLOCKED
        # 는 위 align_tries/forcing 쪽이 이미 상한을 관리하므로 겹치지 않는다.
        self._grasp_fail_tries = 0
        # 재정렬/파지를 다 쓰고도 못 집은 기물 좌표. 다른 후보가 남아 있는
        # 동안은 SEARCH_TARGET 후보에서 뺀다 — 단 그것 말고 후보가 하나도
        # 없어지면 다시 후보로 본다(_nearest_piece/_find_label 의 2단계
        # 탐색 참고, 사용자 지시 2026-09-01).
        # (좌표, 보류한 시각). mcfg.SKIP_EXPIRY_S 가 지나면 저절로 풀린다 —
        # 시효가 없으면 일시적인 고장까지 영구 보류가 되고, 푸는 방법이
        # Mode 버튼 두 번(= reset)뿐이라 아는 사람만 풀 수 있다.
        self.skipped: list[tuple[XY, float]] = []

        # SEARCH_TARGET 에서 후보를 못 고른 이유. 화면과 콘솔이 그대로 쓴다.
        self.search_reason = None

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
        #
        # GridPathPlanner 자신이 "로봇이 실제로 유의미하게 움직였는지"를
        # 보고 그렇지 않으면 직전 결과를 그대로 돌려준다(2026-09-06, 사용자
        # 지시로 근본 수정 — "제자리에서 이상하게 돈다". PATH_REPLAN_MIN_
        # MOVE_M 정의부 주석 참고) — 여기서는 그냥 매 사이클 부르기만 하면
        # 된다.
        sub_goal, corner, blocked_by = self._path_planner.update(
            robot_xy, pose.yaw_deg, target_xy, obstacles)
        self.nav_corner = corner
        self.nav_path = self._path_planner.last_path
        escape_count_before = self._drive.escape_count
        nav = self._drive.update(robot_xy, pose.yaw_deg, sub_goal, [])
        if self._drive.escape_count > escape_count_before:
            # 2026-09-05: 사용자가 "yaw 진동으로 시간이 지체된다"고 보고해서
            # 추가한 계측 — DriveSequencer.escape_count 정의부 참고. 값을
            # 바로 튜닝하지 않고 우선 눈에 보이게만 한다. 이게 자주 뜨면
            # DRIVE_YAW_TOLERANCE_DEG(현재 12도)를 실기 데이터로 다시 볼 것.
            print(f"[navigator] yaw 헌팅 감지 — 정렬 무시하고 "
                  f"{mcfg.ROTATE_OSCILLATION_ESCAPE_CYCLES}사이클 동안 전진합니다 "
                  f"(이번 실행 누적 {self._drive.escape_count}회)", flush=True)
        nav.blocked_by = blocked_by
        self.last_nav = nav
        self.last_cmd = _send_drive(link, pose, self.state.name, nav, target_label=target_label)
        return math.hypot(target_xy[0] - robot_xy[0], target_xy[1] - robot_xy[1])

    def _yaw_error_to_target_deg(self, pose: Pose, robot_xy: XY) -> float:
        """target_xy 를 정면으로 보려면 지금 헤딩에서 얼마나 더 돌아야 하는가.

        `_send_drive`의 ROTATE 관례와 부호를 맞춘다 — 목표-현재, CCW가
        양수. GRASP_REPLAN 재접근의 타이트한 yaw 게이트(_facing_target)와
        그 겨눔 동작(_aim_at_target)이 같이 쓴다.

        PIECE_AIM_YAW_TRIM_DEG(2026-09-02)를 더한다 — 순수 기하로는
        오차 0이어도 실제로는 기물이 그리퍼 화면에서 계속 왼쪽에 남는
        고정 편향이 관찰됐다. 여기 한 곳에서만 더해서 `_facing_target`
        (GRASP_REPLAN 게이트)과 `_aim_at_target`(제자리 겨눔)이 항상
        같은 보정된 오차를 보게 한다."""
        assert self._target_xy is not None
        desired = math.degrees(math.atan2(self._target_xy[1] - robot_xy[1],
                                          self._target_xy[0] - robot_xy[0]))
        raw_err = desired - pose.yaw_deg + mcfg.PIECE_AIM_YAW_TRIM_DEG
        return (raw_err + 180.0) % 360.0 - 180.0

    def _facing_target(self, pose: Pose, robot_xy: XY) -> bool:
        """지금 헤딩이 target_xy 방향으로 GRASP_REPLAN_YAW_TOLERANCE_DEG
        안에 들어와 있는가."""
        return (abs(self._yaw_error_to_target_deg(pose, robot_xy))
                <= mcfg.GRASP_REPLAN_YAW_TOLERANCE_DEG)

    def _aim_at_target(self, pose: Pose, robot_xy: XY, link: VehicleLink) -> None:
        """전후진 없이 제자리에서 target_xy 방향으로 돈다.

        경로 재계산(_approach)을 쓰지 않는 이유: 이미 GRASP_TRIGGER_DIST_M
        안이라 그걸 다시 목표로 걸면 더 다가가 기물을 밀어낼 수 있다.
        필요한 건 방향뿐이다."""
        err = self._yaw_error_to_target_deg(pose, robot_xy)
        cmd = "yaw+" if err >= 0 else "yaw-"
        self.last_cmd = cmd
        link.send(MissionCommand(cmd, "APPROACH_PIECE", pose.x, pose.y,
                                 pose.yaw_deg, target_label=self.target_label))

    def _active_skips(self) -> list[XY]:
        """아직 시효가 안 지난 보류 좌표들. 지난 것은 목록에서도 지운다."""
        now = time.monotonic()
        self.skipped = [(xy, t) for (xy, t) in self.skipped
                        if now - t < mcfg.SKIP_EXPIRY_S]
        return [xy for (xy, _t) in self.skipped]

    def _skip_target(self, why: str) -> None:
        """지금 대상을 보류하고 다음 라운드로 넘어간다.

        홈 복귀를 거칠지는 `mcfg.RETURN_HOME_ENABLED` 가 정한다(2026-09-06
        사용자 지시로 기본 꺼짐). 켜 두면 아래 설명대로 동작한다.

        SEARCH_TARGET 으로 곧장 돌아가지 않는 이유: 포기하는 자리는 실패한
        기물 코앞이거나 이상한 각도로 서 있을 수 있는 자리다. 그대로 다음
        탐색을 시작하면 매번 다른, 예측하기 어려운 자리에서 스캔하게 된다
        — RETURN_HOME 을 한 번 거치면 항상 같은 자리에서 다시 시작한다.

        좌표를 `skipped` 에 남기는 것이 핵심이다 — 안 남기면 SEARCH_TARGET 이
        같은 기물을 또 "가장 가까운 것"으로 골라 무한 반복한다."""
        if self._target_xy is not None:
            self.skipped.append((self._target_xy, time.monotonic()))
        print(f"[mission] {self.target_label} 보류: {why} — 기본 위치로 복귀합니다")
        self._align = None
        self._align_from = None
        self._align_sweep_stage = None
        self._align_sweep_phase_start = None
        self._align_sweep_burst_until = None
        self.target_label = None
        self._target_xy = None
        self.dest_xy = None
        self.dest_box_name = None
        self._face_target_yaw_deg = None
        self.ready_to_advance = False
        self.last_cmd = None
        self._path_planner.reset()
        self._drive.reset()
        self.state = self._next_round_state()

    def _apply_queued_instruction(self) -> None:
        """포기·투하 사이에 들어온 지시를 여기서 적용한다.

        `set_instruction()` 이 GRASP/GRASP_ALIGN 도 "손이 안 비었다"로 보고
        큐에 쌓아 두므로(2026-09-01), 라운드가 끝나는 지점마다 한 번씩
        비워 줘야 한다."""
        if self._queued_instruction_label is not None:
            self._instructed_label = self._queued_instruction_label
            self._instructed_dest_xy = self._queued_instruction_dest_xy
            self._queued_instruction_label = None
            self._queued_instruction_dest_xy = None

    def _next_round_state(self) -> "State":
        """라운드를 끝내고 갈 곳. 홈 복귀를 끄면 그 자리에서 바로 탐색한다.

        ⚠️ 홈 복귀를 건너뛸 때는 큐를 **여기서** 비워야 한다 — 원래는
        RETURN_HOME 완료 시점이 그 자리였다. 안 비우면 대기 중이던 지시가
        영영 적용되지 않는다."""
        if mcfg.RETURN_HOME_ENABLED:
            return State.RETURN_HOME
        self._apply_queued_instruction()
        return State.SEARCH_TARGET

    def begin_carrying(self, label: str) -> bool:
        """차량이 이미 `label` 을 들고 있다고 보고 CARRY_TO_DEST 부터 시작한다.

        중단된 실행을 이어서 끝낼 때 쓴다. 파지까지 성공해 놓고 투하에서
        막히면, 기물은 그리퍼 안에 있어 오버헤드 카메라에 안 보인다 —
        그 상태로 다시 돌리면 Host 는 "작업 영역에 기물 없음"으로 보고
        SEARCH_TARGET 에 영원히 서 있는다. 2026-08-28 실기가 그랬다.

        목적지 매핑이 없는 라벨이면 False. 그때는 어디로 나를지 모르므로
        추측하지 않는다."""
        dest_box = mcfg.PIECE_DEST_BOX.get(label)
        if dest_box is None:
            return False
        self.target_label = label
        self._target_xy = None
        self.dest_xy = _box_front_xy(dest_box)
        self.dest_box_name = dest_box
        self._face_target_yaw_deg = None
        self._align_tries = 0
        self._reaim_tries = 0
        self._nudge_from = None
        self._nudge_yaw_from = None
        self._nudge_plan = None
        self._basket_creep_used = 0.0
        self._basket_yaw_used = 0.0
        self._basket_lost_tries = 0
        self.ready_to_advance = False
        self._path_planner.reset()
        self._drive.reset()
        self.state = State.CARRY_TO_DEST
        return True

    def _retreat_for_overhead_reapproach(self) -> None:
        """바구니 폐루프(PLACE↔NUDGE_BOX)가 막혔다 — 라이다 평면을 못
        잡았거나(lost) 국소 보정 예산을 다 썼는데도 안 풀렸거나, 둘 중
        어느 쪽이든 국소 보정으로는 더 못 고친다는 뜻이다. 접고
        CARRY_TO_DEST 로 돌아가 오버헤드 좌표로 크게 다시 접근한다
        (PLACE 의 두 호출 지점이 똑같은 정리를 하길래 2026-09-03에 묶었다)."""
        self._basket_lost_tries = 0
        self._nudge_from = None
        self._nudge_yaw_from = None
        self._nudge_plan = None
        self._basket_creep_used = 0.0
        self._basket_yaw_used = 0.0
        self.ready_to_advance = False
        self._path_planner.reset()
        self._drive.reset()
        self.state = State.CARRY_TO_DEST

    def _plan_basket_fix(self, fix) -> Optional[tuple]:
        """Pi 가 준 바구니 판독 -> (이동량 m, 방향). 고칠 게 없으면 None.

        방향은 "forward" / "back" / "left" / "right" 다.

        한 번에 하나만 고친다. 거리를 먼저 맞추고 그 다음에 좌우를 본다 —
        Pi 의 좌우 추정은 바구니 가장자리가 방위각 창 안에 들어와야 나오는
        값이라 멀리서는 정확도가 떨어지고(basket_lidar_align 주석), 가까이
        붙고 나서 재는 쪽이 훨씬 믿을 만하다. Pi 의 corrections.from_insert
        가 매기는 우선순위와도 같다.

        총 이동량에 예산을 두는 이유: 판독이 이상해서 같은 방향 보정이 계속
        나오면 차가 바구니를 밀고 들어간다. 예산을 다 쓰면 더 안 움직이고
        Pi 의 거부를 그대로 사람에게 남긴다. 회전(rad)은 병진(m)과 단위가
        달라 예산을 따로 둔다 — 둘 중 하나가 소진돼도 다른 쪽 보정은 여전히
        낼 수 있어야 한다(거리는 맞는데 yaw만 계속 어긋나는 경우, 그 반대도
        마찬가지).

        ⚠️ 10:18 실기까지 yaw_rad 를 아예 안 읽었다 — corrections.from_insert
        는 거리 다음으로 yaw 를 보는데(우선순위는 위 참고), 여기는 lateral_m
        만 읽어서 "거리는 맞는데 라이다 평면 자체가 정면이 아니다"(yaw만
        어긋난 경우)에는 아무 계획도 못 만들고 PLACE 에서 INSERT_BLOCKED만
        반복했다."""
        if fix is None:
            return None
        remaining = mcfg.BASKET_CREEP_BUDGET_M - self._basket_creep_used

        # Pi 가 오차를 직접 계산해 줬으면 그것을 쓴다. 없으면 라이다 판독에서
        # Host 목표를 빼서 낸다(옛 Pi 빌드 대비). 부호는 둘 다 +가 "더 가야
        # 한다"이고, **-면 후진**이다 — 바구니에 너무 붙어 선 경우다.
        error = fix.forward_m
        if error is None and fix.distance_m is not None:
            error = fix.distance_m - mcfg.BASKET_TARGET_LIDAR_M
        if error is not None and remaining > 0.01:
            if abs(error) > mcfg.BASKET_DISTANCE_DEADBAND_M:
                return (min(abs(error), remaining),
                        "forward" if error > 0 else "back")

        if fix.yaw_rad is not None:
            yaw_remaining = mcfg.BASKET_YAW_BUDGET_RAD - self._basket_yaw_used
            if (yaw_remaining > 0.01
                    and abs(fix.yaw_rad) > mcfg.BASKET_YAW_DEADBAND_RAD):
                # yaw_rad 는 `_yaw_error_to_target_deg`와 같은 부호 관례다
                # (+가 CCW) — 그 방향으로 그만큼 더 돌면 라이다 평면 정면과
                # 맞아떨어진다(basket_lidar_align.fit_basket_face 주석:
                # yaw_error = atan2(ny, nx), 로봇을 +yaw_error 만큼 CCW로
                # 돌리면 그 값이 0으로 수렴한다).
                return (min(abs(fix.yaw_rad), yaw_remaining),
                        "rotate_left" if fix.yaw_rad > 0 else "rotate_right")

        if (fix.lateral_m is not None and remaining > 0.01
                and abs(fix.lateral_m) > mcfg.BASKET_LATERAL_DEADBAND_M):
            # lateral_m 은 바구니 중심이 로봇 기준 어디 있는지다(+가 왼쪽) —
            # 그 방향으로 가야 가운데에 선다.
            return (min(abs(fix.lateral_m), remaining),
                    "left" if fix.lateral_m > 0 else "right")
        return None

    def _grasp_sweep_step(self) -> tuple[str, bool]:
        """BACK_OFF 한 걸음 뒤 좌우로 훑는 한 걸음. (cmd, done) 을 낸다.

        Pi 의 관측(observe_target 다중 프레임 합의)은 정지 상태를 전제해서
        회전 "중"에는 물을 수 없다 — 그래서 짧게(GRASP_SWEEP_BURST_SEC) 돌리고
        멈춰서 GRASP 로 돌아가 확인하는 것을 반복한다. done=True 는 "이번
        버스트가 끝났으니 멈추고 확인하라"는 뜻이지, 스윕 전체가 끝났다는
        뜻이 아니다 — 여전히 같은 BACK_OFF 로 돌아오면(못 찾았으면)
        _align_sweep_stage 가 그대로 남아 있어 여기서 이어서 훑는다.

        좌(GRASP_SWEEP_LEFT_SEC)+우(GRASP_SWEEP_RIGHT_SEC) 예산을 다 써도
        못 찾으면 스윕을 접고(_align_sweep_stage=None) 평소 GRASP_ALIGN
        처럼 넘긴다 — 다음 BACK_OFF 는 후진부터 다시 시작한다. align_tries
        는 이 왕복들 내내 계속 누적되므로 GRASP_FORCE 안전망은 그대로
        살아 있다."""
        now = time.monotonic()
        assert self._align_sweep_stage is not None
        assert self._align_sweep_phase_start is not None
        assert self._align_sweep_burst_until is not None
        budget = (mcfg.GRASP_SWEEP_LEFT_SEC if self._align_sweep_stage == "left"
                  else mcfg.GRASP_SWEEP_RIGHT_SEC)
        phase_elapsed = now - self._align_sweep_phase_start
        if phase_elapsed >= budget:
            if self._align_sweep_stage == "left":
                self._align_sweep_stage = "right"
                self._align_sweep_phase_start = now
                self._align_sweep_burst_until = now + mcfg.GRASP_SWEEP_BURST_SEC
                return "yaw-", False
            # 좌우 다 훑었는데도 못 찾음 — 스윕 종료.
            self._align_sweep_stage = None
            self._align_sweep_phase_start = None
            self._align_sweep_burst_until = None
            return "stop", True
        if now >= self._align_sweep_burst_until:
            # 이번 버스트 끝 — 멈추고 GRASP 로 넘겨 확인한다.
            self._align_sweep_burst_until = now + mcfg.GRASP_SWEEP_BURST_SEC
            return "stop", True
        return ("yaw+" if self._align_sweep_stage == "left" else "yaw-"), False

    def step(self, pose: Pose, piece_map: PieceMap, link: VehicleLink) -> State:
        if not pose.ok:
            # 로봇을 잃으면 이번 사이클엔 새로 계산해 명령을 내지 않는다 —
            # 마지막 **유효** 좌표를 기준으로 계속 주행 명령을 내면 안 된다는
            # 원칙은 그대로다. 하지만 "아무 것도 안 보낸다"와 "정지를
            # 보낸다"는 다르다 — 직전 명령이 "go"/"back"/"left"/"right"/
            # "yaw±"처럼 움직이는 값이었다면, 차량 쪽은 새 명령이 올 때까지
            # 그걸 그대로 래치해 둔다(HOST_COMMAND_TIMEOUT_CYCLES, 6사이클
            # ≈0.4초 동안은 신호가 없어도 계속 움직인다는 뜻). "stop"은
            # 좌표를 계산할 필요가 없는 명령이라(HostCommand 에는 애초에
            # 좌표가 없다 — vehicle_link.py 참고) pose 없이도 지금 당장 낼
            # 수 있다.
            #
            # 09-02 실기: NUDGE_BOX로 바구니에 접근하던 중 ArUco 마커를
            # 놓쳐 pose.ok가 False가 됐는데, 그 뒤로 Host가 아무 명령도
            # 안 보내는 바람에 마지막 "go"가 워치독이 걸릴 때까지 그대로
            # 밀었다 — 그 블라인드 구간에 바구니와 충돌했다. 여기서 즉시
            # "stop"을 내면 그 구간이 한두 사이클로 줄어든다.
            if self.last_cmd not in (None, "stop"):
                last = self._last_good_pose
                x, y, yaw = ((last.x, last.y, last.yaw_deg)
                            if last is not None else (0.0, 0.0, 0.0))
                link.send(MissionCommand("stop", self.state.name, x, y, yaw))
                self.last_cmd = "stop"
            if self._align_sweep_phase_start is not None:
                # 스윕(_grasp_sweep_step) 도중 포즈를 잃으면 이 사이클은
                # 위 return으로 건너뛰어 yaw 명령이 실제로는 하나도 안
                # 나가는데, 예산은 벽시계 기준(now - phase_start)이라
                # 그동안도 그냥 흘러버린다 — 포즈가 돌아온 첫 호출에서
                # phase_elapsed가 이미 예산을 넘어 있어, 한 번도 안 돈
                # 채로 스윕이 "다 훑었다"고 착각하고 끝날 수 있다
                # (2026-09-01 코드 리뷰). 기준점을 지금 시각으로 계속
                # 밀어서 "회전하지 않은 이 시간"을 예산에서 뺀다 — 이
                # 스윕을 만든 계기 자체가 BACK_OFF 반복 중 포즈를 잃은
                # 사고였다는 걸 생각하면, 스윕이 실제로 도는 상황이 포즈를
                # 잃기 가장 쉬운 상황과 겹친다는 점도 이 처리가 필요한
                # 이유다.
                now = time.monotonic()
                self._align_sweep_phase_start = now
                self._align_sweep_burst_until = now + mcfg.GRASP_SWEEP_BURST_SEC
            return self.state

        self._last_good_pose = pose

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
            skips = self._active_skips()
            # 사용자 지시(instruction_resolver.py)로 라벨이 지정돼 있으면
            # 그 라벨만 찾는다 — 없으면 평소처럼 최근접 우선(2026-09-01).
            if self._instructed_label is not None:
                found = _find_label(piece_map, self._instructed_label, robot_xy,
                                    skips)
            else:
                found = _nearest_piece(piece_map, robot_xy, skips,
                                       category=self.category)
            self.ready_to_advance = found is not None
            # 왜 후보가 없는지 남긴다. "없음"에는 세 가지 다른 뜻이 있고 사람이
            # 할 일이 각각 다르다 — 기물을 놓아야 하는지, 보류가 풀리길
            # 기다려야 하는지, 영역 안으로 옮겨야 하는지. 이걸 구분해 주지
            # 않아서 2026-09-05 실기에서 검출이 멀쩡한데도 한참 헤맸다.
            if found is not None:
                self.search_reason = None
            elif not piece_map:
                self.search_reason = "지도에 기물 없음"
            elif any(_in_workspace(p) for pts in piece_map.values() for p in pts):
                self.search_reason = (f"보류됨 {len(skips)}개 — "
                                      f"{mcfg.SKIP_EXPIRY_S:.0f}초 뒤 재시도")
            else:
                self.search_reason = "기물이 작업 영역 밖"
            if found is not None and self._should_advance():
                label, xy = found
                # "fetch" 의도(사용자 지시로 목적지 오버라이드가 있음)면 그
                # 고정 좌표로, 아니면(기본값) 기존 라벨별 상자로.
                if self._instructed_dest_xy is not None:
                    dest_xy = self._instructed_dest_xy
                    dest_box = None   # 사용자에게 직접 가져다주는 경우 — 상자 아님
                else:
                    dest_box = mcfg.PIECE_DEST_BOX.get(label)
                    dest_xy = _box_front_xy(dest_box) if dest_box is not None else None
                if dest_xy is None:
                    # 목적지 매핑이 없는 라벨 — 건드리지 않고 다음 후보를 기다린다.
                    self.ready_to_advance = False
                else:
                    self.target_label, self._target_xy = label, xy
                    self.dest_xy = dest_xy
                    self.dest_box_name = dest_box
                    self._face_target_yaw_deg = None
                    self._instructed_label = None   # 소비했으니 초기화
                    self._instructed_dest_xy = None
                    # 재정렬 예산은 **대상 1개** 스코프다. 미션 누적으로 두면
                    # 첫 기물이 예산을 다 쓴 뒤 나머지가 전부 첫 시도에서
                    # 보류된다. 되돌리는 자리는 대상이 바뀌는 여기 하나뿐이다.
                    self._align_tries = 0
                    self._reaim_tries = 0
                    self._align_reaim_backoff = False
                    self._align_sweep_stage = None
                    self._align_sweep_phase_start = None
                    self._align_sweep_burst_until = None
                    self._forcing_grasp = False
                    self._forced_grasp_tries = 0
                    self._align_tries_at_last_force = None
                    self._replan_tries = 0
                    self._align_tries_at_last_replan = None
                    self._tight_yaw_gate = False
                    self._replan_backoff_xy = None
                    self._grasp_fail_tries = 0
                    self._path_planner.reset()   # 새 구간 시작
                    self._drive.reset()
                    self.ready_to_advance = False
                    self.state = State.APPROACH_PIECE

        elif self.state == State.APPROACH_PIECE:
            assert self._target_xy is not None
            dist = math.hypot(self._target_xy[0] - robot_xy[0],
                              self._target_xy[1] - robot_xy[1])
            if dist <= mcfg.GRASP_TRIGGER_DIST_M:
                if not self._facing_target(pose, robot_xy):
                    # 트리거 거리 안이어도 정면이 아니면 아직 GRASP 로 안
                    # 넘어간다 — 제자리에서 겨눈다.
                    #
                    # 2026-09-02까지는 이 게이트를 `_tight_yaw_gate`(GRASP_
                    # REPLAN 이 보낸 재접근에서만)로만 켰다 — 평소
                    # APPROACH_PIECE 는 거리만 보고 바로 넘어갔다(경로
                    # 추종(_approach/DriveSequencer)이 남긴 헤딩을 그대로
                    # 썼다). 07:12 rook 실기(52도 미스얼라인) 같은 큰 사고는
                    # 없었지만, 평소에도 그 헤딩은 목표를 "대충 보는" 정도라
                    # 사용자가 매번 비슷한 크기로 왼쪽 3~5cm 편차를 확인했다
                    # (PIECE_AIM_YAW_TRIM_DEG 주석 참고). 이미 있는 이 게이트
                    # (허용치 GRASP_REPLAN_YAW_TOLERANCE_DEG, ±6도)를 평소
                    # 접근에도 그대로 적용해, 매번 트림이 반영된 정확한
                    # 오차 안에서 GRASP 에 들어가게 한다.
                    self._aim_at_target(pose, robot_xy, link)
                    self.ready_to_advance = False
                    return self.state
                # 트리거 거리 도달 — 여기서 더 다가가면 기물을 밀어낸다.
                # 정지를 보내 제자리에 세우고 GRASP 를(수동 모드면 Next 를)
                # 기다린다. 이 사이클엔 절대 전진 명령을 보내지 않는다.
                self.ready_to_advance = True
                self.last_cmd = "stop"
                link.send(MissionCommand(
                    "stop", "APPROACH_PIECE", pose.x, pose.y, pose.yaw_deg,
                    target_label=self.target_label,
                ))
                if self._should_advance():
                    self.ready_to_advance = False
                    self.state = State.GRASP
            else:
                obstacles = _other_pieces(piece_map, exclude_xy=self._target_xy)
                self._approach(pose, robot_xy, self._target_xy, obstacles,
                               self.target_label, link)
                self.ready_to_advance = False

        elif self.state == State.GRASP_REPLAN:
            # GRASP_ALIGN 을 국소 보정만으로 너무 오래 반복해서, 오버헤드
            # 카메라로 크게 물러났다 다시 세우는 중이다(GRASP_REPLAN_AFTER_
            # TRIES 주석 참고). _approach() 를 그대로 재사용하므로 다른
            # 물체 회피는 평소와 똑같이 GridPathPlanner 가 처리한다.
            assert self._replan_backoff_xy is not None and self._target_xy is not None
            obstacles = _other_pieces(piece_map, exclude_xy=self._target_xy)
            dist = self._approach(pose, robot_xy, self._replan_backoff_xy,
                                  obstacles, self.target_label, link)
            if dist <= mcfg.GRASP_REPLAN_ARRIVE_TOL_M:
                self._replan_backoff_xy = None
                self._tight_yaw_gate = True
                self._path_planner.reset()   # 새 구간 시작(재접근)
                self._drive.reset()
                self.state = State.APPROACH_PIECE

        elif self.state == State.GRASP:
            self.nav_goal = None
            self.nav_corner = None
            self.nav_path = None
            self.last_nav = None
            self.last_cmd = "stop"
            # 재정렬을 GRASP_FORCE_AFTER_TRIES 회 넘게 반복해도 여전히
            # 영역 밖이면, "GRASP" 대신 "GRASP_FORCE" 를 보내 Pi 의 정렬
            # 창 판정을 한 번 건너뛰게 한다(사용자 지시, 2026-08-31 —
            # grippers 저장소 MissionState.GRASP_FORCE 참고). 성공/실패
            # 판정 자체는 Pi 의 기존 두 신호(부하값+뎁스캠 확인) 그대로다.
            status = "GRASP_FORCE" if self._forcing_grasp else "GRASP"
            # ── 좌우 조준을 servo 1 에 맡긴다 (사용자 지시, 2026-09-06) ──
            #
            # 차체 yaw 로는 못 좁힌다. 주행 허용오차가 12도인데(회전이
            # bang-bang 이라 정지 명령 뒤 관성으로 약 10도를 더 돌아서
            # 그보다 좁히면 헌팅이 난다 — DRIVE_YAW_TOLERANCE_DEG 주석),
            # 그리퍼-기물 0.20m 에서 12도면 좌우 42mm 다. VLA 허용치
            # (±41mm)를 이미 넘는다.
            #
            # 그래서 남은 지향 오차를 그대로 실어 보내고 Pi 가 servo 1 로
            # 흡수한다 — 바구니 투하(PLACE)가 이미 같은 필드로 하는 것과
            # 같은 방식이다. 부호 뒤집기와 한계 판정은 Pi 쪽에 있다
            # (baseline_mission._grasp_vla).
            #
            # ⚠️ 매 사이클 새로 계산한다. GRASP 진입 시점 값을 얼려 두면
            # 그사이 차체가 조금이라도 움직였을 때 낡아진다 — 바구니 쪽이
            # 같은 이유로 매 사이클 계산한다.
            # _yaw_error_to_target_deg 를 쓴다 — 그 함수가 PIECE_AIM_YAW_
            # TRIM_DEG 를 더하는 **유일한 지점**이고(그 docstring 참고),
            # 여기서 따로 계산하면 겨눔과 보정이 서로 다른 오차를 보게 된다.
            grasp_yaw_correction_deg = 0.0
            if self._target_xy is not None and pose.ok:
                grasp_yaw_correction_deg = self._yaw_error_to_target_deg(pose, robot_xy)
            link.send(MissionCommand("stop", status, pose.x, pose.y, pose.yaw_deg,
                                      target_label=self.target_label,
                                      yaw_correction_deg=grasp_yaw_correction_deg))
            # poll_status() 는 한 번 물으면 그 응답을 소비한다(다시 물으면
            # IDLE) — 그래서 GRASP_DONE 을 본 뒤로는 다시 안 묻고 그 사실을
            # ready_to_advance 에 붙들어 둔다(수동 모드에서 버튼 누를 때까지
            # 여러 사이클 걸릴 수 있어서, 매번 새로 물으면 신호를 놓친다).
            poll = link.poll_status() if not self.ready_to_advance else "IDLE"
            if not self.ready_to_advance and poll == "GRASP_DONE":
                self.ready_to_advance = True
                if self.target_label is not None and self._target_xy is not None:
                    self.last_grasp_event = (self.target_label, self._target_xy)
            elif self._forcing_grasp and poll == "FAILED":
                # 강제 시도가 실제로 실패했다(부하 미달·뎁스캠에서 확인
                # 안 됨 등, Pi 가 GRASP_FAILED 로 보고). 강제 모드를 풀고
                # Pi 의 다음 재관측(정상 GRASP_ALIGN 한 걸음)을 기다린다
                # — 사용자 지시: 실패하면 재정렬 후 재시도, 그래도 안
                # 되면 포기.
                self._forcing_grasp = False
                print(f"[mission] {self.target_label} 강제 파지 "
                      f"{self._forced_grasp_tries}/{mcfg.GRASP_FORCE_MAX_ATTEMPTS}회 실패 "
                      f"— 재정렬 후 재시도합니다")
            elif poll == "FAILED":
                # 강제가 아닌 일반 실패 — 조건은 다 맞았는데(정렬 창 안,
                # 라벨 인식됨) 팔을 내려도 놓친 경우다(사용자 지시,
                # 2026-09-01). 여기서 아무것도 안 하면 Host 는 다음
                # 사이클에도 그대로 "GRASP" 를 다시 보내고, Pi 는 파지
                # 시퀀스 전체를 처음부터 무한 재시도한다 — mcfg.
                # GRASP_FAIL_MAX_RETRIES 로 상한을 둔다.
                self._grasp_fail_tries += 1
                print(f"[mission] {self.target_label} 파지 실패 "
                      f"{self._grasp_fail_tries}/{mcfg.GRASP_FAIL_MAX_RETRIES}회")
                if self._grasp_fail_tries >= mcfg.GRASP_FAIL_MAX_RETRIES:
                    self._skip_target(
                        f"파지 {self._grasp_fail_tries}회 연속 실패")
                    return self.state

            # Pi 가 "조건이 안 맞는다, 수정된 명령을 달라"고 했으면 재정렬로
            # 넘어간다. 여기서 아무것도 안 하면 Pi 는 계속 기다리고 Host 는
            # 계속 GRASP 를 보내서 영원히 멈춰 있다 — Pi 의 계약이 "스스로
            # 고쳐서 진행하지 않는다"이므로 움직이는 쪽은 Host 뿐이다.
            correction = link.take_correction()
            if correction is not None and not self.ready_to_advance:
                # 이미 실제 파지(팔 내려가 그리퍼 닫기)를 한 번 이상
                # 시도한 뒤에 "정면에서 못 찾음"(BACK_OFF)이 왔다면,
                # 물체가 사라진 게 아니라 이미 턱 사이에 물려 있어서
                # 뎁스캠 화각·거리 판독이 어긋난 것일 수 있다(2026-09-02
                # 실기 — GRASP_REPLAN이 이 경우를 그대로 물러나며 헐겁게
                # 물린 rook을 떨어뜨렸다). 뎁스캠만으로는 "사라짐"과
                # "물렸음"을 가를 수 없으니, 차체를 물러나게 하는 쪽
                # (BACK_OFF 걸음도 GRASP_REPLAN도)은 전부 막고 제자리에서
                # 다시 묻는다 — 아래 두 자리(REPLAN 분기, else 분기)에서 씀.
                grasp_might_be_held = (
                    self._grasp_fail_tries > 0 and correction.kind == BACK_OFF)
                # ⚠️ 2026-08-31 임시 변경(반복 테스트용): 원래는 여기서
                # actionable=False 면 바로 _skip_target 으로 포기했다.
                # 필드에 기물이 하나뿐인 지금은 포기하면 SEARCH_TARGET 이
                # 갈 데가 없어 테스트가 멈춘다 — 그래서 방향을 모르는 경우는
                # 포기 대신 그냥 기다린다. 필드에 기물이 여럿인 정식 시험
                # 으로 돌아가면 이 분기를 원래대로 되돌릴 것.
                if not correction.actionable:
                    print(f"[mission] {self.target_label} 고칠 수 없음(포기 안 하고 "
                          f"대기) — {correction.detail}")
                elif self._forced_grasp_tries >= mcfg.GRASP_FORCE_MAX_ATTEMPTS:
                    # 강제 시도를 다 썼는데 재정렬로도 여전히 안 맞다 —
                    # 여기는 포기한다(사용자 지시 — 반복 테스트 예외 대상이
                    # 아니다. 강제까지 다 쓰고도 안 되면 정말 못 집는 것).
                    self._skip_target(
                        f"강제 파지 {self._forced_grasp_tries}회 소진 — {correction.detail}")
                elif (not grasp_might_be_held
                      and self._replan_tries < mcfg.GRASP_REPLAN_MAX_ATTEMPTS
                      and self._align_tries >= mcfg.GRASP_REPLAN_AFTER_TRIES
                      and (self._align_tries_at_last_replan is None
                           or self._align_tries >= self._align_tries_at_last_replan
                                                    + mcfg.GRASP_REPLAN_AFTER_TRIES)):
                    # GRASP_ALIGN 을 이미 한 다발 반복했다 — Pi 의 좁은 정면
                    # 뎁스캠 국소 보정 대신 오버헤드 카메라로 크게 다시
                    # 세운다(GRASP_REPLAN_AFTER_TRIES 주석 참고). 목표에서
                    # 여유 있게 물러난 지점만 계산해 두고, 장애물 회피는
                    # GRASP_REPLAN 상태의 _approach() 가 평소처럼 처리한다.
                    back_dist = mcfg.GRASP_TRIGGER_DIST_M + mcfg.GRASP_REPLAN_BACKOFF_M
                    dx = robot_xy[0] - self._target_xy[0]
                    dy = robot_xy[1] - self._target_xy[1]
                    away = math.hypot(dx, dy) or 1.0
                    self._replan_backoff_xy = (
                        self._target_xy[0] + dx / away * back_dist,
                        self._target_xy[1] + dy / away * back_dist)
                    self._replan_tries += 1
                    self._align_tries_at_last_replan = self._align_tries
                    print(f"[mission] {self.target_label} 재정렬 {self._align_tries}회 — "
                          f"오버헤드 재계획 {self._replan_tries}/"
                          f"{mcfg.GRASP_REPLAN_MAX_ATTEMPTS} ({correction.detail})")
                    self._align = None
                    self._align_from = None
                    self.ready_to_advance = False
                    self.state = State.GRASP_REPLAN
                elif (self._align_tries >= mcfg.GRASP_FORCE_AFTER_TRIES
                      and (self._align_tries_at_last_force is None
                           or self._align_tries > self._align_tries_at_last_force)):
                    # 재정렬을 충분히 반복했다(그리고 지난 강제 시도 뒤로
                    # 최소 한 걸음은 더 정렬했다) — 한 번 강제로 시도한다.
                    self._forcing_grasp = True
                    self._forced_grasp_tries += 1
                    self._align_tries_at_last_force = self._align_tries
                    print(f"[mission] {self.target_label} 재정렬 {self._align_tries}회 — "
                          f"강제 파지 시도 {self._forced_grasp_tries}/"
                          f"{mcfg.GRASP_FORCE_MAX_ATTEMPTS} ({correction.detail})")
                elif grasp_might_be_held:
                    # BACK_OFF지만 물러나지 않는다 — 위 grasp_might_be_held
                    # 계산 참고. 강제 파지/포기 사다리는 그대로 타야 하니
                    # _align_tries는 세되, 차체는 움직이지 않고 제자리에서
                    # 다시 판정을 받는다(다음 사이클에 그대로 "GRASP" 재전송).
                    print(f"[mission] {self.target_label} 파지 시도 이력 있음 "
                          f"— 물러나지 않고 제자리에서 재확인 ({correction.detail})")
                    self._align_tries += 1
                    self.ready_to_advance = False
                    return self.state
                else:
                    if correction.kind == RE_AIM:
                        self._reaim_tries += 1
                    else:
                        self._reaim_tries = 0

                    self._align_reaim_backoff = False
                    if self._reaim_tries > mcfg.GRASP_REAIM_ESCALATE_AFTER_TRIES:
                        # 같은 자리에서 회전만 반복해도 안 풀린다(사용자
                        # 지시 2026-09-01). BACK_OFF 의 후진 한 걸음만
                        # 빌려 쓴다 — 진짜 BACK_OFF(뎁스캠이 목표를 못 봄)
                        # 와 달리 목표는 여전히 보이므로, 뒤이은 좌우 스윕은
                        # 안 돈다(_align_reaim_backoff 로 구분).
                        print(f"[mission] {self.target_label} 회전 보정 "
                              f"{self._reaim_tries}회 연속 — 후진 한 걸음 뒤 재겨냥")
                        correction = GraspCorrection(
                            BACK_OFF,
                            f"회전 {self._reaim_tries}회 미수렴 — 후진 전환",
                            correction.lateral_mm, correction.forward_mm,
                            correction.yaw_deg)
                        self._align_reaim_backoff = True
                        self._reaim_tries = 0

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
                if self._align.kind != BACK_OFF and self._align_sweep_stage is not None:
                    # 스윕 도중 Pi 가 다른 종류의 보정으로 바꿔 보냈다 —
                    # 스윕 상태는 BACK_OFF 계열에만 유효하므로 정리한다.
                    self._align_sweep_stage = None
                    self._align_sweep_phase_start = None
                    self._align_sweep_burst_until = None

                if self._align.kind == BACK_OFF and self._align_sweep_stage is not None:
                    cmd, done = self._grasp_sweep_step()
                else:
                    moved = math.hypot(pose.x - fx, pose.y - fy)
                    done = moved >= mcfg.GRASP_ALIGN_STEP_M
                    # 뎁스캠이 목표를 못 본 경우도 여기로 온다 — Pi 가 그때
                    # RETREAT 를 보내기 때문이다(방향을 아는 쪽이 방향을 말한다,
                    # domain/task/corrections.from_grasp_precondition). 그래서
                    # Host 는 BACK_OFF 하나만 알면 된다.
                    if (self._align.kind == BACK_OFF and done
                            and not self._align_reaim_backoff):
                        # 후진 한 걸음 끝 — 곧장 GRASP 로 돌아가지 않고 좌우로
                        # 훑는다(2026-09-01 사용자 지시: 후진만 반복해서는
                        # 안 나아지는 게 실기로 확인됨 — mission_config.
                        # GRASP_SWEEP_* 주석 참고).
                        #
                        # _align_reaim_backoff 인 경우는 뺀다 — 회전 미수렴
                        # 때문에 빌려 쓴 후진이라 목표를 못 본 게 아니다.
                        # 스윕(목표 재탐색)을 돌 이유가 없으니 곧장 GRASP 로
                        # 돌아가 다시 정렬을 잰다.
                        now = time.monotonic()
                        self._align_sweep_stage = "left"
                        self._align_sweep_phase_start = now
                        self._align_sweep_burst_until = now + mcfg.GRASP_SWEEP_BURST_SEC
                        cmd = "yaw+"
                        done = False
                    else:
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
            # 2026-09-06 사용자 지시로 발견/수정: 여기만 exclude_xy를 안 넘겨서
            # 이미 그리퍼로 집어 든 self._target_xy(방금 파지한 기물의 원래
            # 판 위 좌표)가 매 사이클 장애물 후보에 다시 섞여 들어갔다 —
            # APPROACH_PIECE(892/903행)는 애초에 자기 목표를 스스로의
            # 장애물로 보지 않으려고 exclude_xy=self._target_xy를 쓰는데,
            # CARRY_TO_DEST만 그 인자를 빠뜨렸다. 실제로 들고 있는 기물은
            # 이미 차 위에 있어 판 위에 없어야 정상이지만, 오버헤드 웹캠이
            # 그 자리에서 간헐적으로(잔상·오검출) 다시 잡으면 GridPathPlanner가
            # 그 유령 장애물을 피하려고 매번 다른 경로를 골라 버벅이며
            # 회전했다가 다시 정상화되는 것처럼 보인다(사용자 실기 관찰).
            # self._target_xy는 PLACE 완료 시점(1723행)까지 그대로 남아 있어
            # CARRY_TO_DEST 전 구간에서 안전하게 쓸 수 있다.
            obstacles = _other_pieces(piece_map, exclude_xy=self._target_xy)
            # 2026-09-05, 사용자 지시: 상자로 향할 때는 dest_xy(_box_front_xy,
            # 상자 중심에서 0.325m 물러난 점)가 아니라 INSERT 목표영역
            # "중심"을 향해 몬다 — dest_xy는 목표중심에서 0.165m 떨어져 있어
            # (기존 PLACE_TRIGGER_DIST_M=0.35m 도착 판정 기준이었다), 그
            # 자리에 서면 새 부채꼴 반경(0.15m) 밖이라 아래 게이트가 영원히
            # 안 열린다 — 실제로 처음 구현했을 때 이 자리에 갇히는 회귀가
            # 났었다. self.dest_xy 필드 자체는 그대로 둔다(다른 코드/테스트가
            # _box_front_xy 값을 그대로 기대한다) — 이번 사이클의 주행
            # 목표만 바꾼다.
            nav_target = (basket_target.target_center(self.dest_box_name)
                          if self.dest_box_name is not None else self.dest_xy)
            dist = self._approach(pose, robot_xy, nav_target, obstacles, None, link)
            if self.dest_box_name is not None:
                # 점(dest_xy) 도착이 아니라 목표영역 중심을 기준으로 한
                # 접근(남쪽) 부채꼴 진입으로 판정한다 —
                # basket_target.check_approach_sector() 참고. 부채꼴 호 위
                # 어디로 들어오든 유효하고, 그 지점마다 다른 정렬각
                # (align_yaw_deg)을 FACE_BOX/NUDGE_BOX에 넘긴다(고정
                # BOX_FACE_YAW_DEG가 아니다 — 정중앙 진입일 때만 우연히 같다).
                sector = basket_target.check_approach_sector(robot_xy, self.dest_box_name)
                self.ready_to_advance = sector.ok
                if self.ready_to_advance and self._should_advance():
                    self._face_target_yaw_deg = sector.align_yaw_deg
                    self.ready_to_advance = False
                    self.state = State.FACE_BOX
            else:
                # 상자가 아니라 사용자에게 직접 가져다주는 경우("가져와") —
                # 고정 지점(dest_xy) 도착 판정을 그대로 쓴다. 정렬 목표는
                # None으로 둬서 FACE_BOX가 mcfg.BOX_FACE_YAW_DEG로 대체하게 한다.
                self.ready_to_advance = dist <= mcfg.PLACE_TRIGGER_DIST_M
                if self.ready_to_advance and self._should_advance():
                    self._face_target_yaw_deg = None
                    self.ready_to_advance = False
                    self.state = State.FACE_BOX

        elif self.state == State.FACE_BOX:
            # 상자 앞엔 도착했지만 아직 방향이 안 맞을 수 있다(어느 축으로
            # 마지막에 들어왔는지에 따라 다름) — PLACE 로 넘어가기 전에
            # 방향을 맞춰 세운다. 2026-09-05부터 이 목표각은 고정
            # BOX_FACE_YAW_DEG("12시"=+y)가 아니라, CARRY_TO_DEST가 부채꼴
            # 진입 지점에서 계산해 넘긴 self._face_target_yaw_deg다(상자가
            # 아니라 사람에게 가져다주는 경우처럼 그 값이 없으면 기존
            # BOX_FACE_YAW_DEG로 대체). next_waypoint 를 또 쓸 필요 없이
            # (목적지에 이미 도착했으므로 이동은 없고 회전만 필요) 방위각
            # 오차만 직접 계산한다.
            self.nav_goal = None
            self.nav_corner = None
            self.nav_path = None
            target_yaw = (self._face_target_yaw_deg
                          if self._face_target_yaw_deg is not None
                          else mcfg.BOX_FACE_YAW_DEG)
            yaw_err = (target_yaw - pose.yaw_deg + 180.0) % 360.0 - 180.0
            # 일반 주행용 DRIVE_YAW_TOLERANCE_DEG가 아니라 박스 전용
            # BOX_FACE_YAW_TOLERANCE_DEG를 쓴다 — safe_300이 드랍 직전
            # servo 1로 잔여 오차를 흡수하므로 여기서 회전으로 무리하게
            # 좁힐 필요가 없다(정의부 주석 참고, 2026-09-05).
            aligned = abs(yaw_err) <= mcfg.BOX_FACE_YAW_TOLERANCE_DEG
            self.ready_to_advance = aligned
            nav = DriveCommand(
                mode=DriveMode.STOP if aligned else DriveMode.ROTATE,
                waypoint=robot_xy, target_yaw_deg=target_yaw,
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
                self._nudge_yaw_from = pose.yaw_deg
                self._nudge_best = 0.0
                self._nudge_best_pi_error = None
                self._nudge_gate_streak = 0
                self._nudge_stall_at = time.monotonic() + mcfg.BASKET_NUDGE_STALL_SEC
                # 이전 진입에서 이미 경고를 찍었어도, 이번 진입은 별개의
                # 시도다 — 2026-09-03 실기: 이걸 리셋 안 해서 두 번째
                # 스톨부터는 메시지도 없이 조용히 멈췄다.
                self._nudge_stall_warned = False
                # ⚠️ 2026-09-04 실기로 드러난 문제: LIDAR_INSERT_CHECK_
                # ENABLED=False로 Pi의 check_insert가 더 이상 "너무 멀다"고
                # 거절하지 않게 되면서, 예전에 NUDGE_BOX<->PLACE 왕복(Pi가
                # INSERT_BLOCKED로 보정 계획을 실어 돌려주는 것)으로 거리를
                # 좁혀 가던 폐루프가 사실상 사라졌다. 그 결과 아래 기본값
                # (고정 5cm)만 쓰면 실제로는 한참 먼 채로 딱 한 번 5cm만
                # 밀고 바로 INSERT를 시도해 버린다(실측: 라이다 0.373m/
                # 0.215m/0.164m — 옛 목표 0.140m와 한참 다름, 매번 정면
                # 앞에서 헛되이 투하). Pi가 더 이상 교정해 주지 않으니,
                # 첫 진입에서부터 Host 스스로 basket_target 게이트가 실제로
                # 만족될 만큼 계획을 세운다 — 5cm보다 짧게는 절대 안 잡는다
                # (그보다 가까우면 원래 로직대로 5cm 직진).
                #
                # 2026-09-05, 사용자 지시("라이다 뺀 상황으로 전제하고 다시
                # 수정해")로 이 계산을 LIDAR_INSERT_CHECK_ENABLED 값과
                # 무관하게 항상 한다 — 원래는 True(Pi가 라이다로 거절·보정)
                # 일 때만 아래 고정 5cm를 썼는데, 그 5cm는 CARRY_TO_DEST가
                # dest_xy(상자 중심에서 0.325m)에 도착했다고 보던 옛
                # 판정에서 "그 정도만 더 가면 충분하다"고 잡은 값이었다.
                # 지금은 CARRY_TO_DEST가 훨씬 더 가까운 접근 부채꼴
                # (basket_target.SOUTH_APPROACH_SECTOR_RADIUS_M=0.15, 목표
                # 중심 기준)에서 바로 넘어오므로, 그 옛 5cm 전제가 더 이상
                # 안 맞는다 — 실측(시뮬레이션) 확인 결과 True일 때 옛 고정
                # 5cm 경로를 그대로 쓰면 라이다가 이미 바구니 앞면을 지나친
                # 채(-2.5cm) 멈췄다. Pi 라이다(있다면)는 이제 순수 보너스
                # 보정으로만 쓴다 — 얼마나 가야 하는지의 1차 판단은
                # 항상 Host 자신의 ArUco 기하(check_basket_insert_gate)가
                # 낸다. LIDAR_INSERT_CHECK_ENABLED는 여전히 "Pi가 거부하면
                # 그 보정 계획(_plan_basket_fix)을 믿을지"만 가른다(아래
                # gate_ok/host_gate_hit 분기 참고) — 첫 계획 자체는 더는
                # 그 플래그를 안 본다.
                if self._nudge_plan is None:
                    dest_box_name = mcfg.PIECE_DEST_BOX.get(self.target_label)
                    if dest_box_name is not None:
                        gate = basket_target.check_basket_insert_gate(
                            robot_xy, pose.yaw_deg, dest_box_name)
                        self._nudge_plan = (
                            max(mcfg.BOX_NUDGE_M, gate.distance_m), "forward")
            # 얼마나 어느 쪽으로 갈지. PLACE 가 Pi 판독을 보고 정해 두거나
            # 위에서 게이트 거리로 정해 뒀으면 그것을 쓰고, 그마저 없으면
            # (dest_box_name을 모르는 라벨 등) 기존 5 cm 직진이다.
            want_m, axis = self._nudge_plan or (mcfg.BOX_NUDGE_M, "forward")
            is_rotate = axis in ("rotate_left", "rotate_right")
            # FACE_BOX가 맞춘 방향 그대로 밀어야 한다 — 2026-09-05부터 이게
            # 고정 BOX_FACE_YAW_DEG가 아니라 진입 지점마다 다른 동적 정렬각
            # (self._face_target_yaw_deg)일 수 있다(basket_target.
            # check_approach_sector 참고). 값이 없으면(상자가 아닌 경우)
            # 기존처럼 BOX_FACE_YAW_DEG로 대체한다.
            nudge_target_yaw = (self._face_target_yaw_deg
                                 if self._face_target_yaw_deg is not None
                                 else mcfg.BOX_FACE_YAW_DEG)
            heading = math.radians(nudge_target_yaw)
            goal = (robot_xy if is_rotate else
                    (self._nudge_from[0] + want_m * math.cos(heading),
                     self._nudge_from[1] + want_m * math.sin(heading)))
            if is_rotate:
                # 제자리 회전이라 위치가 아니라 각도로 진행량을 잰다 —
                # want_m 이 이 axis 에서는 라디안이다(_plan_basket_fix 참고).
                yaw_from = self._nudge_yaw_from
                if yaw_from is None:
                    yaw_from = pose.yaw_deg
                delta_deg = (pose.yaw_deg - yaw_from + 180.0) % 360.0 - 180.0
                moved = abs(math.radians(delta_deg))
            else:
                moved = math.hypot(robot_xy[0] - self._nudge_from[0],
                                   robot_xy[1] - self._nudge_from[1])
            yaw_err = (nudge_target_yaw - pose.yaw_deg + 180.0) % 360.0 - 180.0
            # FACE_BOX와 같은 이유로 여기도 BOX_FACE_YAW_TOLERANCE_DEG를
            # 쓴다(2026-09-05) — 전진 중 직진 유지 기준도 결국 같은 박스
            # 목표 방위 얘기다.
            aligned = abs(yaw_err) <= mcfg.BOX_FACE_YAW_TOLERANCE_DEG
            # 2026-09-02 실기(2건): 여기서 계획 거리(want_m)를 다 채울 때까지
            # Pi 보고를 하나도 안 읽고 있다가, PLACE 에 들어가서야 처음
            # poll_status() 를 불러 확인했다 — ArUco 데드레커닝이 실제와
            # 어긋나면 그사이 이미 바구니에 닿았다. 여기서도 매 사이클
            # 드레인해서, Pi가 실시간 라이다로 "이미 목표창 안"(basket_
            # ready_early) 이라 하거나 "너무 가깝다"고 알려 오면 계획 거리를
            # 마저 채우지 않고 즉시 멈춘다.
            #
            # ⚠️ 09-02 실기 3번째 재발로 확인: `last_basket_fix` 는 PLACE 의
            # _judge_insert(INSERT 명령에 대한 응답)와 여기 라이브 점검
            # (APPROACH_BOX 명령에 대한 응답) 양쪽이 **같은 채널**
            # (Report.INSERT_BLOCKED)로 채운다. PLACE 에서 NUDGE_BOX 로 막
            # 넘어온 순간에는 Pi 가 직전 INSERT 판정에 대해 아직 보내고 있던
            # 오래된 보고(전형적으로 "바구니가 멀다", forward_m 이 양수)가
            # 한두 사이클 뒤늦게 도착할 수 있다 — 그런데 여기서 "값이
            # 있으면 무조건 멈춘다"고 읽으면, 실제로는 한 걸음도 안 갔는데
            # 멈춘 것으로 치고 `_plan_basket_fix`의 예산(BASKET_CREEP_BUDGET_M)
            # 을 또 청구해 몇 사이클 만에 소진시켜 버린다(실측: 0.376m→
            # 0.372m 두 판독이 사실상 제자리인데 예산 0.40m 를 통째로 태워
            # 이후 계속 INSERT_BLOCKED 만 반복). 라이브 점검
            # (baseline_mission.BaselineCarryState)이 실제로 보내는 것은
            # `retreat_if_too_close` 뿐이고, 그건 언제나 forward_m 이
            # 음수다 — 그래서 음수일 때만 "지금 막 온 라이브 신호"로 믿는다.
            # 0 이상(양수/None)은 PLACE 국면의 잔여 보고로 보고 무시한다 —
            # PLACE 가 want_m 완주 후 다시 물어서 정식으로 처리한다.
            link.poll_status()
            fix = link.last_basket_fix
            live_too_close = fix is not None and fix.forward_m is not None \
                and fix.forward_m < 0
            # 2026-09-03 실기(rook/box, 두 바구니 다): 여기서 매 사이클
            # take_basket_ready_early()를 무조건 소비해서 회전(rotate)·좌우
            # (left/right) 축의 "끝났다"에도 같이 넣고 있었다. 그런데
            # APPROACH_BOX_READY는 Pi의 "라이다 거리가 이미 목표창 안"
            # (전후 축) 신호일 뿐이다 — NUDGE_BOX가 바구니 가까이 붙은
            # 뒤에는 이 신호가 사실상 항상 참이라, 회전판을 새로 계획해
            # NUDGE_BOX에 들어가는 바로 그 첫 사이클에 "이미 끝났다"고
            # 오판하고 즉시 PLACE로 돌아갔다. 그 결과가 실측으로도 드러난다
            # — want_m(~0.1rad)을 AGREED_ROTATION_RAD_S(0.25rad/s)로 돌면
            # 최소 0.4초는 걸려야 하는데, 로그의 NUDGE_BOX→PLACE 왕복은
            # 0.1~0.2초 만에 끝났다 — 실제로 거의 안 돈 것이다. 그래서
            # yaw -0.09~-0.13rad대가 수십 번 보정을 거쳐도 전혀 안 줄었다.
            #
            # ready_early는 거리(전후) 축에서만 "끝났다"에 넣는다. 회전·좌우
            # 축은 거리와 무관한 오차라 이 신호로 끝났다고 보면 안 된다.
            # 그래도 매 사이클 take()는 해서(값을 쓰든 안 쓰든) 다음 순수
            # 전후 판정에 낡은 표시가 새지 않게 한다.
            ready_early = link.take_basket_ready_early()
            # 절대 안전 반경 — Pi 라이다 신뢰도와 완전히 무관한, Host 자체
            # ArUco 판정(2026-09-03 실기, 장난감 바구니 우측 입구 충돌 뒤
            # 사용자 지시). 라이다가 "테두리를 넘겨보고 있을 수 있다"고
            # 스스로 불안정을 알리는 바로 그 근접 구간에서, 유일한 안전장치가
            # 그 흔들리는 신호뿐이면 안 된다 — 이 반경 안이면 무조건
            # 멈춘다(mission_config.BASKET_HARD_STOP_MARGIN_M 주석 참고).
            #
            # ⚠️ "back"(후진)만은 예외다(2026-09-03, 사용자 지시) — 완전
            # 정지로 가두면 반경 안에 갇힌 채 스스로 빠져나올 방법이 없다.
            # 후진은 위험 반경에서 멀어지는 방향이니 그대로 허용한다. 실제로
            # `_plan_basket_fix`는 너무 가까우면(forward_m<0) 애초에 axis
            # ="back" 계획을 낸다 — 이 반경 안에서 가장 먼저 나올 계획이
            # 바로 그것이다.
            hard_stop = False
            dest_box_name = mcfg.PIECE_DEST_BOX.get(self.target_label)
            if dest_box_name is not None:
                box_x, box_y, _box_yaw = box_pose(dest_box_name)
                dist_to_box_center = math.hypot(robot_xy[0] - box_x,
                                                robot_xy[1] - box_y)
                hard_radius = cfg.BOX_L / 2.0 + mcfg.BASKET_HARD_STOP_MARGIN_M
                hard_stop = dist_to_box_center <= hard_radius
            # 사용자 지시(2026-09-04): 정면으로 딱 맞춰 서는 것을 강제하지
            # 않는다 — 입구 목표 영역(가로 ±3cm, 안쪽 0~3cm)에서 15cm
            # 이내에 있고 그쪽을 보고만 있어도 충분하다(basket_target.py
            # 참고, 사선 진입 허용).
            #
            # ⚠️ 2026-09-04 밤 실기로 드러난 버그: 이 게이트를 LIDAR_INSERT_
            # CHECK_ENABLED 값과 무관하게 항상 켜 뒀더니, 라이다를 다시 켜도
            # 전후/좌우 축까지 이 느슨한 Host 판정만으로 조기 종료돼 Pi의
            # 실측 라이다 정렬(check_insert가 실제로 요구하는 기준)을
            # 건너뛰었다 — toy 바구니 투하가 입구 오른쪽 바깥으로 나가는
            # 사고로 이어졌다(사용자 보고). LIDAR_INSERT_CHECK_ENABLED가
            # 꺼져 있을 때만(=Pi가 라이다로 거절해 주지 않을 때만) 이
            # 게이트를 쓴다 — 켜져 있으면 예전처럼 Pi 라이다 판정 하나만
            # 믿는다.
            #
            # ⚠️ 2026-09-05 시도했다가 되돌린 것: "회전 축만은 이 게이트를
            # 라이다 스위치와 무관하게 항상 쓰자"(사선이면 정면까지 안
            # 돌아도 되게) — 방향(facing)만 보는 조건을 추가해 봤는데,
            # NUDGE_BOX 회전 진입 시점엔 이미 로봇이 대략 바구니 쪽을
            # 보고 있는 게 보통이라(그래서 회전이 필요해진 것) 그 느슨한
            # ArUco 판정이 거의 매번 즉시 참이 되어 Pi의 정밀 라이다 정렬
            # (BASKET_YAW_DEADBAND_RAD)을 시작도 하기 전에 끝내 버렸다 —
            # 정확히 2026-09-03에 고쳤던 사고(test_nudge_box_rotate_fix.py
            # 참고)와 같은 모양으로 재발했다. gate_ok(Host 전용 ArUco 판정)
            # 로 회전 축을 우회하는 건 여전히 안 한다 — LIDAR_INSERT_CHECK_
            # ENABLED가 꺼져 있을 때만 그 경로로 사선이 허용된다.
            #
            # 대신(같은 날, 사용자 지시 — "무조건 정면은 좀 위험해") 회전
            # 축이 "다 됐다"고 볼 기준 자체를 넓혔다. BASKET_YAW_DEADBAND_
            # RAD(0.04rad≈2.3도)까지 정밀하게 맞추려고 계속 쫓는 대신, **Pi
            # 라이다 실측값**이 NUDGE_ROTATE_DIAGONAL_TOLERANCE_RAD(20도) 안에
            # 들어오면 그 자리에서 그만 돈다 — gate_ok와 달리 이건 여전히
            # Pi의 실측을 보는 것이라 위에서 되돌린 시도와는 다르다. 값 근거는
            # mission_config.py의 그 정의부 코멘트 참고.
            gate_ok = False
            if dest_box_name is not None and not mcfg.LIDAR_INSERT_CHECK_ENABLED:
                gate_ok = basket_target.check_basket_insert_gate(
                    robot_xy, pose.yaw_deg, dest_box_name).ok
            # host_gate_hit — Pi의 실측(라이다 fix/라이브 안전 반경)이 아니라
            # Host 혼자만의 ArUco 판정(gate_ok)만으로 끝났다고 보려는 것인지
            # 표시한다. 이 경로로 끝난 경우만 아래에서 debounce한다
            # (2026-09-05) — "정지 명령을 막 보낸 그 순간의 좌표"가 아니라
            # "실제로 멈춘 뒤 다시 재본 좌표"로 PLACE 전환을 확정하기
            # 위해서다. Pi가 실측으로 확인해 준 경우(fix.yaw_rad 데드밴드·
            # live_too_close·hard_stop·ready_early)는 이미 그 자체로
            # 재확인이니 debounce가 필요 없다.
            if is_rotate:
                if fix is not None and fix.yaw_rad is not None:
                    # 회전판은 Pi 라이다가 직접 확인해 줄 때만 "끝났다"로
                    # 본다 — check_insert가 실제로 판정하는 기준과 맞춘다.
                    # gate_ok를 여기 넣지 않는다 — 위 2026-09-05 코멘트 참고.
                    done_confirmed = (
                        abs(fix.yaw_rad) <= mcfg.NUDGE_ROTATE_DIAGONAL_TOLERANCE_RAD
                        or live_too_close or hard_stop)
                else:
                    # Pi 값이 아직 없을 때만 쓰는 ArUco 폴백. ready_early는
                    # 절대 안 넣는다 — 위 주석 참고.
                    done_confirmed = moved >= want_m or live_too_close or hard_stop
                host_gate_hit = gate_ok
            elif axis in ("left", "right"):
                done_confirmed = moved >= want_m or live_too_close or hard_stop
                host_gate_hit = gate_ok
            elif axis == "back":
                # 후진은 위험 반경에서 빠져나오는 길이다 — 목표 영역
                # 게이트로 조기 종료시키지 않는다(위 hard_stop 관련 주석과
                # 같은 이유).
                #
                # ⚠️ 2026-09-05 실기로 드러난 버그: 여기 live_too_close 를
                # 다른 축(forward/left/right)과 똑같이 "끝났다"에 넣고
                # 있었다. 그런데 back 을 계획하는 이유 자체가 "너무
                # 가깝다"(live_too_close)이므로, back 이 시작되는 바로 그
                # 사이클에 이미 live_too_close=True 라 moved 가 0인 채로
                # 즉시 done=True 가 나 버렸다 — 실제로는 단 한 번도 "back"
                # cmd 를 내보내지 못하고 PLACE 로 곧장 되돌아갔다. 요가 크게
                # 틀어져(0.2~0.48rad) 라이다가 바구니 테두리를 비스듬히
                # 봐서 실제보다 가깝게 잘못 읽히는 상황(라이다 하한
                # 근처에서 진동)에서 이게 154회 연속 반복되며 완전히
                # 멎었다(사용자 보고 — "파지 미세전진 제대로 안됨, INSERT
                # 버그"). live_too_close 를 빼서 실제로 want_m 만큼 물러난
                # 뒤에야 끝났다고 본다 — hard_stop 을 안 넣는 것과 정확히
                # 같은 이유다.
                done_confirmed = moved >= want_m or ready_early
                host_gate_hit = False
            else:   # "forward"
                done_confirmed = (moved >= want_m or ready_early or live_too_close
                                  or hard_stop)
                host_gate_hit = gate_ok
            done = done_confirmed or host_gate_hit
            if host_gate_hit and not done_confirmed:
                self._nudge_gate_streak += 1
            else:
                self._nudge_gate_streak = 0
            advance_ready = done_confirmed or (
                host_gate_hit and self._nudge_gate_streak >= mcfg.NUDGE_GATE_CONFIRM_CYCLES)
            # 전후 이동 중에 방위가 틀어지면 다시 맞춘다 — 5 cm 라도 비스듬히
            # 들어가면 상자 정면에 안 선다. 좌우 이동은 방위를 안 건드리므로
            # (메카넘 횡이동) 회전으로 끊지 않는다 — 여기서 돌면 방금 맞춘
            # 거리와 yaw 가 같이 틀어져 앞 단계를 되돌리게 된다.
            if done:
                mode, cmd = DriveMode.STOP, "stop"
            elif is_rotate:
                # BOX_FACE_YAW_DEG 정렬(아래 elif not aligned)과 다투지 않게
                # 최우선으로 처리한다 — 이 축이 도는 목적 자체가 Pi 라이다
                # 판독을 정면으로 맞추는 것이고, ArUco 기준 BOX_FACE_YAW_DEG
                # 로 되돌아가 버리면 방금 받은 보정을 무시하는 셈이 된다.
                mode = DriveMode.ROTATE
                cmd = "yaw+" if axis == "rotate_left" else "yaw-"
            elif axis in ("left", "right"):
                mode, cmd = DriveMode.FORWARD, axis
            elif not aligned:
                mode = DriveMode.ROTATE
                cmd = "yaw+" if yaw_err >= 0 else "yaw-"
            elif axis == "back":
                mode, cmd = DriveMode.FORWARD, "back"
            else:
                mode, cmd = DriveMode.FORWARD, "go"
            # 진전이 멈추면 멈춘다. 안 움직이는데 계속 미는 것은 어느
            # 원인(바퀴 정지·걸림)에서도 나아지지 않는다.
            #
            # ⚠️ 2026-09-03 실기: moved(ArUco 로 잰 이동량) 만으로 진전을
            # 인정했더니, 차가 육안으로 안 움직이는데도 자세 추정 노이즈만
            # 으로 매번 1cm 문턱을 넘겨서 정지 감시가 6초 안에 단 한 번도
            # 안 걸렸다(13초 넘게 INSERT_BLOCKED 만 반복). Pi 라이다는
            # ArUco 와 다른 센서라 이 노이즈에 안 걸리므로, Pi 값이 있으면
            # (이 축에 해당하는 값이 오면) 그쪽을 우선해서 진전을 판정한다
            # — moved 는 Pi 값이 아직 없을 때(첫 사이클 등)의 보조 수단으로만
            # 남긴다.
            pi_error: Optional[float] = None
            pi_margin = mcfg.BASKET_NUDGE_PROGRESS_M
            if fix is not None:
                if is_rotate and fix.yaw_rad is not None:
                    pi_error, pi_margin = abs(fix.yaw_rad), mcfg.BASKET_YAW_DEADBAND_RAD
                elif axis in ("left", "right") and fix.lateral_m is not None:
                    pi_error, pi_margin = abs(fix.lateral_m), mcfg.BASKET_LATERAL_DEADBAND_M
                elif not is_rotate and axis not in ("left", "right"):
                    if fix.forward_m is not None:
                        pi_error = abs(fix.forward_m)
                    elif fix.distance_m is not None:
                        pi_error = abs(fix.distance_m - mcfg.BASKET_TARGET_LIDAR_M)
                    if pi_error is not None:
                        pi_margin = mcfg.BASKET_DISTANCE_DEADBAND_M
            now = time.monotonic()
            if pi_error is not None:
                if (self._nudge_best_pi_error is None
                        or pi_error < self._nudge_best_pi_error - pi_margin):
                    self._nudge_best_pi_error = pi_error
                    self._nudge_stall_at = now + mcfg.BASKET_NUDGE_STALL_SEC
            elif moved > self._nudge_best + mcfg.BASKET_NUDGE_PROGRESS_M:
                self._nudge_best = moved
                self._nudge_stall_at = now + mcfg.BASKET_NUDGE_STALL_SEC
            if not done and now >= self._nudge_stall_at:
                # 2026-09-03 실기: 여기서 멈추기만 하고 done 을 안 세웠더니
                # NUDGE_BOX 에 영영 눌러앉았다 — 그사이 Pi 는 이미 INSERT를
                # 자체 판단으로 끝내고 IDLE 로 복귀했는데(정지 감시가 걸린
                # 것 자체가 Host 계획이 이미 낡은 판독 기준이었다는 신호였다,
                # 아래 주석 참고) Host 는 그걸 들을 상태(PLACE)에 있지도
                # 않았다. 여기서도 done=True 로 세워 PLACE 로 돌려보낸다 —
                # PLACE 가 poll_status() 로 PLACE_DONE 을 직접 물어보므로,
                # 이미 끝난 배치라면 그대로 다음 기물로 넘어가고, 아직이면
                # 처음부터 다시 계획한다. "바퀴가 진짜 걸렸다"는 가능성은
                # 남아 있지만, 그 경우도 done=False 로 얼어붙는 것보다PLACE
                # 가 다시 INSERT_BLOCKED 를 받아 사람이 볼 수 있는 상태로
                # 표시하는 편이 낫다.
                mode, cmd = DriveMode.STOP, "stop"
                done = True
                # 이 탈출 경로는 그 자체가 이미 "몇 초를 기다려 봤다"는
                # 재확인이다 — host_gate_hit debounce를 또 기다리게 하면
                # 2026-09-03에 고친 "NUDGE_BOX에 영영 눌러앉는" 사고가
                # 다른 경로로 재발한다.
                advance_ready = True
                if not self._nudge_stall_warned:
                    self._nudge_stall_warned = True
                    progress = (f"{math.degrees(moved):.1f}도 밖에 못 돌았습니다"
                                f"(목표 {math.degrees(want_m):.1f}도"
                                if is_rotate else
                                f"{moved * 1000:.0f}mm 밖에 못 갔습니다(목표 "
                                f"{want_m * 1000:.0f}mm")
                    print(f"\n[NUDGE_BOX] {mcfg.BASKET_NUDGE_STALL_SEC:.0f}초 동안 "
                          f"{progress}, 방향 {axis}) — 정지하고 PLACE로 돌아가 "
                          f"다시 확인합니다. 계속 반복되면 바퀴 전원과 걸림을 "
                          f"확인하세요\n", flush=True)

            self.ready_to_advance = advance_ready
            nav = DriveCommand(
                mode=mode, waypoint=goal, target_yaw_deg=nudge_target_yaw,
                yaw_error_deg=yaw_err,
                dist_to_target=max(want_m - moved, 0.0), blocked_by=None,
            )
            self.last_nav = nav
            link.send(MissionCommand(cmd, "NUDGE_BOX", pose.x, pose.y, pose.yaw_deg))
            self.last_cmd = cmd
            if advance_ready and self._should_advance():
                self.ready_to_advance = False
                self._nudge_from = None
                self._nudge_plan = None
                self._nudge_gate_streak = 0
                self.state = State.PLACE

        elif self.state == State.PLACE:
            self.nav_goal = None
            self.nav_corner = None
            self.nav_path = None
            self.last_nav = None
            self.last_cmd = "stop"
            # safe_300 실기 확인(manual_insert_probe.py, 2026-09-05)을 실제
            # 미션 경로에 반영한다 — FACE_BOX/NUDGE_BOX는 그대로 두고(그
            # 회전·정렬 루프는 손대지 않는다), 매 PLACE 사이클마다 그
            # 순간의 잔여 지향오차를 같이 실어 보낸다. FACE_BOX/NUDGE_BOX가
            # 이미 잘 맞춰 왔으면 이 값은 0에 가까워 safe_300 자체가
            # 사실상 건너뛰어지고(BaselineInsertState 참고), 그래도 남는
            # 오차가 있으면(사선 진입 허용, NUDGE_ROTATE_DIAGONAL_TOLERANCE_RAD
            # =20도까지 허용) Pi가 그리퍼를 열기 직전 servo 1로 흡수한다 —
            # 라이다 게이트(check_insert)는 전혀 안 건드린다. 매 사이클
            # 다시 계산하는 이유: PLACE는 INSERT_BLOCKED로 NUDGE_BOX를
            # 여러 번 왕복할 수 있어(아래 else 분기), 최종적으로 Pi가
            # 실제 BaselineInsertState로 넘어가는 그 사이클의 오차라야
            # 의미가 있다 — NUDGE_BOX 진입 시점 값을 얼려 두면 그사이
            # 왕복으로 낡아진다.
            dest_box_name = mcfg.PIECE_DEST_BOX.get(self.target_label)
            yaw_correction_deg = 0.0
            if dest_box_name is not None:
                # 2026-09-05 밤 사용자 지시 — "servo1이 안 돌아도 되는 영역을
                # 따로 만들자 ... 돌아서 투하하는 지점이 가끔 너무 왼쪽이라
                # 아슬아슬해". 지금 자세 그대로도 좁은 영역
                # (basket_target.NO_ROTATION_*)에 들어갈 만하면 servo1은
                # 손대지 않는다 — 매번 정확히 중앙을 노리다가 넓은
                # 목표영역(TARGET_HALF_WIDTH_M) 가장자리(벽 쪽)로 보정이
                # 몰리는 위험을 없앤다.
                if not basket_target.check_no_rotation_zone(
                        robot_xy, pose.yaw_deg, dest_box_name).ok:
                    yaw_correction_deg = basket_target.check_basket_insert_gate(
                        robot_xy, pose.yaw_deg, dest_box_name).facing_error_deg
            link.send(MissionCommand("stop", "PLACE", pose.x, pose.y, pose.yaw_deg,
                                     yaw_correction_deg=yaw_correction_deg))
            status = link.poll_status() if not self.ready_to_advance else "IDLE"
            if status == "PLACE_DONE":
                self.ready_to_advance = True
                # Pi가 명확하게 성공을 확인해 준 경우에만 바구니 위치에서
                # 이 라벨의 트랙을 숨긴다(2026-09-05, 사용자 지시) — 아래
                # FAILED 분기는 일부러 이 이벤트를 안 세운다. 그 분기의
                # 코멘트가 설명하듯 Pi의 FAILED는 오탐일 수 있어서, 실제
                # 안착 여부는 "다음 SEARCH_TARGET에서 다시 보이는가"로
                # 판단하게 이미 설계돼 있다 — 여기서 무조건 숨기면 정말
                # 못 들어가 바닥에 남은 기물까지 다시 못 찾게 된다.
                dest_box_name = mcfg.PIECE_DEST_BOX.get(self.target_label)
                if self.target_label is not None and dest_box_name is not None:
                    box_x, box_y, _box_yaw = box_pose(dest_box_name)
                    self.last_place_event = (self.target_label, (box_x, box_y))
            elif status == "FAILED":
                # 2026-09-03 실기(queen): Pi 가 투하 부하 판정으로 FAILED 를
                # 보고했는데(부하 0.0469 -> 0.0352, RELEASE_LOAD_DROP=0.015
                # 문턱을 못 넘음) 실제로는 바구니에 들어가 있었다 — 이 문턱이
                # 너무 빡빡했던 오탐이었다. 그런데 그 이전엔 이 분기 자체가
                # 없어서 FAILED 가 아래 else(아직 안 끝남, 다시 nudge)로
                # 빠졌고, 그 시점엔 Pi 가 이미 IDLE 로 접어 들어간 뒤라 새
                # fix 가 안 와서 PLACE 가 "stop" 만 영원히 반복하며 얼어붙었다.
                #
                # INSERT 는 판정 결과와 무관하게 그리퍼를 이미 열었다
                # (baseline_mission.py BaselineInsertState 참고) — 그러니
                # FAILED 라고 다시 옮겨 잡을 물건이 없다. "바구니 안착
                # 여부"는 애초에 Pi 가 아니라 오버헤드 카메라를 든 Host 가
                # 판단할 몫이라는 게 그 쪽 docstring의 설계 의도이기도 하다
                # — 그래서 여기서도 그냥 넘어간다: 실제로 안 들어갔으면
                # 다음 SEARCH_TARGET 에서 바닥에 남은 채로 다시 잡힐 것이고,
                # 들어갔으면 더 안 보일 것이다.
                print(f"[mission] 투하 확인 실패(Pi 부하 판정, 오탐 가능) — "
                      f"바구니 안착 여부는 다음 SEARCH_TARGET 에서 다시 "
                      f"보이는지로 판단됩니다. 넘어갑니다.", flush=True)
                self.ready_to_advance = True
            else:
                # Pi 가 "여기서는 못 넣는다"고 하면 그 이유에 실린 숫자를
                # 보고 조금 움직인 뒤 다시 묻는다. 서서 기다리기만 하면
                # 영원히 INSERT_BLOCKED 만 돌아온다 — 2026-08-28 실기가
                # 정확히 그랬다(라이다 0.351m, 요구 0.155m).
                fix = link.take_basket_fix()
                if fix is not None and fix.lost:
                    # Pi가 라이다 평면 자체를 못 찾았다(방향 없음) — 10:41
                    # 실기: 차가 바구니가 아니라 옆의 벽을 보고 있었다.
                    # 국소 보정(전후/좌우/회전)은 "무엇이 어긋났는지"가
                    # 있어야 계획이 서는데, 여기는 그게 아예 없다 — 몇 번
                    # 더 물어서 Pi가 스스로 다시 찾는지 보고, 그래도
                    # 안 되면 오버헤드 카메라로 dest_xy를 향해 크게 다시
                    # 접근한다(GRASP_REPLAN과 같은 이유 — Pi의 좁은 정면
                    # 센서 대신 Host가 아는 좌표로 되돌아간다).
                    self._basket_lost_tries += 1
                    if self._basket_lost_tries >= mcfg.BASKET_LOST_REPLAN_AFTER_TRIES:
                        print(f"[mission] 바구니를 {self._basket_lost_tries}회 "
                              f"연속 못 찾음 — 오버헤드 재접근", flush=True)
                        self._retreat_for_overhead_reapproach()
                        return self.state
                else:
                    plan = self._plan_basket_fix(fix)
                    if plan is not None:
                        self._basket_lost_tries = 0
                        self._nudge_plan = plan
                        self._nudge_from = None
                        self._nudge_yaw_from = None
                        if plan[1] in ("rotate_left", "rotate_right"):
                            self._basket_yaw_used += plan[0]
                        else:
                            self._basket_creep_used += plan[0]
                        self.state = State.NUDGE_BOX
                        return self.state
                    elif fix is not None:
                        # 2026-09-03 실기(soccer 두 번째 시도): fix 는 있다
                        # (바구니는 보인다) — 그런데 오차가 커서 처음 한두
                        # 번의 보정으로 BASKET_CREEP_BUDGET_M 을 다 써버리면
                        # (전후·좌우가 예산을 공유한다, _plan_basket_fix 참고)
                        # plan 이 계속 None 만 나온다. 예전엔 이 경우
                        # _basket_lost_tries 를 안 건드려서 — lost 도 아니고
                        # 진전도 없으니 — PLACE 가 stop 만 8초 넘게 반복하며
                        # 영구히 얼어붙었다. lost 와 같은 카운터로 묶어
                        # 똑같이 오버헤드 재접근으로 빠져나간다.
                        self._basket_lost_tries += 1
                        if self._basket_lost_tries >= mcfg.BASKET_LOST_REPLAN_AFTER_TRIES:
                            print(f"[mission] 바구니 보정이 "
                                  f"{self._basket_lost_tries}회 연속 막힘"
                                  f"(예산 소진 등) — 오버헤드 재접근", flush=True)
                            self._retreat_for_overhead_reapproach()
                            return self.state
                    # fix 가 없으면(아직 새 판독 없음) 카운터를 안 건드리고
                    # 다음 사이클을 기다린다 — "안 왔다"를 실패로도 성공으로도
                    # 치지 않는다.
            if self.ready_to_advance and self._should_advance():
                # 하나 끝났다고 멈추지 않는다 — 다음 기물을 다시 찾는다.
                # 화면에 기물이 더 없으면 SEARCH_TARGET 에서 계속 대기한다.
                #
                # 2026-09-02, 시연용으로 SEARCH_TARGET 에 곧장 가지 않고
                # RETURN_HOME 을 거친다 — _skip_target 과 같은 이유다.
                # 바구니 바로 앞은 매번 각도·거리가 다른 자리라, 거기서 바로
                # 다음 스캔을 시작하면 그때그때 다른 자리에서 SEARCH_TARGET
                # 이 시작된다. RETURN_HOME 을 한 번 거치면 매 라운드가 항상
                # 같은 자리에서 시작해 시연이 예측 가능해진다. 큐에 쌓인
                # 지시(_queued_instruction_label)는 여기서 바로 적용하지
                # 않는다 — RETURN_HOME 완료 시점에 적용하는 기존 경로(아래)
                # 하나로 합친다.
                #
                # ⚠️ 2026-09-02~09-04 사이 여기서 그룹(chess/toy) 소진 여부로
                # AWAIT_CONTINUE(사용자에게 "계속할까요?" 묻기)로 갈지 갈랐던
                # 적이 있다 — 2026-09-04 밤 사용자 지시("AWAIT 다 없애라고.
                # 원래 RETURN_HOME 있던 버전으로 내놔")로 그 기능 전체를
                # 없앴다. State.AWAIT_CONTINUE/AWAIT_COMMAND/IDLE과
                # on_continue()/on_stop()/submit_next_command()도 함께
                # 지웠다 — 이제 무조건 RETURN_HOME 이다.
                self.ready_to_advance = False
                self.target_label = None
                self._target_xy = None
                self.dest_xy = None
                self._path_planner.reset()
                self._drive.reset()
                self.state = self._next_round_state()

        elif self.state == State.RETURN_HOME:
            # 기물을 포기한 뒤(_skip_target) 실패한 자리에 그대로 남지 않고
            # 여기로 먼저 돌아간다 — 다음 SEARCH_TARGET 을 매번 같은 예측
            # 가능한 자리에서 시작하게 한다(사용자 지시, 2026-09-01).
            #
            # 2026-09-02부터 PLACE 완료(하나를 성공적으로 넣은 뒤)도 같은
            # 이유로 여기를 거친다(시연용) — 바구니 앞은 매번 각도·거리가
            # 달라, 거기서 곧장 SEARCH_TARGET을 시작하면 매 라운드가
            # 다른 자리에서 시작된다.
            obstacles = _other_pieces(piece_map)
            dist = self._approach(pose, robot_xy, mcfg.DEFAULT_HOME_XY,
                                  obstacles, None, link)
            self.ready_to_advance = dist <= mcfg.HOME_ARRIVE_TOL_M
            if self.ready_to_advance and self._should_advance():
                self.ready_to_advance = False
                self._path_planner.reset()   # 새 구간 시작
                self._drive.reset()
                # PLACE 완료 경로와 같은 이유로 여기서도 큐를 비운다 —
                # 포기한 기물을 쫓는 동안 새 지시가 들어왔을 수 있다
                # (set_instruction() 이 GRASP/GRASP_ALIGN 도 "손이 안
                # 비었다"로 보고 큐에 쌓아 두므로, 2026-09-01).
                self._apply_queued_instruction()
                self.state = State.SEARCH_TARGET

        return self.state
