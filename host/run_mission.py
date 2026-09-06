"""탑뷰 카메라 2대 + ArUco + geti 로 픽업 -> 이동 -> 내려놓기 미션을 라이브로 돌린다.

Host PC 가 하는 일은 딱 여기까지다: 매 사이클 로봇 pose(ArUco)와 기물 지도
(geti)를 계산해서 "지금 뭘 해야 하는지"(mode)와 "다음 좌표"를 VehicleLink 로
넘기는 것. 실제로 차를 움직이고 집고 내려놓는 건 차량(ROS2, Pi+Hailo)이
SmolVLA(그리퍼캠+차량 RGB캠)로 알아서 한다.

★ 차량에는 라이다가 있고, 여기서 모르는 장애물이 갑자기 나타나면 멈춰서 회피
기동을 하는 반사 안전 레이어가 따로 있다(차량 쪽 ROS2 노드 — 이 저장소 범위
밖). 그 레이어는 Host PC 와 무관하게 항상 최우선으로 작동해야 한다: 라이다는
차량에만 있고, Host PC 를 거치면 지연이 생겨 안전 기능으로 못 쓴다. 그래서 이
스크립트는 그 존재를 몰라도 안전하다 — 매 사이클 "지금 아는 최선의 좌표"만
계속 보내고, 차량이 회피 중이면 그 좌표를 무시하다가 끝나면 최신 좌표를
다시 따라가면 된다.

--vehicle-ip 를 안 주면 ConsoleVehicleLink 로 콘솔에 찍기만 한다(차량 없이
시험용). 주면 UdpVehicleLink 로 실제 UDP 전송한다 — 규격은
VEHICLE_LINK_PROTOCOL.md 참고.

기본은 라벨을 지정하지 않는다 — 화면에 보이는 기물 중 "지금 로봇 위치에서
가장 가까운 것"을 매번 골라서, 그 라벨에 맞는 상자(mission_config.
PIECE_DEST_BOX: 체스말은 chess 상자, 나머지는 toy 상자)로 나른다. 하나
끝나면 멈추지 않고 다음 기물을 또 찾는다 — 화면(작업 영역)에 기물이 하나도
안 남을 때까지 반복.

실행 중 터미널에 문장을 치면(예: "퀸 가져와", "룩 정리해") 그 한 번은 이
"최근접 우선" 규칙을 덮어쓴다(instruction_resolver.py, Claude API로 해석 —
ANTHROPIC_API_KEY 환경변수 필요, 없으면 이 기능만 조용히 꺼진다). "가져와"
류는 사용자 앞(mission_config.DELIVER_HERE_XY)으로, "정리해"류나 라벨만
말한 경우는 평소처럼 라벨별 상자로 나른다. --manual 모드에서는 stdin 을
이미 단계 진행에 쓰고 있어서 이 입력을 안 받는다.

사용법
    python run_mission.py
    python run_mission.py --cams 0 2
    python run_mission.py --show-cams   # 카메라 원본 창도 같이
    python run_mission.py --no-view
    python run_mission.py --mock-complete   # 차량 없이 전체 흐름만 시험
    python run_mission.py --step --mock-complete   # 단계마다 LiveMap 의 Next 버튼으로 직접 진행
    python run_mission.py --vehicle-ip 192.168.0.42   # 실제 차량(Pi)로 UDP 전송

화면은 기본으로 live_map.py 의 2D 지도(로봇/기물/상자/이동경로를 도형으로)
하나만 뜬다. 카메라 원본 + ArUco/geti 오버레이 창은 디버깅용이라 필요할 때만
--show-cams 로 따로 켠다.
"""

from __future__ import annotations

import argparse
import queue
import signal
import math
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

# config.py/localizer.py 는 aruco/ 하위폴더에 있다(팀원이 동기화하는 파일이라
# 건드리지 않는다) — 그 폴더를 경로에 추가해서 기존처럼 bare import 로 쓴다.
sys.path.insert(0, str(Path(__file__).parent / "aruco"))

import config as cfg
from localizer import Camera, RobotLocalizer, detect, make_detector

import geti_detector
import mission_config as mcfg
import mission_log
import pi_lifecycle
import piece_map
import window_layout
from live_map import LiveMap
from mission import MissionFSM, PieceMap, State, visible_labels
from run_localize import draw, open_cams
from vehicle_link import ConsoleVehicleLink, MissionCommand, UdpVehicleLink

# instruction_resolver.py 는 anthropic SDK(및 ANTHROPIC_API_KEY)가 있어야
# 쓸 수 있다 — 없어도 이 스크립트 자체는 죽지 않고 그 기능만 꺼진다
# (2026-09-01, 팀원의 2026-08-31 handoff 델타 이식).
try:
    from instruction_resolver import InstructionResolver
    _instruction_resolver_import_error = None
except Exception as _exc:  # noqa: BLE001 -- 미설치·키 없음 등 다양한 원인
    InstructionResolver = None
    _instruction_resolver_import_error = _exc

_stop = False


def _on_sigint(signum, frame):
    global _stop
    _stop = True



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs="+", default=list(cfg.CAM_INDICES))
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--seconds", type=float, default=None,
                    help="이 시간이 지나면 정지하고 종료한다. 예행연습 안전장치")
    ap.add_argument("--no-stop-on-enter", action="store_true",
                    help="Enter 로 멈추는 감시를 끈다(기본은 켜짐)")
    ap.add_argument("--display", type=int, default=0,
                    help="창을 띄울 화면. 0=주 화면, 1=오른쪽 확장 화면")
    ap.add_argument("--no-tile", action="store_true",
                    help="창 자동 배치를 끄고 OS 기본 위치에 맡긴다")
    ap.add_argument("--cam-width", type=int, default=None,
                    help="카메라 창 가로 크기(px). 안 주면 화면에 맞춰 자동으로 정한다")
    ap.add_argument("--show-cams", action="store_true",
                     help="카메라 원본 + ArUco/geti 오버레이 창도 같이 띄운다 (디버깅용)")
    ap.add_argument("--geti-device", type=str, default="CPU")
    ap.add_argument("--mock-complete", action="store_true",
                     help="차량이 아직 없을 때 GRASP/PLACE 를 즉시 완료된 것으로 흉내낸다(시험용)")
    ap.add_argument("--step", action="store_true",
                     help="단계마다 자동으로 안 넘어가고 LiveMap 의 Next 버튼을 눌러야 진행 "
                          "(조건 충족 여부는 버튼 옆 표시등 초록/빨강으로 보여줌)")
    ap.add_argument("--carrying", type=str, default=None,
                    help="차량이 이미 이 기물을 들고 있다고 보고 운반부터 시작한다 "
                         "(중단된 실행 이어가기. 예: --carrying rook)")
    ap.add_argument("--category", type=str, default=None,
                    choices=sorted(set(mcfg.PIECE_DEST_BOX.values())),
                    help="이 카테고리(예: chess, toy) 기물만 정리한다 — 실행 내내 "
                         "유지되는 필터다. 다른 카테고리 기물은 화면에 보여도 "
                         "무시한다. 터미널에 직접 친 지시(예: '공 가져와')는 "
                         "이 필터와 무관하게 그대로 동작한다.")
    ap.add_argument("--vehicle-ip", type=str, default=None,
                     help="차량(Pi) IP — 주면 실제 UDP로 전송(UdpVehicleLink), "
                          "안 주면 콘솔에만 찍는다(ConsoleVehicleLink)")
    ap.add_argument("--vehicle-cmd-port", type=int, default=5005)
    ap.add_argument("--vehicle-status-port", type=int, default=5006)
    ap.add_argument("--hz-every", type=int, default=20,
                    help="N 사이클마다 루프 Hz 와 단계별 소요를 출력한다(0이면 끄기)")
    # --- 기록과 모니터링 (2026-08-29 실기 준비) ---
    ap.add_argument("--log-file", type=str, default=None,
                    help="상태 전이·Pi 보고를 이 경로에 남긴다(.jsonl 도 같이 생김). "
                         "안 주면 host/logs/ 아래에 시각으로 자동 생성")
    ap.add_argument("--no-log", action="store_true",
                    help="파일 기록을 끈다(기본은 켜짐)")
    ap.add_argument("--quiet-monitor", action="store_true",
                    help="상태 전이·Pi 보고를 터미널에 안 찍는다(파일에는 남는다)")
    ap.add_argument("--manual", action="store_true",
                    help="터미널에서 Enter 를 칠 때마다 다음 단계로 넘어간다. "
                         "b+Enter 는 한 단계 되돌리기, q+Enter 는 정지 후 종료")
    # --- Pi bringup/teardown 자동화 (2026-09-04, 사용자 지시) ---
    #
    # --vehicle-ip 를 줄 때만 의미가 있다 — 콘솔/mock 모드는 Pi 를 안 쓴다.
    # pi_lifecycle.py 의 모듈 docstring 참고.
    ap.add_argument("--no-auto-bringup", action="store_true",
                     help="시작할 때 Pi bringup 을 자동으로 새로 띄우고 끝날 때 "
                          "자동으로 정리하는 것을 끈다(기본은 켜짐, --vehicle-ip "
                          "일 때만 적용). test_ready.sh 로 직접 기동해 둔 상태를 "
                          "그대로 쓰고 싶을 때 사용")
    ap.add_argument("--pi-ssh-host", type=str, default=None,
                     help="Pi 에 ssh 로 접속할 주소 — 안 주면 --vehicle-ip 를 그대로 쓴다")
    ap.add_argument("--pi-ssh-user", type=str, default=pi_lifecycle.PI_USER_DEFAULT)
    ap.add_argument("--bringup-timeout", type=float, default=45.0,
                     help="Pi 쪽 핵심 노드가 뜨는 것을 이만큼(초) 기다린다")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)

    pi_active = False
    if args.vehicle_ip and not args.no_auto_bringup:
        pi_ssh_host = args.pi_ssh_host or args.vehicle_ip
        print(f"[pi] {pi_ssh_host} bringup 확인 중...", flush=True)
        result = pi_lifecycle.bringup(pi_ssh_host, pi_user=args.pi_ssh_user,
                                      ready_timeout=args.bringup_timeout)
        if not result.ok:
            print(f"[pi] bringup 실패 — {result.detail}", flush=True)
            print("[pi] 로봇이 준비되지 않았습니다. tools/ops/test_ready.sh 로 "
                  "직접 상태를 확인하거나, --no-auto-bringup 으로 이 확인을 "
                  "건너뛰고 진행하세요.", flush=True)
            return 1
        print(f"[pi] 준비 완료 — {result.detail}", flush=True)
        pi_active = True

    try:
        return _run_mission(args)
    finally:
        if pi_active:
            pi_ssh_host = args.pi_ssh_host or args.vehicle_ip
            print(f"[pi] {pi_ssh_host} bringup 정리 중...", flush=True)
            td = pi_lifecycle.teardown(pi_ssh_host, pi_user=args.pi_ssh_user)
            print(f"[pi] {'정리 완료' if td.ok else '⚠️ 정리 실패'} — {td.detail}",
                  flush=True)


def _run_mission(args) -> int:
    global _stop

    detector = make_detector()
    cams = [Camera.load(f"cam{i}", i) for i in args.cams]
    # ⚠️ Camera.load() 는 npz 가 없어도 예외를 내지 않고 HFOV 근사 행렬로
    # 조용히 넘어간다(config.py 주석 참고). run_localize.py 는 이 경고를
    # 찍는데 여기에는 없었다 — 미션을 실제로 돌리는 쪽이라 더 중요한데도
    # 근사값으로 도는지 사람이 알 방법이 없었다.
    #
    # 근사값과 실측의 차이는 작지 않다. C920 실측 fx=938.0 인데
    # HFOV 70.4° 근사는 fx=907.3 이다 — 약 3.4% 로, 1.35 m 거리에서
    # 4~5 cm 의 위치 오차가 된다.
    for c in cams:
        if not c.calibrated:
            print(f"⚠️ {c.name}: calib/cam*.npz 가 없어 HFOV {cfg.HFOV_DEG}° "
                  f"근사값을 씁니다 — 위치가 몇 cm 틀립니다. "
                  f"calibrate_camera.py 를 먼저 돌리세요.")
    caps = open_cams(args.cams)
    if not any(c.isOpened() for c in caps):
        print("\n열린 카메라가 하나도 없습니다. --cams 로 인덱스를 바꿔 보세요.")
        for c in caps:
            c.release()
        return 1

    print(f"geti 모델 불러오는 중 ({args.geti_device}, 카메라당 1개)...")
    # 카메라마다 별도 Deployment 인스턴스를 준다 — 하나를 공유하면 두 배경
    # 스레드가 동시에 infer() 를 불러서 "Infer Request is busy" 오류가 난다.
    workers = [geti_detector.GetiWorker(
        geti_detector.load_deployment(device=args.geti_device), c.name) for c in cams]
    print("geti 모델 준비 완료.")

    # 창 배치 계획. 실제로 옮기는 것은 첫 프레임을 그린 **뒤**다 —
    # OpenCV 는 창에 처음 그림을 넣을 때 창 크기를 그림 크기로 되돌린다.
    layout = None
    if not args.no_tile:
        layout = window_layout.plan(
            args.display, len(cams) if args.show_cams else 0,
            want_map=not args.no_view,
            cam_aspect=cfg.IMG_W / cfg.IMG_H,
            cam_width=args.cam_width)
        if layout is None:
            print("[display] 화면 정보를 못 읽어 자동 배치를 건너뜁니다 "
                  "(pyobjc 미설치?)")
    if args.show_cams:
        for cam in cams:
            cv2.namedWindow(cam.name, cv2.WINDOW_NORMAL)

    loc = RobotLocalizer()
    tracker = piece_map.PieceTracker()
    fsm = MissionFSM(manual_mode=args.step or args.manual, category=args.category)
    pmap: PieceMap = {}   # 지시 입력 스레드가 루프 시작 전에 참조할 수 있어 미리 초기화

    resolver = None
    if InstructionResolver is None:
        print(f"[지시] 자연어 지시 비활성화: {_instruction_resolver_import_error}")
    else:
        try:
            resolver = InstructionResolver()
        except Exception as exc:  # noqa: BLE001 -- 키 오류 등 다양한 원인
            print(f"[지시] 자연어 지시 비활성화(초기화 실패): {exc}")
    if args.vehicle_ip:
        link = UdpVehicleLink(args.vehicle_ip, cmd_port=args.vehicle_cmd_port,
                              status_port=args.vehicle_status_port)
        print(f"차량 연결: UDP -> {args.vehicle_ip}:{args.vehicle_cmd_port} "
              f"(상태 수신: :{args.vehicle_status_port})")
    else:
        link = ConsoleVehicleLink(auto_complete=args.mock_complete)

    def _reset_all() -> None:
        # LiveMap 리셋 버튼 콜백 — 화면뿐 아니라 기물 추적/미션 상태도 같이 지운다.
        tracker.reset()
        fsm.reset()
        print("\n[live_map] 리셋됨 — 기물 추적/미션 상태 초기화\n")

    def _toggle_mode() -> None:
        # LiveMap Mode 버튼 콜백 — 자동↔수동 전환, 처음부터 다시 시작.
        fsm.set_manual_mode(not fsm.manual_mode)
        tracker.reset()
        print(f"\n[live_map] 모드 전환 -> {'MANUAL' if fsm.manual_mode else 'AUTO'} (초기화됨)\n")

    live_map = (LiveMap(on_reset=_reset_all, on_next=fsm.request_advance,
                        on_back=fsm.request_back, on_toggle_mode=_toggle_mode)
                if not args.no_view else None)

    if args.carrying:
        if fsm.begin_carrying(args.carrying):
            print(f"[이어서] '{args.carrying}' 을 이미 들고 있다고 보고 "
                  f"{fsm.state.name} 부터 시작합니다")
        else:
            print(f"'{args.carrying}' 의 목적지 상자를 모릅니다 — "
                  f"mission_config.PIECE_DEST_BOX 를 확인하세요")
            for c in caps:
                c.release()
            return 1

    print("\n시작 — 보이는 기물을 가까운 순서대로 라벨별 상자로 나릅니다"
          " (체스말→chess, 나머지→toy).")
    print("q 또는 Ctrl+C 로 종료\n")

    # --- 터미널 입력 한 줄 (2026-08-28 Enter 즉시정지 / 2026-09-01 자연어
    # 지시. 2026-09-02~09-04 사이 AWAIT_CONTINUE·AWAIT_COMMAND 로 잠깐
    # 확장됐다가 2026-09-04 밤 사용자 지시("AWAIT 다 없애라고. 원래
    # RETURN_HOME 있던 버전으로 내놔")로 그 기능 전체가 빠졌다 — 이제
    # 한 그룹을 다 옮겨도 묻지 않고 곧장 RETURN_HOME → SEARCH_TARGET 이다.) ---
    #
    # 실제로 바퀴가 도는 동안 사람이 손닿는 곳에 정지 수단이 있어야 한다.
    # cv2 창의 q 는 창에 포커스가 있어야 먹으므로 터미널에서는 못 쓴다.
    # ⚠️ stdin 이 TTY 가 아니면(백그라운드·nohup·파이프) readline 이 EOF 로
    # 즉시 돌아온다. 그걸 Enter 로 오해하면 기동하자마자 멈춘다 — 실제로
    # 그랬다. 사람이 칠 수 있는 터미널일 때만 감시를 건다.
    #
    # 스레드는 하나만 둔다 — 줄을 읽어 큐에 넣기만 하고, 그 줄이 무슨
    # 뜻인지(다음 단계냐/지시 문장이냐)는 전부 메인 루프에서 그때그때의
    # fsm.state 를 보고 해석한다. 두 스레드가 같은 stdin 을 동시에
    # readline() 하면 어느 쪽이 줄을 가져갈지 알 수 없어서(레이스),
    # 예전처럼 모드별로 스레드를 따로 두지 않는다 — 또한 FSM 을 건드리는
    # 코드는 메인 루프에만 두는 기존 원칙(자연어 지시 결과 처리부 참고)과도
    # 맞다.
    _line_q: "queue.Queue[str]" = queue.Queue()
    _stdin_ready = sys.stdin.isatty()
    # 종료 후 코멘트 프롬프트(아래 "실행 후 코멘트")가 이 스레드가 살아
    # 있는지 확인하고, 새로 읽지 않고 같은 큐에서 받아야 하므로 참조를
    # 남겨 둔다.
    _line_reader_thread: Optional[threading.Thread] = None
    if _stdin_ready:
        def _line_reader():
            global _stop
            while not _stop:
                try:
                    line = sys.stdin.readline()
                except Exception:
                    return
                if line == "":
                    return               # EOF 는 입력이 아니다
                _line_q.put(line.strip())
        _line_reader_thread = threading.Thread(target=_line_reader, daemon=True)
        _line_reader_thread.start()
        if args.manual:
            print("\n>>> 수동 모드 — Enter: 다음 단계 / b: 되돌리기 / q: 정지 <<<\n",
                  flush=True)
        elif not args.no_stop_on_enter:
            print("\n>>> 빈 줄+Enter: 즉시 정지 / 문장+Enter: 그 기물을 지시"
                  "(예: '퀸 가져와', '룩 정리해') <<<\n", flush=True)
    else:
        print("\n[주의] stdin 이 터미널이 아니라 Enter 정지·자연어 지시를 "
              "못 겁니다 — --seconds 로 시간 제한을 두세요\n",
              flush=True)

    # --- 기록 ---
    _log_path = None
    if not args.no_log:
        _log_path = (Path(args.log_file) if args.log_file
                     else mission_log.default_log_path())
    logger = mission_log.MissionLogger(path=_log_path,
                                       echo=not args.quiet_monitor)

    _deadline = (time.time() + args.seconds) if args.seconds else None

    frames_seen = 0
    _placed = False
    _last_hz = None
    # --- 루프 Hz 측정 (2026-08-28 HANDOFF §0-2) ---
    hz_n = 0
    hz_t0 = time.perf_counter()
    hz_acc = {"cap": 0.0, "geti": 0.0, "fsm": 0.0, "view": 0.0}
    try:
        # 라벨을 다 옮겨도 안 끝난다 — 새 기물이 놓이면 계속 반복
        while not _stop:
            if _deadline is not None and time.time() >= _deadline:
                print("\n[STOP] 제한 시간 — 정지합니다", flush=True)
                break
            _t = time.perf_counter()
            grabbed, dets = [], []
            for cap in caps:
                ok, frame = cap.read()
                grabbed.append(frame if ok else None)
                dets.append({} if not ok else
                            detect(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

            pose = loc.update(cams, dets)
            _t_cap = time.perf_counter(); hz_acc["cap"] += _t_cap - _t

            preds = []
            for frame, worker in zip(grabbed, workers):
                if frame is None:
                    preds.append(None)
                    continue
                worker.submit(frame.copy())
                preds.append(worker.latest())

            obs_lists = [piece_map.pieces_from_prediction(cam, pred)
                         for cam, pred in zip(cams, preds)]
            pmap = tracker.update(obs_lists)
            _t_geti = time.perf_counter(); hz_acc["geti"] += _t_geti - _t_cap

            _state_before = fsm.state
            _target_before = fsm._target_xy
            fsm.step(pose, pmap, link)
            # ⚠️ GRASP 진입 순간 물체가 **로봇 기준** 어디에 있는지 남긴다.
            #
            # VLA 정책은 학습 때 본 위치에서만 물체를 집는다 — 좌우 이미지
            # 응답이 액션 std 의 0.62% 라 물체를 눈으로 좇지 못한다(2026-09-04
            # 홀드아웃). 그래서 "팔이 뻗는 양"이 일정해도 차가 서는 자리가
            # 흔들리면 파지가 실패한다.
            #
            # 이 한 줄이 그 둘을 가른다. 여러 회차의 전방/좌우 값이 흩어져
            # 있으면 정지 위치 문제이고, 모여 있는데도 실패하면 정책 문제다.
            # 지금까지는 구분할 방법이 없었다.
            #
            # ⚠️ 2026-09-06: 이 줄이 두 판 연속 **안 찍혔다.** 조건이
            # `_target_before is not None` 이었는데, 그 값은 fsm.step() **전**
            # 스냅샷이라 전환을 만든 그 스텝에서 목표가 정해지면 아직 None
            # 이다. 진입 순간을 잡으려다 진입 순간을 놓친 셈이다.
            # step() 뒤 값으로 물러서고, 그래도 없으면 **왜 없는지**를 찍는다 —
            # 조용히 아무것도 안 나오는 것이 제일 나쁘다.
            if _state_before != fsm.state and fsm.state.name == "GRASP":
                _tgt = _target_before if _target_before is not None else fsm._target_xy
                if _tgt is None or not pose.ok:
                    print(
                        f"\n[파지 진입] 오프셋을 못 잽니다 — "
                        f"목표 {'없음' if _tgt is None else '있음'} · "
                        f"포즈 {'없음' if not pose.ok else '있음'}\n",
                        flush=True,
                    )
                    _tgt = None
            else:
                _tgt = None
            if _tgt is not None:
                _dx = _tgt[0] - pose.x
                _dy = _tgt[1] - pose.y
                _c = math.cos(math.radians(pose.yaw_deg))
                _s = math.sin(math.radians(pose.yaw_deg))
                _fwd = (_dx * _c + _dy * _s) * 1000.0     # 로봇 정면(+) mm
                _lat = (-_dx * _s + _dy * _c) * 1000.0    # 로봇 좌측(+) mm
                print(f"\n[파지 진입] 물체가 로봇 기준 전방 {_fwd:+.0f}mm · "
                      f"좌우 {_lat:+.0f}mm  (yaw {pose.yaw_deg:+.1f}도)\n",
                      flush=True)
            _t_fsm = time.perf_counter(); hz_acc["fsm"] += _t_fsm - _t_geti
            frames_seen += 1

            # 방금 파지/투입에 성공한 그 개체만 지도에서 숨긴다(2026-09-05,
            # 사용자 지시) — mission.py의 last_grasp_event/last_place_event
            # 문서와 piece_map.PieceTracker.suppress_at() 참고. 1회성
            # 이벤트라 읽는 즉시 None으로 되돌린다.
            if fsm.last_grasp_event is not None:
                label, xy = fsm.last_grasp_event
                tracker.suppress_at(label, xy)
                fsm.last_grasp_event = None
            if fsm.last_place_event is not None:
                label, xy = fsm.last_place_event
                tracker.suppress_at(label, xy)
                fsm.last_place_event = None

            # 터미널 입력 한 줄 처리 — 지금 fsm.state 기준으로 해석한다.
            try:
                while True:
                    line = _line_q.get_nowait()
                    if args.manual:
                        key = line.lower()
                        if key in ("q", "quit", "exit"):
                            _stop = True
                            print("\n[STOP] q — 정지하고 종료합니다", flush=True)
                        elif key == "b":
                            fsm.request_back()
                            print("\n[수동] 한 단계 되돌립니다", flush=True)
                        else:
                            fsm.request_advance()
                    elif not args.no_stop_on_enter:
                        if line == "":
                            _stop = True
                            print("\n[STOP] Enter — 정지합니다", flush=True)
                        elif resolver is None:
                            print(f"\n[지시] 무시함 — 자연어 지시 비활성화"
                                  f"({_instruction_resolver_import_error or '초기화 실패'})\n",
                                  flush=True)
                        elif resolver.busy:
                            print("\n[지시] 이전 요청을 아직 처리 중입니다 — 잠시 후 다시 시도하세요\n",
                                  flush=True)
                        else:
                            labels = visible_labels(pmap)
                            if not labels:
                                print("\n[지시] 지금 화면에 보이는 기물이 없습니다\n", flush=True)
                            else:
                                print(f"\n[지시] \"{line}\" 처리 중...\n", flush=True)
                                resolver.submit(line, labels)
            except queue.Empty:
                pass

            # 자연어 지시 결과 — resolver.submit() 은 백그라운드 스레드에서
            # 돌므로, FSM 을 건드리는 건(set_instruction) 여기 메인 루프
            # 안에서만 한다(2026-09-01).
            if resolver is not None:
                result = resolver.poll_result()
                if result is not None:
                    if result.error:
                        print(f"\n[지시] API 오류: {result.error}\n", flush=True)
                    elif not result.matched:
                        print(f"\n[지시] 못 알아들음 — {result.reasoning}\n", flush=True)
                    else:
                        dest_xy = mcfg.DELIVER_HERE_XY if result.intent == "fetch" else None
                        where = "저한테" if result.intent == "fetch" else "정해진 상자로"
                        applied_now = fsm.set_instruction(result.target_label, dest_xy=dest_xy)
                        when = "바로 전환" if applied_now else "지금 든 것 마친 뒤 적용"
                        print(f"\n[지시] '{result.target_label}' 를 {where} "
                              f"({when}) — {result.reasoning}\n", flush=True)

            # 사이클마다 넘긴다 — 무엇이 사건인지는 logger 가 판단한다.
            logger.record(
                state=fsm.state.name, pose=pose, cmd=fsm.last_cmd,
                target=fsm.target_label, report=link.last_report,
                base_alarm=getattr(link, "base_alarm", None),
                ready=fsm.ready_to_advance, hz=_last_hz)

            # ⚠️ pose 를 잃으면 FSM 이 step() 입구에서 그대로 나가고 명령도
            # 안 나간다. 그런데 화면에도 아무것도 안 찍혀서, 밖에서 보면
            # "hz 만 도는데 차가 안 움직임"으로만 보인다 — 2026-09-05 실기에서
            # 이걸 Pi 쪽 문제로 오해해 UDP 수신·워치독·노드 상태까지 뒤진 뒤에야
            # Host 가 애초에 안 보내고 있었다는 걸 알았다. Pi 워치독이 알아서
            # 세우므로 동작은 안전하지만, 이유는 보여 줘야 한다.
            if not pose.ok and frames_seen % 10 == 0:
                line = (f"[로봇 잃음] 마커가 안 보입니다 — 명령을 보내지 않습니다 "
                        f"(cams={pose.n_cams}). 팔이 마커를 가렸거나 작업 영역 밖입니다")
                width = shutil.get_terminal_size((100, 24)).columns - 1
                print("\r" + line[:width].ljust(width), end="", flush=True)

            if fsm.state == State.SEARCH_TARGET and frames_seen % 10 == 0:
                # ⚠️ 한 줄에 담기지 않으면 터미널이 줄을 접고, 그러면 \r 가
                # 시작이 아니라 접힌 줄 머리로 돌아가 이전 내용이 안 지워진다
                # — 같은 문구가 두 번 찍힌 것처럼 보인다(2026-09-05).
                why = fsm.search_reason or "남은 기물 없음"
                line = f"[SEARCH_TARGET] {why} — {pose}"
                width = shutil.get_terminal_size((100, 24)).columns - 1
                print("\r" + line[:width].ljust(width), end="", flush=True)

            if live_map is not None:
                live_map.update(pose, pmap, goal=fsm.nav_goal, nav=fsm.last_nav,
                                 corner=fsm.nav_corner, path=fsm.nav_path,
                                 state_name=fsm.state.name, target_label=fsm.target_label,
                                 ready=(fsm.ready_to_advance if pose.ok else None),
                                 manual_mode=fsm.manual_mode, cmd=fsm.last_cmd)
                if live_map.closed():
                    break
            hz_acc["view"] += time.perf_counter() - _t_fsm

            hz_n += 1
            if args.hz_every and hz_n >= args.hz_every:
                _el = time.perf_counter() - hz_t0
                _ms = {k: v / hz_n * 1000 for k, v in hz_acc.items()}
                _last_hz = hz_n / _el
                print(f"\n[hz] {hz_n / _el:.2f} Hz  ({_el / hz_n * 1000:.0f} ms/사이클)"
                      f"  캡처+ArUco {_ms['cap']:.0f}  geti {_ms['geti']:.0f}"
                      f"  FSM {_ms['fsm']:.0f}  화면 {_ms['view']:.0f} ms", flush=True)
                hz_n = 0
                hz_t0 = time.perf_counter()
                hz_acc = {k: 0.0 for k in hz_acc}

            if args.show_cams:
                for cam, frame, det, pred in zip(cams, grabbed, dets, preds):
                    if frame is None:
                        continue
                    disp = geti_detector.draw(frame, pred) if pred is not None else frame
                    cv2.imshow(cam.name, draw(disp, cam, det, pose))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if layout is not None and not _placed:
                # 창이 다 뜬 지금 한 번만 옮긴다. 매 사이클 옮기면 사람이
                # 손으로 옮겨 놓은 창을 계속 되돌려 버린다.
                print(window_layout.apply(
                    layout, [c.name for c in cams] if args.show_cams else [],
                    live_map.fig if live_map is not None else None), flush=True)
                _placed = True
    finally:
        # ⚠️ 링크를 그냥 닫으면 Pi 워치독(3사이클 = 0.3초)이 설 때까지
        # 바퀴가 돈다. 명시적으로 정지를 여러 번 보내 즉시 세운다.
        # UDP 라 한 발이 유실될 수 있으므로 연발한다.
        try:
            for _ in range(8):
                link.send(MissionCommand("stop", "SEARCH_TARGET", 0.0, 0.0, 0.0))
                time.sleep(0.05)
            print("[STOP] 정지 명령 8회 송신 완료")
        except Exception as exc:
            print(f"[STOP] 정지 명령 실패: {exc} — Pi 워치독이 0.3초 안에 세웁니다")
        for worker in workers:
            worker.stop()
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        if live_map is not None:
            live_map.close()
        if isinstance(link, UdpVehicleLink):
            link.close()
        logger.close()

    # ⚠️ 여기 있던 "실행 후 코멘트" 프롬프트를 뺐다(2026-09-06, Windows).
    #
    # 종료 직후 _line_q 에서 한 줄을 기다리며 부연 설명을 받는 기능이었는데,
    # Windows 에서는 그 큐를 채우는 _line_reader 스레드가 돌지 않아 미션이
    # 끝나도 프롬프트에서 못 빠져나온다.
    #
    # 로그 자체는 그대로 남는다(MissionLogger). 코멘트를 남기려면
    # mission_log.append_annotation 을 따로 부르면 된다.

    print(f"\n\n종료 — 마지막 상태: {fsm.state.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
