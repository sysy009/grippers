"""그리퍼캠 화각(FOV) 측정 — 체스보드 자동 촬영판.

host/aruco/calibrate_camera.py 와 목적이 같지만 두 가지가 다르다.

1. **자동 촬영이다.** SPACE 를 안 눌러도, 판이 충분히 움직이면 알아서 담는다.
   화각 숫자 하나 뽑자고 20번 키를 누르게 할 이유가 없다.
2. **화각을 두 가지로 낸다.** 아래 "왜 두 개인가" 참고.

사용법
    python grippers\\tools\\arm\\gripper_cam_fov.py
    python grippers\\tools\\arm\\gripper_cam_fov.py --board 9 6 --need 25

    체스보드를 카메라 앞에서 **천천히** 움직이면 된다. 화면 구석·기울기·거리를
    바꿔가며 담는 게 중요하다 — 정면 사진만 모이면 왜곡계수가 안 풀리고,
    그러면 화각도 같이 틀린다.

⚠️ --board 는 "칸 수"가 아니라 **내부 코너 수**다. 8x8 칸 체스판이면 7x7 이다.
   틀리면 한 장도 안 잡힌다 — 조용히 틀리지는 않는다.

⚠️ 측정은 **1920x1080** 에서 한다. 이 카메라는 1280x720 을 요청해도 센서를
   자르지 않고 1.50배로 축소하는 것이 실측으로 확인됐으므로(2026-09-04),
   화각은 두 해상도가 같다. 큰 쪽에서 재는 편이 코너 정밀도가 높다.
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

# 이 프로젝트가 쓰는 실물 체스판 — 놀이칸 8x8 이므로 내부 코너 7x7.
BOARD_DEFAULT = (7, 7)
# 한 칸의 실제 길이. ⚠️ 결과에 영향이 없다 — K 와 왜곡계수는 스케일과
# 무관하기 때문이다(host/aruco/calibrate_camera.py 주석에 실측 확인 기록).
# 그래도 넣는 이유는 calibrateCamera 가 물체 좌표를 요구하기 때문이다.
SQUARE_M = 0.035
CAPTURE_W, CAPTURE_H = 1920, 1080
# 이만큼은 움직여야 새 자세로 친다(코너 중심의 픽셀 거리). 너무 작게 잡으면
# 거의 같은 사진 20장을 모으게 되고, 그러면 장수만 많지 왜곡이 안 풀린다.
MIN_MOVE_PX = 80.0


def fov_pinhole(f_px: float, size_px: int) -> float:
    """핀홀 모형 화각(도). 왜곡을 편 뒤의 이상적 카메라 기준이다."""
    return math.degrees(2.0 * math.atan(size_px / (2.0 * f_px)))


def fov_actual(K, dist, w: int, h: int) -> tuple:
    """실제로 담기는 화각(도) — 왜곡을 포함한 값.

    왜 두 개인가: 광각 렌즈는 배럴 왜곡 때문에 **실제로 보이는 범위가 핀홀
    모형보다 넓다.** fx 로만 계산한 값은 "왜곡을 편 이미지"의 화각이라, 카메라가
    실제로 몇 도를 담느냐와 다르다. 팀 공유용으로는 후자가 맞다.

    이미지 테두리 픽셀을 undistortPoints 로 정규화 좌표(=tan 각)로 되돌린 뒤
    최대 각을 좌우/상하로 나눠 재는 방식이다.
    """
    edge = []
    for x in range(0, w, 4):
        edge += [[x, 0], [x, h - 1]]
    for y in range(0, h, 4):
        edge += [[0, y], [w - 1, y]]
    pts = np.array(edge, dtype=np.float32).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    x, y = norm[:, 0], norm[:, 1]
    hfov = math.degrees(math.atan(-x.min()) + math.atan(x.max()))
    vfov = math.degrees(math.atan(-y.min()) + math.atan(y.max()))
    dfov = math.degrees(2.0 * math.atan(np.max(np.hypot(x, y))))
    return hfov, vfov, dfov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="cv2.VideoCapture 인덱스")
    ap.add_argument("--board", nargs=2, type=int, default=BOARD_DEFAULT,
                    metavar=("COLS", "ROWS"), help="내부 코너 수 (칸 수 - 1)")
    ap.add_argument("--need", type=int, default=20, help="모을 장수")
    ap.add_argument("--timeout", type=float, default=180.0, help="최대 초")
    ap.add_argument("--out", default=None, help="결과 json 경로")
    ap.add_argument("--no-window", action="store_true", help="미리보기 창 없이")
    args = ap.parse_args()

    board = tuple(args.board)
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.cam} 를 열지 못했습니다")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"해상도 {w}x{h}  체스보드 내부코너 {board[0]}x{board[1]}")
    print(f"판을 천천히 움직이세요 — 구석·기울기·거리를 바꿔가며 {args.need}장을 모읍니다.")

    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * SQUARE_M

    obj_points, img_points, centers = [], [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    started = time.time()
    last_msg = 0.0

    while len(img_points) < args.need and time.time() - started < args.timeout:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board, flags)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            center = corners.reshape(-1, 2).mean(axis=0)
            # 이전에 담은 자세들과 충분히 떨어져 있을 때만 담는다.
            if all(np.hypot(*(center - c)) > MIN_MOVE_PX for c in centers):
                obj_points.append(objp.copy())
                img_points.append(corners)
                centers.append(center)
                print(f"  담음 {len(img_points)}/{args.need}  중심 ({center[0]:.0f},{center[1]:.0f})")
        if not args.no_window:
            view = cv2.resize(frame, (960, 540))
            if found:
                cv2.drawChessboardCorners(view, board, corners * 0.5, found)
            cv2.putText(view, f"{len(img_points)}/{args.need}", (20, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            cv2.imshow("gripper cam FOV - move the board slowly", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif time.time() - last_msg > 5.0:
            last_msg = time.time()
            print(f"  ... {len(img_points)}/{args.need}  ({time.time()-started:.0f}s)")

    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()

    if len(img_points) < 6:
        raise SystemExit(f"장수 부족({len(img_points)}) — 보정 불가. --board 가 맞는지 확인하세요.")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ph = (fov_pinhole(fx, w), fov_pinhole(fy, h),
          fov_pinhole((fx + fy) / 2, int(math.hypot(w, h))))
    ac = fov_actual(K, dist, w, h)

    print(f"\n장수 {len(img_points)}   재투영 오차 RMS {rms:.3f} px")
    print(f"fx {fx:.1f}  fy {fy:.1f}  cx {cx:.1f}  cy {cy:.1f}")
    print(f"왜곡 k1 {dist[0][0]:+.4f}  k2 {dist[0][1]:+.4f}  p1 {dist[0][2]:+.4f} "
          f"p2 {dist[0][3]:+.4f}  k3 {dist[0][4]:+.4f}")
    print(f"\n{'':14}{'좌우(H)':>10}{'상하(V)':>10}{'대각(D)':>10}")
    print(f"{'실제 담기는':14}{ac[0]:9.1f}°{ac[1]:9.1f}°{ac[2]:9.1f}°   <- 팀 공유용")
    print(f"{'왜곡 편 뒤':14}{ph[0]:9.1f}°{ph[1]:9.1f}°{ph[2]:9.1f}°")

    out = Path(args.out) if args.out else Path(__file__).with_name("gripper_cam_fov.json")
    out.write_text(json.dumps({
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "resolution": [w, h],
        "note": "1280x720 은 이 해상도의 1.50배 축소이므로 화각이 같다",
        "images": len(img_points), "rms_px": rms,
        "K": K.tolist(), "dist": dist.ravel().tolist(),
        "fov_actual_deg": {"h": ac[0], "v": ac[1], "d": ac[2]},
        "fov_pinhole_deg": {"h": ph[0], "v": ph[1], "d": ph[2]},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
