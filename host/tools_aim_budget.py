"""Host 가 낼 수 있는 조준 정밀도를 잰다 — Pi 팀 2026-08-27 요청 §1 에 대한 답.

## 무엇을 재는가

Pi 는 "Host 가 실제로 낼 수 있는 조준 정밀도(전후 ±몇 mm, 좌우/각도 ±몇 mm·도)"
를 물었다. 그 오차는 세 항으로 갈린다.

    ① 로봇 위치 추정 오차      ← ArUco. 실측 0.51mm (2026-08-27, 5084프레임)
    ② 기물 위치 추정 오차      ← geti 박스 아래변 중앙을 바닥에 투영. **미실측**
    ③ 구동 정지 오차           ← Pi 데드밴드(최소 버스트 ~20mm). Host 소유 아님

이 스크립트는 ①과 ②를 **합쳐서** 잰다. 따로 재서 더하면 안 되기 때문이다 —
둘 다 같은 카메라·같은 외부파라미터에서 나오므로 오차가 상관돼 있고, 실제
조준에 쓰이는 것은 절대 좌표가 아니라 **로봇에서 기물까지의 벡터**다. 그 벡터를
매 프레임 다시 계산해서 흩어지는 폭을 보면 ①②가 한 번에 잡힌다.

벡터는 **차체 기준**으로 바꿔서 낸다(전방 +, 왼쪽 +). Pi 의 포획 영역이 차체
기준 직사각형이라 그 축으로 봐야 의미가 있다.

③은 여기서 못 잰다. 차량이 붙어 있어야 한다.

## 쓰는 법

    python tools_aim_budget.py                 # 200 프레임
    python tools_aim_budget.py --frames 500
    python tools_aim_budget.py --settle 30     # 앞 N 프레임은 버린다(트랙 확정 대기)

기물을 **움직이지 말 것.** 흔들림을 재는 것이라 실제로 움직이면 그게 그대로
오차로 잡힌다.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent / "aruco"))

import config as cfg
import geti_detector
import piece_map
from localizer import Camera, RobotLocalizer, detect, make_detector
from run_localize import open_cams

# 같은 기물로 볼 최대 거리(m). PieceTracker 의 트랙 id 를 쓰지 않고 좌표로
# 묶는 이유는, 트랙이 중간에 재생성돼도 같은 물리적 기물로 이어 보기 위해서다.
BIN_M = 0.08


def _robot_frame(px: float, py: float, rx: float, ry: float,
                 yaw_deg: float) -> tuple[float, float]:
    """지도 좌표의 기물을 차체 기준(전방 +x, 왼쪽 +y)으로 옮긴다."""
    dx, dy = px - rx, py - ry
    c = math.cos(math.radians(-yaw_deg))
    s = math.sin(math.radians(-yaw_deg))
    return dx * c - dy * s, dx * s + dy * c


def _spread(vals: list[float]) -> tuple[float, float, float]:
    """(표준편차, 최대편차, 폭) — 전부 mm. 최대편차는 중앙값 기준이다."""
    if len(vals) < 2:
        return 0.0, 0.0, 0.0
    med = statistics.median(vals)
    sd = statistics.stdev(vals) * 1000.0
    worst = max(abs(v - med) for v in vals) * 1000.0
    return sd, worst, (max(vals) - min(vals)) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Host 조준 정밀도 실측")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--settle", type=int, default=20,
                    help="트랙이 확정될 때까지 버릴 앞쪽 프레임 수")
    ap.add_argument("--cams", type=int, nargs="+", default=list(cfg.CAM_INDICES))
    ap.add_argument("--geti-device", type=str, default="CPU")
    args = ap.parse_args()

    detector = make_detector()
    cams = [Camera.load(f"cam{i}", i) for i in args.cams]
    caps = open_cams(args.cams)
    if not any(c.isOpened() for c in caps):
        print("\n열린 카메라가 없습니다.")
        return 1

    print(f"geti 모델 불러오는 중 ({args.geti_device})...")
    workers = [geti_detector.GetiWorker(
        geti_detector.load_deployment(device=args.geti_device), c.name) for c in cams]
    loc = RobotLocalizer()
    tracker = piece_map.PieceTracker()

    poses: list[tuple[float, float, float]] = []
    # (라벨, 좌표bin) -> [(전방 m, 좌측 m), ...]
    rel: dict[tuple[str, tuple[int, int]], list[tuple[float, float]]] = {}
    # 같은 bin 의 **원관측** confidence. 트래커 출력에는 안 실려서 따로 모은다 —
    # 흔들림의 크기를 설명하는 변수가 이것이기 때문이다.
    conf: dict[tuple[str, tuple[int, int]], list[float]] = {}
    kept = 0

    print(f"{args.frames} 프레임 수집 — 기물과 로봇을 움직이지 마세요\n")
    try:
        for n in range(args.frames):
            grabbed, dets = [], []
            for cap in caps:
                ok, frame = cap.read()
                grabbed.append(frame if ok else None)
                dets.append({} if not ok else
                            detect(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

            pose = loc.update(cams, dets)

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

            if n < args.settle or not pose.ok or not pose.fresh:
                continue
            kept += 1
            poses.append((pose.x, pose.y, pose.yaw_deg))
            for label, pts in pmap.items():
                for px, py in pts:
                    key = (label, (round(px / BIN_M), round(py / BIN_M)))
                    rel.setdefault(key, []).append(
                        _robot_frame(px, py, pose.x, pose.y, pose.yaw_deg))
            for lst in obs_lists:
                for o in lst:
                    conf.setdefault(
                        (o.label, (round(o.x / BIN_M), round(o.y / BIN_M))),
                        []).append(o.confidence)

            if (n + 1) % 25 == 0:
                print(f"\r  {n + 1}/{args.frames}  (유효 {kept})", end="", flush=True)
    finally:
        for w in workers:
            w.stop()
        for c in caps:
            c.release()

    print(f"\n\n유효 프레임 {kept}개\n")
    if kept < 10:
        print("표본이 너무 적습니다 — 로봇 마커가 보이는지 확인하세요.")
        return 1

    # --- ① 로봇 위치 추정 반복도 ---
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    yaws = [p[2] for p in poses]
    sx, wx, _ = _spread(xs)
    sy, wy, _ = _spread(ys)
    ymed = statistics.median(yaws)
    yaw_sd = statistics.stdev(yaws) if len(yaws) > 1 else 0.0
    yaw_worst = max(abs(v - ymed) for v in yaws)

    print("=" * 66)
    print("① 로봇 위치 추정 반복도 (지도 좌표)")
    print("=" * 66)
    print(f"  x   표준편차 {sx:5.2f} mm   최대편차 {wx:5.2f} mm")
    print(f"  y   표준편차 {sy:5.2f} mm   최대편차 {wy:5.2f} mm")
    print(f"  yaw 표준편차 {yaw_sd:5.3f}도  최대편차 {yaw_worst:5.3f}도")

    # --- ② 조준 벡터 반복도 (①+② 합산) ---
    print()
    print("=" * 66)
    print("② 조준 벡터 반복도 — 차체 기준 (전방 +, 왼쪽 +)")
    print("   이것이 Host 가 답해야 할 값이다. ①이 이미 포함돼 있다.")
    print("=" * 66)
    print(f"  {'기물':10s} {'n':>4s} {'포착률':>6s} {'conf':>7s} {'전방(mm)':>9s}"
          f" {'전후 σ':>8s} {'전후 최대':>9s} {'좌우 σ':>8s} {'좌우 최대':>9s}")

    # 놓치는 프레임이 있는 트랙은 따로 센다. 실측해 보니 조준 흔들림은 거리나
    # 카메라가 아니라 **검출이 안정적인가**로 갈렸다 — 한 줄로 "최대 몇 mm" 를
    # 내면 불안정한 한 기물이 전체를 대표해 버린다.
    STABLE = 0.95
    stable, flaky = [], []
    for (label, _bin), obs in sorted(rel.items(), key=lambda kv: -len(kv[1])):
        if len(obs) < max(10, kept // 4):
            continue      # 잠깐만 보인 트랙은 통계로 안 쓴다
        fwd = [o[0] for o in obs]
        lat = [o[1] for o in obs]
        sf, wf, _ = _spread(fwd)
        sl, wl, _ = _spread(lat)
        # 트랙 좌표와 원관측 좌표는 bin 하나쯤 어긋날 수 있다(트래커가 여러
        # 관측을 섞어 낸다). 이웃 bin 까지 훑어서 가장 표본이 많은 것을 쓴다.
        cs: list[float] = []
        for dx in (0, -1, 1):
            for dy in (0, -1, 1):
                cand = conf.get((label, (_bin[0] + dx, _bin[1] + dy)), [])
                if len(cand) > len(cs):
                    cs = cand
        cmed = statistics.median(cs) if cs else float("nan")
        row = (label, len(obs), statistics.median(fwd) * 1000, sf, wf, sl, wl, cmed)
        (stable if len(obs) >= kept * STABLE else flaky).append(row)

    if not stable and not flaky:
        print("  안정적으로 보인 기물이 없습니다.")
        return 1

    def _show(rows_):
        for label, n, fwd_mm, sf, wf, sl, wl, cmed in sorted(rows_, key=lambda r: -r[4]):
            print(f"  {label:10s} {n:4d} {n / kept * 100:5.0f}% {cmed:7.3f} {fwd_mm:9.0f}"
                  f" {sf:8.2f} {wf:9.2f} {sl:8.2f} {wl:9.2f}")

    print(f"  ── 매 프레임 잡힌 기물 ({len(stable)}개) ──")
    _show(stable)
    if flaky:
        print(f"  ── 가끔 놓치는 기물 ({len(flaky)}개) ── ★ 검출이 흔들리는 것들이다")
        _show(flaky)

    worst_fwd = max((r[4] for r in stable), default=0.0)
    worst_lat = max((r[6] for r in stable), default=0.0)
    fl_fwd = max((r[4] for r in flaky), default=0.0)
    fl_lat = max((r[6] for r in flaky), default=0.0)

    # --- 오차 예산 ---
    print()
    print("=" * 66)
    print("조준 오차 예산")
    print("=" * 66)
    total_f = math.hypot(worst_fwd, 20.0)
    total_l = math.hypot(worst_lat, 20.0)
    print(f"  ①+② Host 관측 · 안정 기물   전후 ±{worst_fwd:5.1f} mm · 좌우 ±{worst_lat:5.1f} mm")
    if flaky:
        print(f"  ①+② Host 관측 · 흔들림 포함 전후 ±{fl_fwd:5.1f} mm · 좌우 ±{fl_lat:5.1f} mm")
    print( "  ③   구동 정지 (Pi 데드밴드)  ±20.0 mm  ← 이 스크립트로는 못 잰다")
    print(f"  합계 (안정 기물, 제곱합)     전후 ±{total_f:5.1f} mm · 좌우 ±{total_l:5.1f} mm")
    print()
    if total_f > 0:
        verdict = "여유 있음" if total_f < 70 else "⚠️ 부족"
        print(f"  Pi 의 미세전진 상한 70 mm 대비 {verdict} ({70 / total_f:.1f}배)")
    if flaky:
        tf = math.hypot(fl_fwd, 20.0)
        v2 = "여유 있음" if tf < 70 else "⚠️ 부족"
        print(f"  가끔 놓치는 기물까지 넣으면 전후 ±{tf:5.1f} mm — {v2} ({70 / tf:.1f}배)")

    # 흔들림을 설명하는 변수 찾기. 포착률은 프록시로 못 쓴다 — PieceTracker 가
    # PIECE_HOLD_SEC 동안 트랙을 붙들어 주므로, 관측이 튀는 기물도 포착률은
    # 98% 로 멀쩡해 보인다. confidence 와 거리 중 무엇이 설명하는지 직접 본다.
    allrows = stable + flaky
    if len(allrows) >= 4:
        print()
        print("  ── 흔들림은 무엇으로 갈리는가 ──")
        for name, idx in (("conf", 7), ("전방거리", 2)):
            pairs = [(r[idx], r[4]) for r in allrows if r[idx] == r[idx]]
            if len(pairs) < 4:
                print(f"    {name}: 표본 부족({len(pairs)}) — 건너뜀")
                continue
            xs_ = [a for a, _ in pairs]
            ys_ = [b for _, b in pairs]            # 전후 최대편차
            mx, my = statistics.mean(xs_), statistics.mean(ys_)
            num = sum((a - mx) * (b - my) for a, b in zip(xs_, ys_))
            den = math.sqrt(sum((a - mx) ** 2 for a in xs_)
                            * sum((b - my) ** 2 for b in ys_))
            r = num / den if den else 0.0
            print(f"    전후 최대편차 vs {name:8s}  상관 r = {r:+.2f}")
        print("    (음의 상관이 크면 confidence 가 낮은 기물이 더 흔들린다는 뜻)")
    print()
    print("  ⚠️ ③은 추정치다(Pi 문서의 '정지 정밀도 약 ±10mm, 최소 버스트 ~20mm').")
    print("     차량을 붙여 '정렬 후 실제로 몇 mm 어긋나는가'를 재야 확정된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
