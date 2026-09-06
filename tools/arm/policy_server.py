"""VLA 추론 서버 — 프레임과 관절값을 받아 액션 청크를 돌려준다.

Pi 의 `vla_inference_node` 가 `policy_source:=remote` 로 뜨면 이쪽을 부른다.
정책은 여기 한 곳에만 있으면 되고, Pi 는 아무것도 안 깔아도 된다.

## ⚠️ 기본 경로가 아니다 — ACT 는 Pi 에서 직접 돈다

2026-09-05 실측(act_v5_all 120k steps, 180x320):

    Pi 로컬  397~465ms (듀티 14%)  — perception·arm_driver·vla_inference 가
                                    같이 도는 조건에서 잰 값이다
    원격     117ms     (듀티 3.5%) — 리사이즈·전송·추론·회신 왕복 전부 포함

원격이 3.5배 빠르지만 **둘 다 청크 3.33초 안에 여유롭게 들어간다.** 여유가
있는데 시연 경로에 노트북과 네트워크 의존을 하나 더 만들 이유가 없어서
`policy_source` 기본값은 `local` 이다.

그래도 이 경로를 남겨 두는 이유는 둘이다.

1. **Pi 에서 실시간이 안 되는 정책.** SmolVLA 가 그렇다
   (`tools/arm/smolvla_check.py`). 그런 정책은 원격 말고 길이 없다.
2. **체크포인트를 자주 갈아끼울 때.** 200MB 를 Pi 로 옮기지 않아도 된다.

이 서버는 그 예외를 위해 있다.

## 사용법

    python grippers\\tools\\arm\\policy_server.py \\
      --ckpt grippers\\host\\act_v5_all_180_120k_120000

    DP(확산 정책)는 GPU 가 있어야 한다. 노트북 RTX 3050 기준으로
    `.venv-dp` 의 파이썬으로 띄운다 — ACT 가 도는 `.venv`(lerobot 0.4.4)로는
    DP 체크포인트가 안 읽힌다(0.6.x processor 형식):

        .venv-dp/Scripts/python.exe grippers/tools/arm/policy_server.py
          --ckpt ckpt_dp_v5_all/dp_v5_all_180_60k_060000
          --device cuda --scheduler DDIM --denoise 25

    Pi 쪽 — 기본이 local 이므로 remote 를 **명시해야** 부른다:
    ros2 launch grippers_bringup bringup.launch.py use_vla:=true \\
      policy_source:=remote policy_url:=http://192.168.0.2:8770

## ⚠️ 표준 라이브러리만 쓴다

Flask 를 안 쓰는 이유는 의존성 하나를 더 만들지 않기 위해서다. 요청이
3.33초에 한 번뿐이라 성능도 문제가 안 된다.

## ⚠️ 인증이 없다

같은 LAN 안에서만 쓸 것. 기본 바인드가 0.0.0.0 이므로 공용 네트워크에
띄우면 누구나 이 정책을 부를 수 있다. 로봇 팔을 움직이는 값을 내주는
서버라는 점을 기억할 것.
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ros2_ws" / "src" / "grippers_vla"))
from grippers_vla.policy_runner import PolicyRunner  # noqa: E402

RUNNER: PolicyRunner | None = None
STATS = {"calls": 0, "total_ms": 0.0}
#: ⚠️ ThreadingHTTPServer 라 요청이 겹칠 수 있다. 정책과 전처리기는 한 벌뿐이라
#: 두 요청이 동시에 들어가면 조용히 섞인다. 3.33초에 한 번이라 줄 서도 손해가
#: 없으니 한 번에 하나만 들여보낸다. STATS 도 이 락 안에서만 건드린다.
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 요청마다 stderr 로 한 줄씩 찍는 기본 동작을 끈다
        pass

    def _send(self, code, meta, blob=b""):
        header = json.dumps(meta).encode("utf-8")
        body = len(header).to_bytes(4, "big") + header + blob
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.path.startswith("/health"):
            self.send_error(404)
            return
        with LOCK:
            calls, total_ms = STATS["calls"], STATS["total_ms"]
        info = {
            "ok": True,
            "ckpt": RUNNER.ckpt,
            "policy_hw": list(RUNNER.policy_hw),
            "chunk_size": RUNNER.chunk_size,
            "n_action_steps": RUNNER.n_action_steps,
            "calls": calls,
            "avg_ms": round(total_ms / calls, 1) if calls else 0.0,
        }
        payload = json.dumps(info).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if not self.path.startswith("/predict"):
            # 본문을 안 읽고 닫으면 클라이언트가 오류 대신 connection reset 을 본다.
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            if len(raw) < 4:
                raise ValueError(f"본문이 잘렸습니다: {len(raw)}바이트")
            n = int.from_bytes(raw[:4], "big")
            meta = json.loads(raw[4:4 + n].decode("utf-8"))
            if meta.get("dtype", "uint8") != "uint8":
                raise ValueError(f"uint8 픽셀만 받습니다: {meta['dtype']}")
            shape = tuple(int(v) for v in meta["shape"])
            pixels = raw[4 + n:]
            # ⚠️ 길이를 먼저 확인한다. 안 맞으면 numpy 가 내는 reshape 오류가
            # "무엇이 어긋났는지"를 안 알려줘서, 프레임 크기 문제인지 헤더
            # 문제인지 가리는 데 시간이 걸린다.
            want = int(np.prod(shape))
            if len(pixels) != want:
                raise ValueError(
                    f"픽셀 길이가 shape {shape} 와 안 맞습니다: {len(pixels)} != {want}")
            # ⚠️ frombuffer 는 읽기 전용 배열이라 torch.from_numpy 가 요청마다
            # 경고를 뱉는다. 여기서 한 번 복사해 둔다.
            image = np.frombuffer(pixels, dtype=np.uint8).reshape(shape).copy()

            with LOCK:
                started = time.monotonic()
                chunk = RUNNER.predict_chunk(image, meta["state"], meta["task"])
                elapsed = (time.monotonic() - started) * 1000.0
                STATS["calls"] += 1
                STATS["total_ms"] += elapsed
                calls = STATS["calls"]
            print(f"  [{calls:4d}] {meta['task']:24s} {shape[1]}x{shape[0]} "
                  f"-> {chunk.shape}  {elapsed:.0f} ms", flush=True)

            chunk = np.ascontiguousarray(chunk, dtype=np.float32)
            self._send(200, {"shape": list(chunk.shape), "ms": round(elapsed, 1)},
                       chunk.tobytes())
        except Exception as e:  # noqa: BLE001 — 서버는 죽지 않아야 한다
            print(f"  !! 오류: {type(e).__name__}: {e}", flush=True)
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main() -> None:
    global RUNNER
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--device", default="cpu")
    # ── 확산 정책 전용 ─────────────────────────────────────────────────
    # ACT 체크포인트에 주면 아무 일도 안 일어난다(그쪽 config 에 없는 필드다).
    ap.add_argument("--denoise", type=int, default=None,
                    help="DP denoising 스텝. 기본은 학습값(100). 줄이면 그만큼 "
                         "선형으로 빨라지고 궤적이 거칠어진다")
    ap.add_argument("--scheduler", default=None, choices=("DDPM", "DDIM"),
                    help="DP 노이즈 스케줄러. 스텝을 줄일 거면 DDIM 이 맞다 — "
                         "DDPM 은 100스텝을 다 밟으라고 만든 것이다")
    args = ap.parse_args()

    print(f"정책 적재 중: {args.ckpt}")
    t0 = time.monotonic()
    RUNNER = PolicyRunner(args.ckpt, device=args.device,
                          num_inference_steps=args.denoise,
                          noise_scheduler_type=args.scheduler)
    print(f"준비 {time.monotonic() - t0:.1f}s — 입력 {RUNNER.policy_hw}, "
          f"chunk {RUNNER.chunk_size}, n_action_steps {RUNNER.n_action_steps}, "
          f"n_obs_steps {RUNNER.n_obs_steps}")
    if RUNNER.noise_scheduler_type is not None:
        # 실행 청크가 몇 초인지 같이 찍는다 — 듀티를 사람이 암산 안 하게.
        chunk_s = RUNNER.n_action_steps / 30.0
        print(f"확산 정책 — {RUNNER.noise_scheduler_type} "
              f"{RUNNER.num_inference_steps or '기본(학습값)'} 스텝, "
              f"실행 청크 {chunk_s:.2f}초. 아래 ms 가 이보다 크면 팔이 끊긴다")

    # 워밍업. 첫 호출이 느린 것을 파지 중에 겪지 않게 미리 한 번 돌린다.
    RUNNER.predict_chunk(np.zeros((*RUNNER.policy_hw, 3), np.uint8),
                         [0.0] * 6, "warmup")
    print(f"대기: http://{args.host}:{args.port}  (health / predict)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
