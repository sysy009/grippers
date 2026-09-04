"""SmolVLA 체크포인트 점검 - 로드 / 속도 / 언어 반응.

팔을 **전혀 움직이지 않는다.** 정책을 올리고, forward 시간을 재고, 같은 프레임에
지시문만 바꿔 예측이 얼마나 달라지는지 본다.

사용법
    python grippers\\tools\\arm\\smolvla_check.py --ckpt ckpt_smolvla_18k\\pretrained_model
    python grippers\\tools\\arm\\smolvla_check.py --ckpt ... --image frame.png
    python grippers\\tools\\arm\\smolvla_check.py --ckpt ... --camera 0

기본은 카메라 0 에서 한 장 찍는다. --image 를 주면 그 파일을 쓴다.

■ 왜 이 도구가 따로 있는가

rollout_policy.py 는 정책 종류를 체크포인트에서 읽으므로 SmolVLA 도 그대로 돌긴
한다. 다만 이 노트북 CPU 에서는 forward 가 청크 수명보다 오래 걸려서
**실시간 제어가 성립하지 않는다**(아래 듀티 참고). 팔을 걸지 않고 예측값만
비교하는 쪽이 언어 반응을 격리해서 보기에도 더 낫다.

■ 판정 기준

지시문을 바꿨을 때의 변화를 **액션 표준편차와 견줘야** 의미가 생긴다. 그냥
"0 이 아니다"는 아무것도 말해주지 않는다. 학습 데이터의 액션 std 는 약 55.5 이고,
2026-09-04 리눅스 세션이 홀드아웃으로 잰 문장 효과는 그 6.3% 였다 - 반응은
있지만 궤적 선택을 뒤집을 크기는 아니라는 뜻이다.
"""

import argparse
import sys
import time

import numpy as np
import torch

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TASKS = [
    "pick up the queen", "pick up the rook", "pick up the knight",
    "pick up the box", "pick up the star", "pick up the ball",
]
# v5_all 학습 데이터의 액션 표준편차. 변화량을 이 값과 견준다.
ACTION_STD = 55.47


# Windows 한글 콘솔은 cp949 라 em dash 같은 문자를 못 쓴다. 출력 한 줄 때문에
# 측정이 통째로 날아가지 않게 대체 문자로 흘린다(2026-09-04 실측: 3815ms 재고,
# 210s 로드를 다시 해야 했다).
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def strip_unknown_config_fields(ckpt: str) -> None:
    """이 lerobot 버전이 모르는 config 필드를 걷어낸다 (원본은 .orig 로 남긴다).

    학습 쪽(0.6.x)이 저장한 config.json 에는 0.4.4 에 없는 필드가 섞여 온다.
    하나라도 남아 있으면 draccus 가 설정을 통째로 거부한다:
        DecodingError: The fields `pretrained_revision` are not valid for SmolVLAConfig
    rollout_policy.py 의 _strip_unknown_config_fields 와 같은 처리다 - 두 도구로
    같은 체크포인트를 돌리므로 동작이 어긋나면 안 된다.
    """
    import dataclasses
    import json
    import os

    from lerobot.configs.policies import PreTrainedConfig

    path = os.path.join(ckpt, "config.json")
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    cls = PreTrainedConfig.get_choice_class(cfg["type"])
    known = {f.name for f in dataclasses.fields(cls)} | {"type"}
    extra = [k for k in cfg if k not in known]
    if not extra:
        return
    backup = path + ".orig"
    if not os.path.exists(backup):
        os.replace(path, backup)
        with open(backup, encoding="utf-8") as handle:
            cfg = json.load(handle)
    for k in extra:
        cfg.pop(k)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=4, ensure_ascii=False)
    print(f"이 버전이 모르는 config 필드 제거: {extra}  (원본 {os.path.basename(backup)})")


def load_frame(args):
    import cv2
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"이미지를 읽지 못했습니다: {args.image}")
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise SystemExit(f"카메라 {args.camera} 를 열지 못했습니다")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.res[1])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.res[0])
        for _ in range(10):
            ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("프레임을 못 받았습니다")
        # 수집 때와 같은 180도 회전. 이걸 빼면 정책이 처음 보는 화면이 된다.
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        cv2.imwrite("smolvla_check_frame.png", bgr)
        print("촬영 프레임 저장: smolvla_check_frame.png")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"프레임 {rgb.shape[1]}x{rgb.shape[0]}")
    # ⚠️ 리사이즈하지 않는다. SmolVLA 가 내부에서 512x512 로 패딩한다
    # (config.resize_imgs_with_padding). 여기서 미리 줄이면 학습과 달라진다.
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained_model 디렉터리")
    ap.add_argument("--image", default=None, help="쓸 이미지 파일 (없으면 카메라)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--res", nargs=2, type=int, default=(720, 1280), metavar=("H", "W"))
    ap.add_argument("--state", nargs=6, type=float, default=None,
                    help="관절값 6개(도). 없으면 데이터셋 평균 부근의 대기 자세를 쓴다")
    ap.add_argument("--repeat", type=int, default=3, help="속도 측정 반복")
    # 대조군. 문장 효과는 **프레임 효과와 견줘야** 의미가 생긴다 — 같은 정책에서
    # "화면이 바뀌면 얼마나 달라지는가"를 모르면 문장 효과가 큰지 작은지 말할 수 없다.
    ap.add_argument("--image2", default=None, help="대조군 프레임 (다른 회차)")
    ap.add_argument("--state2", nargs=6, type=float, default=None, help="대조군 관절값")
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    strip_unknown_config_fields(args.ckpt)
    t0 = time.perf_counter()
    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.device = "cpu"
    policy = get_policy_class(cfg.type).from_pretrained(args.ckpt, config=cfg)
    policy.to("cpu").eval()
    # 학습 때 device(cuda)가 전처리기에 박혀 있다. 안 덮으면 텐서가 cuda 로 가서 죽는다.
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    n_par = sum(p.numel() for p in policy.parameters())
    print(f"\n정책 {cfg.type}  파라미터 {n_par/1e6:.0f}M  로드 {time.perf_counter()-t0:.1f}s")
    print(f"chunk_size {cfg.chunk_size}  n_action_steps {cfg.n_action_steps}  "
          f"num_steps {getattr(cfg,'num_steps','-')}")

    img = load_frame(args)
    state = torch.tensor(args.state if args.state else
                         [-4.0, -60.0, 40.0, 20.0, 0.0, 50.0], dtype=torch.float32)
    print(f"관절값 {[round(float(v),1) for v in state]}")

    def predict(task, im=None, st=None):
        # 큐를 비워야 매번 진짜 forward 가 돈다 - 안 그러면 캐시된 청크가 나온다.
        for attr in ("_action_queue", "_queues"):
            q = getattr(policy, attr, None)
            if hasattr(q, "clear"):
                q.clear()
        policy.reset()
        with torch.no_grad():
            return post(policy.select_action(pre({
                "observation.state": (state if st is None else st).unsqueeze(0),
                "observation.images.gripper": (img if im is None else im).unsqueeze(0),
                "task": [task]}))).squeeze(0).float()

    print("\n-- 속도 --")
    times = []
    for i in range(args.repeat):
        t = time.perf_counter()
        act = predict(TASKS[0])
        dt = (time.perf_counter() - t) * 1000
        times.append(dt)
        print(f"  {i+1}회  {dt:8.0f} ms" + ("   (첫 회는 워밍업 포함)" if i == 0 else ""))
    steady = np.median(times[1:]) if len(times) > 1 else times[0]
    life = cfg.n_action_steps / 30.0
    print(f"\n  정상 상태 {steady:.0f} ms   청크 수명 {life:.2f}s (30fps x {cfg.n_action_steps}스텝)")
    duty = steady / 1000.0 / life * 100
    print(f"  듀티 {duty:.0f}%  ->  " +
          ("실시간 제어 가능" if duty < 80 else "* 실시간 제어 불가 - 정적 비교로만 쓰세요"))

    print("\n-- 언어 반응 (같은 프레임·같은 관절값, 지시문만 교체) --")
    base = predict(TASKS[0])
    print(f"  기준 \"{TASKS[0]}\"")
    print(f"    예측 {[round(float(v),1) for v in base]}")
    diffs = []
    for t in TASKS[1:]:
        a = predict(t)
        d = float(torch.norm(a - base))
        diffs.append(d)
        print(f"  {t:22s} 차이 {d:7.4f}   ({d/ACTION_STD*100:5.2f}% of std)")
    m = float(np.mean(diffs))
    print(f"\n  평균 문장 효과 {m:.4f}  = 액션 std({ACTION_STD})의 {m/ACTION_STD*100:.1f}%")

    if args.image2:
        print("\n-- 대조군: 프레임 효과 (같은 지시문, 화면만 교체) --")
        import copy
        a2 = copy.copy(args)
        a2.image = args.image2
        img2 = load_frame(a2)
        st2 = (torch.tensor(args.state2, dtype=torch.float32)
               if args.state2 else state)
        fd = float(torch.norm(predict(TASKS[0], img2, st2) - base))
        print(f"  프레임 효과 {fd:.4f}  = 액션 std 의 {fd/ACTION_STD*100:.1f}%")
        print(f"  문장/프레임 비율 {m/fd:.3f}")
        verdict = ("문장이 화면을 이길 수 없습니다 - 언어로 물체를 고르지 못합니다"
                   if m / fd < 0.5 else
                   "문장 효과가 프레임에 필적합니다 - 다물체 실기로 볼 값어치가 있습니다")
    else:
        verdict = ("궤적 선택을 뒤집을 크기가 아닙니다 (언어로 물체 선택 불가)"
                   if m / ACTION_STD < 0.25 else
                   "유의미한 크기입니다 - 다물체 실기로 확인해 볼 값어치가 있습니다")
    print("\n  판정: " + verdict)


if __name__ == "__main__":
    main()
