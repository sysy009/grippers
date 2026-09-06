"""VLA 정책 추론 코어 — ROS 의존성 없음.

ROS 노드가 이걸 감싸기만 한다. 분리한 이유는 **노트북에서 그대로 시험되기
때문**이다. Pi 에 torch 가 올라가기 전에 전처리·단위·출력을 다 확인할 수 있고,
Pi 에서 문제가 나면 "ROS 배선이냐 정책이냐"를 바로 가를 수 있다.

기준 구현은 `tools/arm/rollout_policy.py` 다 — 실기에서 실제로 물체를 집은
경로라, 여기서 그 동작을 바꾸면 안 된다.

⚠️ 세 가지 함정이 실기에서 확인돼 있다. 전부 지키고 있다.

1. **device 를 덮어야 한다.** 학습 때 device(xpu)가 전처리기에 박혀 있어서,
   안 덮으면 `mat1 is on xpu:0` 로 죽는다.

2. **wrist_roll 은 학습 평균으로 덮는다.** 이 관절의 캘리브레이션 범위가
   14틱(2040~2054)뿐이라 1틱만 어긋나도 정규화값이 크게 뛴다. 2026-09-02 실측:
   같은 이미지에 이 값 하나만 바꾸면 정책이 **프레임과 무관하게 동일한 액션**을
   뱉는다 — 트랜스포머가 포화돼 이미지를 아예 무시한다. 학습 평균(0.067)으로
   덮으면 정상 출력과 0.05도 이내로 복구된다.
   실기 데이터에서 이 관절의 action 은 전 프레임 정확히 0 이라 잃는 것이 없다.

3. **DP 는 관측을 n_obs_steps 개 쌓아 받는다.** ACT 는 관측 하나를 받지만
   Diffusion Policy 는 (B, 2, ...) 를 기대한다. 하나만 넣으면 lerobot 의
   `predict_action_chunk` 가 이미지에만 `unsqueeze(1)` 을 걸고 관절값은 그냥
   두어, 두 입력의 축이 어긋난 채로 들어간다. 그리고 `DiffusionConfig` 에는
   **`chunk_size` 가 아예 없다** — ACT 만 있는 필드라 그대로 읽으면 적재
   단계에서 죽는다. 둘 다 아래에서 정책 종류를 안 보고 처리한다.

4. **n_action_steps 를 함부로 줄이지 말 것.** ACT 는 시간을 안 본다. 시작
   자세에서는 관측이 몇 초간 거의 같아서 "언제 펴는가"를 관측으로 정할 수 없고,
   그 시간 정보가 **청크 안에** 들어 있다. v5 의 첫 청크(100스텝)는 "가만히
   있다가 50스텝쯤부터 그리퍼를 연다"이다. 2026-09-02 실측: 30 으로 줄였더니
   그리퍼가 열리기 직전에 잘렸고, 같은 관측으로 다시 예측하니 같은 청크가 나와
   0~29 를 무한 반복했다. 팔이 25초 동안 한 번도 안 움직였다.
"""
from __future__ import annotations

import numpy as np

#: 학습 데이터의 관절 순서. arm servo 1..6 과 같다.
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
#: wrist_roll 을 덮을 값(학습 평균). 위 함정 2 참고.
WRIST_ROLL_FREEZE = 0.067


class PolicyRunner:
    """체크포인트 하나를 들고 있다가 (이미지, 관절값, 지시문) -> 액션 청크."""

    @staticmethod
    def _resize_from_train_config(ckpt: str):
        import json
        from pathlib import Path
        try:
            tf = json.loads((Path(ckpt) / "train_config.json").read_text(encoding="utf-8"))
            tf = tf["dataset"]["image_transforms"]
        except Exception:
            return None
        if not tf.get("enable") or "resize" not in tf.get("tfs", {}):
            return None
        size = tf["tfs"]["resize"]["kwargs"]["size"]
        return (int(size[0]), int(size[1]))

    def __init__(self, ckpt: str, device: str = "cpu",
                 freeze_wrist_roll: float | None = WRIST_ROLL_FREEZE,
                 n_action_steps: int | None = None,
                 policy_hw: tuple | None = None,
                 num_inference_steps: int | None = None,
                 noise_scheduler_type: str | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self._torch = torch
        self.ckpt = ckpt
        self.device = device
        self.freeze_wrist_roll = freeze_wrist_roll

        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.device = device
        if n_action_steps is not None:
            # 위 함정 4. 줄이려면 실기로 확인하고 줄일 것.
            cfg.n_action_steps = n_action_steps
        # ── 확산 정책 전용 손잡이 ─────────────────────────────────────────
        # 둘 다 ACT 에는 없는 필드라, 주지 않으면 아무 일도 안 일어난다.
        #
        # DP 는 U-Net 을 num_inference_steps 번 **순차** 통과한다. 그것이
        # 추론 시간의 거의 전부라, 이 숫자가 곧 듀티다. 기본값은 학습 때의
        # num_train_timesteps(100)이고, 줄이면 그만큼 선형으로 빨라지는 대신
        # 궤적이 거칠어진다.
        #
        # ⚠️ **DDIM 으로 바꾸지 말 것.** 일반론은 "스텝을 줄일 거면 DDIM"
        # 이지만 이 모델에서는 반대로 측정됐다(2026-09-06, RTX 3050).
        # 같은 스텝 수에서 시간은 똑같고(둘 다 813ms) 원본 궤적에서 벗어난
        # 정도는 DDIM 이 두 배가 넘는다 — DDPM 10 이 DDIM 25 보다 빠르면서
        # 더 정확해서, DDIM 이 이기는 지점이 아예 없다.
        #
        #     기준선 0.175  = DDPM 100 을 두 무리로 갈라 평균끼리 비교한 값.
        #                     "같은 설정으로도 이만큼은 벌어진다"는 바닥이다.
        #     DDPM 50  0.106 (0.6배)   DDIM 25  0.708 (4.1배)
        #     DDPM 25  0.328 (1.9배)   DDIM 10  0.821 (4.7배)
        #     DDPM 10  0.585 (3.3배)
        #
        # 이유는 이 모델의 학습 체인이 100스텝밖에 안 되기 때문으로 본다
        # (num_train_timesteps=100). DDIM 은 보통 1000스텝짜리 긴 체인을
        # 건너뛰라고 만든 것인데, 이미 100 이면 건너뛸 것이 별로 없고 DDIM
        # 자신의 근사 오차만 얹힌다.
        #
        # 그래서 이 인자는 "DDIM 이 왜 나쁜지 다시 재 보라"고 남겨 둔 것이다.
        if noise_scheduler_type is not None:
            cfg.noise_scheduler_type = noise_scheduler_type
        if num_inference_steps is not None:
            cfg.num_inference_steps = int(num_inference_steps)
        self.cfg = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
        self.policy.to(device).eval()
        # 위 함정 1.
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        # ⚠️⚠️ 입력 해상도는 **config.json 이 아니라 train_config.json** 에 있다.
        #
        # config.json 의 `input_features` 는 데이터셋 원본 크기(3, 720, 1280)를
        # 그대로 적어 둔다. 실제로 정책이 본 것은 데이터셋 변환이 줄인 180x320 이다.
        # config.json 을 믿고 720x1280 을 넣으면 **정책이 한 번도 본 적 없는
        # 해상도**가 들어가는데, 오류가 안 나서 조용히 틀린다.
        # (rollout_policy.py 도 같은 이유로 train_config.json 을 읽는다.)
        self.policy_hw = policy_hw or self._resize_from_train_config(ckpt)
        if self.policy_hw is None:
            raise RuntimeError(
                f"{ckpt}/train_config.json 에서 학습 리사이즈를 찾지 못했습니다. "
                "policy_hw 로 직접 주십시오 — 원본 크기로 넣으면 조용히 틀립니다."
            )
        # ⚠️ `chunk_size` 는 ACT 에만 있다. DP 의 같은 자리는 `horizon` 이고,
        # 둘 다 없는 정책이 오면 n_action_steps 로 물러선다. 이 값은 원격
        # 클라이언트가 health() 로 받아 가는 것이라 없으면 안 된다.
        self.chunk_size = int(
            getattr(cfg, "chunk_size", None)
            or getattr(cfg, "horizon", None)
            or cfg.n_action_steps
        )
        self.n_action_steps = int(cfg.n_action_steps)
        # ACT 는 1, DP 는 2. 위 함정 3.
        self.n_obs_steps = int(getattr(cfg, "n_obs_steps", 1) or 1)
        self.num_inference_steps = getattr(cfg, "num_inference_steps", None)
        self.noise_scheduler_type = getattr(cfg, "noise_scheduler_type", None)

    def _batch(self, image_bgr: np.ndarray, state6):
        torch = self._torch
        state = torch.tensor([float(v) for v in state6], dtype=torch.float32)
        if self.freeze_wrist_roll is not None:
            state[4] = self.freeze_wrist_roll
        # ⚠️ BGR 그대로 넣는다. 학습 때 lerobot 이 OpenCV 프레임을 그대로 썼고,
        # 여기서 RGB 로 바꾸면 학습과 다른 색을 보게 된다.
        img = torch.from_numpy(np.ascontiguousarray(image_bgr)).permute(2, 0, 1).float() / 255.0
        if (img.shape[1], img.shape[2]) != self.policy_hw:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=self.policy_hw, mode="bilinear",
                align_corners=False).squeeze(0)
        return state, img

    def predict_chunk(self, image_bgr: np.ndarray, state6, task: str) -> np.ndarray:
        """액션 청크 [n_action_steps, 6]. 단위는 LeRobot 정규화 단위다.

        `select_action` 이 아니라 `predict_action_chunk` 를 쓴다 — 큐에서 한
        스텝씩 빼는 게 아니라 청크를 통째로 받아 arm_driver 에 넘기기 때문이다
        (ExecuteJointChunk).
        """
        torch = self._torch
        state, img = self._batch(image_bgr, state6)
        if self.n_obs_steps > 1:
            # ⚠️ 관측 하나를 n_obs_steps 번 **복제**한다. 위 함정 3.
            #
            # 왜 이전 프레임을 기억해 쓰지 않는가 — 이 서버는 청크마다 한 번
            # 불리므로 "직전 관측"이 1.07초 전이다. 학습 때 짝지어진 두 관측은
            # 33ms 간격(연속 프레임)이라, 1.07초짜리 짝은 정책이 한 번도 본 적
            # 없는 입력이다. 그것보다는 복제가 낫다 — lerobot 자신도 회차
            # 시작에서 큐를 채울 때 첫 관측을 그대로 복제한다
            # (modeling_diffusion.select_action docstring: "for the first steps,
            # the observation is copied n_obs_steps times").
            #
            # 즉 복제는 **매 청크를 회차 시작처럼 보이게** 하는 것이고, 그
            # 조건은 정책이 실제로 겪어 본 조건이다. 대신 두 관측의 차이가 0
            # 이라 속도 정보가 없다 — 정말 필요해지면 Pi 가 연속 두 프레임을
            # 보내도록 프로토콜을 늘리는 것이 정답이고, 그때는 이 복제가
            # 부족한 관측 수를 채우는 자리로 물러난다.
            states = torch.stack([state] * self.n_obs_steps, dim=0).unsqueeze(0)
            images = torch.stack([img] * self.n_obs_steps, dim=0).unsqueeze(0)
        else:
            states = state.unsqueeze(0)
            images = img.unsqueeze(0)
        batch = {
            "observation.state": states,
            "observation.images.gripper": images,
            "task": [task],
        }
        with torch.no_grad():
            # ⚠️ 큐가 비어 있어야 lerobot 이 위 배치를 그대로 쓴다. 큐가 차
            # 있으면 그쪽을 쌓아 쓰는데(modeling_diffusion.predict_action_chunk),
            # 이 경로는 select_action 을 안 부르므로 원래 비어 있다. 그래도
            # 매번 비우는 편이 싸고, 앞선 호출이 남긴 것이 섞이지 않는다.
            self.policy.reset()
            chunk = self.policy.predict_action_chunk(self.pre(batch))
            # ⚠️ post 는 dict 가 아니라 액션 텐서를 받는다(rollout_policy.py 기준).
            # 청크는 [B, T, D] 인데 정규화는 마지막 축에만 걸리므로 시간축을
            # 배치로 접어 넣었다가 되돌린다 — 접지 않고 넣으면 조용히 다른 축에
            # 걸릴 수 있다.
            b, t, d = chunk.shape
            out = self.post(chunk.reshape(b * t, d)).reshape(b, t, d)[0]
        return out.float().cpu().numpy()[: self.n_action_steps]
