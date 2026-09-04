"""녹화용 카메라의 초점을 고정한다 — 화질이 조용히 반토막 나던 원인.

## 무엇이 있었나

2026-09-01 과 2026-09-02 이후의 녹화 화질이 다르다. 인접 픽셀 차이 평균으로
잰 선명도:

    09-01 17:32  msmf_1280x720.png  (PNG 무압축)   11.35
    09-01 18:07  check_cam0.png     (PNG 무압축)   11.84
    09-01        퀸 녹화 (AV1 압축)                11.42   <- 압축을 거쳐도 유지
    09-02 12:18  cam_check.png      (PNG 무압축)    5.11   <- 무압축인데 이미 절반
    09-02~03     룩·나이트·상자·별·공 녹화     1.96 ~ 3.02

퀸은 AV1 압축을 거치고도 원본 PNG 와 같은 11 을 유지한다. **압축이 원인이면
이럴 수 없다.** 09-02 정오의 무압축 PNG 가 이미 5.11 인 것이 결정적이다 -
카메라가 내보내는 그림 자체가 뭉개져 있었다. 그래서 인코더 qp 를 낮춰도
돌아오지 않는다. 이미 찍은 다섯 데이터셋은 되살릴 수 없다.

## 왜 그렇게 됐나

C920 은 오토포커스가 켜져 있으면 초점이 계속 움직인다. 바닥처럼 평평하고
무늬가 옅은 장면에서는 엉뚱한 거리에 맞기 쉽다.

`aruco/camera_devices.py:153-156` 은 오토포커스를 끄지만 **초점값을 안 건다** -
`config.py` 에 `CAM_FOCUS` 가 없어서 `getattr(cfg, "CAM_FOCUS", None)` 이 None 이다.
오토포커스만 끄면 **그 순간의 초점에 그대로 굳는다.** 맞은 상태에서 끄면
계속 선명하고, 어긋난 상태에서 끄면 계속 흐리다.

그리고 lerobot 의 녹화 경로(`OpenCVCamera.connect`)는 초점을 아예 안 만진다.
해상도·FPS·FOURCC 만 건다(`camera_opencv.py:182`). 그래서 녹화 화질이
"직전에 어떤 프로그램이 카메라를 어떤 상태로 두고 갔느냐"에 좌우된다.

## 무엇을 하나

녹화 시작 때 오토포커스를 끄고 **초점을 정해진 값으로 고정**한다. 값은 이
파일을 직접 실행해 찾는다:

    python grippers/tools/arm/camera_focus.py

초점을 0~255 로 훑으며 선명도를 재고 가장 선명한 값을 알려 준다. 그 값을
녹화에 넘긴다:

    .\rec_piece.ps1 knight 20 -Focus 30

⚠️ 카메라나 거치대를 옮기면 다시 재야 한다. 초점 거리가 바뀐다.
"""
from __future__ import annotations

import sys


def sharpness(bgr) -> float:
    """인접 픽셀 차이의 평균. 초점이 맞을수록 크다.

    라플라시안 분산이 더 흔하지만 노이즈에 민감하다. 이 장면은 저대비
    바닥이 넓어서, 노이즈에 덜 흔들리는 1차 차분 쪽이 안정적이었다.
    """
    import numpy as np

    g = bgr.astype(np.float64).mean(2)
    h, w = g.shape
    # 그리퍼 턱과 바닥이 같이 보이는 중앙부만 본다. 가장자리는 왜곡이 크다.
    r = g[h // 5: h * 4 // 5, w // 5: w * 4 // 5]
    return float(np.abs(np.diff(r, axis=0)).mean() + np.abs(np.diff(r, axis=1)).mean())


def patch_camera_focus(focus: int | None, sharpen: int | None = None) -> bool:
    """lerobot 이 카메라를 연 뒤 오토포커스를 끄고 초점을 고정한다.

    `dshow_patch` 와 같은 함수를 감싼다. **그 뒤에** 적용해야 FOURCC 재설정
    다음에 초점이 걸린다 - 순서가 바뀌면 협상 과정에서 되돌아갈 수 있다.
    """
    if focus is None:
        return False

    import cv2
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    if getattr(OpenCVCamera, "_focus_patched", False):
        return False
    original = OpenCVCamera._configure_capture_settings

    def configure(self):
        result = original(self)
        cap = self.videocapture
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, float(focus))
        if sharpen is not None:
            cap.set(cv2.CAP_PROP_SHARPNESS, float(sharpen))
        got = cap.get(cv2.CAP_PROP_FOCUS)
        af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
        if abs(got - focus) > 2 or af not in (0, -1):
            raise RuntimeError(
                f"{self}: 초점 고정 실패 - 요청 {focus}, 실제 {got}, 오토포커스 {af}. "
                "이대로 찍으면 초점이 움직여 화질이 회차마다 달라집니다."
            )
        return result

    OpenCVCamera._configure_capture_settings = configure
    OpenCVCamera._focus_patched = True
    return True


def sweep(index: int = 0, width: int = 1280, height: int = 720,
          settle: int = 8, step: int = 5) -> int:
    """초점을 훑으며 선명도를 재고 가장 선명한 값을 돌려준다."""
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"카메라 {index} 를 열 수 없습니다. 다른 프로그램이 잡고 있지 않은지 확인하세요.")
    # 녹화와 같은 순서로 건다 (dshow_patch 참고: FOURCC 가 마지막)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"카메라 {index}  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  "
          f"{cap.get(cv2.CAP_PROP_FPS):.0f}fps")

    # 먼저 지금 상태 - 아무것도 안 건드렸을 때가 녹화가 받던 그림이다
    for _ in range(settle):
        cap.read()
    ok, f = cap.read()
    base = sharpness(f) if ok else 0.0
    print(f"손대지 않은 현재 상태: 선명도 {base:.2f}  "
          f"(오토포커스 {cap.get(cv2.CAP_PROP_AUTOFOCUS):.0f}, 초점 {cap.get(cv2.CAP_PROP_FOCUS):.0f})")
    print("  참고: 9/1 정상 촬영이 11 대, 9/2 이후 망가진 녹화가 2~3 대였습니다.")
    print()

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    best, rows = (None, -1.0), []
    print("초점  선명도")
    for v in range(0, 256, step):
        cap.set(cv2.CAP_PROP_FOCUS, float(v))
        for _ in range(settle):
            cap.read()
        ok, f = cap.read()
        if not ok:
            continue
        s = sharpness(f)
        rows.append((v, s))
        bar = "#" * int(s * 4)
        mark = ""
        if s > best[1]:
            best = (v, s); mark = "  <-"
        print(f"{v:4d}  {s:6.2f}  {bar}{mark}")
    cap.release()

    print()
    print(f"가장 선명한 초점: {best[0]}  (선명도 {best[1]:.2f}, 현재 상태 대비 {best[1]/max(base,1e-9):.1f}배)")
    if best[1] < 8:
        print("⚠️ 최고값이 8 미만입니다. 초점만의 문제가 아닐 수 있습니다 -")
        print("   렌즈 오염, 조명, 카메라 자체 설정(sharpness/exposure)도 확인하세요.")
    print()
    print(f"녹화할 때:  .\rec_piece.ps1 <물건> <회차> -Focus {best[0]}")
    return best[0]


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sweep(idx)
