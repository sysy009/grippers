"""재시도를 켜고 시연을 녹화한다 — `lerobot-record` 를 감싼 것.

## 왜 필요한가

`calibrate_retry.py` 와 **같은 이유**다. LeRobot 의 모터 통신은 `num_retry`
기본값이 0 이라 **한 번 실패하면 그 자리에서 죽는다.** 캘리브레이션에서
두 번 그렇게 죽었고, 실측해 보니 통신 자체는 멀쩡했다 — 팔을 움직이는 중
70,260회 읽기에 실패 0, 전압 11.8~11.9V 안정.

캘리브레이션은 죽어도 다시 하면 그만이지만, **녹화는 다르다.** 25분짜리
30 에피소드를 찍는 도중에 죽으면 그때까지가 날아간다(`--resume=true` 로
이어붙일 수는 있지만 그 에피소드는 버려진다). 그래서 처음부터 감싼다.

## wrist_roll 을 상수로 못 박는다

5번 관절(wrist_roll)은 이 과제에서 쓰지 않는다. v3 데이터셋에서도 실제 범위가
-15.16 ~ 8.31 로 사실상 고정이었다. 그런데 교시 중 손이 스치면 값이 흔들리고,
그 흔들림이 학습 데이터에 그대로 들어간다.

팔로워 쪽은 서보 가동범위를 좁혀 막아 두었다(`grippers_arm.json` 의 wrist_roll
range 2040~2054, raw 2047 부근 ±7틱). **그것만으로는 부족하다** —
`lerobot_record.py:399-411` 이 데이터셋에 기록하는 `action` 은 실제로 전송된
값이 아니라 **리더가 읽힌 값**이기 때문이다(코드에도 TODO 로 남아 있다).

    action_values  = act_processed_teleop      # 리더가 읽힌 값 → 기록됨
    _sent_action   = robot.send_action(...)    # 실제 전송값 → 버려짐

그대로 두면 `observation.state` 의 wrist_roll 은 고정인데 `action` 만 흔들리는
데이터가 된다. 없애려던 잡음이 남는 데다 **상태-액션 불일치**까지 더해진다.
정책은 "wrist_roll 을 명령해도 아무 일도 안 일어난다"를 배운다.

그래서 리더가 읽는 값 자체를 여기서 상수로 덮는다. 기록·전송·실제 동작 셋이
같아진다.

⚠️ 값을 바꾸려면 `WRIST_ROLL_FREEZE_DEG` 하나만 고칠 것. 서보 가동범위와
   **같은 자세**를 가리켜야 한다 — 어긋나면 리더가 팔로워가 갈 수 없는 곳을
   명령하고, 그 명령이 데이터에 남는다.

## 쓰는 법

`lerobot-record` 의 인자를 **그대로** 준다.

    python tools/arm/record_retry.py \
      --robot.type=so101_follower --robot.port=COM8 --robot.id=grippers_arm \
      --dataset.repo_id=lsy0284/gripper_pick_v1 \
      --dataset.single_task="pick up the queen" ...

⚠️ 대화형이다. **진짜 터미널 창**에서 실행할 것.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dshow_patch import patch_dshow_property_order
from rerun_downscale import MAX_EDGE, patch_rerun_downscale
from batch_encode_patch import patch_batch_encode
from camera_focus import patch_camera_focus
from feetech_retry import RETRY, patch_bus

#: 리더가 읽힌 wrist_roll 을 이 값으로 덮는다. 팔로워 서보 가동범위
#: (raw 2040~2054, 정중앙 2047)가 가리키는 자세와 같은 0도.
#: None 으로 두면 덮지 않는다(관절을 다시 쓰게 되면 그렇게 바꾼다).
WRIST_ROLL_FREEZE_DEG: float | None = 0.0


def _patch_leader_wrist_roll() -> bool:
    """리더가 읽힌 wrist_roll 을 상수로 덮는다.

    `SOLeader.get_action()` 은 `{"<motor>.pos": val}` 를 돌려주고, record 루프가
    그 값을 **그대로 데이터셋에 기록**한다. 그러니 여기서 덮으면 기록·전송이
    한꺼번에 상수가 된다. 팔로워 서보 가동범위 제한과 짝이다 — 위 docstring 참고.
    """
    if WRIST_ROLL_FREEZE_DEG is None:
        return False

    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    original = SOLeader.get_action

    def wrapper(self):
        action = original(self)
        if "wrist_roll.pos" in action:
            action["wrist_roll.pos"] = WRIST_ROLL_FREEZE_DEG
        return action

    SOLeader.get_action = wrapper
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    patch_bus()
    print(f"모터 통신을 재시도 {RETRY}회로 감쌌습니다.")
    if patch_dshow_property_order():
        print("카메라: 해상도를 FOURCC 보다 먼저 적용합니다 (DSHOW MJPG 협상용).")
    if patch_rerun_downscale():
        print(f"rerun 이미지를 긴 변 {MAX_EDGE}px 로 줄여 보냅니다 (루프 예산 보호).")
    if patch_batch_encode():
        print("배치 영상 인코딩을 동작하는 구현으로 교체했습니다.")
    # 초점은 GRIP_CAM_FOCUS 환경변수로 받는다. rec_piece.ps1 의 -Focus 가 이걸 채운다.
    # 값이 없으면 아무것도 하지 않는다 - 카메라 상태를 조용히 바꾸지 않기 위해서다.
    _focus = os.environ.get("GRIP_CAM_FOCUS")
    if _focus and patch_camera_focus(int(_focus)):
        print(f"카메라 초점을 {_focus} 로 고정합니다 (오토포커스 끔).")
    elif not _focus:
        print("경고: 초점을 고정하지 않습니다. 화질이 직전 프로그램이 남긴 상태에 좌우됩니다.")
        print("      camera_focus.py 로 값을 찾아 -Focus 로 넘기세요.")
    if _patch_leader_wrist_roll():
        print(f"wrist_roll 을 {WRIST_ROLL_FREEZE_DEG}도로 고정합니다 (리더 입력을 덮어씀).")
    else:
        print("wrist_roll 고정: 꺼짐 (WRIST_ROLL_FREEZE_DEG=None)")
    print()

    # draccus 가 sys.argv 를 그대로 읽으므로 우리 인자를 그 자리에 둔다.
    from lerobot.scripts.lerobot_record import main as record_main
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    record_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
