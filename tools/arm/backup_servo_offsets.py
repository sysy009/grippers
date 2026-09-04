"""캘리브레이션 전에 서보 EEPROM 값을 백업한다.

## 왜 필요한가

LeRobot 캘리브레이션은 파일만 쓰는 게 아니라 **서보 안의 `Homing_Offset` 을
직접 덮어쓴다**(`lerobot/motors/feetech/feetech.py:275`).

    Present_Position = Actual_Position - Homing_Offset

그런데 이 팔에는 그리퍼 프로젝트의 교시 자세가 **RAW 서보값**으로 박혀 있다
(`floor_grasp_profiles.py` 의 HORIZONTAL_SAFE_145_RAW · IDLE_CRADLE_RAW ·
CARRY_RAW 등). Homing_Offset 이 바뀌면 **같은 RAW 값이 다른 물리 자세**가 되어
실기로 얻은 파지 자세가 전부 어긋난다.

이 스크립트는 그 값들을 JSON 으로 남겨, 나중에 그리퍼 미션을 되살릴 수 있게 한다.

    python backup_servo_offsets.py COM8            # 백업
    python backup_servo_offsets.py COM8 --restore <파일>   # 복구

복구는 `Homing_Offset` 과 **위치 한계(Min/Max_Position_Limit)** 를 되돌린다.
속도 상한은 `--with-velocity` 를 줘야 한다.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# SO-101 의 관절 구성. so_follower.py 와 같은 순서·모델.
MOTORS = {
    "shoulder_pan": (1, "sts3215"),
    "shoulder_lift": (2, "sts3215"),
    "elbow_flex": (3, "sts3215"),
    "wrist_flex": (4, "sts3215"),
    "wrist_roll": (5, "sts3215"),
    "gripper": (6, "sts3215"),
}

# 백업할 레지스터. Homing_Offset 이 핵심이고 나머지는 참고용이다.
# 아래 둘째 묶음은 **lerobot 이 절대 덮어쓰지 않는** 값들이다. connect() ->
# configure() 가 P/I/D 와 Acceleration 은 매번 6축에 같은 값으로 덮어쓰지만,
# Maximum_Velocity_Limit 과 Torque_Limit 은 손대지 않는다. 그래서 EEPROM 에
# 남은 값이 그대로 살아 있고, 팔을 바꾸거나 초기화하면 조용히 되돌아간다.
#
# 2026-09-01: 팔로워 6축이 Maximum_Velocity_Limit=65 로 묶여 있어 리더를 크게
# 움직여도 굼뜨게 따라오는 증상이 있었다(리더는 250). 그때 이 스크립트가 이
# 값을 안 떠서 "원래 65였는가"를 확인할 방법이 없었다. 그래서 추가한다.
FIELDS = ["Homing_Offset", "Min_Position_Limit", "Max_Position_Limit",
          "Present_Position",
          # lerobot 이 안 건드리는 값 — 이력이 없으면 원인 추적이 불가능하다
          "Maximum_Velocity_Limit", "Torque_Limit", "Max_Torque_Limit",
          # lerobot 이 매 connect 마다 덮어쓰는 값 — 비교용 참고치
          "Acceleration", "Maximum_Acceleration", "P_Coefficient",
          "Present_Temperature"]

OUT_DIR = Path(__file__).parent / "servo_backup"


def make_bus(port: str) -> FeetechMotorsBus:
    return FeetechMotorsBus(
        port=port,
        motors={n: Motor(i, m, MotorNormMode.RANGE_M100_100)
                for n, (i, m) in MOTORS.items()},
    )


def backup(port: str) -> int:
    bus = make_bus(port)
    bus.connect(handshake=False)
    data = {"port": port, "when": datetime.now().isoformat(timespec="seconds"),
            "motors": {}}
    for name in MOTORS:
        row = {}
        for f in FIELDS:
            try:
                row[f] = int(bus.read(f, name, normalize=False))
            except Exception as e:
                row[f] = f"읽기 실패: {type(e).__name__}"
        data["motors"][name] = row
        print(f"  {name:14s} {row}")
    bus.disconnect()

    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"servo_{port}_{stamp}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {path}")
    return 0


def restore(port: str, src: str) -> int:
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    bus = make_bus(port)
    bus.connect(handshake=False)
    for name, row in data["motors"].items():
        v = row.get("Homing_Offset")
        if isinstance(v, int):
            bus.write("Homing_Offset", name, v, normalize=False)
            print(f"  {name:14s} Homing_Offset <- {v}")
        # 위치 한계도 되돌린다. 예전에는 Homing_Offset 만 복구했는데, 그러면
        # **오프셋은 맞는데 관절이 안 움직이는** 상태가 된다 - 2026-09-03 에
        # 그리퍼가 정책 단위 27 에서 막혀 롤아웃이 통째로 헛돌았다. 원인은
        # 남의 캘리브레이션이 남긴 Max_Position_Limit=2090 이었고, 오프셋만
        # 복구해서는 그게 안 지워졌다.
        #
        # wrist_roll 은 더 위험하다. 이 팔은 그 관절을 2040~2054(14카운트)로
        # 좁혀 잠가 두는데(record_retry.py 참고), 잠금이 풀리면 정규화가
        # 14카운트 기준이라 raw 가 조금만 움직여도 정규화값이 튄다.
        for reg in ("Min_Position_Limit", "Max_Position_Limit"):
            lim = row.get(reg)
            if isinstance(lim, int):
                bus.write(reg, name, lim, normalize=False)
                print(f"  {name:14s} {reg} <- {lim}")
        if "--with-velocity" in sys.argv:
            mv = row.get("Maximum_Velocity_Limit")
            if isinstance(mv, int):
                bus.write("Maximum_Velocity_Limit", name, mv, normalize=False)
                print(f"  {name:14s} Maximum_Velocity_Limit <- {mv}")
    bus.disconnect()
    print("\n복구 완료. 그리퍼 미션의 교시 자세를 실제로 확인할 것.")
    if "--with-velocity" not in sys.argv:
        print("(속도 상한은 복구하지 않았다. 필요하면 --with-velocity)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    p = sys.argv[1]
    if "--restore" in sys.argv:
        sys.exit(restore(p, sys.argv[sys.argv.index("--restore") + 1]))
    sys.exit(backup(p))
