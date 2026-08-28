"""Host PC -> 차량(Pi) 링크. 여기가 두 저장소가 만나는 **유일한 자리**다.

## 무엇이 바뀌었나 (2026-08-27 병합)

예전에는 이 파일이 자기만의 전선 규격(`cmd`/`status`/`robot_x`/`robot_y`/
`robot_yaw_deg`/`target_label`)을 정의했고, 그 규격이 `VEHICLE_LINK_PROTOCOL.md`
와 `PI_BRIDGE_TASK.md` 에 문서로만 적혀 있었다. 그런데 Pi 쪽은 2026-08-26 팀
확정으로 **다른 규격**(`state` + 속도 넷)을 쓰기 시작했고, 두 규격이 같은
포트(5005/5006)를 쓰면서 서로 못 알아듣는 상태였다 — Pi 의 `UdpHostLink._parse()`
는 `state` 가 없는 패킷을 전부 버린다.

이제 전선 규격은 **`domain/ports/baseline_ports.py` 를 직접 import** 한다.
문서 두 벌을 손으로 맞추는 대신 **양쪽이 같은 파일을 읽는다** — 그 파일이
스스로 경고하는 "직렬화 규약이 어긋나는 사고"(BoxColor -> Destination 개명 때
두 번 났다는)를 구조적으로 못 나게 만드는 것이 목적이다.

`baseline_ports.py` 와 `domain/task/motion.py` 는 `abc` · `dataclasses` · `math`
만 import 하는 순수 파이썬이라, ROS2 가 없는 이 Windows Host 에서도 그대로
로드된다.

## Host 내부 어휘는 그대로다

`MissionCommand`("go"/"stop"/"yaw+"/"yaw-" + mission.State 이름)는 **남는다.**
`mission.py` 가 계산해서 내놓는 것, `live_map.py` 가 화면에 찍는 것, `run_sim.py`
가 가상 차량을 굴리는 것이 전부 이 어휘다. 바뀐 것은 **전선에 실릴 때의 모양**
뿐이고, 변환은 `UdpVehicleLink` 안에서만 일어난다.

경계를 여기 하나로 몰아둔 이유: 링크 구현체를 바꾸는 것만으로 Host FSM 전체를
건드리지 않고 규격을 바꿀 수 있어야 하기 때문이다. `ConsoleVehicleLink` 와
`run_sim.SimVehicleLink` 는 이 변경의 영향을 전혀 받지 않는다.

## 역할 분담 — 좌표는 전선에 싣지 않는다

Host 가 물체 좌표 · 차량 좌표와 방향 · 경로 계산 · 차량 제어 명령을 전부
소유하고, Pi 는 그 명령을 실행하고 상태를 보고만 한다. 그래서 `HostCommand`
에는 좌표가 하나도 없다 — 로봇 pose 를 "참고용"으로라도 실어 보내면 Pi 가
그것을 읽기 시작하는 순간 역할 분담이 무너진다(`baseline_ports.py` 참고).

예전 규격이 보내던 `robot_x`/`robot_y`/`robot_yaw_deg`/`target_label` 은 그래서
전선에서 **빠진다.** 라벨도 마찬가지다 — 무엇을 집을지는 내려가는 팔이 자기
카메라로 확인한다(`baseline_mission.py` 의 `_OBJECT_WIDTH_MM`). 디버깅에 필요한
값은 `detail` 문자열로만 흘려보낸다.
"""

from __future__ import annotations

import json
import re
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 레포 루트를 경로에 얹어 domain/ 을 직접 쓴다. host/ 는 grippers 레포의
# 하위 디렉터리이므로 parent 하나면 된다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domain.ports.baseline_ports import HostCommand, MissionState, Report
from domain.task.motion import AGREED_LINEAR_MPS, AGREED_ROTATION_RAD_S


@dataclass
class MissionCommand:
    """Host 내부 표현. **전선 규격이 아니다** — 전선으로 나가는 모양은
    `HostCommand` 이고, 변환은 `UdpVehicleLink.send()` 가 한다.

    `robot_*` 와 `target_label` 은 화면 표시와 로그용으로 남아 있다.
    """

    cmd: str                           # "go" | "stop" | "yaw+" | "yaw-"
    status: str                        # 지금 미션 단계 (mission.State 이름)
    robot_x: float
    robot_y: float
    robot_yaw_deg: float
    target_label: Optional[str] = None
    fresh: bool = True
    t: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# 어휘 대응표 — 이 두 표가 병합의 실체다
# ---------------------------------------------------------------------------

# mission.State 이름 -> MissionState (Pi 가 아는 이름)
#
# SEARCH_TARGET -> IDLE : Host 가 다음 기물을 고르는 동안 Pi 는 할 일이 없다.
#     MissionState 에 SEARCH 를 새로 넣지 않는 이유는, 넣어봤자 Pi 쪽
#     BaselineIdleState 가 하는 일과 똑같기 때문이다 — 상태를 늘리면 양쪽이
#     맞춰야 할 이름만 하나 더 는다.
# FACE_BOX -> CARRY : 아직 물체를 들고 제자리 회전 중이다. `BaselineCarryState`
#     가 CARRY/APPROACH_BOX 를 둘 다 받는다.
# NUDGE_BOX -> APPROACH_BOX : 바구니 앞 미세전진. Pi 쪽 APPROACH_BOX 의 의미와
#     정확히 같다.
# PLACE -> INSERT : 이름만 다르고 같은 동작이다.
_STATE_TO_PI = {
    "SEARCH_TARGET":  MissionState.IDLE,
    "APPROACH_PIECE": MissionState.APPROACH,
    "GRASP":          MissionState.GRASP,
    # GRASP_ALIGN -> APPROACH : Host 가 GRASP_BLOCKED 를 받고 차를 다시 세우는
    #     중이다. 이때 Pi 를 GRASP 로 두면 매 사이클 파지 판정(1.7초짜리)을
    #     다시 돌려서 차가 움직이는 동안 계속 BLOCKED 를 뱉는다. APPROACH 는
    #     Host 속도대로 주행만 하고, 다시 GRASP 가 올 때 한 번만 판정한다 —
    #     "관측 -> 소이동 -> 재관측" 폐루프가 성립하는 것이 이 매핑 덕이다.
    "GRASP_ALIGN":    MissionState.APPROACH,
    "CARRY_TO_DEST":  MissionState.CARRY,
    "FACE_BOX":       MissionState.CARRY,
    "NUDGE_BOX":      MissionState.APPROACH_BOX,
    "PLACE":          MissionState.INSERT,
    # INSERT_ALIGN -> APPROACH_BOX : GRASP_ALIGN 과 정확히 같은 이유다. Pi 를
    #     INSERT 로 둔 채 차를 움직이면 매 사이클 투하 판정을 다시 돌려서
    #     움직이는 내내 BLOCKED 를 뱉고, 그 판정에 쓰인 라이다 값은 어차피
    #     "판독이 흔들린다"로 스스로 무효가 된다. APPROACH_BOX 로 주행만 하고
    #     다시 PLACE 가 갈 때 한 번만 판정하게 한다.
    "INSERT_ALIGN":   MissionState.APPROACH_BOX,
    # HALTED -> CARRY : 물체를 든 채 사람을 기다리는 중이다. INSERT 로 두면
    #     Pi 가 투하를 다시 시도할 수 있고, IDLE 로 두면 팔을 접는다. 둘 다
    #     사람이 오기 전에 일어나면 안 된다. CARRY + stop 이 "들고 서 있기"다.
    "HALTED":         MissionState.CARRY,
    "DONE":           MissionState.DONE,
}

# Pi 가 돌려주는 Report -> mission.py 가 기다리는 옛 문자열
#
# mission.py 는 "GRASP_DONE" 과 "PLACE_DONE" 두 개만 본다. 나머지는 여기서
# 흡수하되 **버리지 않는다** — `last_report` 에 남기고 경고를 찍는다.
# GRASP_BLOCKED / INSERT_BLOCKED 에 실제로 대응하는 로직(수정된 명령을 다시
# 내는 것)은 다음 단계에서 mission.py 에 들어간다. 지금은 그 신호가 오고
# 있다는 사실이 보이게만 해 둔다.
_BLOCKING_REPORTS = {
    Report.GRASP_BLOCKED, Report.GRASP_CENTERING, Report.INSERT_BLOCKED,
}

# 한 번 나오면 mission.py 의 상태 전이를 좌우하는 값들. `poll_status()` 에서
# 다른 보고에 덮이면 안 된다.
_TERMINAL = {"GRASP_DONE", "PLACE_DONE", "FAILED"}


# ---------------------------------------------------------------------------
# GRASP_BLOCKED 보정 요청 — Pi 의 한글 사유를 mission.py 가 쓸 값으로 옮긴다
# ---------------------------------------------------------------------------
#
# Pi 는 "왜 못 내려가는지"를 사람이 읽는 문장으로 보낸다
# (`preconditions.PreconditionReport.detail`, `grasp_alignment.judge` 의 reason).
# 그중 **Host 가 차를 다시 세워서 고칠 수 있는 것은 세 가지**뿐이고, 나머지는
# Host 가 아무리 움직여도 안 풀린다(E-STOP·미실측 상수·그리퍼가 안 비었음).
#
# ⚠️ 문자열 매칭이라 깨지기 쉽다. Pi 쪽 `grasp_alignment.py` 의 문구를 누가
#    고치면 여기가 조용히 실패한다 — Host 가 UNFIXABLE 로 보고 대상을 포기해
#    버린다. **제대로 된 해법은 `baseline_ports.py` 에 보정 종류 상수를 두고
#    Pi 가 그 코드를 `detail` 과 함께 보내는 것**이고, 그건 양쪽 합의가
#    필요해서 지금은 안 했다. 그때까지의 임시 다리다.
#    아래 `_CORRECTION_KEYS` 의 문구는 `grasp_alignment.judge()` 의 리터럴을
#    그대로 옮긴 것이다 — 그 파일을 고치면 여기도 같이 고칠 것.

BACK_OFF = "BACK_OFF"      # 물체가 턱 선보다 가깝다 -> 뒤로
CREEP_IN = "CREEP_IN"      # 물체가 전진 거리 밖이다 -> 앞으로
RE_AIM = "RE_AIM"          # 물체가 턱 폭 밖이다 -> 좌우로 다시 겨눔
UNFIXABLE = "UNFIXABLE"    # Host 가 움직여서 고칠 수 있는 게 아니다

_CORRECTION_KEYS = (
    ("후진 필요", BACK_OFF),
    ("재직진 필요", CREEP_IN),
    ("재회전 필요", RE_AIM),
    # servo 1 이 거부했거나 팔 길이가 미실측이라 Pi 가 못 고치는 좌우 치우침도
    # 결국 차를 다시 겨누는 것으로 푼다.
    ("재회전", RE_AIM),
    ("Pi가 못 고친다", RE_AIM),
)

_LATERAL_RE = re.compile(r"좌우\s*([+-]?\d+(?:\.\d+)?)\s*mm")


@dataclass(frozen=True)
class GraspCorrection:
    """Pi 가 요청한 재정렬. `kind` 는 위 네 상수 중 하나다.

    `lateral_mm` 은 **+ 가 왼쪽**이다(Pi `TargetObservation.lateral_m` 규약).
    RE_AIM 일 때 회전 방향이 여기서 나온다 — 부호를 못 읽으면 어느 쪽으로
    돌지 모르므로 보정하지 않는 편이 낫다(반대로 돌면 더 나빠진다).
    """

    kind: str
    detail: str = ""
    lateral_mm: Optional[float] = None

    @property
    def actionable(self) -> bool:
        """Host 가 차를 움직여 고칠 수 있는가."""
        if self.kind == UNFIXABLE:
            return False
        if self.kind == WAIT:
            return False   # 움직여서 고치는 게 아니다. 기다린다
        if self.kind in (RE_AIM, SHIFT) and self.lateral_mm is None:
            return False   # 방향을 모른다 — 찍어서 움직이지 않는다
        return True

    @property
    def transient(self) -> bool:
        """가만히 있으면 저절로 풀리는가. WAIT 는 실패가 아니라 대기다."""
        return self.kind == WAIT


def classify_correction(detail: str) -> GraspCorrection:
    """Pi 의 `detail` 문장 -> `GraspCorrection`. 모르면 UNFIXABLE."""
    m = _LATERAL_RE.search(detail or "")
    lateral = float(m.group(1)) if m else None
    for key, kind in _CORRECTION_KEYS:
        if key in (detail or ""):
            return GraspCorrection(kind, detail, lateral)
    return GraspCorrection(UNFIXABLE, detail, lateral)

# ---------------------------------------------------------------------------
# INSERT_BLOCKED 보정 요청 — 바구니 앞은 GRASP 와 사유 어휘가 다르다
# ---------------------------------------------------------------------------
#
# GRASP 쪽 분류기를 그대로 쓰면 안 된다. 사유 문구가 다르기도 하지만, 더 중요한
# 차이가 하나 있다 — **INSERT 차단 사유에는 "움직이면 안 되고 기다려야 하는
# 것"이 섞여 있다.**
#
#     "차체가 아직 정지하지 않았다"
#     "직전 판독이 없다 — 한 사이클 더 확인해야 한다"
#     "판독이 흔들린다 (+Nmm) — 아직 움직이는 중이거나 관측이 불안정하다"
#
# 셋 다 **Host 가 가만히 있으면 저절로 풀립니다.** 여기에 대고 차를 3cm 움직이면
# 판독이 또 흔들려서 조건이 영영 안 맞는다. 그래서 WAIT 를 따로 둔다.
#
# 문구 출처는 `domain/task/preconditions.py::check_insert`. GRASP 쪽과 같은
# 취약점을 공유한다 — Pi 가 문구를 고치면 여기가 조용히 실패한다. 정식 해법도
# 같다(`baseline_ports.py` 에 사유 코드 상수).

SHIFT = "SHIFT"            # 좌우로 밀렸다 -> 메카넘 횡이동. lateral_mm 부호로 방향
WAIT = "WAIT"              # 기다리면 풀린다. 움직이면 오히려 나빠진다

_INSERT_KEYS = (
    # 기다리면 풀리는 것 — 반드시 먼저 본다
    ("차체가 아직 정지하지 않았다", WAIT),
    ("직전 판독이 없다", WAIT),
    ("판독이 흔들린다", WAIT),
    # 차를 움직여 고치는 것
    ("바구니가 멀다", CREEP_IN),
    ("라이다 판독이 하한보다 가깝다", BACK_OFF),
    # 테두리를 스치는 중이라는 조기 신호다. 더 가면 절벽(0.125m)이라 물러난다.
    ("정면 점이 부족하다", BACK_OFF),
    ("정렬이 틀어졌다", RE_AIM),
    ("좌우로 밀려 있다", SHIFT),
)

# "yaw +0.123rad" 에서 부호를 읽는다. 차량 정면이 +x, 왼쪽이 +y 이고
# yaw_error = atan2(ny, nx) 이므로(basket_lidar_align.py:351), **양수면 바구니
# 면의 법선이 왼쪽을 향한다 = 반시계(yaw+)로 돌아야 한다.**
_YAW_RE = re.compile(r"yaw\s*([+-]?\d+(?:\.\d+)?)\s*rad")


def classify_insert_correction(detail: str) -> GraspCorrection:
    """Pi 의 INSERT_BLOCKED `detail` -> `GraspCorrection`. 모르면 UNFIXABLE.

    `detail` 에는 사유가 **여러 개 한꺼번에** 올 수 있다(`check_insert` 가
    reasons 를 모아서 낸다). 그래서 `_INSERT_KEYS` 순서가 곧 우선순위다 —
    기다리면 풀리는 것이 하나라도 섞여 있으면 **움직이지 않는다.** 판독이
    아직 안 정해진 상태에서 낸 거리·yaw 값을 믿고 움직이면 안 되기 때문이다.
    """
    text = detail or ""
    lateral = None
    m = _LATERAL_RE.search(text)
    if m:
        lateral = float(m.group(1))
    else:
        # "좌우로 밀려 있다 (+85mm > ±70mm)" 형식도 받는다.
        m2 = re.search(r"좌우로 밀려 있다\s*\(([+-]?\d+(?:\.\d+)?)\s*mm", text)
        if m2:
            lateral = float(m2.group(1))

    for key, kind in _INSERT_KEYS:
        if key in text:
            if kind is RE_AIM:
                my = _YAW_RE.search(text)
                if my is None:
                    return GraspCorrection(UNFIXABLE, detail, lateral)
                # yaw 부호를 lateral_mm 자리에 실어 보낸다 — mission.py 의
                # 회전 방향 판정이 "부호가 양수면 yaw+" 로 GRASP 와 같다.
                return GraspCorrection(RE_AIM, detail, float(my.group(1)))
            return GraspCorrection(kind, detail, lateral)
    return GraspCorrection(UNFIXABLE, detail, lateral)


# 같은 경고를 이 간격보다 자주 찍지 않는다. REJECTED 는 Pi 워치독이 발동할
# 때마다 나오는데, Host 주기가 워치독 한계보다 느리면 초당 여러 번이 된다 —
# 그대로 찍으면 콘솔이 묻히고, 진짜 인코더 버그가 났을 때 그 한 줄이 안 보인다.
_WARN_REPEAT_SEC = 5.0


def encode(cmd: MissionCommand) -> HostCommand:
    """Host 내부 명령 -> 전선에 실릴 `HostCommand`.

    네 가지 동작이 속도 넷으로 어떻게 옮겨지는가:

        go     -> linear_x = +AGREED_LINEAR_MPS
        back   -> linear_x = -AGREED_LINEAR_MPS
        left   -> linear_y = +AGREED_LINEAR_MPS       (메카넘 횡이동)
        right  -> linear_y = -AGREED_LINEAR_MPS
        stop   -> stop = True            (나머지 셋을 무시하는 가장 센 명령)
        yaw+   -> angular_z = +AGREED_ROTATION_RAD_S   (반시계)
        yaw-   -> angular_z = -AGREED_ROTATION_RAD_S   (시계)

    Host 는 회전과 병진을 **절대 섞지 않는다** — `_send_drive()` 가 셋 중
    하나만 고르므로, Pi 의 `resolve_motion()` 이 "제자리회전에 병진이 섞였다"
    로 거부하는 경로에 걸릴 일이 없다.

    속도 크기는 `domain/task/motion.py` 의 합의값을 그대로 가져온다. 여기에
    숫자를 다시 적으면 두 벌이 되고, 갈라지는 순간 Pi 가 조용히 잘라낸 값으로
    돌아 Host 의 경로 계산과 실제 주행이 어긋난다. 안전 한계 자체는 여전히
    Pi 가 집행한다 — Host 가 무엇을 보내든 바퀴를 돌리는 쪽이 자른다.
    """
    state = _STATE_TO_PI.get(cmd.status)
    if state is None:
        # 모르는 상태 이름을 추측해서 보내지 않는다. 정지가 안전하다.
        return HostCommand(state=MissionState.IDLE, stop=True)

    if cmd.cmd == "go":
        return HostCommand(state=state, linear_x=AGREED_LINEAR_MPS)
    if cmd.cmd == "back":
        # 예전 4어휘(go/stop/yaw+/yaw-)에는 후진이 없었다. 속도 형식으로
        # 바뀌면서 부호만 뒤집으면 되는 것이 됐다 — Pi 의 `_clamp` 가
        # copysign 이라 음수 크기를 그대로 잘라 준다. GRASP_ALIGN 이 쓴다.
        return HostCommand(state=state, linear_x=-AGREED_LINEAR_MPS)
    if cmd.cmd == "left":
        # 메카넘 횡이동. 다섯 필드에 이미 있는 linear_y 라 프로토콜 확장이
        # 아니다. INSERT_ALIGN 이 쓴다 — 바구니 앞 좌우 오차는 회전으로 고치면
        # 거리와 yaw 가 같이 틀어져서 세 조건을 동시에 흔든다.
        return HostCommand(state=state, linear_y=AGREED_LINEAR_MPS)
    if cmd.cmd == "right":
        return HostCommand(state=state, linear_y=-AGREED_LINEAR_MPS)
    if cmd.cmd == "yaw+":
        return HostCommand(state=state, angular_z=AGREED_ROTATION_RAD_S)
    if cmd.cmd == "yaw-":
        return HostCommand(state=state, angular_z=-AGREED_ROTATION_RAD_S)
    # "stop" 과 모르는 값 전부 — 모르면 정지한다.
    return HostCommand(state=state, stop=True)


class VehicleLink:
    """전송 어댑터의 추상 인터페이스."""

    #: 마지막으로 받은 Pi 보고 (report, state, detail). 아직 없으면 None.
    last_report: Optional[tuple[str, str, str]] = None

    #: 마지막 GRASP_BLOCKED 가 요청한 재정렬. mission.py 의 GRASP 가 읽고
    #: GRASP_ALIGN 으로 넘어간다. **읽은 쪽이 지운다**(take_correction) —
    #: 한 번의 요청으로 한 번만 움직이기 위해서다.
    last_correction: Optional[GraspCorrection] = None

    #: 마지막 INSERT_BLOCKED 가 요청한 재정렬. PLACE 가 읽는다.
    #:
    #: ⚠️ **GRASP 것과 반드시 따로 둔다.** 예전에는 슬롯이 하나여서, PLACE 중에
    #: 온 INSERT_BLOCKED 를 아무도 안 읽고 남겼다가 다음 기물의 GRASP 가
    #: 집어갔다 — 바구니 얘기를 기물 얘기로 읽고 엉뚱하게 3cm 움직이거나,
    #: actionable=False 면 멀쩡한 기물을 이유 없이 보류했다.
    last_insert_correction: Optional[GraspCorrection] = None

    def take_correction(self) -> Optional[GraspCorrection]:
        """GRASP 보정 요청을 **소비한다.** 없으면 None.

        지우지 않고 두면 Host 가 한 번의 BLOCKED 로 계속 움직인다 — Pi 는
        재관측할 때마다 새로 보고하므로, 매 요청당 한 걸음이 맞다."""
        c, self.last_correction = self.last_correction, None
        return c

    def take_insert_correction(self) -> Optional[GraspCorrection]:
        """INSERT 보정 요청을 소비한다. 없으면 None."""
        c, self.last_insert_correction = self.last_insert_correction, None
        return c

    def send(self, cmd: MissionCommand) -> None:
        raise NotImplementedError

    def poll_status(self) -> str:
        """차량이 보고하는 상태.

        "IDLE" | "BUSY" | "GRASP_DONE" | "PLACE_DONE" | "FAILED" 중 하나.
        """
        raise NotImplementedError


class ConsoleVehicleLink(VehicleLink):
    """전송 없이 콘솔에만 찍는다. 차량 없이 mission.py 로직만 시험할 때 쓴다.

    GRASP/PLACE 명령을 보내는 즉시 완료된 것으로 치고 다음 상태로 넘어간다.
    """

    def __init__(self, auto_complete: bool = True) -> None:
        self._auto_complete = auto_complete
        self._pending_done: Optional[str] = None

    def send(self, cmd: MissionCommand) -> None:
        extra = f"target={cmd.target_label}" if cmd.target_label else ""
        print(f"\r[vehicle_link] {cmd.cmd:5s} [{cmd.status:14s}] "
              f"robot=({cmd.robot_x:6.3f},{cmd.robot_y:6.3f},{cmd.robot_yaw_deg:6.1f}°) "
              f"{extra}   ",
              end="", flush=True)
        if self._auto_complete and cmd.status in ("GRASP", "PLACE"):
            self._pending_done = f"{cmd.status}_DONE"

    def poll_status(self) -> str:
        if self._pending_done:
            status, self._pending_done = self._pending_done, None
            return status
        return "IDLE"


class UdpVehicleLink(VehicleLink):
    """실제 차량(Pi)과 UDP+JSON 으로 말한다. 명령 5005 송신 / 보고 5006 수신.

    ## 왜 최신 것만 보는가

    이 링크가 실어 나르는 것은 **그 순간의 속도 명령**이라 오래된 패킷은
    쓸모가 없다. TCP 로 재전송을 기다리는 것보다 다음 사이클 것을 쓰는 쪽이
    항상 낫다. 그래서 수신도 큐를 쌓지 않고 마지막 것만 본다.

    ## 안 닿아도 예외를 내지 않는다

    UDP 라 Pi 가 아직 안 켜져 있어도 `send()` 는 조용히 나간다. 링크가
    끊긴 것을 판정하는 것은 **받는 쪽(Pi)의 워치독**이다 — Host 가 말을
    멈추면 차량도 멈춘다.
    """

    def __init__(self, pi_ip: str, cmd_port: int = 5005, status_port: int = 5006,
                 bind_ip: str = "0.0.0.0", verbose: bool = True) -> None:
        self.pi_ip = pi_ip
        self.cmd_port = cmd_port
        self.verbose = verbose
        self.last_report: Optional[tuple[str, str, str]] = None
        self._warn_seen: dict[str, tuple[float, int]] = {}

        # INSERT 는 두 번 보고된다: INSERT_DONE(또는 INSERT_FAILED) 다음에
        # 반드시 IDLE_DONE 이 온다(baseline_mission.BaselineInsertState).
        # 팔이 접히기 전에 차를 움직이면 안 되므로 **IDLE_DONE 을 완료 신호로
        # 쓰고**, 그 직전 결과를 여기 기억해 성패를 가른다.
        self._insert_ok: Optional[bool] = None

        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.setblocking(False)
        self._recv_sock.bind((bind_ip, status_port))

    # --- 송신 ---------------------------------------------------------

    def send(self, cmd: MissionCommand) -> None:
        host_cmd = encode(cmd)
        payload = json.dumps({
            "state":     host_cmd.state,
            "linear_x":  host_cmd.linear_x,
            "linear_y":  host_cmd.linear_y,
            "angular_z": host_cmd.angular_z,
            "stop":      host_cmd.stop,
        }).encode("utf-8")
        try:
            self._send_sock.sendto(payload, (self.pi_ip, self.cmd_port))
        except OSError as exc:
            # 네트워크가 잠깐 끊겨도 미션 루프는 안 죽어야 한다 — 다음
            # 사이클에 다시 시도된다.
            self._warn(f"전송 실패 — {exc}")

    # --- 수신 ---------------------------------------------------------

    def poll_status(self) -> str:
        """논블로킹. 그 사이 쌓인 보고를 전부 읽되 **완료 신호는 놓치지 않는다.**

        여러 개가 와 있으면 마지막 것만 쓰는 것이 이 프로젝트의 관례지만,
        보고는 속도 명령과 달리 **사건**이라 덮어쓰면 안 된다 — INSERT_DONE
        과 IDLE_DONE 이 한 사이클 안에 같이 도착하는 일이 실제로 생긴다.

        ⚠️ 여기서 한 번 더 나눈다: 완료/실패(`_TERMINAL`)는 **그 밖의 값에
        절대 덮이지 않는다.** 그냥 "마지막 것"을 돌려주면 이런 순서에서
        신호가 통째로 사라진다:

            GRASP_DONE  ->  STATE  ->  REJECTED     (한 사이클에 같이 도착)
                            ^^^^^^^^^^^^^^^^^^ 이게 덮어써서 "BUSY" 가 나감

        mission.py 의 GRASP 는 `poll_status() == "GRASP_DONE"` 한 번을 보고
        전이하는데, 그 한 번을 놓치면 **영원히 GRASP 에 머문다.** 그리고
        REJECTED 는 워치독이 발동할 때마다 나오므로(Host 주기가 Pi 워치독
        한계보다 느리면 초당 여러 번) 이 순서는 드문 사고가 아니라 상시
        상황이다.
        """
        terminal = None      # 완료/실패 — 최우선
        other = "IDLE"       # BUSY/IDLE — 참고용
        while True:
            try:
                data, _addr = self._recv_sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError as exc:
                self._warn(f"수신 오류 — {exc}")
                break
            translated = self._handle(data)
            if translated is None:
                continue
            if translated in _TERMINAL:
                terminal = translated
            else:
                other = translated
        return terminal if terminal is not None else other

    def _handle(self, data: bytes) -> Optional[str]:
        """보고 하나를 옛 문자열로 옮긴다. 옮길 게 없으면 None."""
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            self._warn("Pi 보고 파싱 실패 — 버림")
            return None

        report = msg.get("report")
        state = msg.get("state", "")
        detail = msg.get("detail", "")
        if not isinstance(report, str):
            self._warn("Pi 보고에 report 가 없다 — 버림")
            return None
        self.last_report = (report, state, detail)

        if report == Report.GRASP_DONE:
            return "GRASP_DONE"
        if report == Report.GRASP_FAILED:
            self._warn(f"파지 실패 — {detail}")
            return "FAILED"

        # INSERT: 결과를 기억해 두고 IDLE_DONE 에서 판정한다.
        if report == Report.INSERT_DONE:
            self._insert_ok = True
            return "BUSY"
        if report == Report.INSERT_FAILED:
            self._insert_ok = False
            self._warn(f"투하 실패 — {detail}")
            return "BUSY"
        if report == Report.IDLE_DONE:
            ok, self._insert_ok = self._insert_ok, None
            if ok is False:
                return "FAILED"
            return "PLACE_DONE"

        if report == Report.GRASP_BLOCKED:
            self.last_correction = classify_correction(detail)
        elif report == Report.INSERT_BLOCKED:
            self.last_insert_correction = classify_insert_correction(detail)

        if report in _BLOCKING_REPORTS:
            # Pi 가 "조건이 안 맞는다, 수정된 명령을 달라"고 말하는 중이다.
            # 지금 Host 에는 그 요청에 응답하는 로직이 없다 — 기다리기만 한다.
            # 다음 단계에서 mission.py 에 GRASP_ALIGN 을 넣어 대응한다.
            self._warn(f"Pi 가 대기 중: {report} [{state}] {detail}")
            return "BUSY"

        if report == Report.REJECTED:
            # Pi 가 명령 자체를 실행할 수 없다고 되돌려줬다. 링크 문제가
            # 아니라 **Host 인코더 버그** 신호다 — 조용히 넘기면 안 된다.
            self._warn(f"⚠️ Pi 가 명령을 거부했다: [{state}] {detail}")
            return "BUSY"

        if report == Report.GRASP_READY or report == Report.INSERT_READY:
            return "BUSY"
        if report == Report.STATE:
            return "IDLE" if state == MissionState.IDLE else "BUSY"

        self._warn(f"모르는 Pi 보고: {report} [{state}] {detail}")
        return None

    def close(self) -> None:
        self._send_sock.close()
        self._recv_sock.close()

    def _warn(self, message: str) -> None:
        """같은 문구는 `_WARN_REPEAT_SEC` 마다 한 번만, 그동안 몇 번 더 났는지와
        함께 찍는다. 눌러 버리지 않고 **세어서 보여주는** 이유: 워치독 발동이
        상시가 됐다는 사실 자체가 진단 정보이기 때문이다."""
        if not self.verbose:
            return
        now = time.monotonic()
        last, count = self._warn_seen.get(message, (0.0, 0))
        if now - last < _WARN_REPEAT_SEC:
            self._warn_seen[message] = (last, count + 1)
            return
        suffix = f"  (직전 {_WARN_REPEAT_SEC:.0f}초간 {count}회 더)" if count else ""
        print(f"\n[vehicle_link] {message}{suffix}")
        self._warn_seen[message] = (now, 0)
