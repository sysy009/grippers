"""arm_driver_node — SO-ARM101 실제 하드웨어를 쥔 노드.
soarm_lab.arm을 그대로 감싼다. 새 IK/서보 로직은 없음.

⚠️ 두 가지를 반드시 지킬 것 (디버깅으로 찾아낸 함정):

1. `soarm.grip()` 은 `real` 인자를 받지 않고 내부에서 항상
   `self._backend(False)` (SimBackend)를 쓴다 — 실물 명령으로 절대
   못 쓴다. 실물 그리퍼는 `soarm._backend(real=True).drv.set_position(6, ...)`
   로 서보에 직접 명령해야 한다 (soarm_lab/arm.py의 Arm.grip 정의 참고).

2. 실물 포트는 `arm_port` 노드 파라미터로 받는다 — 심볼릭 링크로 포트를
   고정하던 방식은 폐기했다. `Arm._backend(real=True)` 는 `self._real` 이
   비어 있을 때만 기본 포트(`/dev/soarm`)로 `RealBackend()` 를 새로
   만들므로, 커스텀 포트를 쓰려면 첫 호출 전에 `soarm._real` 을 미리
   채워 둬야 한다 — __init__ 에서 그렇게 한다.
"""

import fcntl
import json
import math
import os
import sys
import time

sys.path.insert(0, "/third_party/soarm_provided_d")  # PYTHONPATH 미설정 환경 대비 안전장치

import rclpy  # noqa: I001
import rclpy.logging  # main()이 노드 생성 실패 시 노드 없이 로그를 남긴다
from grippers_interfaces.action import (
    ExecuteJointChunk,
    MoveToCartesian,
    MoveToFloorPose,
    ReorientArm,
)
from grippers_interfaces.srv import (
    GetArmState,
    GetLoad,
    GetPolicyState,
    OffsetBaseYaw,
    SetGripper,
)
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

# 아래 3개는 순서가 고정이다(위 import rclpy의 noqa: I001이 이 블록 전체의
# 자동 정렬을 막아 준다) — 알파벳 순으로 바꾸면 깨진다.
# soarm 자체가 Arm() 싱글턴이다 — 이 뒤로는 soarm.go()이지 soarm.arm.go()가 아니다.
# soarm_lab을 import하면 soarm_lab/__init__.py가 자기 디렉터리를 sys.path에 얹어
# 둬서(내부 모듈끼리 flat import 되게) 그 다음부터 real/driver_sdk를 이렇게 바로
# 가져올 수 있다 — 반드시 soarm_lab import 다음에 와야 한다.
from soarm_lab import arm as soarm
from real import RealBackend

from .gripper_calibration import (
    GRIPPER_CLOSED_MM,
    GRIPPER_GRASP_MIN_MM,
    GRIPPER_OPEN_MM,
    position_from_width,
    width_from_position,
)
from .floor_grasp_profiles import (
    BASKET_DROP_195_RAW,
    CARRY_RAW,
    HORIZONTAL_GRASP_POSES_DEG,
    HORIZONTAL_SAFE_145_RAW,
    IDLE_CRADLE_RAW,
    TAUGHT_HOMING_OFFSETS,
)
from . import calib_identity

WRIST_SERVO_ID = 4
GRIPPER_SERVO_ID = 6
ALL_SERVO_IDS = range(1, 7)

# STS3215 PRESENT_LOAD 는 하위 10비트가 크기(0~1023), 0x400 비트가 방향이다
# (third_party/soarm_provided_d/soarm_lab/driver_sdk.py get_load).
# 도메인 계층은 0~1 비율만 안다 — 서보 각도 변환과 같은 이유로 원시값 →
# 비율 변환은 이 노드가 담당한다 (class_diagram.md §2).
GRIPPER_LOAD_MAX_RAW = 1023.0
# STS3215 위치 레지스터의 정의역. 이 밖의 값은 읽기가 깨진 것이다.
POSITION_RAW_MAX = 4095

# 그리퍼 닫힘 명령 후 부하가 정착할 때까지의 대기 시간.
# 실측(2026-08-18): 0.26~0.51s 는 이동 중 포화(±500)라 빈 채와 물체가 구분되지
# 않고, 0.77s 에 거의 안정, 1.03s 이후로는 10초까지 한 틱도 변하지 않았다.
# 1.0s 안정 + 여유로 1.5s. 정착 타이밍은 서보 물리 지식이므로 도메인이 아니라
# 여기 둔다 — GraspState 에 sleep 을 넣지 않는다.
GRASP_SETTLE_SEC = 1.5
# 그리퍼 개폐 "이동이 끝났는가" 판정 — 시간이 아니라 위치 정지로 본다
# (_wait_gripper_motion_settled 참고, 2026-08-24 실기로 필요성 확인).
GRIPPER_MOTION_POLL_SEC = 0.1
GRIPPER_MOTION_SETTLED_RAW = 3  # 이 폭 안에서만 변하면 멈춘 것으로 본다
GRIPPER_MOTION_TIMEOUT_SEC = 4.0  # 최대 행정(168mm↔9mm, 약 850raw)보다 넉넉하게
# servo 6도 속도를 상속하지 않고 매번 직접 쓴다(_on_set_gripper 주석 참고).
# 850 raw를 GRIPPER_MOTION_TIMEOUT_SEC의 절반 안에 끝낼 수 있는 값으로 잡는다 —
# 그래야 위 타임아웃이 "느려서"가 아니라 "정말 뭔가에 물려서"만 걸린다.
GRIPPER_SPEED_RAW = 600
GRIPPER_ACCEL_RAW = 30
FLOOR_POSE_STEPS = 30
FLOOR_POSE_STEP_SEC = 0.10
# 서보 goal_speed / acceleration — _glide_to_raw_positions가 매 이동마다 다시
# 쓴다(왜 상속하면 안 되는지는 그 함수의 주석 참고).
#
# 단위는 실측으로 확인된 대로 대략 raw/s다(2026-08-24: 레지스터 150에서 실측
# 153 raw/s). 이 값이 정하는 것은 **궤적의 모양이 아니라 상한**이다 — 실제
# 움직임은 여전히 FLOOR_POSE_STEPS개의 waypoint를 FLOOR_POSE_STEP_SEC 간격으로
# 찍는 보간이 만든다. 즉 이 상한은 보간이 요구하는 속도를 막지만 않으면 된다.
# 가장 긴 이동은 IDLE->safe의 servo 2(1663 raw)로 3.0s 안에 끝내려면 554 raw/s가
# 필요하다 — 2배 여유를 두고 잡는다. 팔이 검증된 것보다 빨라지는 게 아니라,
# 원래 의도된 보간 궤적을 따라가지 못하게 막던 병목을 치우는 것이다.
FLOOR_POSE_SPEED_RAW = 1200
FLOOR_POSE_ACCEL_RAW = 30

# **IDLE로 돌아오는 이동에서만** 나머지 관절이 완전히 멈춘 뒤 별도 구간에서
# 단독으로 움직일 관절. 최종 자세는 바뀌지 않는다 — 경로만 두 구간으로 쪼갠다.
#
# ⚠️ 2026-08-24 실기, 세 번에 걸쳐 좁혀 온 값이다.
#
# (1) 룩을 문 채 safe -> idle로 복귀하는데 그리퍼가 차체 전면을 긁어 룩을
#     놓쳤다. 선형 보간이 모든 관절을 같은 비율로 동시에 움직이는 게 원인이다.
#     이 구간의 이동량은
#
#         servo 2(Shoulder)     -1663 raw   팔을 내림
#         servo 4(Wrist Pitch)  +1618 raw   손목을 접음  <- 142도
#         servo 3(Elbow)         +579 raw
#         servo 5(Wrist Roll)     +64 raw   (사실상 안 움직인다)
#
#     즉 어깨가 내려가는 **동안** 손목이 같은 비율로 접히니, 물체를 문
#     그리퍼가 팔이 충분히 물러나기 전에 차체 전면에 닿는다. 사용자가 처음
#     지목한 servo 5는 64 raw밖에 안 움직여 대상이 아니다 — 물리적 관찰
#     (손목이 일찍 접힌다)은 정확했고 번호만 어긋났다.
#
# (2) 손목에 부분 지연(진행률 45%까지 정지)만 줬더니 **여전히 심하게 긁었다**.
#     겹치는 구간이 조금이라도 남으면 소용이 없다는 뜻이라, 아예 별도 구간으로
#     분리했다.
#
# (3) 그런데 그걸 **모든 이동에 전역으로** 걸었더니 IDLE로 돌아오는 쪽은
#     고쳐졌는데 IDLE에서 나가는 쪽이 새로 긁기 시작했다. 방향이 반대면
#     안전한 관절 순서도 반대이기 때문이다 — 돌아올 때는 어깨가 먼저 물러난
#     뒤 손목이 접혀야 하고, 나갈 때는 그 반대다. 그래서 지연은 전역 설정이
#     아니라 **이동마다 호출부가 정하는 인자**이고(_glide_to_raw_positions의
#     defer_joints), IDLE을 목표로 하는 이동에만 건다.
#
# 지연이 실제로 의미를 갖는 구간은 safe<->idle뿐이다 — 다른 구간의 servo 4
# 이동량은 44~87 raw로 FLOOR_POSE_START_TOLERANCE_RAW 이하라 어차피 건너뛴다.
RETURN_TO_IDLE_DEFERRED_JOINTS = (4,)
# 보간이 끝난 뒤 "실제 도달"을 기다리는 값 — 고정 sleep이 아니라 위치 폴링이다
# (_wait_floor_pose_arrived 참고, 2026-08-24 실기로 필요성 확인).
FLOOR_POSE_ARRIVE_POLL_SEC = 0.1
# 진전이 이만큼(raw)도 없이 FLOOR_POSE_STALL_SEC가 지나면 걸린 것으로 본다.
# 폴링 간격 0.1s에 실측 최저 속도 153 raw/s면 정상 이동 중에는 스텝당 15 raw쯤
# 줄어드므로, 3 raw는 센서 잡음만 걸러내고 진짜 진전은 놓치지 않는 수준이다.
FLOOR_POSE_PROGRESS_RAW = 3
FLOOR_POSE_STALL_SEC = 1.5
# 어떤 경우에도 여기서 영원히 매달리지 않게 하는 최후의 한계선.
FLOOR_POSE_ARRIVE_MAX_SEC = 25.0
# 시리얼 패킷 유실 한 번으로 이동이 끊기지 않게 하는 재시도
# (_read_joint_positions 주석 참고).
JOINT_READ_ATTEMPTS = 3
JOINT_READ_RETRY_SEC = 0.05

# ── VLA 관절 청크 재생(ExecuteJointChunk) ──────────────────────────────────
# STS3215 한 바퀴 = 4096 카운트지만 LeRobot 이 정규화에 쓰는 분모는
# `model_resolution_table[...] - 1` = 4095 다(motors_bus._normalize).
# 여기서도 같은 값을 써야 왕복 변환이 딱 떨어진다.
STS3215_RESOLUTION_MINUS_1 = 4095
# Goal_Velocity. 0 = 무제한이고, 수집 때 팔로워가 리더를 쫓던 조건이 그것이다.
#
# ⚠️ 여기를 0 이 아닌 값으로 두면 정책이 조용히 망가진다. 2026-09-02 실기:
# align_to_idle 이 남긴 150 raw/s 가 서보 RAM 에 남아 있어서, 정책이
# shoulder_lift 를 50도로 부르는데 팔은 12.1도/s 로만 갔다. 스텝당 이동
# 상한이 **현재 위치** 기준이라, 못 따라간 만큼 다음 목표가 더 잘리고 그게
# 다시 뒤처짐을 키운다. 그래서 재생 시작마다 명시적으로 다시 쓴다.
VLA_SPEED_RAW = 0
VLA_ACCEL_RAW = 30
# 스텝당 관절 이동 상한 기본값(도). rollout_policy.py 의 --max-rel 기본값과
# 같은 자리다 — 정책이 분포 밖 입력에 튀어도 한 스텝에 갈 수 있는 거리를 묶는다.
VLA_MAX_STEP_DEG_DEFAULT = 5.0
# 청크 하나의 상한. 100스텝 x 30fps = 3.33초가 정상이고, 그 몇 배를 넘는
# 요청은 인터페이스를 잘못 쓴 것으로 보고 거부한다.
VLA_MAX_STEPS = 500

MAX_FLOOR_POSE_SERVO2_TEMP_C = 50
FLOOR_POSE_START_TOLERANCE_RAW = 120
# ⚠️ 위 120 raw(약 10.5도)는 **교시 자세 단계 사이**를 오갈 때 쓰라고 정한
# 값이다(_wait_floor_pose_arrived 주석 참고: 다음 단계의 시작 자세 게이트와
# 같은 값을 써야 한다). GRASP 좌우 정렬 보정처럼 몇 도짜리 미세 이동에
# 그대로 쓰면 이동 자체가 "이미 도착"으로 걸러진다.
#
# 2026-08-28 실기에서 정확히 그 일이 났다. 뎁스캠이 룩을 좌우 +27mm로 보고
# servo 1을 +5.2도(59 raw) 돌리라고 했는데, 59 < 120이라 _glide_phase가
# 관절을 통째로 버렸다. 로그에는 `offset_base_yaw: servo 1 2067 -> 2067
# (+5.2도)`가 찍히고 서비스는 ok=True를 냈다 — 움직이지 않았는데 성공이다.
# Pi는 매 사이클 같은 +27mm를 다시 보고 같은 보정을 다시 걸어, 60초 내내
# GRASP_CENTERING만 반복하고 하강으로 넘어가지 못했다.
#
# 좌우 정렬 허용오차는 GRASP_CENTERING_TOLERANCE_M = 10mm이고, servo 1 축에서
# 턱까지가 294mm(baseline_constants.SERVO1_AXIS_TO_JAW_MM, 실측 2026-08-26)
# 이므로 10mm는 약 1.95도 = 22 raw다. 즉 이 보정은 22 raw 단위를 분간해야
# 하는 동작이라 120 raw 격자로는 원리적으로 불가능하다.
#
# ⚠️ 이 주석은 2026-08-29까지 214mm(= 30 raw)로 적혀 있었다. 실측 전의
# 어림값이 남아 있던 것이고, 결론(120으로는 불가능)은 어느 값으로도 같지만
# 숫자를 인용하면 틀린 값이 퍼진다. 팔 길이의 단일 출처는 baseline_constants다.
BASE_YAW_TOLERANCE_RAW = 10  # 약 0.9도 = 294mm 거리에서 약 4.5mm
# 좌우 보정 글라이드의 waypoint 간격(raw). 교시 자세 이동은 FLOOR_POSE_STEPS
# (30)로 고정이지만, 이 보정은 5도짜리도 있고 15도짜리도 있어서 고정 단계
# 수를 쓰면 짧은 이동에 같은 3.0초를 쓴다. 12 raw(약 1도)마다 한 점씩 찍으면
# 5.2도 보정이 5단계 = 0.5초로 끝나 호출자의 3.0초 대기 안에 들어온다.
BASE_YAW_RAW_PER_STEP = 12
# recover_idle이 "지금 어느 등록 자세에서 시작하는가"를 판정하는 허용치.
# 정상 경로의 게이트(120)보다 넉넉하다 — 복구가 필요한 상황은 정의상 팔이
# 목표에 못 미친 상황이라 120으로는 아무 자세에도 안 붙는다. 대신 이 값을
# 넘으면 추측해서 움직이지 않고 사람에게 넘긴다(_move_floor_stage 참고).
RECOVER_MATCH_TOLERANCE_RAW = 500

# 기동 시 IDLE 대비 편차를 로그로만 남기는 임계값 — 절대 자동 이동의 기준이
# 아니다. tools/align_to_idle.py가 실제 정렬을 담당한다.
IDLE_OFFSET_WARN_RAW = 120
IDLE_OFFSET_ERROR_RAW = 800

# --- 첫 이동 자동 IDLE 정렬 ----------------------------------------------
#
# 2026-08-25 사용자 지시: "맨처음 이동 게이트에서 최초 로봇암의 자세를 파악하고
# 무조건 자동으로 align_idle을 할 수 있게". 그전까지는 _log_idle_offset이
# 편차를 로그로만 남기고, 정렬은 사람이 tools/align_to_idle.py를 따로 돌려야
# 했다 — 그리고 그걸 잊으면 첫 safe 이동이 "등록된 이전 단계가 아닙니다"로
# 거부되거나, torque가 꺼진 채라 _require_operational_servos에서 막혔다.
#
# **기동 시점이 아니라 첫 이동 요청 시점**에 정렬한다. 노드가 뜨자마자 팔이
# 움직이면 운영자가 손을 대고 있을 수 있고, 정렬이 필요한지 아닌지는 실제로
# 움직이려는 순간에 판정하는 편이 정확하다.
#
# ⚠️ 정렬 경로는 반드시 아래 세 갈래로 나뉜다. 어떤 자세에서든 IDLE로 직선
# 보간하면 안 된다 — 팔이 바닥 높이에 있을 때 그렇게 하면 그리퍼가 바닥을
# 쓸고 간다(2026-08-24 실기, RETURN_TO_IDLE_DEFERRED_JOINTS 주석 참고).
#
#   (a) 등록 자세 근처(RECOVER_MATCH_TOLERANCE_RAW 이내) -> recover_idle과
#       똑같이 검증된 상승 체인을 탄다.
#   (b) 등록 자세 어디에도 안 붙지만 servo 2가 아래로 뻗어 있다
#       (AUTO_ALIGN_LIFT_VIA_SAFE_SERVO2_RAW 이상) -> 먼저 safe로 **들어올린
#       뒤** safe -> idle 체인을 탄다. safe로 가는 이동은 정의상 팔을 드는
#       방향이라 바닥을 쓸지 않는다.
#   (c) 그 외(이미 접힌 영역) -> IDLE로 직접 보간한다.
#
# servo 2 문턱값 근거: IDLE 829, safe 2492, grasp 3027~3136. 2200은 safe보다
# 조금 아래로, "팔이 앞으로 뻗어 있다"와 "접혀 있다"를 가르는 자리다.
AUTO_ALIGN_LIFT_VIA_SAFE_SERVO2_RAW = 2200
# 정렬 이동은 출발 자세가 검증되지 않았으므로 정상 이동의 절반 속도로 간다.
AUTO_ALIGN_SPEED_RAW = 600
# 접기 전에 그리퍼를 닫을지 가르는 폭. 실제로 물체를 문 닫힘 폭은 가장 넓은
# 것이 soccer의 31.0mm라, 그보다 한참 위인 이 값을 넘으면 아무것도 물고 있지
# 않은 '열린' 상태로 본다(_close_gripper_before_folding 참고).
AUTO_ALIGN_GRIPPER_CLOSE_ABOVE_MM = 45.0

CRADLE_XYZ_M = [0.15, 0.0, 0.20]  # TODO: INSERT 후 복귀 경로 별도 실측 필요

# MentorPi 베이스 보드가 잡는 장치. arm_port가 이걸 가리키면 팔 드라이버가
# 베이스 보드의 시리얼을 열어 버려 주행 명령이 통째로 깨진다.
BASE_BOARD_DEVICE = "/dev/rrc"


class ArmPortConflictError(RuntimeError):
    """arm_port가 베이스 보드 장치를 가리킬 때 __init__에서 올린다.
    main()이 잡아서 노드를 띄우지 않고 종료한다."""


class ArmCalibrationMismatchError(RuntimeError):
    """팔에 실린 캘리브레이션이 교시 자세와 다르다.

    하드웨어 고장이 아니다 — 팔은 멀쩡하고, 다만 이 코드가 아는 자세가
    아니다. 그래서 ArmHardwareUnavailableError 와 따로 둔다: 사람이 할 일이
    '고치기'가 아니라 '오프셋 되돌리기 또는 다시 교시하기'다."""


class ArmHardwareUnavailableError(RuntimeError):
    """SO-ARM101 또는 서보 버스가 응답하지 않거나 동작 불가 상태일 때 발생한다."""


class ArmDriverNode(Node):
    def __init__(self):
        super().__init__("arm_driver_node")
        cb_group = ReentrantCallbackGroup()

        self.declare_parameter("arm_port", "/dev/soarm")
        self.declare_parameter("enable_torque_on_start", False)
        self.declare_parameter("auto_align_on_first_move", True)
        # 교시 자세와 다른 캘리브레이션에서 기동을 거부한다. 끄는 것은
        # 팔을 다시 교시하는 중처럼 자세가 무효인 줄 알고 있을 때만이다.
        self.declare_parameter("verify_calibration", True)
        # 그리퍼 명령 폭의 하한. 기본값은 파지 전용 하한이고, 지금은 그것이
        # GRIPPER_CLOSED_MM과 같아 동작이 바뀌지 않는다
        # (gripper_calibration.GRIPPER_GRASP_MIN_MM 주석 참고).
        #
        # 파라미터로 뺀 이유는 tools/gripper_force_probe.py가 "턱이 실제로
        # 어디서 멈추는가"를 재려면 이 하한 아래로 명령해 봐야 하기 때문이다.
        # 런타임에 ros2 param set으로만 내릴 수 있게 해서, 평소 경로에서는
        # 실수로 낮은 값이 쓰이지 않는다.
        self.declare_parameter("min_gripper_width_mm", GRIPPER_GRASP_MIN_MM)
        # VLA 정책이 학습된 LeRobot 캘리브레이션 파일 경로.
        #
        # 비워 두면 ExecuteJointChunk 가 아예 거부된다 — 기본값을 지어내면
        # servo 5 가 85도 어긋난 데로 가기 때문에(ExecuteJointChunk.action 의
        # 표 참고) 없으면 없는 대로 멈추는 쪽이 맞다.
        #
        # 노트북의 원본은 ~/.cache/huggingface/lerobot/calibration/robots/
        # so_follower/grippers_arm.json 이고, Pi 에는 이 파일을 같이 배포해야
        # 한다. 정책을 다시 학습하면 그때 쓴 캘리브레이션으로 같이 바꿀 것.
        self.declare_parameter("policy_calibration_file", "")

        arm_port = self.get_parameter("arm_port").value
        enable_torque_on_start = bool(self.get_parameter("enable_torque_on_start").value)
        # 첫 바닥 자세 이동 요청 때 한 번만 IDLE로 자동 정렬한다.
        self._auto_align_pending = bool(self.get_parameter("auto_align_on_first_move").value)

        # RealBackend(port=...) 는 생성 즉시 시리얼 포트를 연다 — 검사는 반드시
        # 그 앞에 와야 한다. 뒤에 두면 이미 베이스 보드를 열어 버린 뒤가 된다.
        self._reject_base_board_port(arm_port)
        soarm._real = RealBackend(port=arm_port)
        if not soarm._real.drv.is_connected():
            raise ArmHardwareUnavailableError(f"SO-ARM101 serial connection failed: {arm_port}")
        self._claim_serial_port(soarm._real, arm_port)

        self.get_logger().info(f"arm_port={arm_port}")

        self._check_startup_torque(
            soarm._real,
            enable_torque_on_start=enable_torque_on_start,
        )
        self._check_taught_calibration(soarm._real)
        self._log_idle_offset(soarm._real)

        self._move_action_server = ActionServer(
            self,
            MoveToCartesian,
            "arm_driver/move_to_cartesian",
            execute_callback=self._execute_move,
            callback_group=cb_group,
        )
        self._reorient_action_server = ActionServer(
            self,
            ReorientArm,
            "arm_driver/reorient",
            execute_callback=self._execute_reorient,
            callback_group=cb_group,
        )
        self._floor_pose_action_server = ActionServer(
            self,
            MoveToFloorPose,
            "arm_driver/move_to_floor_pose",
            execute_callback=self._execute_floor_pose,
            callback_group=cb_group,
        )
        self._policy_calibration = self._load_policy_calibration(soarm._real)
        # 다른 액션과 달리 cancel_callback을 준다 — 재생 중 E-STOP이 여기로
        # 들어온다. 기본값은 거부라서 명시하지 않으면 취소가 안 먹는다.
        self._joint_chunk_action_server = ActionServer(
            self,
            ExecuteJointChunk,
            "arm_driver/execute_joint_chunk",
            execute_callback=self._execute_joint_chunk,
            cancel_callback=lambda _goal_handle: CancelResponse.ACCEPT,
            callback_group=cb_group,
        )
        self.create_service(
            GetPolicyState,
            "arm_driver/get_policy_state",
            self._on_get_policy_state,
            callback_group=cb_group,
        )
        self.create_service(
            SetGripper,
            "arm_driver/set_gripper",
            self._on_set_gripper,
            callback_group=cb_group,
        )
        self.create_service(
            GetLoad,
            "arm_driver/get_load",
            self._on_get_load,
            callback_group=cb_group,
        )
        self.create_service(
            GetArmState,
            "arm_driver/get_arm_state",
            self._on_get_arm_state,
            callback_group=cb_group,
        )
        self.create_service(
            Trigger,
            "arm_driver/fold_to_cradle",
            self._on_fold_to_cradle,
            callback_group=cb_group,
        )
        self.create_service(
            Trigger,
            "arm_driver/hold_position",
            self._on_hold_position,
            callback_group=cb_group,
        )
        self.create_service(
            OffsetBaseYaw,
            "arm_driver/offset_base_yaw",
            self._on_offset_base_yaw,
            callback_group=cb_group,
        )
        self.get_logger().info(
            "arm_driver_node ready — "
            f"auto_align_on_first_move={self._auto_align_pending}"
        )

    def _claim_serial_port(self, backend, arm_port: str) -> None:
        """이 포트를 쓰는 arm_driver가 하나뿐이도록 배타 잠금을 건다.

        ⚠️ 2026-08-25 실기에서 이것 없이 하루를 태웠다. 재기동 스크립트의
        pkill 패턴이 `ros2 run` 래퍼만 잡고 설치된 노드 실행 파일은 놓쳐서,
        arm_driver 세 개가 동시에 같은 시리얼 포트를 읽고 쓰고 있었다.
        증상이 하드웨어 고장과 구분되지 않는다는 것이 문제였다:

            - get_arm_state가 10~80%씩 무작위로 실패한다
            - 실패 서보 목록이 호출마다 바뀐다([6] → [2,4,5,6] → [1,2,4,5])
            - **깨진 값이 정상인 척 통과한다** — servo 3 위치가 55841로
              돌아온 적이 있다. 다른 서보의 응답 바이트가 섞인 것이다.
            - 드라이버는 "multiple access on port?"라고 정확히 말해 주는데,
              그 로그는 노드 stdout에 묻혀 아무도 안 본다

        flock은 권고적 잠금이라 파일을 여는 것 자체를 막지는 않지만, 같은
        규칙을 지키는 두 번째 arm_driver는 여기서 즉시 멈춘다. 프로세스가
        죽으면 커널이 알아서 놓아 주므로 남은 잠금을 치울 일이 없다.

        잠금 파일 핸들은 노드 수명 동안 살아 있어야 한다 — 지역 변수로 두면
        GC가 닫아 버려 잠금이 조용히 풀린다.
        """
        serial_handle = getattr(backend.drv, "serial", None)
        if serial_handle is None:
            self.get_logger().warn("시리얼 핸들을 찾지 못해 포트 배타 잠금을 걸지 못했습니다")
            return
        self._port_lock_file = serial_handle
        try:
            fcntl.flock(serial_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise ArmPortConflictError(
                f"{arm_port}를 이미 다른 프로세스가 쓰고 있습니다 ({e}). "
                "arm_driver가 두 개 이상 떠 있으면 시리얼 응답이 서로 섞여 "
                "위치·부하 값이 무작위로 깨집니다. 기존 프로세스를 먼저 종료하세요: "
                "ps -eo pid,args | grep arm_driver | grep -v grep"
            ) from e
        self.get_logger().info(f"{arm_port} 배타 잠금 확보")

    def _check_taught_calibration(self, backend) -> None:
        """이 팔의 Homing_Offset 이 교시 당시와 같은지 본다.

        ⚠️ 2026-08-29 에 VLA 시연 수집을 준비하며 LeRobot 캘리브레이션을
        돌렸고, 그때 서보의 Homing_Offset 이 덮여 썼다. 오프셋이 바뀌면
        floor_grasp_profiles.py 의 RAW 자세가 **같은 숫자로 다른 물리
        자세**를 가리킨다.

        오프셋은 서보 EEPROM 에 있지 git 에 있지 않아서, 브랜치를 바꿔도
        팔은 안 바뀐다. 코드는 베이스라인인데 팔은 VLA 캘리브레이션인
        조합이 아무 경고 없이 만들어진다 — 그 상태로 움직이면 차체·라이다에
        막히는 범위로 들어간다(shoulder_pan 가동폭 2493 -> 2087, sysy009
        실측 2026-08-29).

        그래서 경고가 아니라 거부다."""
        if not bool(self.get_parameter("verify_calibration").value):
            self.get_logger().warn(
                "verify_calibration=false — 교시 자세가 유효한지 확인하지 않습니다")
            return

        # 단발 읽기로는 안 된다. 이 버스는 패킷을 이따금 흘리고, 서보 6개
        # 연속 읽기라 묶음이 깨질 확률이 그만큼 쌓인다(_read_with_retry 주석
        # 참고). 재시도가 없으면 패킷 하나 유실이 그대로 기동 거부가 되는데,
        # 그건 이 검사가 막으려는 위험과 아무 상관이 없는 실패다.
        current = {
            servo_id: self._read_with_retry(backend.drv.get_homing_offset, servo_id)
            for servo_id in sorted(TAUGHT_HOMING_OFFSETS)
        }
        result = calib_identity.verdict(current, TAUGHT_HOMING_OFFSETS)
        if result.ok:
            self.get_logger().info(f"캘리브레이션 확인 — {result.message()}")
            return
        raise ArmCalibrationMismatchError(result.message())

    def _check_startup_torque(
        self,
        backend,
        *,
        enable_torque_on_start: bool,
    ) -> None:
        """기동 시 6개 서보의 통신·torque 상태를 확인한다."""
        states = {servo_id: backend.drv.get_torque(servo_id) for servo_id in ALL_SERVO_IDS}

        unreadable = [servo_id for servo_id, enabled in states.items() if enabled is None]
        if unreadable:
            raise ArmHardwareUnavailableError(
                "SO-ARM101 torque 상태를 읽지 못했습니다 — " f"servo IDs: {unreadable}"
            )

        disabled = [servo_id for servo_id, enabled in states.items() if not enabled]
        if not disabled:
            self.get_logger().info("SO-ARM101 torque 상태 정상 — all enabled")
            return

        self.get_logger().warn(f"torque OFF servo IDs: {disabled}")

        if not enable_torque_on_start:
            self.get_logger().warn(
                "enable_torque_on_start=False — 자동 torque enable을 수행하지 않습니다"
            )
            return

        self.get_logger().warn(
            "enable_torque_on_start=True — 1초 후 전체 servo torque를 활성화합니다"
        )
        time.sleep(1.0)

        enable_failed = [
            servo_id for servo_id in ALL_SERVO_IDS if not backend.drv.set_torque(servo_id, True)
        ]
        if enable_failed:
            raise ArmHardwareUnavailableError(
                "SO-ARM101 torque enable 명령 실패 — " f"servo IDs: {enable_failed}"
            )

        still_disabled = [
            servo_id for servo_id in ALL_SERVO_IDS if backend.drv.get_torque(servo_id) is not True
        ]
        if still_disabled:
            raise ArmHardwareUnavailableError(
                "torque enable 후 상태 확인 실패 — " f"servo IDs: {still_disabled}"
            )

        self.get_logger().info("SO-ARM101 torque enable 완료")

    def _log_idle_offset(self, backend) -> None:
        """기동 시 현재 자세와 IDLE 사이 편차를 로그로만 남긴다 — 절대 서보를
        움직이지 않는다.

        ⚠️ 여기서 움직이지 않는 것은 여전히 의도다. 실제 정렬은 **첫 이동
        요청 시점**에 _auto_align_to_idle이 한다(2026-08-25). 노드가 뜨자마자
        팔이 움직이면 운영자가 아직 팔에 손을 대고 있을 수 있어서, 정렬은
        누군가 실제로 움직이라고 요청한 순간으로 미룬다. 이 로그는 그 전에
        "지금 얼마나 벗어나 있는지"를 남기는 기록이다.

        use_fake_arm 파라미터가 로그가 없어 며칠간 묻힌 사고(#128)의 재발
        방지책이다: 안전 관련 상태는 반드시 실제 값을 로그로 남긴다."""
        targets = {servo_id: IDLE_CRADLE_RAW[servo_id - 1] for servo_id in range(1, 6)}
        targets[GRIPPER_SERVO_ID] = position_from_width(GRIPPER_CLOSED_MM)

        offsets = {}
        unreadable = []
        for servo_id, target in targets.items():
            present = backend.drv.get_position(servo_id)
            if present is None:
                unreadable.append(servo_id)
                continue
            offsets[servo_id] = present - target

        if unreadable:
            self.get_logger().error(
                f"IDLE offset 확인 실패 — present position 읽기 불가 servo IDs: {unreadable}"
            )
        if not offsets:
            return

        summary = " ".join(f"s{sid}={offsets[sid]:+d}" for sid in sorted(offsets))

        breaches = {sid: off for sid, off in offsets.items() if abs(off) > IDLE_OFFSET_WARN_RAW}
        if not breaches:
            self.get_logger().info(f"IDLE offset: {summary} -> OK")
            return

        self.get_logger().warn(f"IDLE offset: {summary}")
        for servo_id in sorted(breaches):
            offset = breaches[servo_id]
            if abs(offset) > IDLE_OFFSET_ERROR_RAW:
                self.get_logger().error(
                    f"ERROR: s{servo_id} offset {offset:+d} exceeds {IDLE_OFFSET_ERROR_RAW}. "
                    "첫 이동 요청 때 자동 정렬이 시도됩니다"
                    "(auto_align_on_first_move=False면 tools/align_to_idle.py를 먼저 돌리세요)."
                )
            else:
                self.get_logger().warn(
                    f"WARN: s{servo_id} offset {offset:+d} exceeds {IDLE_OFFSET_WARN_RAW}. "
                    "첫 이동 요청 때 자동 정렬이 시도됩니다."
                )

    def _require_operational_servos(self, servo_ids=ALL_SERVO_IDS) -> None:
        """모션 전후 대상 서보가 통신 가능하고 torque ON인지 확인한다."""
        backend = soarm._backend(real=True)

        unavailable = []
        torque_off = []

        for servo_id in servo_ids:
            enabled = backend.drv.get_torque(servo_id)
            if enabled is None:
                unavailable.append(servo_id)
            elif not enabled:
                torque_off.append(servo_id)

        if unavailable:
            raise ArmHardwareUnavailableError(
                "SO-ARM101 servo 통신 실패 — " f"servo IDs: {unavailable}"
            )

        if torque_off:
            raise ArmHardwareUnavailableError("SO-ARM101 torque OFF — " f"servo IDs: {torque_off}")

    def _reject_base_board_port(self, arm_port: str) -> None:
        """arm_port가 MentorPi 베이스 보드와 같은 장치를 가리키면 노드를 띄우지 않는다.

        udev 규칙에 따라 /dev/rrc 는 /dev/ttyACM0 같은 실제 장치를 가리키는 심볼릭
        링크라, 경로 문자열만 비교하면 충돌을 놓친다 — realpath로 양쪽을 풀어서
        같은 장치인지 본다.

        /dev/rrc 가 없는 환경(개발 머신, CI, 시뮬레이션)에서는 비교할 대상이 없으니
        검사를 건너뛴다 — 베이스 보드가 없다면 충돌할 것도 없다."""
        if not os.path.exists(BASE_BOARD_DEVICE):
            return
        if os.path.realpath(arm_port) != os.path.realpath(BASE_BOARD_DEVICE):
            return
        raise ArmPortConflictError(
            f"arm_port({arm_port})가 {BASE_BOARD_DEVICE}(MentorPi 베이스 보드)와 같은 "
            f"장치입니다. SO-ARM101 포트를 확인하세요 — 이 상태로 진행하면 베이스 보드 "
            f"시리얼 통신이 깨집니다."
        )

    def _execute_move(self, goal_handle):
        req = goal_handle.request
        xyz = [req.target.x, req.target.y, req.target.z]
        result = MoveToCartesian.Result()
        try:
            self._require_operational_servos(range(1, 6))

            # grip=None — 그리퍼는 이 액션이 건드리지 않는다. GRASP는
            # move_to_cartesian()과 set_gripper()를 분리 호출한다
            # (state_machine.md §3 GRASP 계약).
            angles_deg, err = soarm.go(
                xyz,
                grip=None,
                real=True,
                down=req.down,
                secs=1.2,
            )
            time.sleep(1.2)  # RealBackend.move는 즉시 반환하므로 정착 시간만큼 대기

            # 명령 직후 USB가 빠지거나 torque가 꺼진 경우도 성공으로 보고하지 않는다.
            # vendored RealBackend.move()는 servo write 결과를 반환하지 않으므로
            # ROS2 경계에서 통신 상태를 다시 확인한다.
            self._require_operational_servos(range(1, 6))

            # ⚠️ MoveToCartesian.action의 Result에는 distance_remaining이 없다
            # (Feedback에만 있음) — IK 잔차는 로그로만 남긴다.
            self.get_logger().info(f"move_to_cartesian 완료 — IK 잔차 {err * 1000:.1f}mm")
            result.reached = True
            goal_handle.succeed()
        except ValueError as e:
            # Arm.go()가 IK 잔차 초과 시 던지는 예외 — "팔 범위 밖" 같은 정상적인
            # 실패 경로다.
            self.get_logger().warn(f"도달 불가: {e}")
            result.reached = False
            goal_handle.abort()
        except Exception as e:
            # 시리얼 연결 끊김 등 하드웨어 예외 — 노드가 죽으면 안 되므로 여기서
            # 잡아 실패 응답으로 돌린다.
            self.get_logger().error(f"move_to_cartesian 하드웨어 오류: {e}")
            result.reached = False
            goal_handle.abort()
        return result

    def _execute_reorient(self, goal_handle):
        # TODO: 손목 φ 재조정 IK/모션 로직. POSE_PLAN이 아직 ⏸ 보류라 phi는
        # 지금 항상 0.0으로 들어온다(domain/task/states.py PosePlanState.
        # _solve_phi). soarm_lab에는 손목 단독 회전 프리미티브가 없어 φ≠0을
        # 실제로 지원하려면 이 노드에서 직접 φ 제약을 포함한 IK를 풀어야
        # 한다 — 재도입 시 구현. 지금은 정직하게 스텁으로 항상 성공 응답.
        phi = goal_handle.request.phi
        self.get_logger().warn(
            f"reorient(phi={phi:.3f}rad): 손목 재조정 미구현 — settled=True로 스텁 응답"
        )
        result = ReorientArm.Result()
        result.settled = True
        result.current_phi = phi
        result.wrist_load = self._read_load(WRIST_SERVO_ID)
        goal_handle.succeed()
        return result

    def _execute_floor_pose(self, goal_handle):
        req = goal_handle.request
        result = MoveToFloorPose.Result()
        try:
            if req.profile not in HORIZONTAL_GRASP_POSES_DEG:
                raise ValueError(f"알 수 없는 수평 파지 profile: {req.profile}")
            if req.stage not in {"idle", "carry", "safe", "grasp", "midpoint", "drop", "recover_idle"}:
                raise ValueError(f"알 수 없는 수평 파지 stage: {req.stage}")

            # 세션 첫 이동이면 여기서 IDLE로 자동 정렬한다(사용자 지시,
            # 2026-08-25). _require_operational_servos보다 **앞**에 와야
            # 한다 — 정렬이 필요한 전형적 상황이 전원 투입 직후 torque OFF
            # 상태이고, 그건 정확히 저 검사가 막는 상태다. 정렬 자체가
            # goal<-present latch로 torque를 켜므로 순서를 뒤집으면 자동
            # 정렬이 영영 불려 오지 못한다.
            #
            # recover_idle은 제외한다 — 그쪽은 이미 같은 일을 하는 복구
            # 경로라, 앞에 정렬을 한 번 더 붙이면 같은 이동을 두 번 한다.
            if self._auto_align_pending and req.stage != "recover_idle":
                self._auto_align_to_idle()
                self._auto_align_pending = False

            self._require_operational_servos(range(1, 6))
            backend = soarm._backend(real=True)
            # 내려가기 직전에만 온도 상한을 적용한다. 물체를 든 뒤 온도가
            # 상한을 넘었더라도 midpoint/safe 상승은 막지 않아야 바닥 가까이에
            # 물체를 든 채 정지하는 더 위험한 상태를 피할 수 있다.
            if req.stage == "grasp":
                servo2_temp = backend.drv.get_temperature(2)
                if servo2_temp is None or servo2_temp > MAX_FLOOR_POSE_SERVO2_TEMP_C:
                    raise ArmHardwareUnavailableError(
                        f"servo 2 온도 {servo2_temp}°C — 수평 파지 시작 상한 "
                        f"{MAX_FLOOR_POSE_SERVO2_TEMP_C}°C"
                    )

            self._move_floor_stage(backend, req.profile, req.stage)
            self._require_operational_servos(range(1, 6))
            result.reached = True
            goal_handle.succeed()
        except ValueError as e:
            self.get_logger().warn(f"수평 파지 자세 거부: {e}")
            result.reached = False
            goal_handle.abort()
        except Exception as e:
            self.get_logger().error(f"수평 파지 자세 하드웨어 오류: {e}")
            result.reached = False
            goal_handle.abort()
        return result

    # ── VLA 관절 청크 재생 ──────────────────────────────────────────────────

    def _load_policy_calibration(self, backend):
        """정책이 학습된 LeRobot 캘리브레이션 -> servo별 변환표. 없으면 None.

        ⚠️ 이 노드의 raw 와 정책의 raw 는 같은 물리 자세에서도 값이 다르다.
        Feetech 는 ``Present_Position = Actual_Position - Homing_Offset`` 이고
        (lerobot feetech.py `_get_half_turn_homings` 의 docstring), 정책 학습에
        쓴 LeRobot 캘리브레이션과 지금 서보에 올라간 Homing_Offset 이 다르기
        때문이다. 두 세계가 같은 팔을 보므로 차이는 관절마다 **상수**이고,
        그것이 아래 delta 다.

        ⚠️ 기준을 TAUGHT_HOMING_OFFSETS 상수가 아니라 **서보 레지스터 실측값**
        으로 잡는다. 오프셋은 서보 EEPROM 에 있지 git 에 있지 않아서
        (_check_taught_calibration 주석), 코드가 뭐라고 적혀 있든 실제로 무엇이
        올라가 있느냐만이 raw 의 뜻을 정한다. 실측을 쓰면 팔이 이미 LeRobot
        캘리브레이션을 들고 있을 때 delta 가 저절로 0 이 되고, 교시본을 복원하면
        저절로 그만큼 붙는다 — 어느 쪽이든 맞는다.
        """
        path = str(self.get_parameter("policy_calibration_file").value or "").strip()
        if not path:
            self.get_logger().info(
                "policy_calibration_file 미설정 — ExecuteJointChunk(VLA 재생)는 거부됩니다."
            )
            return None

        try:
            with open(os.path.expanduser(path), encoding="utf-8") as handle:
                entries = json.load(handle)
        except (OSError, ValueError) as e:
            self.get_logger().error(f"정책 캘리브레이션을 읽지 못했습니다({path}): {e}")
            return None

        table = {}
        try:
            for name, entry in entries.items():
                servo_id = int(entry["id"])
                low, high = int(entry["range_min"]), int(entry["range_max"])
                if high <= low:
                    raise ValueError(f"{name}: range_min({low}) >= range_max({high})")
                # 단발 읽기로는 안 된다 — 이 버스는 패킷을 이따금 흘린다
                # (_check_taught_calibration 주석 참고).
                live = self._read_with_retry(backend.drv.get_homing_offset, servo_id)
                if live is None:
                    raise ValueError(f"{name}: servo {servo_id} 의 Homing_Offset 읽기 실패")
                table[servo_id] = {
                    "name": name,
                    "min": low,
                    "max": high,
                    "delta": int(entry["homing_offset"]) - int(live),
                }
        except (KeyError, TypeError, ValueError) as e:
            self.get_logger().error(f"정책 캘리브레이션 형식 오류({path}): {e}")
            return None

        missing = [servo_id for servo_id in ALL_SERVO_IDS if servo_id not in table]
        if missing:
            self.get_logger().error(f"정책 캘리브레이션에 servo {missing} 가 없습니다({path})")
            return None

        summary = ", ".join(
            f"s{servo_id} {table[servo_id]['delta']:+d}" for servo_id in ALL_SERVO_IDS
        )
        self.get_logger().info(f"정책 캘리브레이션 적재({path}) — 좌표계 보정(raw): {summary}")
        return table

    def _policy_to_taught_raw(self, values):
        """LeRobot 정규화 값 6개 -> 이 노드의 raw 목표 6개.

        정규화 규칙은 lerobot ``MotorsBus._unnormalize`` 를 그대로 따른다 —
        servo 1..5 는 DEGREES, servo 6 은 RANGE_0_100 이다
        (so_follower.py: `use_degrees` 기본값이 True).

        ⚠️ 관절에는 가동범위 클램프를 **일부러 걸지 않는다.** lerobot 도 안 건다
        (DEGREES 분기에는 bounded_val 이 없다). 여기서만 range_min/max 로 자르면
        실기로 검증한 롤아웃과 동작이 달라지고, 특히 wrist_roll 은 기록된 범위가
        14 raw(약 1.2도)뿐이라 사실상 관절이 고정돼 버린다. 절대 한계는 서보의
        Min/Max_Position_Limit 레지스터가 잡는다 —
        tools/arm/backup_servo_offsets.py 가 복원하는 그 값이고, 롤아웃이 기대던
        보호 장치도 같은 것이다. 스텝당 이동량은 호출부의 max_step_deg 가 묶는다.
        """
        table = self._policy_calibration
        goal = {}
        for servo_id in ALL_SERVO_IDS:
            entry = table[servo_id]
            low, high = entry["min"], entry["max"]
            value = float(values[servo_id - 1])
            if servo_id == GRIPPER_SERVO_ID:
                bounded = min(100.0, max(0.0, value))
                policy_raw = (bounded / 100.0) * (high - low) + low
            else:
                mid = (low + high) / 2.0
                policy_raw = value * STS3215_RESOLUTION_MINUS_1 / 360.0 + mid
            goal[servo_id] = int(policy_raw) + entry["delta"]
        return goal

    def _taught_raw_to_policy(self, raw):
        """이 노드의 raw 6개 -> LeRobot 정규화 값 6개. _policy_to_taught_raw 의 역이다.

        두 방향이 정확히 서로의 역이어야 한다 — 정책은 자기가 낸 명령을 다음
        관측으로 다시 본다. 한쪽만 어긋나면 정책이 "명령대로 안 갔다"고 읽고
        같은 방향으로 계속 더 밀어붙인다.
        """
        table = self._policy_calibration
        values = []
        for servo_id in ALL_SERVO_IDS:
            entry = table[servo_id]
            low, high = entry["min"], entry["max"]
            policy_raw = float(raw[servo_id - 1] - entry["delta"])
            if servo_id == GRIPPER_SERVO_ID:
                value = (policy_raw - low) / (high - low) * 100.0
            else:
                value = (policy_raw - (low + high) / 2.0) * 360.0 / STS3215_RESOLUTION_MINUS_1
            values.append(float(value))
        return values

    def _on_get_policy_state(self, request, response):
        del request
        try:
            if self._policy_calibration is None:
                raise ValueError("policy_calibration_file 이 없어 정책 좌표계로 못 바꿉니다")
            backend = soarm._backend(real=True)
            raw = self._read_joint_positions(backend)
            if raw is None:
                raise ArmHardwareUnavailableError("관절 위치 읽기 실패")
            gripper_raw = backend.drv.get_position(GRIPPER_SERVO_ID)
            if gripper_raw is None:
                raise ArmHardwareUnavailableError("그리퍼 위치 읽기 실패")
            ordered = [raw[servo_id] for servo_id in range(1, 6)] + [gripper_raw]
            response.positions = self._taught_raw_to_policy(ordered)
            response.ok = True
            response.message = ""
        except ValueError as e:
            self.get_logger().warn(f"정책 관절 상태 거부: {e}")
            response.ok = False
            response.message = str(e)
        except Exception as e:
            self.get_logger().error(f"정책 관절 상태 하드웨어 오류: {e}")
            response.ok = False
            response.message = str(e)
        return response

    def _execute_joint_chunk(self, goal_handle):
        req = goal_handle.request
        result = ExecuteJointChunk.Result()
        feedback = ExecuteJointChunk.Feedback()
        sent = 0
        try:
            if self._policy_calibration is None:
                raise ValueError(
                    "policy_calibration_file 이 없어 정책 좌표계를 해석할 수 없습니다 — "
                    "추측해서 재생하면 servo 5 가 약 86도 어긋납니다"
                )
            if req.fps <= 0.0:
                raise ValueError(f"fps 는 양수여야 합니다: {req.fps}")
            count = len(req.positions)
            if count == 0 or count % len(ALL_SERVO_IDS) != 0:
                raise ValueError(f"positions 길이는 6의 배수여야 합니다: {count}")
            steps = count // len(ALL_SERVO_IDS)
            if steps > VLA_MAX_STEPS:
                raise ValueError(f"청크가 너무 깁니다: {steps} 스텝 (상한 {VLA_MAX_STEPS})")

            max_step_deg = (
                float(req.max_step_deg) if req.max_step_deg > 0.0 else VLA_MAX_STEP_DEG_DEFAULT
            )
            max_step_raw = max(1, round(max_step_deg * STS3215_RESOLUTION_MINUS_1 / 360.0))

            self._require_operational_servos(ALL_SERVO_IDS)
            backend = soarm._backend(real=True)

            # ⚠️ 재생 시작마다 명시적으로 다시 쓴다 — 앞서 무엇이 돌았든
            # 상관없게(VLA_SPEED_RAW 주석의 2026-09-02 사례).
            for servo_id in ALL_SERVO_IDS:
                backend.drv.set_speed(servo_id, VLA_SPEED_RAW)
                backend.drv.set_acceleration(servo_id, VLA_ACCEL_RAW)

            # 스텝 제한의 기준점은 처음 한 번만 실제로 읽고, 그 뒤로는 우리가
            # 보낸 목표를 이어 쓴다. 30Hz 재생 중에 매 스텝 서보 6개를 되읽으면
            # 시리얼이 못 버티고, 늦은 읽기는 그 자체로 궤적을 흔든다.
            #
            # 명령을 기준으로 삼는 것이 안전한 이유는 Goal_Velocity 가 0(무제한)
            # 이라 팔이 뒤처지지 않기 때문이다. 저 값을 묶으면 이 가정이 깨진다.
            start = self._read_joint_positions(backend)
            if start is None:
                raise ArmHardwareUnavailableError("시작 관절 위치 읽기 실패")
            last = dict(start)
            gripper_raw = backend.drv.get_position(GRIPPER_SERVO_ID)
            if gripper_raw is None:
                raise ArmHardwareUnavailableError("그리퍼 위치 읽기 실패")
            last[GRIPPER_SERVO_ID] = gripper_raw

            clamped_steps = 0
            period = 1.0 / float(req.fps)
            started_at = time.monotonic()

            for step in range(steps):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.ok = False
                    result.steps_executed = sent
                    result.message = f"취소됨 — {sent}/{steps} 스텝 재생"
                    self.get_logger().warn(f"관절 청크 {result.message}")
                    return result

                offset = step * len(ALL_SERVO_IDS)
                goal = self._policy_to_taught_raw(req.positions[offset:offset + len(ALL_SERVO_IDS)])
                step_clamped = False
                for servo_id in ALL_SERVO_IDS:
                    move = goal[servo_id] - last[servo_id]
                    if move > max_step_raw:
                        goal[servo_id] = last[servo_id] + max_step_raw
                        step_clamped = True
                    elif move < -max_step_raw:
                        goal[servo_id] = last[servo_id] - max_step_raw
                        step_clamped = True
                    if not backend.drv.set_position(servo_id, goal[servo_id]):
                        raise ArmHardwareUnavailableError(
                            f"servo {servo_id} write 실패 — step {step + 1}/{steps}"
                        )
                    last[servo_id] = goal[servo_id]
                clamped_steps += int(step_clamped)
                sent = step + 1

                feedback.step = step
                goal_handle.publish_feedback(feedback)

                # 자기 시계로 재생한다. 스텝마다 period 를 통째로 자면 쓰기에 든
                # 시간이 누적돼 청크가 요청보다 길어지고, 정책이 가정한 fps 와
                # 어긋난다.
                remaining = started_at + (step + 1) * period - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

            self._require_operational_servos(ALL_SERVO_IDS)
            elapsed = time.monotonic() - started_at
            result.ok = True
            result.steps_executed = sent
            result.message = (
                f"{sent} 스텝 재생 {elapsed:.2f}s (요청 {steps / req.fps:.2f}s), "
                f"스텝 제한 걸린 스텝 {clamped_steps}"
            )
            # 제한이 자주 걸렸다는 것은 정책이 max_step_deg 보다 빠르게 움직이려
            # 했다는 뜻이다 — 궤적이 눌려서 느려지고, 그만큼 정책 의도와 벌어진다.
            if clamped_steps > steps // 2:
                self.get_logger().warn(
                    f"관절 청크: {clamped_steps}/{steps} 스텝이 max_step_deg="
                    f"{max_step_deg:.1f}도에 걸렸습니다 — 궤적이 눌리고 있습니다"
                )
            goal_handle.succeed()
        except ValueError as e:
            self.get_logger().warn(f"관절 청크 거부: {e}")
            result.ok = False
            result.message = str(e)
            result.steps_executed = sent
            goal_handle.abort()
        except Exception as e:
            self.get_logger().error(f"관절 청크 하드웨어 오류: {e}")
            result.ok = False
            result.message = str(e)
            result.steps_executed = sent
            goal_handle.abort()
        return result

    def _glide_to_joint_angles(self, backend, angles_deg) -> None:
        """실측 시험과 같은 선형 waypoint로 servo 1..5를 천천히 이동한다."""
        goal = {
            servo_id: backend.drv.degrees_to_position(angles_deg[servo_id - 1])
            for servo_id in range(1, 6)
        }
        self._glide_to_raw_positions(backend, goal)

    @staticmethod
    def _tolerance_for(tolerance_raw, servo_id: int) -> int:
        """관절 하나의 허용오차. 정수면 전 관절 공통, dict면 관절별이다.

        dict에 없는 관절은 기본값을 쓴다 — 미세 이동이 필요한 관절만
        따로 조이고 나머지는 원래 게이트를 그대로 두려는 것이다. 전
        관절을 같이 조이면, 실제로 움직이지 않는 관절(중력을 받는 어깨
        같은)의 몇 raw짜리 처짐까지 실패로 잡는다."""
        if isinstance(tolerance_raw, dict):
            return tolerance_raw.get(servo_id, FLOOR_POSE_START_TOLERANCE_RAW)
        return tolerance_raw

    def _glide_to_raw_positions(
        self, backend, goal, defer_joints=(), speed_raw=FLOOR_POSE_SPEED_RAW,
        tolerance_raw=FLOOR_POSE_START_TOLERANCE_RAW, steps=FLOOR_POSE_STEPS
    ) -> None:
        """servo 1..5 raw 목표로 이동한다.

        defer_joints에 든 관절은 **나머지가 완전히 멈춘 뒤** 별도 구간에서
        단독으로 움직인다. 기본값은 빈 튜플 — 즉 아무 데도 안 주면 예전과
        똑같은 한 번의 선형 보간이다.

        ⚠️ 지연은 **방향마다 다르게** 걸어야 한다. 전역 설정으로 모든 이동에
        걸었더니 IDLE로 돌아오는 쪽은 고쳐졌는데 IDLE에서 나가는 쪽이 새로
        긁기 시작했다(2026-08-24 실기). 어느 쪽으로 가느냐에 따라 안전한
        관절 순서가 반대이기 때문이다 — 그래서 호출부가 정한다."""
        deferred = tuple(servo_id for servo_id in defer_joints if servo_id in goal)
        if not deferred:
            self._glide_phase(backend, goal, speed_raw=speed_raw,
                              tolerance_raw=tolerance_raw, steps=steps)
            return

        start = self._read_joint_positions(backend)
        if start is None:
            raise ArmHardwareUnavailableError("시작 관절 위치 읽기 실패")

        # 1구간: 지연 관절은 출발 위치에 **고정**하고 나머지만 목표로.
        hold_deferred = {**goal, **{servo_id: start[servo_id] for servo_id in deferred}}
        self._glide_phase(backend, hold_deferred, label="1/2 지연관절 고정", speed_raw=speed_raw,
                          tolerance_raw=tolerance_raw, steps=steps)
        # 2구간: 나머지는 이미 목표에 있으므로 지연 관절만 실제로 움직인다.
        self._glide_phase(
            backend, goal, label=f"2/2 servo {list(deferred)} 단독", speed_raw=speed_raw,
            tolerance_raw=tolerance_raw, steps=steps
        )

    def _glide_phase(self, backend, goal, label=None, speed_raw=FLOOR_POSE_SPEED_RAW,
                     tolerance_raw=FLOOR_POSE_START_TOLERANCE_RAW,
                     steps=FLOOR_POSE_STEPS) -> None:
        """한 번의 선형 보간 구간 — 목표에 이미 있으면 아무것도 하지 않는다.

        "이미 있다"의 기준은 `tolerance_raw`다. 미세 이동을 시키려면 이
        값을 같이 줄여야 한다 — 안 그러면 이동이 조용히 버려진다
        (BASE_YAW_TOLERANCE_RAW 주석의 2026-08-28 사례)."""
        start = self._read_joint_positions(backend)
        if start is None:
            raise ArmHardwareUnavailableError("시작 관절 위치 읽기 실패")

        moving = {
            servo_id: goal[servo_id] - start[servo_id]
            for servo_id in range(1, 6)
            if abs(goal[servo_id] - start[servo_id])
            > self._tolerance_for(tolerance_raw, servo_id)
        }
        if not moving:
            return  # 이미 도착해 있다 — 빈 구간에 시간을 쓰지 않는다
        if label is not None:
            self.get_logger().info(f"glide[{label}]: 이동 관절 {moving}")

        # ⚠️ 반드시 매 이동마다 명시적으로 다시 쓴다 — 상속하면 안 된다.
        # STS3215의 goal_speed는 서보 레지스터에 남는 상태값이라(driver_sdk의
        # set_position docstring: "Speed/acceleration are set separately and
        # cached by the controller"), 이 노드가 안 쓰면 **마지막으로 누가
        # 무슨 값을 썼는지에 따라** 팔 속도가 조용히 달라진다.
        #
        # 2026-08-24 실기에서 정확히 이 일이 났다: tools/align_to_idle.py가
        # 의도적으로 느린 SPEED_RAW=150을 쓰고 그 값이 레지스터에 남아,
        # 이후 arm_driver의 IDLE->safe 이동(servo 2가 1663 raw)이 3.0s 글라이드
        # 안에 끝나지 못했다. 실측 속도는 servo 2 = 153 raw/s, servo 4 =
        # 151 raw/s로 레지스터 값에 정확히 붙어 있었고, 도달 대기 4.0s까지
        # 다 쓰고도 각각 591 / 564 raw가 남아 safe 단계가 실패했다.
        for servo_id in range(1, 6):
            backend.drv.set_speed(servo_id, speed_raw)
            backend.drv.set_acceleration(servo_id, FLOOR_POSE_ACCEL_RAW)

        for step_index in range(1, steps + 1):
            ratio = step_index / steps
            for servo_id in range(1, 6):
                position = round(start[servo_id] + ratio * (goal[servo_id] - start[servo_id]))
                if not backend.drv.set_position(servo_id, position):
                    raise ArmHardwareUnavailableError(
                        f"servo {servo_id} write 실패 — step {step_index}/{steps}"
                    )
            time.sleep(FLOOR_POSE_STEP_SEC)

        self._wait_floor_pose_arrived(backend, goal, tolerance_raw=tolerance_raw)

    def _wait_floor_pose_arrived(self, backend, goal,
                                 tolerance_raw=FLOOR_POSE_START_TOLERANCE_RAW) -> None:
        """보간이 끝난 뒤 servo 1..5가 실제로 goal에 도달할 때까지 기다린다.

        ⚠️ 2026-08-24 실기에서 확인한 문제: 예전에는 여기서 그냥
        ``time.sleep(FLOOR_POSE_SETTLE_SEC)``만 하고 끝냈다. 보간 waypoint를
        다 써 넣었다는 것은 "명령을 다 보냈다"는 뜻일 뿐 "팔이 그 자세에
        도달했다"는 뜻이 아닌데, 그 상태로 result.reached=True를 돌려주고
        있었다. 그래서 safe 단계는 성공했다고 보고해 놓고, 곧이어 들어온
        grasp 단계가 ``_near_pose(actual, safe)``(±120 raw)에서 떨어져
        "grasp 이동 시작 자세가 등록된 이전 단계가 아닙니다"로 거부됐다
        (/tmp/arm.log 1787562322). 즉 실패 지점과 실패 원인이 서로 다른
        단계에 있어 로그만 봐서는 원인을 알 수 없었다.

        도달 판정 기준은 ``FLOOR_POSE_START_TOLERANCE_RAW``로 다음 단계의
        시작 자세 게이트와 **같은 값을 쓴다** — 이 함수가 통과시킨 자세는
        정의상 다음 단계가 받아들이는 자세여야 하기 때문이다.

        ⚠️ 기다리는 방식은 **고정 시간이 아니라 진전 기준**이다. 2026-08-24
        실기에서 고정 4.0s로 했다가 servo 2만 계속 592 raw를 남기고 실패했는데,
        타임아웃 뒤에 확인해 보니 목표에서 +5 raw에 도착해 있었다 — 멈춘 게
        아니라 **느렸을 뿐**이었다. 어깨(servo 2)는 팔 전체를 중력에 맞서
        들어올리므로 goal_speed를 1200으로 올려도 실측 153 raw/s가 한계였다
        (같은 거리를 움직이는 servo 4는 230 raw/s로 빨라졌다). 즉 이 관절의
        속도는 명령이 아니라 토크가 정한다. IDLE->safe의 1663 raw는 그
        속도로 10.9s가 걸리는데 예산은 7.0s였다.

        고정 상한을 그 10.9s에 맞춰 늘리는 건 답이 아니다 — 배터리 전압이나
        적재 무게가 달라지면 그 수치도 같이 달라지기 때문이다. 그래서 **잔차가
        줄고 있는 동안에는 계속 기다리고**, 진전이 FLOOR_POSE_STALL_SEC 동안
        멈추면 그때 실패로 본다. 진짜로 걸린 경우에는 여전히 빨리 실패하고,
        느릴 뿐인 경우에는 끝까지 기다린다. FLOOR_POSE_ARRIVE_MAX_SEC는 어떤
        경우에도 영원히 매달리지 않게 하는 최후의 한계선이다.
        """
        started = time.monotonic()
        stall_deadline = started + FLOOR_POSE_STALL_SEC
        hard_deadline = started + FLOOR_POSE_ARRIVE_MAX_SEC
        best_error = None
        residual = {}
        while True:
            actual = self._read_joint_positions(backend)
            if actual is None:
                # 폴링 중 한 프레임을 놓친 것뿐일 수 있다 — 다음 폴에서 다시
                # 본다. 정말 통신이 끊겼다면 아래 stall/hard 마감이 잡는다.
                if time.monotonic() >= hard_deadline:
                    raise ArmHardwareUnavailableError(
                        f"{FLOOR_POSE_ARRIVE_MAX_SEC}s 동안 관절 위치를 읽지 못했습니다"
                    )
                time.sleep(FLOOR_POSE_ARRIVE_POLL_SEC)
                continue
            residual = {
                servo_id: actual[servo_id] - goal[servo_id] for servo_id in range(1, 6)
            }
            if all(abs(error) <= self._tolerance_for(tolerance_raw, servo_id)
                   for servo_id, error in residual.items()):
                return

            now = time.monotonic()
            worst_error = max(abs(error) for error in residual.values())
            if best_error is None or best_error - worst_error > FLOOR_POSE_PROGRESS_RAW:
                # 아직 줄고 있다 — 느린 것이지 걸린 게 아니다. 시계를 다시 준다.
                best_error = worst_error
                stall_deadline = now + FLOOR_POSE_STALL_SEC
            if now >= stall_deadline or now >= hard_deadline:
                break
            time.sleep(FLOOR_POSE_ARRIVE_POLL_SEC)

        waited = time.monotonic() - started
        worst = max(residual, key=lambda servo_id: abs(residual[servo_id]))
        reason = "진전이 멈췄습니다" if waited < FLOOR_POSE_ARRIVE_MAX_SEC else "최대 대기 시간을 넘겼습니다"
        raise ArmHardwareUnavailableError(
            f"{waited:.1f}s 기다렸으나 목표 자세에 도달하지 못했습니다({reason}) — "
            f"잔차(raw) {residual}, 최악 servo {worst} {residual[worst]:+d} "
            f"(허용 ±{self._tolerance_for(tolerance_raw, worst)})"
        )

    @staticmethod
    def _read_joint_positions(backend, attempts=JOINT_READ_ATTEMPTS):
        """servo 1..5 위치를 읽는다 — 하나라도 못 읽으면 None.

        ⚠️ 2026-08-24 실기: 이동 중 폴링에서 servo 3만 한 번 None이 나와
        (``{1: 2071, 2: 2488, 3: None, 4: 2598, 5: 3056}``) 복구 이동이
        **한복판에서** 예외로 끊겼다. 팔은 이미 움직이던 중이라 어중간한
        자세에 멈춰 섰다 — 즉 한 번의 시리얼 패킷 유실이 팔을 오도가도
        못하게 만들었다. 읽기 실패는 그 자체로는 하드웨어 고장이 아니므로
        몇 번 다시 시도하고, 그래도 안 되면 호출부가 판단하게 None을 준다.
        """
        for attempt in range(attempts):
            positions = {servo_id: backend.drv.get_position(servo_id) for servo_id in range(1, 6)}
            if all(position is not None for position in positions.values()):
                return positions
            if attempt + 1 < attempts:
                time.sleep(JOINT_READ_RETRY_SEC)
        return None

    def _raw_goals(self, backend, angles_deg):
        return {
            servo_id: backend.drv.degrees_to_position(angles_deg[servo_id - 1])
            for servo_id in range(1, 6)
        }

    @staticmethod
    def _tuple_goals(raw_positions):
        return {servo_id: raw_positions[servo_id - 1] for servo_id in range(1, 6)}

    @staticmethod
    def _near_pose(actual, expected):
        return all(
            abs(actual[servo_id] - expected[servo_id]) <= FLOOR_POSE_START_TOLERANCE_RAW
            for servo_id in range(1, 6)
        )

    def _move_floor_stage(self, backend, profile, stage) -> None:
        """검증된 자세 사이에서만 움직인다. 수평 IDLE↔SAFE는 직접 전환한다.

        servo1(Base)은 safe/grasp/midpoint 사이를 오가는 동안 절대 움직이지
        않는다 — APPROACH 단계가 이미 물체를 정면에 맞춰 놨는데, 여기서
        HORIZONTAL_SAFE_145_RAW/HORIZONTAL_GRASP_POSES_DEG에 박혀 있는
        servo1 절대값으로 다시 돌리면 그 정렬이 깨진다(사용자 지시,
        2026-08-24). idle/drop은 그대로 등록된 절대 servo1 값을 쓴다 —
        idle은 물체를 다 옮기고 나서의 중립 자세(CARRY_IDLE)라 실제로
        정면 정렬 값(IDLE_CRADLE_RAW)으로 되돌아가야 하고, drop도 별개의
        고정 전달 자세이기 때문이다."""
        actual = self._read_joint_positions(backend)
        if actual is None:
            raise ArmHardwareUnavailableError("현재 관절 위치 읽기 실패")

        frozen_servo1 = actual[1]

        def _freeze_servo1(goals):
            return {**goals, 1: frozen_servo1}

        idle = self._tuple_goals(IDLE_CRADLE_RAW)
        carry = self._tuple_goals(CARRY_RAW)
        drop = self._tuple_goals(BASKET_DROP_195_RAW)
        safe = _freeze_servo1(self._tuple_goals(HORIZONTAL_SAFE_145_RAW))
        grasp = _freeze_servo1(self._raw_goals(backend, HORIZONTAL_GRASP_POSES_DEG[profile]))
        midpoint = {
            servo_id: round((grasp[servo_id] + safe[servo_id]) / 2.0) for servo_id in range(1, 6)
        }

        if stage == "idle":
            if self._near_pose(actual, idle):
                return
            if not (self._near_pose(actual, safe) or self._near_pose(actual, drop)
                    or self._near_pose(actual, carry)):
                raise ValueError("idle 복귀는 safe/drop/carry 자세에서만 시작할 수 있습니다")
            self._glide_to_raw_positions(backend, idle, defer_joints=RETURN_TO_IDLE_DEFERRED_JOINTS)
            return

        if stage == "carry":
            # 물체를 든 채 주행할 자세. IDLE과 같은 게이트를 쓰고 **같은 손목
            # 지연을 반드시 건다** — safe에서 오는 servo 4 이동량이 +1381 raw
            # (121도)로 idle로 갈 때(+1618)와 같은 성격이다. 지연 없이 가면
            # 어깨가 내려가는 동안 손목이 같이 접혀 그리퍼가 차체 전면을
            # 긁는다(RETURN_TO_IDLE_DEFERRED_JOINTS 주석의 2026-08-24 사고).
            if self._near_pose(actual, carry):
                return
            if not (self._near_pose(actual, safe) or self._near_pose(actual, drop)
                    or self._near_pose(actual, idle)):
                raise ValueError("carry 이동은 safe/drop/idle 자세에서만 시작할 수 있습니다")
            self._glide_to_raw_positions(backend, carry,
                                          defer_joints=RETURN_TO_IDLE_DEFERRED_JOINTS)
            return

        if stage == "recover_idle":
            # 실패 복구 전용 경로 — 등록된 시작 자세 게이트를 **일부러**
            # 건너뛴다(사용자 요청, 2026-08-24: "실패하면 알아서 IDLE로
            # 돌아가게").
            #
            # 왜 "idle" 단계로는 안 되는가: 이동이 실패하면 팔은 정의상
            # 등록된 자세들 **사이**에 멈춰 선다. 바로 그 상태가 "idle"의
            # 시작 게이트에 걸려 거부되므로, 정작 복구가 필요한 순간에만
            # 복구가 막히는 모순이 생긴다.
            #
            # ⚠️⚠️ 2026-08-24 실기 사고 — 처음 구현은 여기서 곧장 idle로
            # 보간했다. 근거로 "실패는 IDLE<->safe 경로 위에서 나니 되짚기만
            # 하면 된다"고 적었는데 **그 전제가 틀렸다**. 실제 실패는 그리퍼
            # 닫기 단계, 즉 팔이 **바닥에 내려간 grasp 자세**에서 났고, 거기서
            # idle로 직선 보간하자 그리퍼가 바닥을 긁으며 쓸려 갔다. 사용자
            # 지적: "이렇게 움직이는건 절대로 안돼".
            #
            # 교훈: 팔이 **어느 자세에 도착하는가**만으로는 부족하고 **가는
            # 경로 자체가 안전 요구사항**이다. 이 로봇의 작업 공간이 곧
            # 바닥이기 때문이다. 그래서 복구도 정상 경로와 똑같이 **먼저
            # 들어올린 뒤** 등록된 waypoint를 밟아 올라간다.
            #
            # 어느 자세에서 시작하는지 모르므로 가장 가까운 등록 자세를 찾아
            # 거기서부터 검증된 상승 체인을 탄다. 어느 자세에도 못 붙으면
            # **아무것도 하지 않는다** — 경로를 추측해 움직이는 것보다
            # 자동 복구를 포기하고 사람에게 넘기는 쪽이 낫다.
            if self._near_pose(actual, idle):
                return

            named = {"grasp": grasp, "midpoint": midpoint, "safe": safe,
                     "drop": drop, "carry": carry}
            distances = {
                name: max(abs(actual[servo_id] - pose[servo_id]) for servo_id in range(1, 6))
                for name, pose in named.items()
            }
            nearest = min(distances, key=distances.get)
            if distances[nearest] > RECOVER_MATCH_TOLERANCE_RAW:
                raise ValueError(
                    "등록된 자세 어디에도 가깝지 않아 안전한 복구 경로를 정할 수 없습니다 "
                    f"(현재 {actual}, 가장 가까운 것은 {nearest}로 {distances[nearest]} raw 차이, "
                    f"허용 {RECOVER_MATCH_TOLERANCE_RAW}). 팔을 직접 보면서 "
                    "tools/align_to_idle.py로 정렬하세요"
                )

            # 검증된 상승 체인 — 바닥에서 곧장 idle로 쓸어가지 않는다.
            chain = {
                "grasp": (midpoint, safe, idle),
                "midpoint": (safe, idle),
                "safe": (idle,),
                "drop": (idle,),
                "carry": (idle,),
            }[nearest]
            self.get_logger().warn(
                f"recover_idle: 가장 가까운 등록 자세 '{nearest}'({distances[nearest]} raw)에서 "
                f"{len(chain)}단계로 들어올려 복귀합니다"
            )
            for waypoint in chain:
                defer = (RETURN_TO_IDLE_DEFERRED_JOINTS
                         if waypoint is idle or waypoint is carry else ())
                self._glide_to_raw_positions(backend, waypoint, defer_joints=defer)
            return

        if stage == "safe":
            if self._near_pose(actual, idle):
                self._glide_to_raw_positions(backend, safe)
                return
            if self._near_pose(actual, grasp):
                self._glide_to_raw_positions(backend, midpoint)
                self._glide_to_raw_positions(backend, safe)
                return
            if self._near_pose(actual, midpoint):
                self._glide_to_raw_positions(backend, safe)
                return
            if self._near_pose(actual, safe):
                return
            if self._near_pose(actual, drop):
                self._glide_to_raw_positions(backend, safe)
                return
            if self._near_pose(actual, carry):
                self._glide_to_raw_positions(backend, safe)
                return
            raise ValueError(
                "safe 이동 시작 자세가 등록된 idle/carry/grasp/midpoint/drop이 아닙니다")

        if stage == "drop":
            if not (self._near_pose(actual, idle) or self._near_pose(actual, safe)
                    or self._near_pose(actual, carry)):
                raise ValueError("drop 이동은 idle/safe/carry 자세에서만 시작할 수 있습니다")
            self._glide_to_raw_positions(backend, drop)
            return

        expected_start = safe if stage == "grasp" else grasp
        if not self._near_pose(actual, expected_start):
            raise ValueError(f"{stage} 이동 시작 자세가 등록된 이전 단계가 아닙니다")
        self._glide_to_raw_positions(backend, grasp if stage == "grasp" else midpoint)

    # --- 첫 이동 자동 IDLE 정렬 -------------------------------------------

    def _latch_torque_at_present(self, backend) -> dict:
        """servo 1..6의 goal에 자기 present 값을 그대로 write한다.

        ⚠️ STS3215는 goal_position write에 torque를 **자동으로 켠다**(펌웨어
        레벨 거동이라 driver_sdk 소스에는 안 나온다 — 2026-08-21 Pi 실기로
        확인). 즉 늘어져 있는 관절에 목표 자세를 곧장 write하면, write가
        도달하는 순간 torque가 켜지면서 그 목표를 향해 급하게 움직인다.
        goal == present로 먼저 한 번 써 두면 torque만 켜지고 이동량은 0이다.

        tools/align_to_idle.py의 latch_torque_at_present와 같은 계약이다 —
        그쪽은 사람이 돌리는 도구이고 이쪽은 노드 안에서 자동으로 도는
        경로라 코드를 공유하지 않는다(그 도구는 driver_sdk에 직접 붙는데,
        이 노드가 포트를 쥐고 있는 동안에는 그럴 수 없다).
        """
        present = {}
        for servo_id in ALL_SERVO_IDS:
            position = backend.drv.get_position(servo_id)
            if position is None:
                raise ArmHardwareUnavailableError(
                    f"servo {servo_id} present position 읽기 실패 — torque latch 중단"
                )
            if not backend.drv.set_position(servo_id, position):
                raise ArmHardwareUnavailableError(
                    f"servo {servo_id} goal<-present write 실패 — torque latch 중단"
                )
            present[servo_id] = position

        still_off = [
            servo_id for servo_id in ALL_SERVO_IDS if backend.drv.get_torque(servo_id) is not True
        ]
        for servo_id in still_off:
            backend.drv.set_torque(servo_id, True)
        failed = [
            servo_id for servo_id in still_off if backend.drv.get_torque(servo_id) is not True
        ]
        if failed:
            raise ArmHardwareUnavailableError(f"torque latch 후에도 OFF인 servo IDs: {failed}")

        self.get_logger().info(f"자동 정렬: goal<-present latch 완료(이동 없음) — {present}")
        return present

    def _align_named_poses(self, backend, frozen_servo1):
        """정렬이 현재 자세를 대볼 등록 자세들.

        어느 profile로 파지하다 멈췄는지 모르므로 **여섯 profile의 grasp/
        midpoint를 전부** 후보에 넣는다. 정상 경로(_move_floor_stage)는
        요청에 profile이 들어 있어 하나만 보면 되지만, 정렬은 아무 정보
        없이 불려 온다."""
        safe = {**self._tuple_goals(HORIZONTAL_SAFE_145_RAW), 1: frozen_servo1}
        named = {"safe": safe, "drop": self._tuple_goals(BASKET_DROP_195_RAW)}
        for profile, angles in HORIZONTAL_GRASP_POSES_DEG.items():
            grasp = {**self._raw_goals(backend, angles), 1: frozen_servo1}
            named[f"grasp:{profile}"] = grasp
            named[f"midpoint:{profile}"] = {
                servo_id: round((grasp[servo_id] + safe[servo_id]) / 2.0)
                for servo_id in range(1, 6)
            }
        return safe, named

    def _close_gripper_before_folding(self, backend) -> None:
        """접기 전에 활짝 열린 그리퍼만 닫는다.

        "다음 동작이 요구하는 형상을 그 동작 전에 만든다"는 이 프로젝트의
        규칙(사용자 지시 2026-08-25)을 정렬에도 적용한다 — 벌어진 손가락 판을
        단 채 IDLE로 접으면 차체에 닿는다.

        ⚠️ 다만 **무조건 닫지는 않는다**. 정렬이 불려 오는 시점에 그리퍼가
        물체를 문 채일 수 있고, 그때 GRIPPER_CLOSED_MM(9.0)을 명령하면 물체를
        으깬다 — servo 6에는 토크 제한 레지스터가 없어 위치 오차가 곧 힘이다.
        기준은 폭 하나로 충분하다: 실제로 무언가를 쥐고 있는 닫힘 폭은 가장
        넓은 것이 soccer의 31.0mm이므로, 그보다 한참 위인 45mm를 넘는 폭은
        정의상 아무것도 물고 있지 않은 '열린' 상태다.
        """
        present = backend.drv.get_position(GRIPPER_SERVO_ID)
        if present is None:
            self.get_logger().warn("자동 정렬: servo 6 위치를 못 읽어 그리퍼를 건드리지 않습니다")
            return
        width_mm = width_from_position(present)
        if width_mm <= AUTO_ALIGN_GRIPPER_CLOSE_ABOVE_MM:
            self.get_logger().info(
                f"자동 정렬: 그리퍼 {width_mm:.1f}mm — 이미 접기에 알맞아 그대로 둡니다"
            )
            return
        self.get_logger().warn(
            f"자동 정렬: 그리퍼가 {width_mm:.1f}mm로 열려 있습니다 — "
            f"접기 전에 {GRIPPER_CLOSED_MM}mm로 닫습니다"
        )
        backend.drv.set_speed(GRIPPER_SERVO_ID, GRIPPER_SPEED_RAW)
        backend.drv.set_acceleration(GRIPPER_SERVO_ID, GRIPPER_ACCEL_RAW)
        backend.drv.set_position(GRIPPER_SERVO_ID, position_from_width(GRIPPER_CLOSED_MM))
        self._wait_gripper_motion_settled(backend)

    def _auto_align_to_idle(self) -> None:
        """세션 첫 이동 직전에 팔을 IDLE로 자동 정렬한다.

        2026-08-25 사용자 지시. 예전에는 _log_idle_offset이 편차를 로그로만
        남기고 정렬은 사람이 tools/align_to_idle.py로 따로 돌려야 했는데,
        그걸 잊으면 첫 safe 이동이 자세 게이트에서 거부되거나 torque OFF로
        막혔다. 이제 이 함수가 그 자리를 대신한다 — align_to_idle.py는 노드를
        띄우지 않은 상태에서 쓰는 수동 경로로 남는다.

        ⚠️ 경로 선택이 이 함수의 전부다. 어디서 시작하든 IDLE로 직선 보간하는
        것은 절대 안 된다 — 팔이 바닥 높이에 있을 때 그렇게 하면 그리퍼가
        바닥을 쓸고 간다(AUTO_ALIGN_LIFT_VIA_SAFE_SERVO2_RAW 주석 참고).
        """
        backend = soarm._backend(real=True)
        actual = self._read_joint_positions(backend)
        if actual is None:
            raise ArmHardwareUnavailableError("자동 정렬 전 관절 위치 읽기 실패")

        idle = self._tuple_goals(IDLE_CRADLE_RAW)
        offsets = {servo_id: actual[servo_id] - idle[servo_id] for servo_id in range(1, 6)}
        summary = " ".join(f"s{servo_id}={offsets[servo_id]:+d}" for servo_id in sorted(offsets))

        # 어떤 목표를 쓰기 전에 반드시 먼저 온다.
        self._latch_torque_at_present(backend)

        if self._near_pose(actual, idle):
            self.get_logger().info(f"자동 정렬: 이미 IDLE입니다({summary}) — 이동 없음")
            return

        safe, named = self._align_named_poses(backend, actual[1])
        distances = {
            name: max(abs(actual[servo_id] - pose[servo_id]) for servo_id in range(1, 6))
            for name, pose in named.items()
        }
        nearest = min(distances, key=distances.get)

        if distances[nearest] <= RECOVER_MATCH_TOLERANCE_RAW:
            kind = nearest.split(":")[0]
            chain = {
                "grasp": (named[nearest.replace("grasp:", "midpoint:")], safe, idle),
                "midpoint": (safe, idle),
                "safe": (idle,),
                "drop": (idle,),
            }[kind]
            route = f"등록 자세 '{nearest}'({distances[nearest]} raw)에서 상승 체인"
        elif actual[2] >= AUTO_ALIGN_LIFT_VIA_SAFE_SERVO2_RAW:
            # 등록 자세 어디에도 안 붙는데 팔이 앞·아래로 뻗어 있다. IDLE로
            # 곧장 가면 바닥을 쓴다 — safe로 먼저 **들어올린** 뒤 접는다.
            chain = (safe, idle)
            route = (
                f"미등록 자세이나 servo2={actual[2]}가 "
                f"{AUTO_ALIGN_LIFT_VIA_SAFE_SERVO2_RAW} 이상 — safe로 먼저 들어올림"
            )
        else:
            chain = (idle,)
            route = f"미등록 자세이나 servo2={actual[2]}로 이미 접힌 영역 — IDLE 직행"

        self.get_logger().warn(f"자동 정렬 시작 — IDLE 편차 {summary} / {route}")
        self._close_gripper_before_folding(backend)
        for waypoint in chain:
            defer = RETURN_TO_IDLE_DEFERRED_JOINTS if waypoint is idle else ()
            self._glide_to_raw_positions(
                backend, waypoint, defer_joints=defer, speed_raw=AUTO_ALIGN_SPEED_RAW
            )

        final = self._read_joint_positions(backend)
        if final is None or not self._near_pose(final, idle):
            raise ArmHardwareUnavailableError(
                f"자동 정렬 후에도 IDLE에 도달하지 못했습니다 — 현재 {final}. "
                "팔을 직접 보면서 tools/align_to_idle.py로 정렬하세요"
            )
        residual = {servo_id: final[servo_id] - idle[servo_id] for servo_id in range(1, 6)}
        self.get_logger().info(f"자동 정렬 완료 — 잔차 {residual}")

    @staticmethod
    def _read_with_retry(reader, servo_id, attempts=JOINT_READ_ATTEMPTS):
        """레지스터 하나를 읽는다 — 실패하면 몇 번 다시 시도한다.

        get_arm_state는 서보 6개 × 레지스터 4개 = 24회 연속 읽기라, 읽기
        하나의 실패 확률이 낮아도 묶음 전체가 깨질 확률은 그만큼 쌓인다.
        _read_joint_positions가 이동 중 폴링에 재시도를 두는 것과 같은 이유다.

        ⚠️ 2026-08-25에 이 재시도를 **잘못된 진단으로** 넣었다는 것을 남겨
        둔다. 그날 관측된 10~80%의 실패는 패킷 유실이 아니라 arm_driver가
        세 개 떠서 같은 시리얼 포트를 동시에 쓰고 있었기 때문이었고, 진짜
        해결은 _claim_serial_port의 배타 잠금이다. 프로세스를 하나로 정리한
        뒤 실패율은 30회 중 0이었다. 재시도는 그대로 두되(묶음 읽기에 맞는
        대비이긴 하다) **재시도가 늘면 경합을 의심할 것** — 재시도로 덮이는
        실패는 원인이 따로 있다는 신호다.
        """
        for attempt in range(attempts):
            value = reader(servo_id)
            if value is not None:
                return value
            if attempt + 1 < attempts:
                time.sleep(JOINT_READ_RETRY_SEC)
        return None

    def _on_get_arm_state(self, request, response):
        """servo 1..6의 위치·부하·온도·torque를 한 번에 돌려준다.

        이 노드가 /dev/soarm을 독점하므로 검증 도구가 서보를 실측할 수 있는
        유일한 경로다. 읽기에 실패한 서보는 online=False로 표시하고 나머지
        값은 0으로 둔다 — 못 읽은 것을 0으로 보고하면 '부하 없음'과 구분되지
        않기 때문이다."""
        try:
            backend = soarm._backend(real=True)
        except Exception as e:  # 포트가 끊긴 경우
            response.ok = False
            response.message = f"백엔드 접근 실패: {e}"
            return response

        online, positions, loads, temperatures, torques = [], [], [], [], []
        for servo_id in ALL_SERVO_IDS:
            position = self._read_with_retry(backend.drv.get_position, servo_id)
            # ⚠️ 범위 밖 값은 실패로 본다. 2026-08-25에 servo 3 위치가
            # 55841로 돌아온 적이 있다 — 다른 서보의 응답 바이트가 섞인
            # 것인데, 그대로 통과시키면 호출부가 그 숫자를 믿는다.
            # STS3215의 위치는 정의상 0..4095다.
            if position is not None and not 0 <= int(position) <= POSITION_RAW_MAX:
                self.get_logger().warn(
                    f"servo {servo_id} 위치 {position} — "
                    f"0..{POSITION_RAW_MAX} 밖이라 버립니다(응답이 섞인 것으로 봅니다)"
                )
                position = None
            if position is None:
                online.append(False)
                positions.append(0)
                loads.append(0.0)
                temperatures.append(0)
                torques.append(False)
                continue
            raw_load = self._read_with_retry(backend.drv.get_load, servo_id)
            temperature = self._read_with_retry(backend.drv.get_temperature, servo_id)
            online.append(True)
            positions.append(int(position))
            loads.append(0.0 if raw_load is None else abs(raw_load) / GRIPPER_LOAD_MAX_RAW)
            temperatures.append(0 if temperature is None else int(temperature))
            torques.append(self._read_with_retry(backend.drv.get_torque, servo_id) is True)

        response.online = online
        response.position_raw = positions
        response.load_ratio = loads
        response.temperature_c = temperatures
        response.torque_on = torques
        offline = [servo_id for servo_id, ok in zip(ALL_SERVO_IDS, online, strict=True) if not ok]
        response.ok = not offline
        response.message = "" if not offline else f"읽기 실패 servo IDs: {offline}"
        return response

    def _on_set_gripper(self, request, response):
        min_width_mm = float(self.get_parameter("min_gripper_width_mm").value)
        width_mm = max(min_width_mm, min(GRIPPER_OPEN_MM, request.width_mm))
        raw_position = position_from_width(width_mm, min_width_mm=min_width_mm)
        if width_mm < GRIPPER_CLOSED_MM:
            self.get_logger().warn(
                f"set_gripper: {width_mm:.1f}mm — 빈 닫힘 하한 {GRIPPER_CLOSED_MM}mm보다 "
                "좁습니다. 턱 사이에 물체가 있을 때만 안전합니다"
            )
        try:
            self._require_operational_servos((GRIPPER_SERVO_ID,))

            # soarm.grip()을 쓰지 않는다 — 항상 SimBackend를 움직이는 함정이 있다
            # (모듈 docstring 참고). 실물 백엔드의 드라이버에 직접 명령한다.
            backend = soarm._backend(real=True)
            # 관절 이동과 같은 이유로 속도를 상속하지 않고 직접 쓴다
            # (_glide_to_raw_positions 주석 참고). 2026-08-24 실기에서
            # align_to_idle의 SPEED_RAW=150이 servo 6에도 남아, 완전 개방
            # (168mm)에서 파지(15mm)까지의 약 820 raw 행정이 150 raw/s로
            # 5.5s가 걸렸다 — GRIPPER_MOTION_TIMEOUT_SEC(4.0s)을 넘겨
            # "그리퍼 닫기 실패"로 끝났다.
            backend.drv.set_speed(GRIPPER_SERVO_ID, GRIPPER_SPEED_RAW)
            backend.drv.set_acceleration(GRIPPER_SERVO_ID, GRIPPER_ACCEL_RAW)
            if not backend.drv.set_position(GRIPPER_SERVO_ID, raw_position):
                self.get_logger().error("set_gripper 실패: servo 6 position write 실패")
                response.ok = False
                response.load_ratio = 0.0
                return response

            # 정착 전에 읽으면 빈 채와 물체가 구분되지 않는다 — 이동 중에는
            # 양쪽 다 포화값(±500)이 나온다. set_position()은 즉시 반환하므로
            # 여기서 기다린다.
            #
            # 이 대기는 응답의 load_ratio 뿐 아니라 **그 다음 get_load() 를 위한
            # 것이기도 하다.** GraspState 는 set_gripper() 응답을 쓰지 않고
            # get_load() 를 따로 부르는데(ArmDriver.set_gripper 는 None 반환),
            # 그 호출이 정착 후에 도착하려면 set_gripper 가 정착까지 붙들고
            # 있어야 한다.
            #
            # 개방(OPEN_MM) 명령에도 똑같이 걸린다 — 개폐 판정이 필요 없는
            # 경로에서는 순수 지연이지만, 분기해서 "어떤 호출은 안 기다린다"를
            # 만드는 것보다 한 가지 계약(반환 시점 = 정착 완료)이 낫다.
            #
            # ⚠️ 2026-08-24 실기: 고정 sleep만으로는 부족하다. preopen을
            # 80mm→168mm(GRIPPER_OPEN_MM)로 올리면서 닫힘 행정이 319raw에서
            # 820raw로 2.6배 길어졌는데, GRASP_SETTLE_SEC(1.5s)은 짧은 행정
            # 기준으로 잡힌 값이라 **아직 닫히는 중에 반환**됐다. 그 결과
            # load_ratio가 "물체를 쥔 부하"가 아니라 "이동 중 모터 토크"
            # (0.0704)로 읽혔고, 호출자(grasp_test_console 5단계·GraspState)가
            # 곧바로 들어올리기를 시작해 닫힘과 상승이 겹쳤다 — 물체를 놓칠
            # 확률이 매우 높은 순서다. 실제로 들어올린 뒤 load가 0.0352로
            # 반토막 났다.
            #
            # 행정 길이는 요청 폭에 따라 달라지므로 상수를 다시 튜닝하는 대신
            # **위치가 멈출 때까지 기다린 뒤** 부하 정착을 기다린다.
            self._wait_gripper_motion_settled(backend)
            time.sleep(GRASP_SETTLE_SEC)
            response.ok = True
            response.load_ratio = self._read_load()
        except Exception as e:
            self.get_logger().error(f"set_gripper 실패: {e}")
            response.ok = False
            response.load_ratio = 0.0
        return response

    def _wait_gripper_motion_settled(self, backend) -> None:
        """servo 6의 위치가 더 이상 변하지 않을 때까지 기다린다(최대
        GRIPPER_MOTION_TIMEOUT_SEC).

        "닫힘 완료"를 시간이 아니라 **관측**으로 판정하는 게 요점이다 —
        요청 폭에 따라 행정 길이가 3배 가까이 달라지므로 어떤 고정 sleep도
        한쪽 경우에는 틀린다(위 _on_set_gripper 주석의 2026-08-24 실기 참고).

        물체를 쥐어 목표 위치까지 못 가고 멈추는 것도 '정착'이다 — 목표
        도달 여부가 아니라 **정지 여부**를 본다. 위치를 못 읽으면(통신
        순간 오류) 그 폴링만 건너뛰고, 타임아웃까지도 안 멈추면 조용히
        빠져나온다 — 여기서 예외를 내면 정상 파지까지 실패로 만든다."""
        previous = None
        deadline = time.monotonic() + GRIPPER_MOTION_TIMEOUT_SEC
        while time.monotonic() < deadline:
            time.sleep(GRIPPER_MOTION_POLL_SEC)
            current = backend.drv.get_position(GRIPPER_SERVO_ID)
            if current is None:
                continue
            if previous is not None and abs(current - previous) <= GRIPPER_MOTION_SETTLED_RAW:
                return
            previous = current
        self.get_logger().warn(
            f"set_gripper: servo 6이 {GRIPPER_MOTION_TIMEOUT_SEC}s 안에 멈추지 않았습니다 "
            "— 부하 판정이 이동 중 값일 수 있습니다"
        )

    def _on_get_load(self, request, response):
        response.load_ratio = self._read_load()
        return response

    def _on_fold_to_cradle(self, request, response):
        """팔을 교시 IDLE(IDLE_CRADLE_RAW)로 접는다. 도달을 확인하고 답한다.

        ⚠️ 예전 구현은 IDLE 로 가지 않았다. CRADLE_XYZ_M([0.15, 0, 0.20])
        으로 역기구학 이동을 한 뒤 1.2초 자고 **무조건** success=True 를
        냈다. 그 좌표에는 "INSERT 후 복귀 경로 별도 실측 필요"라는 TODO 가
        달려 있었다 — 교시된 자세가 아니라 자리표시자다.

        2026-08-28 실기에서 그 대가를 치렀다. 이 서비스가 성공을 반환한
        직후 팔은 IDLE 에서 s3=-856 s5=-935 raw(각각 약 75도, 82도)
        떨어진 자세에 서 있었다. 성공을 믿고 다음 동작을 시키면 자세
        게이트에서 거부되거나, 더 나쁘게는 그 자세에서 바닥으로 내려간다.

        이제 `_auto_align_to_idle()` 에 위임한다. 그 함수는 어디서
        시작하든 안전한 경로를 고르고(바닥 높이면 safe 를 경유해 들어
        올린다), 열린 그리퍼를 접기 전에 닫고, **도달할 때까지 기다렸다가**
        잔차를 로그로 남긴다. 도달 못 하면 예외가 나므로 success=False 가
        정직하게 나간다."""
        try:
            self._require_operational_servos(range(1, 6))
            self._auto_align_to_idle()
            self._require_operational_servos(range(1, 6))
            response.success = True
            response.message = "IDLE 복귀 완료"
        except Exception as e:
            self.get_logger().error(f"fold_to_cradle 실패: {e}")
            response.success = False
            response.message = str(e)
        return response

    # servo 1 좌우 보정의 한계각. 이보다 크게 돌려야 할 만큼 어긋났다면
    # 그건 차량이 잘못 선 것이라 Host가 다시 세워야 한다(사용자 지시
    # 2026-08-26의 "영역 밖" 갈래).
    #
    # 팔 길이 294mm(실측 2026-08-26) 기준 15도면 약 79mm다. 이 값이 체스말
    # 턱 폭 허용치(±76mm)와 거의 같은 것이 우연이 아니라 중요하다 — 턱 폭
    # 안에 들어온 물체는 servo 1으로 전부 중앙에 맞출 수 있고, 한계각에
    # 걸리는 경우와 영역 밖인 경우가 사실상 같은 지점에서 갈린다
    # (baseline_constants.SERVO1_AXIS_TO_JAW_MM 주석).
    #
    # ⚠️ 2026-08-29까지 여기 240mm(= 64mm)로 적혀 있었다. 실측 전 어림값이다.
    MAX_BASE_YAW_OFFSET_RAD = math.radians(15.0)

    # STS3215는 한 바퀴가 4096 카운트다.
    RAW_PER_RADIAN = 4096.0 / (2.0 * math.pi)

    def _on_offset_base_yaw(self, request, response):
        """servo 1만 현재 위치에서 offset_rad만큼 돌린다.

        GRASP 하강 **전에** 부른다. servo 1은 safe/grasp/midpoint 사이를
        오가는 동안 `_freeze_servo1`이 현재값을 그대로 물려주므로, 여기서
        한 번 돌려 두면 그 뒤 하강 경로가 그 각도를 이어받는다 — 교시 자세의
        servo 1 절대값으로 되돌아가지 않는다.

        한계각을 넘거나 관절 범위를 벗어나면 **움직이지 않고 거부한다.**
        여기서 무리하게 돌리면 팔이 차체 옆을 치거나, 하강 경로가 교시된
        평면에서 벗어난다."""
        response.ok = False
        response.position_raw = 0
        offset = float(request.offset_rad)
        if not math.isfinite(offset):
            response.message = "offset_rad가 수치가 아닙니다"
            return response
        if abs(offset) > self.MAX_BASE_YAW_OFFSET_RAD:
            response.message = (
                f"보정각 {math.degrees(offset):+.1f}도가 한계 "
                f"±{math.degrees(self.MAX_BASE_YAW_OFFSET_RAD):.0f}도를 넘습니다 — "
                "차량을 다시 세워야 합니다")
            return response

        try:
            backend = soarm._backend(real=True)
            actual = self._read_joint_positions(backend)
            if actual is None:
                response.message = "현재 관절 위치 읽기 실패"
                return response

            target = int(round(actual[1] + offset * self.RAW_PER_RADIAN))
            if not (0 <= target <= POSITION_RAW_MAX):
                response.message = (
                    f"servo 1 목표 {target}가 관절 범위(0~{POSITION_RAW_MAX}) 밖입니다")
                return response

            # ⚠️ 한계각은 **교시 정면(IDLE)으로부터의 절대 각도**로 본다.
            # 예전에는 한 번의 요청 크기만 봤는데, 이 서비스는 현재 위치를
            # 기준으로 상대 회전을 하므로 같은 요청이 반복되면 servo 1이
            # 한 번에 15도를 넘지 않으면서도 얼마든지 멀리 걸어간다.
            #
            # 반복은 정상 동작이다 — 이 보정은 "관측 -> 소이동 -> 재관측"
            # 폐루프라 여러 사이클에 걸쳐 수렴한다(_judge_alignment 주석).
            # 2026-08-28 실기에서 좌우 오차는 3회 보정(합 약 14도) 만에
            # 56.9mm -> 26.7mm 로 줄어 READY 가 났다. 즉 루프는 닫힌다.
            #
            # 문제는 수렴하지 못하는 경우다. 물체가 애초에 팔이 닿는 범위
            # 밖이면 같은 방향 보정이 계속 나오는데, 상대 회전이라 한 번에
            # 15도를 안 넘으면서 얼마든지 멀리 걸어갈 수 있다. 위 실측이
            # 이미 14도까지 갔으니 여유가 크지 않다.
            #
            # 한계에 걸리면 ok=False 가 나가고 Pi 는 "servo 1이 거부했다,
            # 재회전 필요"로 Host 에 차량 재정렬을 요청한다 — 팔로 못 고칠
            # 만큼 틀어졌으면 차를 다시 세우는 것이 맞고, 이미 있는 경로다.
            drift_raw = target - IDLE_CRADLE_RAW[0]
            limit_raw = self.MAX_BASE_YAW_OFFSET_RAD * self.RAW_PER_RADIAN
            if abs(drift_raw) > limit_raw:
                response.message = (
                    f"servo 1이 교시 정면에서 "
                    f"{math.degrees(drift_raw / self.RAW_PER_RADIAN):+.1f}도까지 "
                    f"벌어집니다 — 한계 ±{math.degrees(self.MAX_BASE_YAW_OFFSET_RAD):.0f}도. "
                    "차량을 다시 세워야 합니다")
                self.get_logger().warn(f"offset_base_yaw: {response.message}")
                return response

            goal = {servo_id: actual[servo_id] for servo_id in range(1, 6)}
            goal[1] = target
            # servo 1만 미세 허용오차로 옮긴다. 기본 120 raw를 그대로 쓰면
            # 몇 도짜리 정렬 보정이 통째로 버려진다(BASE_YAW_TOLERANCE_RAW
            # 주석의 2026-08-28 사례). 나머지 관절은 제자리를 지키기만
            # 하면 되므로 원래 게이트를 그대로 둔다.
            # 단계 수를 이동 거리에 맞춘다. 교시 자세용 기본값
            # FLOOR_POSE_STEPS(30)를 그대로 쓰면 5도짜리 보정에도 3.0초
            # (30 x FLOOR_POSE_STEP_SEC)를 쓴다. 그러면 호출자의 서비스
            # 대기(_ros_call.SERVICE_TIMEOUT_SEC = 3.0s)를 넘겨, 팔은 제대로
            # 돌았는데 Pi 는 실패로 받는다 — 2026-08-28 실기에서 실제로
            # 세 번 다 그렇게 나왔다.
            move_raw = abs(target - actual[1])
            steps = max(3, min(FLOOR_POSE_STEPS,
                               int(math.ceil(move_raw / BASE_YAW_RAW_PER_STEP))))
            self._glide_to_raw_positions(
                backend, goal, tolerance_raw={1: BASE_YAW_TOLERANCE_RAW}, steps=steps)

            # 도달을 **확인하고** ok를 정한다. 예전에는 무조건 True였다 —
            # 그래서 위 버그로 팔이 한 raw도 안 움직였는데도 성공이 나갔고,
            # Pi는 보정이 먹은 줄 알고 같은 관측·같은 보정을 무한히 반복했다.
            # 못 돌렸으면 거부해야 Host가 차량을 다시 세우는 대안 경로
            # (_judge_alignment 의 "servo 1이 거부했다, 재회전 필요")로 넘어간다.
            settled = self._read_joint_positions(backend)
            if settled is None:
                response.message = "보정 후 관절 위치 읽기 실패 — 돌았는지 확인 불가"
                self.get_logger().error(f"offset_base_yaw: {response.message}")
                return response
            response.position_raw = int(settled[1])
            error = response.position_raw - target
            response.ok = abs(error) <= BASE_YAW_TOLERANCE_RAW
            response.message = (
                f"servo 1 {actual[1]} -> {response.position_raw} "
                f"(목표 {target}, 잔차 {error:+d} raw, {math.degrees(offset):+.1f}도)")
            if response.ok:
                self.get_logger().info(f"offset_base_yaw: {response.message}")
            else:
                self.get_logger().warn(
                    f"offset_base_yaw: 도달 실패 — {response.message} "
                    f"(허용 ±{BASE_YAW_TOLERANCE_RAW} raw)")
            return response
        except Exception as exc:  # noqa: BLE001 -- 서비스 경계
            response.message = f"servo 1 보정 실패: {exc}"
            self.get_logger().error(response.message)
        return response

    def _on_hold_position(self, request, response):
        # 현재 관절 자세를 래치한다 — E-STOP 시 파지물 낙하 방지용
        # (states.py EstopState가 호출). 새 목표를 보내지 않고 전 관절
        # 토크를 켜 두는 것만으로 STS3215가 지금 위치를 유지한다.
        try:
            backend = soarm._backend(real=True)
            failed_ids = [
                servo_id for servo_id in ALL_SERVO_IDS if not backend.drv.set_torque(servo_id, True)
            ]
            if failed_ids:
                response.success = False
                response.message = "torque enable 실패 — " f"servo IDs: {failed_ids}"
                self.get_logger().error(response.message)
                return response

            response.success = True
        except Exception as e:
            self.get_logger().error(f"hold_position 실패: {e}")
            response.success = False
            response.message = str(e)
        return response

    def _read_load(self, servo_id: int = GRIPPER_SERVO_ID) -> float:
        """서보 원시 부하값을 0~1 비율로 정규화해 돌려준다.

        ⚠️ 부호를 버리고 **절대값**을 쓴다. PRESENT_LOAD 의 0x400 비트는 부하의
        방향인데, 실측에서 같은 '빈 채로 닫음' 조건이 음수(-88)로도 양수로도
        나와 방향이 일관되지 않았다. 파지 판정에 필요한 건 '얼마나 버티고
        있나'(크기)이지 어느 쪽으로 밀리나(방향)가 아니므로 크기만 본다.

        읽기에 실패하면(None) 0.0 — 즉 '빈 채'로 본다. 부하를 못 읽는 상태에서
        파지 성공으로 판정해 물체를 든 줄 알고 이송하는 것보다, 실패로 보고
        재시도하는 쪽이 안전하다."""
        backend = soarm._backend(real=True)
        raw = backend.drv.get_load(servo_id)
        if raw is None:
            self.get_logger().warn(f"servo {servo_id} load read 실패 — 안전값 0.0으로 처리")
            return 0.0
        return abs(raw) / GRIPPER_LOAD_MAX_RAW


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ArmDriverNode()
    except (ArmPortConflictError, ArmHardwareUnavailableError) as e:
        # 노드를 띄우면 안 되는 설정/하드웨어 오류다. 장치가 없거나 서보 버스가
        # 응답하지 않는데 계속 실행하면 이후 명령이 성공처럼 보일 수 있다.
        rclpy.logging.get_logger("arm_driver_node").fatal(str(e))
        rclpy.shutdown()
        return 1
    from rclpy.executors import MultiThreadedExecutor

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
