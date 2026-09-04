"""perception_node — 카메라 기반 인식.

scan_floor는 Hailo-10H YOLO로 실제 검출을 반환할 수 있지만 `scan_floor_enabled`
파라미터(기본값 False)로 잠겨 있다(2026-08-21, 구조 검증용 — 모듈 하단 HAILO_*
상수 블록의 경고 참고: pose_m은 자리표시자고 클래스 매핑도 불완전하다). 게이트를
켜지 않으면 지금까지처럼 빈 목록을 반환한다. find_box/measure_opening은 아직
정직한 미구현 스텁.

⚠️ 안전 원칙 (domain/ports/perception.py의 Perception ABC 계약, 실측 전까지 절대
어기면 안 됨):
- monitor_clearance: 모르면 항상 contact_risk=True(정지)로 응답한다. False로 두면
  실제 장애물을 못 보고 밀고 지나가는 사고로 직결된다.
- scan_floor: 모르면 빈 목록으로 응답한다 — SCAN이 이걸 '대상 없음'으로 해석해
  DONE으로 유도한다.
- find_box: 모르면 found=False로 응답한다 — TRANSPORT가 이걸 받으면 대상을
  보류 등록하고 SCAN으로 복귀한다.
- confirm_grasp: 카메라를 못 열거나 기준 프레임이 없으면 confirmed=False다 —
  다른 관측 포트와 같은 "모르면 실패" 관례(domain/ports/perception.py 참고).
"""

import math
import statistics
import threading
import time

import rclpy
from geometry_msgs.msg import Point, Vector3
from grippers_interfaces.msg import Detection, DetectionArray
from grippers_interfaces.srv import (
    ConfirmGrasp,
    MonitorClearance,
    ObserveTarget,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

# 클래스 이름·매핑은 각 백엔드별 매핑 모듈로 뽑았다 — rclpy 없이 순수 pytest로
# 테스트하려면(2026-08-22, PR #185 리뷰 후속) 이 파일 밖에 둬야 한다.
# perception_node.py는 rclpy를 무조건 import해서 ROS2 없이는 아예 못 불러온다.
from grippers_perception.cpu_yolo_scan_mapping import (
    object_class_for_cpu_yolo_class_name,
)
from grippers_perception.floor_consensus import CONF_THRESHOLD, confirmed_tracks, track_bbox_xyxy
from grippers_perception.hailo_scan_mapping import HAILO_CLASS_NAMES, object_class_for_hailo_id
from grippers_perception.gripper_cam_geometry import orient

try:
    from sensor_msgs.msg import CameraInfo, Image

    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False

try:
    import cv2
    import numpy as np

    _GRASP_CAM_CV_AVAILABLE = True
except ImportError:
    _GRASP_CAM_CV_AVAILABLE = False

try:
    from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice

    _HAILO_AVAILABLE = True
except ImportError:
    _HAILO_AVAILABLE = False

# 🔴 임시 비활성화 (2026-08-22, 배포 전용 — 커밋 안 함): AI HAT+2가 부팅마다
# "Timeout waiting for firmware file / Failed writing SOC firmware on stage 2"로
# 죽는다 (재부팅 2회 + 완전 파워사이클 1회로도 재현, hailortcli scan에 장치 자체가
# 안 잡힘). Hailo 커뮤니티의 동일 증상 스레드가 온보드 DDR 손상으로 결론났다.
# 새 보드가 들어오기 전까지 여기서 강제로 꺼서 scan_floor가 Hailo 경로를 안 탄다 —
# 아래 CPU YOLO 경로가 대신한다. 하드웨어 복구되면 이 한 줄만 지우면 된다.
_HAILO_AVAILABLE = False

try:
    from ultralytics import YOLO

    _CPU_YOLO_AVAILABLE = True
except ImportError:
    _CPU_YOLO_AVAILABLE = False


# ── confirm_grasp (1단계, classical CV 임시 구현) ───────────────────────────
# YOLO가 아직 안 붙어서(2026-08-21 기준) 실기 로그 수집을 시작하려고 정교한
# 검출 없이 "기준(빈 그리퍼) 프레임과 지금 프레임이 얼마나 다른가"만 본다.
# GRASP는 이 결과를 아직 판정에 안 쓴다(domain/task/states.py GraspState 참고) —
# 로그만 쌓아서 나중에 임계값을 실측으로 잡는다.
GRIPPER_CAM_DEVICE_DEFAULT = "/dev/gripper_cam"
# ⚠️ VLA 학습 데이터와 같은 해상도여야 한다. 정책은 1280x720 으로 찍은 화면만
# 보고 배웠고, 이 카메라(ABKO APC-850)는 **4:3 을 요청하면 좌우를 잘라낸다** —
# 640x480 은 축소가 아니라 크롭이라 가로 화각의 25% 가 사라진다(RECORDING.md).
# 그 상태로 정책에 넣으면 학습 때 못 본 화각의 화면이 들어간다.
#
# 640x480 은 2026-08-21(커밋 36e9d6e) 에 icSpring 카메라 기준으로 들어온 값이고,
# 8/30 광각 교체 커밋(5d7c63d)은 회전만 정리하고 이 상수는 그대로 뒀다.
GRIPPER_CAM_WIDTH = 1280
GRIPPER_CAM_HEIGHT = 720
GRIPPER_CAM_WARMUP_FRAMES = 5  # 노출 자동조정 전 프레임은 검게 나온다 (실기 확인됨)
# 그리퍼캠 프레임을 토픽으로 내보낼지. 0 = 끔(기본). VLA 를 쓸 때만 켠다.
# 정책은 청크 하나(100스텝 = 3.33초)당 관측 한 장만 쓰므로 높을 이유가 없다.
GRIPPER_CAM_PUBLISH_HZ_DEFAULT = 0.0
GRIPPER_CAM_TOPIC_DEFAULT = "gripper_cam/image_raw"
# 손가락은 프레임 하단 중앙 일부에만 작게 잡힌다(2026-08-21 실기 스냅샷 확인) —
# 전체 프레임으로 diff를 내면 배경(의자·책상) 변화에 신호가 희석된다. 정확한
# 비율은 카메라 장착이 바뀌면 같이 바뀌니 재장착 후 스냅샷으로 재확인할 것.
GRIPPER_CAM_ROI = (0.30, 0.55, 0.70, 1.00)  # (x0, y0, x1, y1), 프레임 폭/높이 비율
# 실측 4건(2026-08-21, n=1 각각) 기준 임시치 — "실측 확정"은 아니지만 최소한
# 관측값 안쪽에 두는 게 관측값 밖(15.0)보다 낫다:
#   빈 그리퍼 4.65 · 축구공(위치 이탈) 1.88 · 별 7.46 · 큐브 10.97
# 15.0은 네 값 전부보다 커서 confirmed가 상시 False였다(PR #185 리뷰 지적).
# TODO: 실측 — 케이스를 더 모아 재보정한다. confirm_grasp 로그(diff_score)를
# 실제 파지 성공/실패 케이스별로 모은다. ros2 param set으로 재배포 없이
# 튜닝할 수 있게 파라미터로도 노출한다.
CONFIRM_GRASP_DIFF_THRESHOLD_DEFAULT = 6.0

# ── scan_floor (2026-08-21 신설, 2026-08-23 갱신) ────────────────────────────
# ⚠️ 클래스별 거리 보정값(CLASS_DISTANCE_CALIBRATION_SQRT_PX_M, 아래 "RGB
# bbox 면적 기반 거리 추정" 참고)이 전부 미실측(None)인 동안은 모든 CPU YOLO
# 검출이 _approach_pose_m()에서 걸러져 scan_floor가 항상 빈 목록을 반환한다.
# Hailo 경로는 하드웨어 고장(#189)으로 애초에 안 쓴다.
#
# 클래스 매핑도 불완전하다. domain.values.ObjectClass는 GABE/CHESS_PIECE
# 둘뿐인데 Hailo 모델은 7종(container/knight/queen/rook/box/soccer/star)이다.
# knight/queen/rook은 CHESS_PIECE로, box/soccer/star는 GABE로 매핑했지만
# "cube"는 애초에 학습 클래스에 없고, container(Hailo 전용)는 목적지 상자로
# 보여서 **바닥 물체 후보에서 제외**했다 — 확실하지 않은 매핑을 코드에 박지
# 않는다(cpu_yolo_scan_mapping.py/hailo_scan_mapping.py 참고).
#
# 🔴 안전 게이트 (PR #185 리뷰 지적, 2026-08-21): 위 두 가지(거리 보정값
# 미실측, 클래스 매핑) 때문에 지금 이 게이트가 없어도 사실상 안전하지만,
# "우연히 안전한" 상태에 기대지 않는다 — 별도 파라미터로 기본값을 꺼둔다.
# 구조 검증(SCAN→SELECT→APPROACH 실기 경로 확인)이 필요할 때만 명시적으로
# `-p scan_floor_enabled:=true`로 켤 것.
SCAN_FLOOR_ENABLED_DEFAULT = False
HAILO_HEF_PATH_DEFAULT = "/tmp/best_640.hef"
# 2026-08-22: 0.35였다가 0.8로 올림 — 실기 검증 중 물체를 놓기 전인데도
# CPU YOLO가 0.35~0.48대 confidence로 오검출을 냈다(사용자 실측 지적).
# 오검출이 SELECT/APPROACH로 흘러 들어가는 게 더 위험하므로, 놓친 검출(재스캔하면
# 그만) 쪽보다 오검출(엉뚱한 좌표로 주행) 쪽을 훨씬 더 강하게 억제한다.
HAILO_SCORE_THRESHOLD = 0.8
# HAILO_CLASS_NAMES · 클래스 매핑은 hailo_scan_mapping.py 참고 (모듈 상단 import).

# CPU YOLO(ultralytics) 폴백 — 2026-08-22, Hailo-10H 하드웨어 고장(위 _HAILO_AVAILABLE
# 강제 비활성화 참고)으로 임시 대체. 클래스 구성이 Hailo 모델과 다르다(6종,
# "container" 없음) — cpu_yolo_scan_mapping.py 참고.
# ⚠️ 2026-08-24: 기본 경로가 /tmp/best_cpu.pt였다. 그런데 이 파일은 그날까지
# 컨테이너 /tmp에만 있었고 저장소·호스트 어디에도 백업이 없었다 — 컨테이너를
# 다시 만들거나 /tmp가 비워지는 순간 인식이 통째로 죽고 복구할 원본도 없는
# 상태였다. 시연 전에 감당할 위험이 아니라 /grippers/models(호스트
# ~/docker/shared/grippers/models에 바인드 마운트되어 컨테이너 수명과
# 무관하게 남는다)로 옮기고 기본값을 그쪽으로 돌린다.
#
# ## 배포된 가중치 (2026-08-26 train-9로 교체)
#
# 경로는 고정하고 파일만 갈아 끼운다 — 경로를 버전마다 바꾸면 코드와 실기가
# 어긋날 때 어느 쪽이 맞는지 알 수 없게 된다. 이전 버전은 같은 디렉터리에
# 이름을 붙여 남기므로 되돌리려면 그 파일을 best.pt로 덮으면 된다.
#
# ⚠️ 2026-08-27 사용자 지시로 파일명을 best_cpu.pt에서 best.pt로 통일했다
# ("_cpu" 접미사가 혼란을 줘서 뺐다). 아래 sha256과 지표는 파일명과 무관하게
# 그대로 유효하다 — 내용은 안 바뀌었다.
#
#   train-9  2026-08-26  sha256 bd13ae42b9a080d85a9c620b983d7c4ad45d6f69ccc0a99da6c44cb0ce6490c8
#   train-8  2026-08-21  sha256 9680cf7d156c32cdc8082214108451aa3e110598c0ce7ee3cf541791d173182c
#            되돌리기: models/best_cpu_train8_20260821.pt (이 백업 파일명은 안 바꿈)
#
# 클래스 구성은 두 버전이 **완전히 같다**(6종, 인덱스까지 동일:
# 0 knight / 1 queen / 2 rook / 3 box / 4 soccer / 5 star). 매핑 코드를
# 손댈 필요가 없다는 뜻이다.
#
# 검증 지표는 전 항목이 올랐다:
#
#            train-8   train-9
#   precision  0.932    0.973
#   recall     0.862    0.956    <- +9.3%p
#   mAP50      0.935    0.984
#   mAP50-95   0.829    0.948    <- +11.9%p
#
# ⚠️ recall이 크게 오른 것은 **게이트 임계를 다시 봐야 한다는 뜻**이기도 하다.
# 지금 걸려 있는 오검출 게이트(conf 0.70 · bottom-y 290 · 5중 3 합의)는
# train-8의 검출 분포를 보고 잡은 값이다. train-9는 진짜 물체를 더 높은
# 신뢰도로 잡을 가능성이 크므로 게이트가 헐거워졌을 수 있다 —
# tools/grasp_geometry_calibrate.py --mode gate로 여유를 다시 재 볼 것.
#
# (맥 ~/Downloads/grippers_model_backup/ 에 두 버전 모두 보관)
CPU_YOLO_MODEL_PATH_DEFAULT = "/grippers/models/best.pt"
# ⚠️ 2026-08-23: 단일 프레임 신뢰도 임계값을 0.8까지 올려 오검출을 억누르던
# 방식(2026-08-22 시도)을 폐기했다 — 대신 HANDOFF.md가 실기로 검증한 2단계
# 게이트를 쓴다: 프레임당 admission은 conf 0.45(floor_consensus.CONF_
# THRESHOLD)로 넉넉하게 열어두고, 여러 프레임에 걸친 다중 프레임 합의
# (k-of-n·순도·산포·거리 게이트, floor_consensus.py)로 오검출을 거른다.
# 물체 없는 장면의 오탐 4개(벽 밑동·바구니 주변)는 배경이 정지해 있어
# 합의 필터로 원리적으로 못 거르므로, 거리 게이트(y≥290)가 대신 막는다
# (HANDOFF.md "인식 > 한계" 참고).
CONSENSUS_N_FRAMES = 10  # tools/perception/approach.py --frames 기본값과 동일
CONSENSUS_COLLECT_TIMEOUT_SEC = 2.5  # SERVICE_TIMEOUT_SEC(3.0s, _ros_call.py)보다 짧게

# 진짜 3D 위치가 아니다 — 위 경고 참고. base_link 앞 임의 고정점.
# Hailo 경로는 아직 이 자리표시자를 쓴다 — 하드웨어가 고장(#189)이라 bbox 좌표계를
# 실기로 검증할 방법이 없다. 복구되면 아래 CPU YOLO와 같은 방식으로 바꿀 것.
FAKE_POSE_M = (0.3, 0.0, 0.0)
FAKE_DIMS_M = (0.05, 0.05, 0.05)

# ── 접근 자세(standoff/theta) 계산 (2026-08-22) ─────────────────────────────
# 사용자 지시: pose_m을 고정값 대신 실측 거리로 계산해서, 베이스가 도착했을
# 때 물체가 차체 앞 APPROACH_STANDOFF_M 지점에 오도록 전진 거리를 역산한다.
# 이어서 사용자가 "물체가 정면이 아니라 좌우로 벗어나 있으면?"이라고 물어서
# 좌우(y) 오프셋도 카메라 fx·cx로 같이 계산하게 넓혔다 — 픽셀 오프셋과 거리로
# 카메라 광학축 기준 좌우 각도를 구하는 표준 핀홀 역투영이라 메카넘(홀로노믹)
# 베이스가 곧장 옆으로 스트레이프해 정렬할 수 있다.
#
# ⚠️ 범위: 처음엔 "도착 위치(x, y)"만 풀고 방위각(theta)은 이슈 #171 팀 결정
#   전이라 0으로 미뤄뒀다. 그런데 사용자가 "파지를 위해 물체와 일직선상으로
#   마주보게 자세를 잡을 것"이라고 명시적으로 지시해서(2026-08-22), 도메인
#   코드 오너 본인의 지시로 이 자리에서 theta까지 함께 푼다 — #171을 팀
#   대신 여기서 결정하는 게 아니라, "파지하려면 물체를 정면으로 마주봐야
#   한다"는 이번 사용자 지시를 그대로 구현하는 것이다. 계산: 물체 원시 위치
#   (x_obj, y_obj)에서 베어링각 phi=atan2(y_obj, x_obj)만큼 회전해 마주보고,
#   그 방향으로 APPROACH_STANDOFF_M만큼 물러난 지점에 도착한다.
#     x_final = x_obj - STANDOFF*cos(phi), y_final = y_obj - STANDOFF*sin(phi),
#     theta_final = phi
#   이렇게 하면 도착 지점에서 물체까지 거리가 정확히 STANDOFF이고, 로봇이
#   phi만큼 돌아 있어 물체가 정면(차체 중심선상)에 온다.
#   - 카메라 장착 위치(차체 기준 오프셋)를 실측한 상수가 없어 차체 기준점과
#     같다고 근사한다. 카메라 광학축이 차체 정면 중심선과 나란하다고도
#     가정한다(둘 다 실측 전 근사 — 오차 요인).
#   - pose_m은 (state_machine.md의 "base_link 로부터 최단 거리" 관례와 일치하게)
#     **스캔 시점 base_link 기준 상대 좌표**다.
#   ⚠️ 2026-08-23: ApproachState는 더 이상 이 pose_m을 base.drive_to()에
#     넘기지 않는다(domain/task/states.py, domain/ports/base_driver.py의
#     `approach` 참고 — 실기 검증된 시각 서보 폐루프로 교체됐다). 이제 이
#     pose_m의 유일한 소비자는 SelectState의 최단 거리 정렬뿐이라, 위 odom
#     원점 정합 문제는 더 이상 치명적이지 않다 — 후보 우선순위가 조금
#     틀려도 시각 서보가 알아서 수렴한다.
APPROACH_STANDOFF_M = 0.18

# ── RGB bbox 면적 기반 거리 추정 (2026-08-23, depth 폐기 후 대체) ───────────
# depth 카메라(구조광 IR)는 체스말·축구공처럼 작고 광택 있는 물체를 거리와
# 무관하게 거의 못 본다(flying-pixel — 프레임 전체를 훑어도 물체의 진짜
# 거리값이 어디에도 없었다, 41cm/75cm 양쪽 실측 확인). 반대로 벽처럼 크고
# 평평한 면은 1% 이내로 맞았다 — 센서 자체는 정상이고 "작고 광택 있는 바닥
# 소품"이라는 이번 데모의 실제 대상에만 근본적으로 안 맞는다. baseline 역산·
# 패럴랙스 보정으로도 없는 값을 만들어낼 수는 없으므로 depth를 포기하고
# RGB만으로 거리를 추정한다.
#
# 원리: 같은 물체는 카메라에 가까울수록 bbox가 커진다 — 핀홀 모델에서
# 물체의 화면상 선형 치수는 거리에 반비례하므로 면적(=선형치수²)은 거리제곱에
# 반비례한다. 즉 distance_m = K_class / sqrt(bbox_area_px). K_class는 클래스별
# 실제 크기에 좌우되는 상수라 **클래스마다 따로 실측**해야 한다 — 이게 depth
# 방식과의 핵심 차이다: depth의 결손은 센서 한계라 고칠 방법이 없지만, 이
# 상수는 물체 하나 놓고 거리 한 번 재면 바로 채워진다.
# ⚠️ 2026-08-24: 모델을 z = K/sqrt(hw)에서 z = K/(sqrt(hw) - PADDING)으로 바꿨다.
#
# 룩을 0.40m·0.70m·1.04m 세 거리에서 재면 역산 K가 35.49 → 36.78 → 37.40으로
# 거리에 따라 단조 증가했다. 순수 핀홀이라면 K는 상수여야 하므로, 이는 검출
# bbox가 물체 실루엣보다 **항상 일정 픽셀만큼 크게** 잡히기 때문이다(검출기의
# 성질이지 물체의 성질이 아니다). 그 여유분은 물체가 작아질수록(= 멀수록)
# 상대적으로 크게 작용해 가까운 거리를 과대평가하게 만든다 — 실제로 1.04m에서
# 보정한 단일 상수를 0.40m에 쓰면 +2.1cm(+5.4%) 과대였다.
#
# sqrt(hw)에서 이 여유분을 빼주면 세 점이 전부 ±0.4cm(RMS 0.32cm) 안에
# 들어온다. 여유분은 검출기 성질이라 클래스와 무관하다고 보고 룩 3점에서
# 구한 값을 공통으로 쓰고, 나머지 클래스의 K는 각자의 실측 1점을 그대로
# 재현하도록 다시 계산했다 — 즉 이 변경은 각 클래스의 보정 거리에서는
# 이전과 완전히 같은 값을 내고, 그 거리에서 멀어질수록 룩 데이터가 옳다고
# 말하는 방향으로만 달라진다.
BBOX_PADDING_PX = 2.5

# ── observe_target 전용 오검출 게이트 (2026-08-26) ────────────────────────
#
# observe_target은 **단일 프레임** 경로라 scan_floor의 다중 프레임 합의를
# 통째로 건너뛴다. 원래는 시각 서보 루프가 매 반복 부르는 저지연 관측이라
# 그게 맞았는데, 그 루프가 Host로 넘어가면서 이제 이 서비스를 쓰는 곳은
# GRASP 진입 판정과 파지 확인 — **차가 멈춰 있는 순간**뿐이다. 그래서
# 억제를 다시 걸 여유가 생겼다.
#
# 걸지 않으면 어떻게 되는지 2026-08-26에 실기로 봤다. CARRY 자세에서
# 사무실 배경을 향해 관측하자 **닫힌 노트북을 rook 0.60으로**, 다른
# 노트북을 knight 0.49로 잡았다. identify_target은 모든 클래스 중 가장 큰
# 검출을 고르므로, 그대로 두면 노트북을 집으러 내려간다.
#
# 세 겹으로 막는다.

# (1) 신뢰도 — 사용자 지시 2026-08-26. floor_consensus.CONF_THRESHOLD(0.45)는
# 다중 프레임 합의가 뒤에서 걸러 주는 것을 전제로 넉넉히 연 값이라 여기엔
# 맞지 않는다.
#
# 실측 여유(2026-08-26): 파지 자리의 진짜 rook은 conf 0.93~0.94로 나온다.
# 여유 +0.23이라 이 게이트가 진짜 물체를 막지 않는다.
OBSERVE_CONF_THRESHOLD = 0.70

# (2) 화면상 위치 — 파지 거리(0.15~0.4m)의 바닥 물체는 화면 아래쪽에 온다.
# 배경·먼 물체는 위쪽이다. 위 오검출 5개의 bbox 아래끝이 144~240px로 전부
# 이 선 위였다(화면 높이 480).
#
# 실측 여유(2026-08-26): 파지 자리의 진짜 rook은 아래끝 383.7~383.9px로
# 나온다. 여유 +94px.
#
# ⚠️ **이 게이트가 없으면 안 된다.** 같은 실측에서 배경 노트북이 rook
# conf 0.80~0.81로 잡혔다 — 신뢰도 게이트(0.70)를 **넘는다.** 그것을 막는
# 것은 아래끝 206~215px, 즉 이 게이트뿐이다. 두 게이트가 서로 다른 오검출을
# 맡고 있으므로 어느 하나도 뺄 수 없다.
OBSERVE_MIN_BOTTOM_Y_PX = 290.0

# (3) 다중 프레임 합의 — 배경 오검출은 프레임마다 깜빡인다. 2026-08-26의
# 노트북 오검출은 7프레임 중 2번만 나왔다.
OBSERVE_CONSENSUS_FRAMES = 5
OBSERVE_CONSENSUS_MIN_HITS = 3

# 표본을 이만큼 재사용한다. identify_target이 클래스 6개를 연달아 묻는데,
# 그때마다 5프레임을 새로 뜨면 6배가 든다 — 같은 순간의 같은 표본으로
# 답하는 것이 맞고 더 빠르다.
#
# ⚠️ 수집 자체가 약 1.7초 걸린다(5프레임 x CPU 추론, 2026-08-26 실측).
# 창이 그보다 짧으면 캐시가 **항상** 만료돼 있어 아무 효과가 없다 — 처음
# 1.0초로 뒀다가 실기 로그에서 6번 연속 재수집하는 것을 보고 늘렸다.
# 창은 수집 시간보다 넉넉히 길어야 하고, 만료 기준 시각도 수집을 **마친**
# 시점이어야 한다.
OBSERVE_CACHE_SEC = 3.0

# 표본을 모으는 데 허용하는 상한. 프레임은 약 7Hz로 오고 그때마다 CPU
# 추론이 붙으므로 5장에 1~2초가 든다.
OBSERVE_COLLECT_TIMEOUT_SEC = 4.0
CLASS_DISTANCE_CALIBRATION_SQRT_PX_M = {
    # 2026-08-23 실측(핫스팟 연결 실기, observe_target 서비스로 단일 프레임
    # h×w 직접 측정 — scan_floor의 consensus 게이트(MIN_BOTTOM_Y_PX=290)는
    # 이 거리대(0.66~1.13m)의 물체를 전부 걸러내 우회했다. 줄자 실측:
    # 축구공 0.66m, 나이트 0.84m, 룩 1.04m, 퀸 1.13m(전방 거리, base_link
    # 기준 — 좌우 오프셋은 z_m 보정과 무관해 무시). K = distance_m *
    # sqrt(bbox_area_px), bbox_area_px = h*w(observe_target 응답).
    #
    # 2026-08-24에 K = distance_m * (sqrt(bbox_area_px) - BBOX_PADDING_PX)로
    # 다시 계산했다(위 주석 참고). 각 클래스의 실측 1점은 그대로 재현된다.
    # 2026-08-27: --mode scale로 17cm/27cm(줄자 100mm) 재니 f = 0.908 —
    # 9.2% 어긋나 있었다. 보정 = 35.9307 / 0.908.
    "knight": 39.5578,  # 보정 2026-08-27 (구 35.9307 = 0.84m 1점)
    # ⚠️ 2026-08-26: 28.3382에서 고침. 1.13m **한 점**으로 잡은 K를 파지
    # 거리 0.15m까지 7.5배 외삽하고 있었고, 실제로 깨져 있었다.
    #
    # --mode scale(줄자로 100mm 밀고 판독 변화를 봄)에서 f = 0.807 —
    # 수평으로 100mm 움직였는데 판독은 80.7mm만 변했다. 2026-08-25에
    # 같은 물리 18cm를 14.4cm로 읽었던 것(배율 0.800)과 같은 값이고,
    # 그 둘은 **서로 다른 모델(train-8 / train-9)에서 나왔다.**
    # 즉 모델 교체 탓이 아니라 원래부터 K가 틀려 있었다.
    #
    # 보정 = 28.3382 / 0.807. 보정 뒤 두 가지가 저절로 맞는다:
    #   턱 선   rook 0.1757 vs queen 0.1761  (물리적으로 같은 자리, 차이 0.4mm)
    #   좌우 영점 29.5 / 31.4 / 29.7 mm      (카메라 옆 오프셋, 폭 1.9mm)
    # 독립적으로 잰 값들이 두 물리량으로 수렴하므로 보정이 옳다고 본다.
    # 2026-08-27: 35.1155에서 다시 고침. --mode scale로 17cm/27cm(줄자 100mm)
    # 두 자리를 재니 f = 0.916 — 아직 8.4% 어긋나 있었다. 보정 = 35.1155 / 0.916.
    #
    # ⚠️ tools/grasp_geometry_calibrate.py의 CURRENT_K 사본이 이때 35.1155로
    # 안 갈아 끼워져 있어서, 그 도구가 낸 "보정값"이 스테일 28.3382 기준으로
    # 계산돼 있었다(30.94 — 틀린 값). 여기 실제 배포값을 기준으로 다시 계산해
    # 38.3357을 쓴다. 두 파일을 항상 같이 고칠 것.
    "queen": 38.3357,  # 보정 2026-08-27 (구 35.1155, 2026-08-26 1점)
    # 2026-08-27: --mode scale로 재니 f = 0.922 — 7.8% 어긋나 있었다.
    # 3점 최소제곱이라 먼 거리(0.40~1.04m)에서는 안정적이었지만, 파지
    # 거리대(0.18m)로 외삽하면서 오차가 드러났다. 보정 = 34.8340 / 0.922.
    "rook": 37.7658,  # 보정 2026-08-27 (구 34.8340 = 0.40/0.70/1.04m 3점)
    # 2026-08-27 --mode k 최초 계측(train-9 모델 — 2026-08-26에 검출 자체는
    # 이미 고쳐졌다, 60프레임 0회는 train-8 시절 얘기). 0.13/0.18/0.23/0.25m
    # 4점 최소제곱. 0.10m도 재봤지만 문서상 뎁스캠 근거리 컷오프(~0.128m)
    # 아래라 그 점만 22mm 벗어났다 — 빼고 풀었더니 나머지 넷이 3mm 안으로
    # 들어왔다(포함해서 풀면 22.8918, 빼고 풀면 23.2733).
    "box": 23.2733,
    # 2026-08-27: --mode scale로 재니 f = 0.733 — **26.7% 어긋나 있었다**,
    # 네 클래스 중 가장 컸다. 보정 = 18.9592 / 0.733.
    "soccer": 25.8794,  # 보정 2026-08-27 (구 18.9592 = 0.66m 1점)
    # 2026-08-27 --mode k 최초 계측. 0.10/0.15/0.18/0.25m 4점 수집. box와 같은
    # 이유로 0.10m이 근거리 컷오프(~0.128m) 아래라 27mm 벗어나 제외 — 나머지
    # 세 점(0.15/0.18/0.25m) 최소제곱으로 풀면 잔차 ≤1.7mm(포함해서 풀면
    # 23.4568, 빼고 풀면 24.1690).
    "star": 24.1690,
}
# 이보다 작은 bbox는 너무 멀거나 오검출일 가능성이 높아 거리 추정을 시도하지
# 않는다 — "모르면 제외"(hailo_scan_mapping.py와 같은 관례).
MIN_BBOX_AREA_PX = 25.0
# 2026-08-23 실기 확인(ros2 topic list) — 실제 네임스페이스는 "ascamera_hp60c"가
# 아니라 "ascamera"다(depth_cam_rotate_node.py의 같은 날짜 경고 참고).
RGB_CAMERA_INFO_TOPIC_DEFAULT = "/ascamera/camera_publisher/rgb0/camera_info"


def _standoff_arrival_pose(x_obj, y_obj):
    """물체 원시 위치(x_obj=전방 m, y_obj=좌측 m, base_link 기준)에서, 물체를
    정면으로 마주보고 APPROACH_STANDOFF_M만큼 물러난 최종 도착 자세
    (x, y, theta_rad)를 계산한다.

    사용자 지시(2026-08-22) — "파지를 위해 물체와 일직선상으로 마주보게
    자세를 잡을 것": 베어링각 phi=atan2(y_obj, x_obj)만큼 회전해 물체를
    정면에 두고, 그 방향으로 STANDOFF만큼 물러난 지점에 도착한다. 도착
    지점에서 물체까지 거리는 정확히 STANDOFF이고, theta=phi라 도착 시
    로봇 정면(차체 중심선)이 물체를 향한다.

    rclpy 없이 순수 계산이라 실기 스크립트(자로 잰 값을 직접 넣는 수동
    테스트 등)에서도 그대로 재사용한다."""
    phi = math.atan2(y_obj, x_obj)
    x_final = x_obj - APPROACH_STANDOFF_M * math.cos(phi)
    y_final = y_obj - APPROACH_STANDOFF_M * math.sin(phi)
    return x_final, y_final, phi


def _bgr_from_image_msg(msg):
    """Image 메시지를 BGR cv2 배열로 바꾼다 — cv_bridge를 쓰지 않는다.

    ⚠️ 2026-08-23 실기 확인: cv_bridge의 컴파일된 확장(cv_bridge_boost.so)이
    numpy 1.x ABI로 빌드돼 있어 이 환경의 numpy 2.x와 안 맞는다
    (`AttributeError: _ARRAY_API not found` → 세그폴트, ultralytics/torch가
    numpy 2.x를 끌어온 뒤 발생). tools/perception/floor_observer.py의
    to_bgr()과 같은 방식(원시 바이트 버퍼를 numpy로 직접 해석)으로 우회
    한다 — 같은 환경에서 이미 동작이 증명된 경로다."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1] if enc == "rgb8" else img
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"지원하지 않는 인코딩: {msg.encoding}")


def _image_msg_from_bgr(frame, stamp, frame_id="gripper_cam"):
    """BGR cv2 배열을 Image 메시지로 바꾼다 — `_bgr_from_image_msg` 의 역방향.

    여기서도 **cv_bridge 를 쓰지 않는다.** 같은 이유다 — 이 환경의 cv_bridge 는
    numpy 2.x 에서 세그폴트를 낸다(위 함수 주석 참고). `gripper_cam_publisher_node`
    는 CvBridge 를 쓰므로 이 컨테이너에서는 켜면 안 된다.
    """
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(frame).tobytes()
    return msg


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        cb_group = ReentrantCallbackGroup()

        self._latest_frame = None
        self._frame_seq = 0
        # observe_target 다중 프레임 표본 캐시 — (표본, 수집 시각)
        self._observe_samples_cache = None
        self._observe_samples_at = 0.0
        self._rgb_fx = self._rgb_cx = None
        if _CV_AVAILABLE:
            # depth_cam_rotate_node가 내보내는 회전 보정된 컬러 스트림.
            # (예전엔 "camera/color/image_raw"를 구독했는데, 실제로 이 이름으로
            # 퍼블리시하는 노드가 없어 _on_image가 한 번도 안 불렸다 — 2026-08-21 확인)
            self.create_subscription(
                Image,
                "depth_cam/rgb/image_rotated",
                self._on_image,
                10,
                callback_group=cb_group,
            )
            # ⚠️ 토픽명은 2026-08-23 실기로 확인됨(ros2 topic list). 다만 이
            # 토픽은 회전 보정 전(depth_cam_rotate_node 이전) intrinsics다 —
            # 180도 회전이면 cx/cy가 뒤집혀야 하는데(cx'=width-cx 등) 그
            # 보정은 아직 안 했다. 좌우(y) 계산 정확도에 영향을 준다. RGB
            # bbox 면적 자체(z_m 계산)는 이 값과 무관해 문제없다.
            self.create_subscription(
                CameraInfo,
                RGB_CAMERA_INFO_TOPIC_DEFAULT,
                self._on_rgb_camera_info,
                10,
                callback_group=cb_group,
            )
        else:
            self.get_logger().warn("sensor_msgs 미설치 — 카메라 구독 비활성화")

        self.create_service(
            MonitorClearance,
            "perception/monitor_clearance",
            self._on_monitor_clearance,
            callback_group=cb_group,
        )
        self.create_service(
            ConfirmGrasp,
            "perception/confirm_grasp",
            self._on_confirm_grasp,
            callback_group=cb_group,
        )
        self.create_service(
            ObserveTarget,
            "perception/observe_target",
            self._on_observe_target,
            callback_group=cb_group,
        )

        self.declare_parameter("gripper_cam_device", GRIPPER_CAM_DEVICE_DEFAULT)
        self.declare_parameter("confirm_grasp_diff_threshold", CONFIRM_GRASP_DIFF_THRESHOLD_DEFAULT)
        # confirm_grasp 와 발행 타이머가 같은 VideoCapture 를 만진다. 콜백이
        # ReentrantCallbackGroup 이고 MultiThreadedExecutor 로 스핀하므로 정말로
        # 동시에 들어온다 — read() 중에 다른 쪽이 release() 하면 죽는다.
        self._grasp_cam_lock = threading.Lock()
        self._grasp_cam = None
        self._grasp_cam_reference = None  # 기준(빈 그리퍼) 프레임 — 그레이스케일
        if _GRASP_CAM_CV_AVAILABLE:
            self._grasp_cam_reference = self._capture_grasp_frame()
            if self._grasp_cam_reference is None:
                self.get_logger().warn(
                    "confirm_grasp: 기준 프레임 캡처 실패 — 그리퍼캠 연결/조명 확인 필요"
                )
        else:
            self.get_logger().warn("opencv 미설치 — confirm_grasp 항상 confirmed=False 반환")

        # 그리퍼캠 프레임을 토픽으로도 내보낸다 — VLA 정책(vla_inference)이 쓴다.
        #
        # 왜 여기냐면, **이 노드가 이미 장치를 소유하고 있기 때문**이다.
        # `gripper_cam_publisher_node` 를 따로 띄우면 같은 `/dev/gripper_cam` 을
        # 두 번 열어 `Device or resource busy` 가 난다(그 노드 docstring 의 경고).
        # HLD 노드 표도 perception 의 책임을 "카메라 소유"로 적고 있다.
        #
        # 기본은 0 = 끔이다. 켜기 전에는 이 노드의 동작이 전과 완전히 같다.
        self.declare_parameter("gripper_cam_publish_hz", GRIPPER_CAM_PUBLISH_HZ_DEFAULT)
        self.declare_parameter("gripper_cam_topic", GRIPPER_CAM_TOPIC_DEFAULT)
        self._gripper_cam_pub = None
        publish_hz = float(self.get_parameter("gripper_cam_publish_hz").value)
        if publish_hz > 0.0:
            if not (_CV_AVAILABLE and _GRASP_CAM_CV_AVAILABLE):
                self.get_logger().warn(
                    "gripper_cam_publish_hz 가 켜져 있지만 sensor_msgs/opencv 가 없어 발행하지 않습니다"
                )
            else:
                topic = self.get_parameter("gripper_cam_topic").value
                self._gripper_cam_pub = self.create_publisher(Image, topic, 1)
                self.create_timer(1.0 / publish_hz, self._on_gripper_cam_timer)
                self.get_logger().info(
                    f"그리퍼캠 발행: {topic} @ {publish_hz:.1f}Hz "
                    f"({GRIPPER_CAM_WIDTH}x{GRIPPER_CAM_HEIGHT}, 회전 적용됨)"
                )

        self.declare_parameter("scan_floor_enabled", SCAN_FLOOR_ENABLED_DEFAULT)
        self.declare_parameter("hailo_hef_path", HAILO_HEF_PATH_DEFAULT)
        self.declare_parameter("cpu_yolo_model_path", CPU_YOLO_MODEL_PATH_DEFAULT)
        self._hailo_model = None
        self._hailo_output_shape = None
        self._hailo_input_size = None
        self._cpu_yolo_model = None
        if _HAILO_AVAILABLE:
            self._load_hailo_model()
        elif _CPU_YOLO_AVAILABLE:
            self._load_cpu_yolo_model()
        else:
            self.get_logger().warn(
                "hailo_platform·ultralytics 둘 다 미설치 — scan_floor 항상 빈 목록 반환"
            )

        if self._hailo_model is not None:
            backend_state = "Hailo"
        elif self._cpu_yolo_model is not None:
            backend_state = "CPU YOLO(ultralytics, Hailo 하드웨어 고장 임시 대체 — 이슈 #189)"
        else:
            backend_state = "백엔드 없음 → 항상 빈 목록"
        scan_floor_state = (
            f"{backend_state} (게이트 켜짐)"
            if self.get_parameter("scan_floor_enabled").value
            else f"{backend_state} · 게이트 꺼짐 → 빈 목록 반환"
        )
        self.get_logger().info(
            "perception_node ready "
            f"(scan_floor: {scan_floor_state}, "
            "find_box/measure_opening/monitor_clearance: NOT IMPLEMENTED)"
        )

    def _load_hailo_model(self):
        """VDevice/ConfiguredInferModel을 한 번만 만든다. 물리 Hailo-10H가
        1개뿐이라, tools/hailo/live_yolo_demo.py 같은 다른 프로세스가 이미
        VDevice를 쥐고 있으면 HAILO_OUT_OF_PHYSICAL_DEVICES로 실패한다 —
        둘 중 하나만 켜둘 것.

        ⚠️ vdevice/infer_model을 self에 안 붙이고 지역 변수로만 두면 이 함수가
        끝나는 순간 가비지 컬렉션되고, 나중에 self._hailo_model.run()을 부를 때
        "Lost communication with the server. This may happen if VDevice is
        released while the CIM is in use."로 죽는다 (2026-08-21 실기 확인 —
        MultiThreadedExecutor 탓으로 오진했다가, 단일 스레드로 바꿔도 똑같이
        죽는 걸 보고서야 진짜 원인을 찾았다). configure()가 반환하는
        ConfiguredInferModel이 부모를 안 붙잡아 주므로 노드 수명 내내
        직접 붙잡아야 한다."""
        hef_path = self.get_parameter("hailo_hef_path").value
        try:
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            self._hailo_vdevice = VDevice(params)
            self._hailo_infer_model = self._hailo_vdevice.create_infer_model(hef_path)
            self._hailo_infer_model.input().set_format_type(FormatType.UINT8)
            self._hailo_model = self._hailo_infer_model.configure()
            self._hailo_output_shape = self._hailo_infer_model.output().shape
            self._hailo_input_size = self._hailo_infer_model.input().shape[0]
            self.get_logger().info(
                f"scan_floor: Hailo-10H 모델 로드됨 {hef_path} (입력={self._hailo_input_size})"
            )
        except Exception as exc:  # noqa: BLE001 -- 장치 경합 등 다양한 원인을 전부 접는다
            self.get_logger().warn(f"scan_floor: Hailo 모델 로드 실패, 빈 목록으로 접음 ({exc})")
            self._hailo_model = None

    def _load_cpu_yolo_model(self):
        """ultralytics YOLO를 CPU로 로드한다. Hailo-10H 하드웨어 고장(이슈 #189)
        임시 대체 — 모듈 상단 CPU_YOLO_* 상수 경고 참고. 클래스 구성이 Hailo
        모델과 다르다(cpu_yolo_scan_mapping.py 참고)."""
        model_path = self.get_parameter("cpu_yolo_model_path").value
        try:
            self._cpu_yolo_model = YOLO(model_path)
            self.get_logger().info(f"scan_floor: CPU YOLO 모델 로드됨 {model_path}")
        except Exception as exc:  # noqa: BLE001 -- 파일 없음 등 다양한 원인을 전부 접는다
            self.get_logger().warn(f"scan_floor: CPU YOLO 모델 로드 실패, 빈 목록으로 접음 ({exc})")
            self._cpu_yolo_model = None

    def _on_image(self, msg):
        self._latest_frame = msg
        self._frame_seq += 1

    def _on_rgb_camera_info(self, msg):
        self._rgb_fx = msg.k[0]
        # 2026-08-23: 이 camera_info는 회전 보정 전(depth_cam_rotate_node
        # 이전) 원본 스트림의 것인데, bbox_xyxy는 180도 회전된 프레임
        # (depth_cam/rgb/image_rotated) 기준이다. 180도 회전에서 cx는
        # width - cx로 뒤집힌다(fx는 회전과 무관해 그대로 둔다). 실기
        # 확인된 해상도 640×480(ascamera_node 로그, hp60c 기본값) 기준.
        self._rgb_cx = msg.width - msg.k[2]

    def _approach_pose_m(self, class_name, bbox_xyxy):
        """검출 bbox(회전 보정된 RGB 프레임 기준 픽셀)로 최종 도착 자세
        (x, y, theta)를 구한다 — 단위 m/rad, 스캔 시점 base_link 기준.

        모듈 상단 "RGB bbox 면적 기반 거리 추정" 경고 참고 — depth 카메라는
        이번 데모 소품(작고 광택 있는 바닥 물체)을 근본적으로 못 봐서 폐기
        했다. 대신 bbox_area_px = 폭×높이에서 distance_m = K_class /
        sqrt(bbox_area_px)로 거리(z_m)를 구하고, 원래 RGB bbox 중심의 픽셀
        오프셋을 핀홀 역투영해 좌우(y_obj)를 구한다. 그 다음 물체 원시 위치
        (x_obj=z_m, y_obj)에서 베어링각 phi=atan2(y_obj, x_obj)만큼 회전해
        물체를 정면으로 마주보고, 그 방향으로 APPROACH_STANDOFF_M만큼 물러난
        지점을 반환한다(사용자 지시 — "파지를 위해 물체와 일직선상으로
        마주보게").

        다음 중 하나라도 있으면 **`None`** — 호출자가 이 검출을 후보에서
        제외해야 한다는 신호다:
        - bbox_area_px < MIN_BBOX_AREA_PX(너무 멀거나 오검출 가능성)
        - class_name의 K_class가 아직 실측 안 됨(CLASS_DISTANCE_CALIBRATION_
          SQRT_PX_M이 None) — 이 경우 bbox_area_px를 경고 로그에 남겨
          실측 자료로 쓸 수 있게 한다
        - RGB camera_info를 아직 못 받음(self._rgb_fx가 None)"""
        x1, y1, x2, y2 = bbox_xyxy
        bbox_area_px = (x2 - x1) * (y2 - y1)
        if bbox_area_px < MIN_BBOX_AREA_PX:
            return None

        k_class = CLASS_DISTANCE_CALIBRATION_SQRT_PX_M.get(class_name)
        if k_class is None:
            self.get_logger().warn(
                f"scan_floor: {class_name} 거리 보정값 미실측 — 후보에서 제외 "
                f"(실측용: bbox_area_px={bbox_area_px:.1f})"
            )
            return None

        if self._rgb_fx is None:
            return None

        # MIN_BBOX_AREA_PX=25(sqrt=5)가 이미 BBOX_PADDING_PX보다 크게 걸러주지만,
        # 분모가 0에 붙으면 거리가 발산하므로 명시적으로 막는다.
        effective_px = math.sqrt(bbox_area_px) - BBOX_PADDING_PX
        if effective_px <= 0.0:
            return None
        z_m = k_class / effective_px
        u = (x1 + x2) / 2.0
        y_obj = -(u - self._rgb_cx) * z_m / self._rgb_fx
        return _standoff_arrival_pose(z_m, y_obj)

    def _scan_floor_detections_hailo(self, frame):
        canvas = self._letterbox(frame, self._hailo_input_size)

        bindings = self._hailo_model.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(canvas))
        bindings.output().set_buffer(np.empty(self._hailo_output_shape, dtype=np.float32))
        self._hailo_model.run([bindings], timeout=1000)
        detections_by_class = bindings.output().get_buffer()

        detections = []
        track_id = 0
        for class_id, dets in enumerate(detections_by_class):
            object_class = object_class_for_hailo_id(class_id)
            if object_class is None:
                continue  # 매핑 미확정 클래스(container/box 등) — 바닥 후보에서 제외
            class_name = HAILO_CLASS_NAMES[class_id]
            for det in dets:
                score = float(det[4])
                if score < HAILO_SCORE_THRESHOLD:
                    continue
                track_id += 1
                detections.append(self._make_detection(track_id, object_class, score))
                self.get_logger().info(
                    f"scan_floor(Hailo): {class_name}->{object_class} score={score:.2f} "
                    f"track_id={track_id}"
                )
        return detections

    def _yolo_raw_frame_detections(self, frame):
        """프레임 한 장의 ultralytics 추론 결과를 합의 필터 입력 형식
        `[(raw_cls, conf, bbox_xyxy), ...]`으로 바꾼다. domain ObjectClass
        매핑이나 거리 계산은 하지 않는다 — 합의로 확정된 뒤에만 한다
        (미확정 검출에 계산을 낭비하지 않는다, floor_consensus.py 참고).

        HANDOFF.md 동작점의 conf 0.45(CONF_THRESHOLD)로 admission을 넉넉하게
        열어둔다 — 오검출 억제는 여기가 아니라 다중 프레임 합의가 한다."""
        results = self._cpu_yolo_model.predict(frame, verbose=False)[0]
        raw_detections = []
        for box in results.boxes:
            score = float(box.conf[0])
            if score < CONF_THRESHOLD:
                continue
            class_name = results.names[int(box.cls[0])]
            bbox_xyxy = tuple(float(v) for v in box.xyxy[0])
            raw_detections.append((class_name, score, bbox_xyxy))
        return raw_detections

    def _collect_cpu_yolo_frames(self, n_frames, timeout_sec):
        """정지 상태를 전제로 서로 다른 n_frames개 프레임에 대해 YOLO 추론을
        돌려 raw 검출 리스트를 모은다(다중 프레임 합의 필터의 입력 형식,
        floor_consensus.confirmed_tracks 참고).

        카메라 콜백(_on_image)은 MultiThreadedExecutor의 다른 스레드에서
        계속 돈다 — 여기서 rclpy.spin_once()를 부르면 이 실행기 설정에서는
        불필요한 중첩 스핀이라 안 쓴다. 대신 짧게 자면서 self._latest_frame
        이 새 메시지 객체로 바뀌었는지 identity로 확인한다(같은 프레임을
        두 번 세지 않기 위함)."""
        frames = []
        last_msg = None
        deadline = time.monotonic() + timeout_sec
        while len(frames) < n_frames and time.monotonic() < deadline:
            current = self._latest_frame
            if current is None or current is last_msg:
                time.sleep(0.01)
                continue
            last_msg = current
            frame = _bgr_from_image_msg(current)
            frames.append(self._yolo_raw_frame_detections(frame))
        return frames

    def _scan_floor_detections_cpu_yolo(self):
        """ultralytics CPU 추론 + 다중 프레임 합의 필터(HANDOFF.md 2026-08-23
        검증: 산포 0.2~1.1px, 순도 1.00). 단일 프레임 검출을 그대로 쓰지
        않는다 — 정지 상태에서 CONSENSUS_N_FRAMES장을 새로 모아
        floor_consensus.confirmed_tracks()로 확정된 물체만 후보로 낸다."""
        frames = self._collect_cpu_yolo_frames(CONSENSUS_N_FRAMES, CONSENSUS_COLLECT_TIMEOUT_SEC)
        if len(frames) < CONSENSUS_N_FRAMES:
            self.get_logger().warn(
                f"scan_floor(CPU YOLO): {CONSENSUS_COLLECT_TIMEOUT_SEC}s 안에 "
                f"{len(frames)}/{CONSENSUS_N_FRAMES}프레임만 모임 — 있는 만큼으로 합의 시도"
            )
        if not frames:
            return []

        tracks = confirmed_tracks(frames, CONSENSUS_N_FRAMES)

        detections = []
        track_id = 0
        for t in tracks:
            object_class = object_class_for_cpu_yolo_class_name(t.label)
            if object_class is None:
                continue  # 매핑 미확정 클래스(box 등) — 바닥 후보에서 제외

            bbox_xyxy = track_bbox_xyxy(t)
            approach_pose = self._approach_pose_m(t.label, bbox_xyxy)
            if approach_pose is None:
                # 상세 사유(bbox 너무 작음/보정값 미실측/camera_info 없음)는
                # _approach_pose_m 내부에서 필요한 경우에만 따로 경고한다.
                continue
            x_final, y_final, theta_final = approach_pose

            track_id += 1
            score = max(t.confs)
            detections.append(
                self._make_detection(
                    track_id, object_class, score, pose_m=(x_final, y_final, 0.0), yaw_rad=theta_final
                )
            )
            self.get_logger().info(
                f"scan_floor(CPU YOLO, 합의 {len(t.frames)}/{CONSENSUS_N_FRAMES}프레임): "
                f"{t.label}->{object_class} score={score:.2f} purity={t.purity:.2f} "
                f"spread={t.spread:.1f}px x={x_final:.3f} y={y_final:.3f} "
                f"theta={theta_final:.3f} track_id={track_id}"
            )
        return detections

    @staticmethod
    def _make_detection(track_id, object_class, score, pose_m=None, yaw_rad=0.0):
        """Detection 메시지를 만든다.

        `pose_m`을 주면 그걸 쓴다(CPU YOLO — RGB bbox 면적 기반 접근 자세,
        모듈 상단 "RGB bbox 면적 기반 거리 추정" 경고 참고). 안 주면 자리표시자 FAKE_POSE_M을
        쓴다(Hailo — 하드웨어 고장(#189)으로 bbox 좌표계를 실기로 검증할 방법이
        없어 아직 자리표시자에 머문다). `yaw_rad`도 같은 이유로 CPU YOLO는
        _standoff_arrival_pose()가 계산한 값을, Hailo는 기본값 0.0을 쓴다."""
        x, y, z = pose_m if pose_m is not None else FAKE_POSE_M
        return Detection(
            track_id=track_id,
            cls=object_class,
            pose=Point(x=x, y=y, z=z),
            dims=Vector3(x=FAKE_DIMS_M[0], y=FAKE_DIMS_M[1], z=FAKE_DIMS_M[2]),
            yaw_rad=yaw_rad,
            confidence=score,
        )

    def _gate_observe_detections(self, detections):
        """observe_target 전용 오검출 게이트 — 신뢰도와 화면상 위치.

        걸러낸 것을 세어 돌려준다. 조용히 버리면 "왜 아무것도 안 잡히나"를
        실기에서 추적할 수 없다."""
        kept, weak, high = [], 0, 0
        for class_name, score, bbox in detections:
            if score < OBSERVE_CONF_THRESHOLD:
                weak += 1
                continue
            if bbox[3] < OBSERVE_MIN_BOTTOM_Y_PX:
                # 화면 위쪽 = 멀거나 바닥이 아니다. 파지 거리의 물체는
                # 아래쪽에 온다.
                high += 1
                continue
            kept.append((class_name, score, bbox))
        return kept, weak, high

    def _observe_samples(self):
        """정지 전제 다중 프레임 표본. 캐시가 살아 있으면 재사용한다.

        캐시를 두는 이유: identify_target이 클래스 6개를 연달아 묻는데,
        그때마다 5프레임을 새로 뜨면 6배가 든다. 같은 순간을 묻는 질문이니
        같은 표본으로 답하는 것이 맞고 더 빠르다."""
        now = time.monotonic()
        if (self._observe_samples_cache is not None
                and now - self._observe_samples_at < OBSERVE_CACHE_SEC):
            return self._observe_samples_cache

        frames = self._collect_cpu_yolo_frames(
            OBSERVE_CONSENSUS_FRAMES, OBSERVE_COLLECT_TIMEOUT_SEC)
        gated, weak_total, high_total = [], 0, 0
        for frame_detections in frames:
            kept, weak, high = self._gate_observe_detections(frame_detections)
            gated.append(kept)
            weak_total += weak
            high_total += high
        if weak_total or high_total:
            self.get_logger().info(
                f"[observe] 게이트 탈락 — 신뢰도<{OBSERVE_CONF_THRESHOLD} {weak_total}건, "
                f"화면 위쪽(y<{OBSERVE_MIN_BOTTOM_Y_PX:.0f}) {high_total}건 "
                f"({len(frames)}프레임)")
        self._observe_samples_cache = gated
        # 수집을 **마친** 시각을 쓴다. 시작 시각을 쓰면 수집에 걸린 시간이
        # 창에서 먼저 깎여 나가 캐시가 거의 즉시 만료된다.
        self._observe_samples_at = time.monotonic()
        return gated

    def _on_observe_target(self, request, response):
        """정면 목표 하나를 관측한다. GRASP 진입 판정과 파지 확인이 쓴다.

        ⚠️ 2026-08-26에 단일 프레임에서 다중 프레임 합의로 바꿨다. 원래는
        시각 서보 루프가 매 반복 부르는 저지연 관측이라 최신 한 장이면
        충분했는데(노이즈는 다음 반복이 고친다), 그 루프가 Host로 넘어가면서
        이제 이 서비스를 쓰는 곳은 **차가 멈춰 있는 판정 순간**뿐이다.
        되돌리는 반복이 없으므로 한 장의 오검출이 그대로 결정이 된다.

        같은 날 실기: CARRY 자세에서 사무실 배경을 향해 관측하자 닫힌
        노트북을 rook 0.60으로 잡았다. 7프레임 중 2번만 나왔으므로 합의가
        걸러 낸다.

        CPU YOLO 백엔드 전용. 모델 미로드·프레임 없음·합의 미달은 전부
        found=False — "모르면 실패" 관례. 여러 후보가 있으면 가장 큰(=가까운)
        것을 고른다."""
        response.found = False
        response.x = 0.0
        response.h = 0.0
        response.w = 0.0
        response.metric_ok = False
        response.forward_m = 0.0
        response.lateral_m = 0.0
        if self._cpu_yolo_model is None or self._latest_frame is None:
            return response

        samples = self._observe_samples()
        boxes = []
        for frame_detections in samples:
            candidates = [d for d in frame_detections if d[0] == request.raw_cls]
            if candidates:
                # 가장 큰 높이 = 가장 가까운 것.
                boxes.append(max(candidates, key=lambda d: d[2][3] - d[2][1])[2])

        if len(boxes) < OBSERVE_CONSENSUS_MIN_HITS:
            if boxes:
                self.get_logger().info(
                    f"[observe] {request.raw_cls} {len(boxes)}/{len(samples)}프레임 — "
                    f"합의 미달(최소 {OBSERVE_CONSENSUS_MIN_HITS}) → 검출 없음으로 본다")
            return response

        # 프레임마다 조금씩 흔들리므로 좌표별 중앙값을 쓴다 — 한 프레임이
        # 크게 튀어도 결과가 끌려가지 않는다.
        bbox = tuple(statistics.median(b[i] for b in boxes) for i in range(4))
        x1, y1, x2, y2 = bbox
        response.found = True
        response.x = (x1 + x2) / 2.0
        response.h = y2 - y1
        response.w = x2 - x1

        # 미터 환산 — GRASP 진입 판정이 "물체가 턱이 쓸고 갈 영역 안에
        # 있는가"를 재려면 픽셀이 아니라 거리가 필요하다. 실패하면 값을
        # 지어내지 않고 metric_ok=False로 남긴다.
        metric = self._target_offsets_m(request.raw_cls, bbox)
        if metric is not None:
            response.metric_ok = True
            response.forward_m, response.lateral_m = metric
        return response

    def _target_offsets_m(self, class_name, bbox_xyxy):
        """bbox -> (전방 거리 m, 좌우 오프셋 m). 모르면 **None**.

        `_approach_pose_m`과 같은 수식을 쓰되 정지 거리 보정 없이 물체 자체의
        위치를 낸다 — 그쪽은 "어디에 서야 하는가"를, 이쪽은 "물체가 어디에
        있는가"를 답한다."""
        ## ⚠️ 이 카메라는 정면 아래로 11.3도 기울어져 있다 (사용자 2026-08-26)
        #
        # 라이다와 **같은 각도**다. 그동안 코드 어디에도 안 적혀 있었다.
        # 아래 두 줄은 지금 그대로가 맞다 — 다만 이유를 모르면 나중에 누군가
        # "기울었으니 cos을 곱해야지"라고 고쳐서 망가뜨리기 딱 좋은 자리다.
        #
        # **좌우는 기울기와 무관하다.** 순수 pitch는 카메라 자신의 x축을
        # 회전시키지 않으므로 좌우 축은 수평으로 남는다. 아래 핀홀 좌우 식과
        # 클래스별 좌우 영점은 그대로 유효하다 — 좌우에 cos을 곱하면 틀린다.
        #
        # **전방은 조심해야 하지만, 실측이 지금 방식을 지지한다.** 기울어진
        # 광선을 따라 잰 거리는 수평 거리가 아니고 둘은 가까울수록 크게
        # 갈린다 — 바닥 물체가 카메라보다 0.10~0.15m 아래라면 수평으로 1mm
        # 움직일 때 광선 거리는 0.18m에서 0.77~0.87mm만 변한다(K를 잡은
        # 0.66~1.04m에서는 0.98~0.995mm로 사실상 같다).
        #
        # 그런데 이 비율은 **1을 넘을 수 없다.** 2026-08-26에 줄자로 재며
        # 측정한 값이 1.033 ± 0.023이었다. 즉 이 판독은 기울어진 광선 거리가
        # 아니라 수평 변위를 거의 1:1로 따라간다 — bbox 면적 모델이
        # 경험적으로 그렇게 맞춰져 있다. **cos(11.3도)을 곱하면 안 된다.**
        #
        # 같은 이유로 척도 실측의 +3.3%도 기울기로는 설명되지 않는다.
        # 기울기는 비율을 1 아래로만 끌 수 있다.
        #
        # (뎁스 **영상**에서 직접 읽은 거리는 다르다. 그쪽은 진짜 기울어진
        # 광선 거리라 수평 거리나 라이다 값과 비교하려면 기하를 적용해야 한다.)
        x1, y1, x2, y2 = bbox_xyxy
        bbox_area_px = (x2 - x1) * (y2 - y1)
        if bbox_area_px < MIN_BBOX_AREA_PX:
            return None
        k_class = CLASS_DISTANCE_CALIBRATION_SQRT_PX_M.get(class_name)
        if k_class is None or self._rgb_fx is None:
            return None
        effective_px = math.sqrt(bbox_area_px) - BBOX_PADDING_PX
        if effective_px <= 0.0:
            return None
        z_m = k_class / effective_px
        u = (x1 + x2) / 2.0
        return z_m, -(u - self._rgb_cx) * z_m / self._rgb_fx

    def _on_monitor_clearance(self, request, response):
        # 안전 원칙: 실제 측정 전까지는 항상 정지 신호. 절대 False로 바꾸지 말 것.
        self.get_logger().warn(
            "monitor_clearance: 비전 파이프라인 미구현 — contact_risk=True(정지) 반환"
        )
        response.front = 0.0
        response.left = 0.0
        response.right = 0.0
        response.top = 0.0
        response.contact_risk = True
        return response

    # ---- confirm_grasp (1단계, classical CV 임시 구현) ----
    def _open_grasp_cam(self):
        device = self.get_parameter("gripper_cam_device").value
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, GRIPPER_CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GRIPPER_CAM_HEIGHT)
        return cap

    def _capture_grasp_frame(self):
        """그리퍼캠에서 그레이스케일 프레임 한 장을 잡는다. 카메라가 없거나
        읽기에 실패하면 **None** — 호출자가 confirmed=False로 접는다.

        열려 있던 캡처가 죽어 있으면(핫플러그 재연결 등) 한 번 다시 연다.
        노출 자동조정이 끝나기 전 프레임은 실기에서 검게 나오는 게 확인됐으므로
        (2026-08-21) 앞쪽 몇 프레임은 버린다."""
        frame = self._capture_grasp_bgr()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _capture_grasp_bgr(self, warmup: bool = True):
        """그리퍼캠 프레임 한 장을 **BGR 그대로** 잡는다. 회전은 적용한 상태.

        `_capture_grasp_frame` 과 발행 타이머가 같은 캡처를 공유하기 위해
        분리했다. **장치를 여는 곳은 이 노드 하나여야 한다** — perception 이
        `/dev/gripper_cam` 을 계속 붙들고 있어서, 다른 노드가 같은 장치를 열면
        `Device or resource busy` 가 난다.

        `warmup` 은 처음 열었을 때 노출이 안정될 때까지 버리는 용도다. 연속
        발행에서는 매번 6장을 잡을 이유가 없어 끈다 — 15Hz 로 발행하면 초당
        90장을 grab 하게 되어 Pi CPU 를 그냥 태운다.
        """
        if not _GRASP_CAM_CV_AVAILABLE:
            return None
        with self._grasp_cam_lock:
            if self._grasp_cam is None or not self._grasp_cam.isOpened():
                self._grasp_cam = self._open_grasp_cam()
                warmup = True  # 방금 열었으면 노출이 안 잡혀 있다
            if self._grasp_cam is None:
                return None
            if warmup:
                for _ in range(GRIPPER_CAM_WARMUP_FRAMES):
                    self._grasp_cam.grab()
            ok, frame = self._grasp_cam.read()
            if not ok or frame is None:
                self._grasp_cam.release()
                self._grasp_cam = None
                return None
        # 장착이 뒤집혀 있다. 여기서 돌려야 GRIPPER_CAM_ROI(하단 중앙)가
        # 계속 손가락을 가리키고, VLA 정책도 학습 때와 같은 방향을 본다
        # — gripper_cam_geometry 참고.
        return orient(frame)

    def _on_gripper_cam_timer(self):
        """그리퍼캠 프레임을 토픽으로 내보낸다. 발행이 꺼져 있으면 안 불린다."""
        frame = self._capture_grasp_bgr(warmup=False)
        if frame is None:
            # 매 주기 경고를 찍으면 로그가 잠긴다 — throttle 로 낮춘다.
            self.get_logger().warn(
                "그리퍼캠 프레임 읽기 실패", throttle_duration_sec=5.0
            )
            return
        self._gripper_cam_pub.publish(
            _image_msg_from_bgr(frame, self.get_clock().now().to_msg())
        )

    @staticmethod
    def _letterbox(frame, size):
        """정사각형 size x size로 비율 유지 레터박스한다 (tools/hailo/
        live_yolo_demo.py의 letterbox()와 동일 로직)."""
        h, w = frame.shape[:2]
        scale = min(size / h, size / w)
        resized = cv2.resize(frame, (round(w * scale), round(h * scale)))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        y0 = (size - resized.shape[0]) // 2
        x0 = (size - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        return canvas

    def _grasp_roi(self, gray_frame):
        """GRIPPER_CAM_ROI(비율)를 실제 픽셀 슬라이스로 잘라낸다."""
        h, w = gray_frame.shape
        x0, y0, x1, y1 = GRIPPER_CAM_ROI
        return gray_frame[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]

    def _on_confirm_grasp(self, request, response):
        frame = self._capture_grasp_frame()
        if frame is None or self._grasp_cam_reference is None:
            self.get_logger().warn("confirm_grasp: 프레임/기준 없음 — confirmed=False 반환")
            response.confirmed = False
            response.confidence = 0.0
            return response

        # 기준(빈 그리퍼) 프레임과의 평균 절대 밝기 차이 — 정교한 검출이 아니라
        # "뭔가 달라졌다"만 보는 1단계 임시 신호다. GRIPPER_CAM_ROI로 손가락
        # 부근만 잘라서 비교한다 — 전체 프레임으로 하면 배경 변화에 묻힌다
        # (2026-08-21 실기 확인: 전체 프레임 diff는 물체 유무와 무관하게 ~1로 고정).
        # threshold는 미실측 자리 표시자이니 로그(diff_score)를 실제 파지
        # 성공/실패와 대조해 재보정한다.
        diff_score = float(
            np.mean(cv2.absdiff(self._grasp_roi(frame), self._grasp_roi(self._grasp_cam_reference)))
        )
        threshold = self.get_parameter("confirm_grasp_diff_threshold").value
        confirmed = diff_score > threshold
        confidence = max(0.0, min(1.0, diff_score / (2.0 * threshold)))

        self.get_logger().info(
            f"confirm_grasp: diff_score={diff_score:.2f} threshold={threshold:.2f} "
            f"confirmed={confirmed} confidence={confidence:.3f}"
        )
        response.confirmed = confirmed
        response.confidence = confidence
        return response

    def destroy_node(self):
        if self._grasp_cam is not None:
            self._grasp_cam.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    # 2026-08-21 실기 디버깅 메모: scan_floor의 Hailo 추론이 "Lost communication
    # with the server..."로 죽어서 한때 이 MultiThreadedExecutor가 원인인 줄
    # 알았다(서비스 콜백이 __init__과 다른 워커 스레드에서 돈다고 추정) —
    # 단일 스레드(rclpy.spin)로 바꿔도 똑같이 죽는 걸 보고 오진이었다는 걸
    # 확인했다. 진짜 원인은 _load_hailo_model()이 vdevice/infer_model을
    # self에 안 붙이고 지역 변수로 둬서 가비지 컬렉션된 것이었다 (그 함수의
    # docstring 참고). 그래서 원래대로 되돌린다 — monitor_clearance 같은
    # 안전 판정이 다른 서비스 처리에 밀리지 않게 유지한다.
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
