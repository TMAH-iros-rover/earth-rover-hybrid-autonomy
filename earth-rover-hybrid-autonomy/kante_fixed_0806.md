# 2026-08-06 작업 요약

브랜치: `feature/genie-cluster-planner-and-gps-guard`

이번에 추가/수정한 내용을 정리한 문서입니다. 기존 기본 동작(`planner.mode: motion_primitives`,
공식 SAM-TP 재현 로더, GPS 무필터)은 전부 그대로 유지되며, 아래 세 가지는 모두 **명시적으로 켜야만
작동하는 opt-in 기능**입니다.

## 1. GeNIE 논문 스타일 경로 계획 모드 (`planner.mode: genie_cluster`)

- 파일: `src/earth_rover/planning/motion_primitive_planner.py`
- GeNIE 논문(Sec III-D, Algorithm 1)의 "후보 경로 샘플링 → top-K 선별 → 클러스터링(실루엣 기반
  adaptive k-means) → 근접 클러스터 병합 → GPS 목표 방향과 각도가 가장 가까운 클러스터 선택"
  파이프라인을 image-space로 이식했습니다. (`scripts/06_path_planning.py`가 만든 BEV/미터 단위
  재현과 같은 알고리즘이지만, 카메라 캘리브레이션이 없는 라이브 시스템 특성상 픽셀 좌표에서 동작.)
- 기존 `motion_primitives`(고정 7~11개 heading + 가중합 점수 + EMA/커밋타임 히스테리시스)는
  전혀 건드리지 않았고, `genie_cluster`는 완전히 별도 모드로 추가해서 `--planner-mode genie_cluster`
  로 A/B 테스트하고 언제든 롤백할 수 있게 했습니다.
- 클러스터 구성이 매 프레임 바뀔 수 있어서, "논문 방식으로 매 프레임 새로 고르되 확신 있게 유지"하는
  자체 hold/confirm 로직을 얹었습니다. 안전 관련 near-field 하드컷은 홀드 중에도 매 프레임
  새로 검사합니다.
- 설정: `configs/default.yaml`의 `planner:` 섹션에 `genie_n_candidates`, `genie_top_k`,
  `genie_k_max`, `genie_waypoint_count`, `genie_curvature_jitter`,
  `genie_merge_threshold_ratio`, `genie_switch_heading_deadband_deg` 추가.

## 2. 직접 학습한 SAM-TP 체크포인트 연결 (`--predictor-backend hf`)

- 파일: `training/hf_sam_tp_predictor.py` (신규), `training/run_sam_tp_sdk_shadow.py`,
  `scripts/run_sam_tp_sdk_shadow.sh`
- `runs/sam_tp/best_sam_tp.pt`(HuggingFace `transformers.Sam2Model` 기반, prompt 없이
  `not_a_point_embed`/`no_mask_embed`를 학습된 prompt token으로 사용)를
  `earth-rover-hybrid-autonomy/checkpoints/sam_tp/best_sam_tp.pt`로 복사해서 배치
  (git에는 추적 안 함 — `.gitignore`에 `checkpoints/` 추가, 다른 체크포인트들과 동일한 원칙).
- 기존 `SamTpPredictor`는 외부 upstream 저장소(`sam2.sam_tp`, 별도 GPU 환경)가 있어야만
  동작하는 "공식 재현" 로더라서 이 체크포인트를 못 씀 — `HfSamTpPredictor`를 새로 만들어서
  같은 `SamTpPrediction` 인터페이스로 low-res 마스크 로짓을 원본 프레임 해상도로 업샘플링해 반환.
- 실행: `PREDICTOR_BACKEND=hf ./scripts/run_sam_tp_sdk_shadow.sh` (외부 upstream repo, conda env
  없이 프로젝트 기본 파이썬 환경에서 바로 실행됨).

## 3. GPS 위경도 스파이크(순간 튐) 방어 로직

- 파일: `src/earth_rover/navigation/checkpoint_route.py` (`_sanitize_position`)
- 문제: 라이브 로그에서 GPS 위경도가 한 샘플만 수십 미터씩 순간 이동했다가 되돌아오는 현상 확인.
  기존에는 heading(나침반) 값에만 이상 회전속도 거부 로직(`_sanitize_heading`,
  `max_heading_rate_deg_per_sec`)이 있었고, 위치(lat/lon) 자체에는 아무 필터가 없어서
  `target_bearing_deg`/`distance_to_target_m`에 스파이크가 그대로 들어가고 있었음.
- 해결: 직전에 받아들인 위치 대비 이번 fix까지의 haversine 거리 ÷ 경과시간 = "함의된 속도"가
  `max_gps_jump_speed_mps`를 넘으면 이번 fix를 버리고 직전 위치를 유지. 단, 기준 시각은
  갱신하지 않아서 같은 새 위치가 계속 들어오면 dt가 커지며 결국 진짜 이동으로 받아들여짐
  (heading 필터와 동일한 anti-lockout 설계).
- **기본값은 `null`(비활성)**입니다. heading은 로버의 물리적 회전 한계라는 명확한 상한이 있지만,
  GPS는 정지 상태에서도 몇 미터씩 흔들리는 게 정상이라 실제 로그(스파이크 크기/주기)를 보고
  값을 정해야 합니다. 값 확정하려면 실제 GPS 로그를 공유해주세요.

## 테스트

`PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider -q` 기준 364개 통과,
신규 테스트 29개 추가(`test_motion_primitive_planner.py`, `test_hf_sam_tp_predictor.py`,
`test_sam_tp_sdk_shadow.py`, `test_checkpoint_route.py`). 이번 작업과 무관한 기존 실패 1개
(`test_mission1_live_profile_has_bounded_deadzone_compensation` — `mission1_live.yaml`의
`max_linear=0.3`이 테스트 기준 0.15 초과)는 그대로 남아있음, 손대지 않았습니다.
