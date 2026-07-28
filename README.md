# GeoTrafficView-3D v2

위성 지도 위에 **실측 CCTV**를 올리고, 영상에서 추출한 교통 객체를 **실세계 좌표 3D**로 표출한 뒤
**공간 DB(PostGIS)** 에 적재해 **로컬 LLM(Ollama Qwen)** 으로 자연어 질의하는 교통 디지털 트윈.

> v1(Cesium 뷰어 + 정밀도로지도 점군/벡터 렌더링)에서 **MapLibre 웹맵 스택만** 추출한 버전입니다.
> 무거운 점군·Cesium 의존성을 제거하고, 위성 타일맵 + 이중 패널(CCTV↔3D) + DB + LLM에 집중합니다.
>
> 📘 **설계·측정·근거**
> - [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) — **I권 · 초기 구축**: 수집·탐지·기하변환·웹맵·DB·LLM,
>   TOPIS+HD맵 cm급 결합, ITS 마커 오차 실측(93~793m), 캘리브레이션 3단계 스펙트럼
> - [TECHNICAL_DOCS_2.md](TECHNICAL_DOCS_2.md) — **II권 · 차로 정렬**: 정밀도로지도 6종 병합,
>   HMM 맵매칭·차로 고정, 노면표시 자동 정합, 정적 오탐 마스크, GT 라벨링과 그 한계,
>   운동 필드 학습, deck.gl 3D 표출. **효과가 없었던 시도도 근거와 함께 기록**

```
┌ MapLibre 위성맵 (Esri World Imagery / Sentinel-2 토글)
│   └ CCTV 마커 (ITS 전국 + TOPIS 서울, 실측 좌표)
│        └ 클릭 → 이중 패널
│             ├ 좌: CCTV 영상 + 핑크 3D 박스(bbox_3d)
│             └ 우: 지도 위 fill-extrusion 3D 박스(실좌표) — 영상 클럭으로 동기
├ 💬 LLM 채팅 (Qwen NL2SQL + 지도 제어 "송도IC로 이동")
└ 🎯 GCP 캘리브레이션 페이지 (카메라 선택 · ROI · 차선방향 · HD맵 스냅 · 자동 부트스트랩)
```

---

## 1. 빠른 시작

```bash
# 0) 파이썬 의존성
pip install -r requirements.txt

# 1) 웹맵 실행
cd webmap && npm install && npm run dev      # → http://localhost:5174

# 2) (선택) PostGIS + LLM
docker run -d --name geotraffic-db -e POSTGRES_PASSWORD=geotraffic -e POSTGRES_DB=geotraffic \
  -p 5434:5432 <postgis-image>
ollama pull qwen3:8b
```

`.env.local` (리포 루트, gitignore됨)에 인증키:
```
ITS_API_KEY=...      # https://www.its.go.kr/opendata (무료)
UTIC_API_KEY=...     # (선택) UTIC은 키+IP 인증
```
> **TOPIS(서울)는 키가 필요 없습니다.**

---

## 2. 프로젝트 구조

```
webmap/                     MapLibre 웹앱 (포트 5174)
  index.html / src/main.js    위성맵 · CCTV 마커 · 이중 패널
  src/player.js               영상+캔버스 3D박스 ↔ 지도 fill-extrusion 동기 재생
  src/chat.js                 LLM 채팅 패널(질의 + 지도 제어)
  calibrate.html / src/calibrate.js   GCP·ROI·차선방향 캘리브레이션
  src/homography.js           DLT 호모그래피 + 저차원 보정(하이브리드)
  vite.config.js              /api/{gproj,save-gproj,autocalib,ask} 개발 서버 엔드포인트
  public/data/                cameras.geojson · replay/*.json.gz · footage/*.mp4 · hdmap/

trafficlab/                 TrafficLab-3D 엔진 이식본(비-GUI)
  projection/g_projection.py   왜곡보정→호모그래피→시차→3D 리프팅
  motion/kinematics.py         속도·방향 스무딩(TrackSmoother)
  inference/pipeline.py        YOLO+ByteTrack → .json.gz
  io/                          G-Projection 스키마 / gzip replay

data_ingest/                CCTV 소스 클라이언트
  fetch_its_cctv.py            ITS OpenAPI(전국 고속·국도, HLS)
  fetch_topis_cctv.py          서울 TOPIS(공개·키불필요, HLS)
  fetch_utic_cctv.py           UTIC(도심, 키+IP 인증)
  match_streams.py             경찰청 CCTV ↔ ITS 스트림 공간매칭
  grab_frame.py                스트림 → 스냅샷/클립

tools/                      파이프라인 도구
  its_grab.py                  ITS 최근접 카메라 클립 캡처
  add_location.py              지역 추가(캡처→캘리브→추론→웹맵→DB) 원스톱
  run_city.py                  다중 카메라 배치 자동화
  make_calibration_template.py placeholder 캘리브레이션 생성
  auto_calibrate_vp.py         자동 캘리브레이션(소실점+IPM+OSM 방위)
  export_hdmap_snap.py         정밀도로지도 다중 데이터셋 병합 + 주행방향 검증
  reproject.py                 캘리브레이션 변경 시 재투영(YOLO 재실행 없이)
  lane_snap.py                 차선 정렬(그리디) — HD맵↔OSM 채점 폴백
  mapmatch.py                  HMM 맵매칭(Viterbi) + Frenet 재구성 + 차로 고정
  lane_overlay.py              캘리브레이션 검수 게이트(차선을 영상에 순투영)
  autocalib_lanes.py           노면표시 자동 정합(카메라 자세/유사변환 Chamfer-DT)
  gcp_candidates.py            GCP 클릭 지점 추천(모서리형·지면·구역 분산)
  static_mask.py               박힌 안내문구 오탐 마스크(후보 제시 → 사람 승인)
  hdmap_align.py               HD맵 ↔ 위성영상 Chamfer/DT 정합(표시 전용)
  motion_field.py              운동 필드 v(link, s) 학습
  turn_counts.py               링크 전이 · 방향별 회전교통량
  bridge_to_webmap.py          결과 → 웹맵 등록(+PostGIS 적재)
  refresh_markers.py           ITS/TOPIS 마커 갱신(replay 카메라 보존·dedup)
  eval/extract_frames.py       평가용 프레임 추출
  eval/evaluate.py             GT 대비 검출(AP50/P/R)·추적(MOTA/IDF1/IDSW) 평가
  eval/check_calibration.py    캘리브레이션 타당성 자동 점검(GT 불필요)
  eval/sim_track_test.py       추적 A/B 통제 실험(정답을 아는 시뮬레이션)
  eval/label_feasibility.py    라벨 가능 구간 진단(사람 시간 쓰기 전에)
  retrack.py                   검출 고정·연관만 교체(ByteTrack ↔ BEV ↔ BEV+차로) A/B/C

trafficlab/motion/bev_tracker.py   BEV(지면) 추적기 — OC-SORT(ORU/OCM/OCR) + 투영공분산 마할라노비스
trafficlab/motion/lane_graph.py    차선 그래프 + Frenet(투영·접선·경로거리·전진)
trafficlab/motion/map_matching.py  Viterbi 맵매칭 · 차로 고정 · 등장성 회귀
trafficlab/motion/motion_field.py  운동 필드 빌드/조회
webmap/src/vehicles.js             deck.gl 3D 차량 메시 + 트랙 시간축 보간

webmap/label.html + src/label.js   GT 라벨러(박스·트랙ID, 이전프레임 복사)
eval/<LOC>/labels.json             정답 라벨 · metrics.json 평가 결과

db/    schema.sql · ingest_postgis.py · ingest_sqlite.py
llm/   nl2sql.py (Ollama Qwen → SQL → 답변, PostGIS/SQLite 자동 전환)
location/<LOC>/   G_projection_<LOC>.json · cctv_<LOC>.png · footage/clip.mp4
                  static_mask.png(정적 오탐) · gcp_candidates.json(클릭 추천)
output/           추론 결과 .json.gz (모델/설정별)
models/           yolov8s-visdrone.pt 등
```

---

## 3. 워크플로우 — 카메라 한 대 추가하기

```bash
# ① 지역 추가 (ITS 최근접 카메라 자동 선택)
python tools/add_location.py --loc PANGYO_2 --center-lon 127.1004 --center-lat 37.3997
#    또는 직접 스트림 지정 (TOPIS 등)
python data_ingest/fetch_topis_cctv.py --bbox 126.86 126.92 37.55 37.60 --name sangam
python tools/add_location.py --loc SANGAM01 --url <hlsUrl> --lon 126.8829 --lat 37.5807

# ② 캘리브레이션 — 검수 게이트부터
python tools/lane_overlay.py    --loc PANGYO_2          # 차선이 도로에 얹히는가? (통과 못 하면 아래로)
python tools/autocalib_lanes.py --loc PANGYO_2 --apply  # 노면표시 자동 정합(QC 이미지 확인 후)
python tools/gcp_candidates.py  --loc PANGYO_2          # 영상에서 클릭할 지점 추천
#    → http://localhost:5174/calibrate.html
#      카메라 선택 → 🧲 HD맵 스냅 + 🎯 GCP 후보 안내 → GCP 1~3점 → [계산 & 저장]
#      ⚠ 4점 이상 클릭하면 전체 재계산되어 자동 정합 결과가 버려집니다

# ③ 정적 오탐 마스크 (카메라당 1회 · 사람 승인 필수)
python tools/static_mask.py --loc PANGYO_2 --review      # 후보 크롭 확인
python tools/static_mask.py --loc PANGYO_2 --accept 0 1  # 글자만 승인(차량이면 기각)

# ④ 본 파이프라인 — 순서가 중요합니다 (YOLO 재실행 불필요)
python tools/reproject.py   --loc PANGYO_2 --source webmap/public/data/replay/pangyo_2.json.gz
python tools/static_mask.py --loc PANGYO_2 --apply output/reprojected/PANGYO_2/clip.json.gz
python tools/mapmatch.py    --loc PANGYO_2 --src output/reprojected/PANGYO_2/masked.json.gz --center-lane
python tools/bridge_to_webmap.py --replay output/mapmatch/PANGYO_2/clip.json.gz --loc PANGYO_2 --ingest

# ⑤ 자연어 질의
python llm/nl2sql.py "가장 혼잡한 카메라는?"      # 또는 웹맵 좌하단 💬 채팅
```

> **마스킹은 검출을 지우므로 맨 앞**, 재투영은 캘리브레이션을 바꾼 뒤 반드시, 맵매칭은 그 위에.
> `mapmatch.py`를 `--src` 없이 돌리면 **이미 맵매칭된 웹맵 replay를 다시 읽어 이중 적용**됩니다.
> HD맵 커버리지가 없는 카메라는 `mapmatch`가 자동으로 OSM 중심선으로 폴백합니다.

정밀 추론(원거리 소형차 개선)이 필요하면:
```bash
python run_inference_sahi.py --loc PANGYO_2 --source location/PANGYO_2/footage/clip.mp4 --slice 384
```

---

## 4. 핵심 기능

### 4.1 위성 basemap + CCTV 마커
- **Esri World Imagery**(Maxar/Airbus/CNES 계열 고해상) · **Sentinel-2 cloudless**(EOX) 토글. 둘 다 토큰 불필요.
- 마커: **ITS**(전국 고속·국도) + **TOPIS**(서울, 공개) — `tools/refresh_markers.py`로 갱신.
- replay(추론 완료) 카메라는 연두/시안, 나머지는 파랑. **dedup**으로 같은 카메라 중복·튐 방지.

### 4.2 이중 패널 (TrafficLab Visualization 재현)
- 좌: `<video>` + `<canvas>` — `bbox_3d` 8점을 핑크 3D 박스로.
- 우: 지도 `fill-extrusion` — `sat_floor_box`(로컬 미터) → UTM52N → WGS84 → 클래스별 높이 압출.
- `video.currentTime × fps`를 마스터 클럭으로 좌우 프레임 동기.

### 4.3 캘리브레이션 (3단계 스펙트럼)
| 방식 | 클릭 | 정확도 | 사용 |
|------|------|--------|------|
| 완전 수동 GCP | **6~8점** | 최고(LOO ≤1m 목표) | `calibrate.html` ① GCP |
| **하이브리드** | **1~3점** | 높음 | 🤖 자동 → GCP 1점(평행이동)/2점(유사변환) |
| 완전 자동 | 0점 | 부트스트랩(~20m) | `auto_calibrate_vp.py` (VP+IPM+OSM 방위) |

#### 재작업 가이드 (깊이 스케일 오류를 잡는 핵심)
화면 우측 패널이 **실시간 품질 진단 + 다음 행동**을 제시합니다.
- 🎯 **권장 GCP 구역**: 세로 3구간(먼거리/중간/근거리) × 가로 3구간 오버레이 → **핑크 구역이 없어질 때까지** 클릭
- **LOO 교차검증 정확도**: 한 점을 빼고 예측한 오차(m) = **진짜 정확도**(5점 이상에서 산출)
  > **4점 GCP는 호모그래피 자유도(8)와 동수라 적합 잔차가 수학적으로 항상 0** — "RMS 0.00m"은 정확도가 아니라 *검증 불가*입니다.
  > 실측 데모: 같은 7점이라도 세로 분포 52% → LOO **4.47m**, 세로 2%로 뭉치면 → LOO **52.88m** (12배 악화)
- 📏 **투영 격자 검증**: 저장된 H로 차로(3.5m)·깊이(10m) 격자를 영상에 되그려 실제 도로와 육안 대조
- CLI: `python tools/eval/calib_guide.py --loc <LOC>` / `--all` → 분포·잉여도·LOO·기하 타당성 + 재작업 절차
- **목표치**: 점 6~10개 · 세로 분포 50%+ · **`check_calibration.py` ✓ (가장 중요)**
  > LOO는 원거리 점에서 본래 커진다(1px ≈ 수 m). 실측 예: 10점 캘리브가 LOO 59m여도 기하 타당성 ✓이고
  > 다운스트림(투영 잡음·추적)은 모두 개선됐다. **LOO 수치만 보고 점을 지우지 말 것**(깊이 고정점이 사라짐).
- 찍기 좋은 지점: 정지선 끝, 차선 점선 시작/끝, 횡단보도 모서리, 노면 화살표 꼭짓점
  (위성에서도 같은 '점'을 정확히 찍을 수 있어야 함 — 차량·그림자 금지)

- **HD맵 스냅(🧲)**: 정밀도로지도 정점(B2 노면표시·A2 차선·신호등)으로 GCP 지도클릭을 cm급 스냅.
  `python tools/export_hdmap_snap.py --dir <HDMap_UTM52N_...>`
- **ROI**: 도로영역 폴리곤 → 추론 시 도로 밖 검출 폐기(실측 19% 필터).
- **차선 방향**: CCTV에 차로선 → 실좌표 heading → `TrackSmoother` 프라이어(정지·가림 차량 방향 안정화).

### 4.4 객체 탐지
| 조합 | 검출(450f) | 3D박스 | 트랙 |
|------|-----------|--------|------|
| YOLO11x 풀프레임(COCO) | 1,830 | 1,093 | 24 |
| **VisDrone + SAHI 슬라이싱** | **7,904** | **6,288** | **152** |

### 4.5 차로 정렬 — 이탈을 구조적으로 막는다

세 단계로 강해집니다. 자세한 근거는 [TECHNICAL_DOCS_2.md §2](TECHNICAL_DOCS_2.md).

**① `lane_snap.py` — 링크 방향이 곧 정면**
정밀도로지도 `A2_LINK`는 방향성 링크이고, `FromNodeID→ToNodeID`가 주행 방향입니다.
**보유 6개 데이터셋 전부에서 역방향 0건**으로 검증됐습니다(송도 898·송도항동 1491·경부선 639·
외곽순환 1195·판교 618). 덕분에 **정지 차량도 정면/후면이 확정**됩니다.

교차로에서는 직진·좌회전 링크가 물리적으로 교차해 최근접 스냅이 틀립니다(실측: 링크 위 점의
16%가 1m 이내에서 140°+ 다른 링크에 붙음, 그중 31건 역방향). 방향 일관성을 비용에 넣어
**평균 방위오차 18.77° → 1.94°, 역방향 31 → 0건**.

HD맵이 카메라가 보는 도로를 담고 있지 않을 수 있으므로(실측 YEONSU_JCT: HD맵 방향차 89.5° vs
OSM 2.2°) **HD맵과 OSM을 같은 기준으로 채점해 좋은 쪽을 채택**합니다.

**② `mapmatch.py` — 트랙 전체를 Viterbi로**
프레임별 독립 선택은 나란한 차로 사이를 깜빡입니다. HMM(Newson & Krumm)으로 한 번에 풀면
**링크 플리커가 63~90% 감소**합니다.

**③ 차로 고정(`--lane-lock`, 기본 ON)**
그래도 남는 문제를 각각 막습니다 — 사슬 고정(횡방향 점프) · 등장성 회귀(s 후진) ·
속도 평활 재적분(정지·급변).

| | 적용 전 | 적용 후 |
|---|---|---|
| 횡방향 점프(차선 건너뛰기) | 29건 | **0건** |
| s 후진(앞뒤 떨림) | 122건 | **0건** |
| 횡오프셋 \|d\| 95% | 10.65 m | **0.000 m** |
| 속도 변화 중앙 | 10.8 km/h/frame | **2.16** |

> **한계**: 이탈을 못 하게 만든 것이지 **올바른 차로를 고른 것은 아닙니다.** 캘리브레이션이
> 부정확하면 실제와 다른 차로에 놓입니다. 또 한 트랙은 한 사슬에 머물러 **차로 변경·회전을
> 표현하지 않습니다.**

### 4.6 캘리브레이션 검수 · 자동 정합

**검수 게이트가 먼저입니다.** `lane_overlay.py`가 차선을 영상에 순투영해 도로에 얹히는지
보여 줍니다. **현재 5대 중 4대가 탈락**합니다.

`H = K·[r1|r2|t]` 분해는 카메라 높이를 그냥 내놓아 **3줄짜리 타당성 검사**가 됩니다.

| 카메라 | 분해 높이 | 판정 |
|---|---|---|
| PANGYO_2 | 38.2 m | 비현실적 |
| OKRYEON_IC | −2.7 m | 비현실적(지면 아래) |
| YEONSU_JCT / SANGAM01 / SONGDO_L020101 | 8.3 / 11.6 / 11.8 m | 타당 |

**`autocalib_lanes.py` — 노면표시 자동 정합.** 영상 차선과 HD맵 차선을 Chamfer/DT로 맞춥니다.
전제 3가지가 모두 필요합니다: 배경 중앙값 프레임 · 정적 오탐 마스크 · **도로 영역 제한**
(차량 검출이 지나간 자리 = 도로. 이게 결정적이라 마스크가 1,513 → 9,865px으로 살아났습니다).

```
정합 전 17.26px → 자세 7자유도 8.88px → 8자유도 8.54px   (곡면 대비 27.9%)
```

> **자동 정합은 거친 정렬까지입니다.** 페인트가 횡방향을 구속하지 못합니다(±8.75m 밀어도
> 비용 변동 3%). 다만 **원근 구조 오차를 지면 평면 오차로 바꿔 주므로**, 이어서
> **GCP 1~3점**(`correction3` 하이브리드)이면 마무리됩니다. 4점 이상은 전체 재계산이라 자동
> 결과가 버려집니다.

### 4.7 정적 오탐 마스크 (`tools/static_mask.py`)

ITS/TOPIS 영상은 지점명·방면을 인코딩 전에 합성하고, YOLO는 그 글자를 차량으로 잡습니다 —
**고정 위치 반복 검출이 전체의 11.8%**.

정지 차량과의 구분은 **시간축 표준편차**로 하되, 자동 확정하지 않고 **크롭을 사람이 승인**합니다.
이 장치가 실제로 작동했습니다: SANGAM01의 후보 5개는 전부 정체 차량이라 기각했습니다
(클립 2초라 신호 대기 차량이 정적으로 보임).

| GT 대비 | AP50 | P | R | MOTA | IDF1 |
|---|---|---|---|---|---|
| 원본 | 0.555 | 0.394 | 0.596 | **−0.362** | 0.441 |
| 마스크 | 0.562 | **0.683** | **0.596** | **+0.277** | **0.591** |

**재현율이 그대로**입니다 — 실제 차량은 하나도 지우지 않았습니다.
`pipeline.py`가 `location/<LOC>/static_mask.png`를 자동으로 읽어 추론 단계에서 제외합니다.

### 4.8 평가 (GT 라벨링 + MOTA/IDF1) — `tools/eval/`

**측정 없이는 개선을 증명할 수 없습니다.**

```bash
python tools/eval/label_feasibility.py --all        # ① 어디를 라벨해야 의미가 있는가
python tools/eval/extract_frames.py --loc PANGYO_2 --n 30 --stride 3
#    http://localhost:5174/label.html?loc=PANGYO_2   (드래그=박스 · C=이전프레임 복사 · S=저장)
python tools/eval/evaluate.py --loc PANGYO_2 --iou 0.5 --roi 355 275 695 445     --pred A.json.gz --pred B.json.gz
```

| 지표 | 의미 |
|------|------|
| **AP50 / P / R** | 검출 품질(class-agnostic — GT와 모델 클래스 체계가 달라도 유효) |
| **MOTA** | 1 − (FN+FP+IDSW)/GT (음수 가능) |
| **IDF1** | ID 일관성(헝가리안 전역 매칭) — 파편화에 민감 |

**`--roi` 는 부분 라벨링에 필수입니다.** GT와 예측을 같은 영역으로 제한하지 않으면 라벨 안 한
영역의 정상 검출이 전부 FP로 잡힙니다.

**어디까지 라벨할 수 있는가** — 720×480 오블리크 CCTV에서 객체는 근거리 30×27px, 중거리
20×14px, **원거리 11×8px**(최소변 p10 6.6px)입니다. 원거리는 6배 확대해도 경계 식별이 안 되고,
상시 점유 차로는 배경 중앙값 자체가 오염돼 반자동 제안도 불가합니다.
`label_feasibility.py`가 카메라별로 판정합니다(실측: PANGYO_2 불가 48% · YEONSU_JCT 불가 78%).

### 4.9 BEV 추적 A/B/C — **GT가 이전 결론을 뒤집었습니다**

```bash
python tools/retrack.py --loc PANGYO_2 --lane --max-age 90    # A/B/C 한 번에
```

파편화 지표만 보면 BEV가 크게 이깁니다:

| (SONGDO_IC, max_age 90) | 트랙수 | 수명중앙 | 0.5초미만 |
|---|---|---|---|
| A. ByteTrack | 152 | 0.78s | 42% |
| B. BEV | 66 | **5.00s** | 14% |
| C. BEV+차로구속 | 69 | 5.37s | 10% |

**그러나 근거리 GT로 재면 반대입니다:**

| (PANGYO_2, GT 47박스/9트랙/29프레임) | MOTA | IDF1 | IDSW |
|---|---|---|---|
| A. ByteTrack | **−0.362** | **0.441** | **2** |
| B. BEV | −0.426 | 0.390 | 5 |
| C. BEV+차로구속 | −0.404 | 0.407 | 4 |

**수명 증가의 일부는 잘못된 병합이었습니다.** 프록시 지표(>180km/h 연관)로는 못 잡았고
GT로만 드러났습니다. 단서 둘: ① 표본이 작습니다(IDSW 2 vs 5 = 3건 차이) ② **BEV의 원래 주장은
원거리인데 라벨은 근거리만 있습니다** — 그 구간은 현재 라벨 불가입니다.

→ **BEV 채택을 원거리 성능으로 정당화하지 마세요.** 차로 구속(C)도 3개 지점 × 4개 설정에서
이득이 없어 기본 OFF입니다.

**투영 발산 게이트**: 지면 투영이 지평선 근처에서 발산합니다(SONGDO_IC 검출의 13.7%가 300m 밖,
최대 33km). `--max-range 250 --max-sigma 25`로 걸러냅니다.

### 4.10 3D 이동체 표출 (`webmap/src/vehicles.js`)

deck.gl `MapboxOverlay`(interleaved) + `SimpleMeshLayer` 3장.

- **절차적 차량 메시** — 외부 glTF 없이 코드로 생성(라이선스·다운로드 없음).
  차체/캐빈 + **전조등·미등을 별도 레이어**로 그려 정면/후면이 눈에 보입니다.
- **시간축 보간** — 30fps 키프레임을 rAF에서 Catmull-Rom(위치) + 최단호(heading)로 채웁니다.
  키프레임 재현 오차 **0.000000 m**(5,503점 검증).
- 초기화·렌더 실패 시 기존 fill-extrusion 박스로 자동 폴백.

> **좌표 규약**: `sat_coords` heading은 `atan2(dy,dx)`(동쪽 0°·반시계), deck.gl yaw도 반시계.
> 메시를 +X 정면/+Z 상방으로 만들어 `yaw = heading`, `roll = 0`.
> 카메라마다 world 원점이 다르므로 전환 시 `setProjector(toLL)` 갱신이 필수입니다.

### 4.11 DB + LLM
- `db/ingest_postgis.py` — `geom(4326)` + `geom_utm(32652)`, GIST/시간 인덱스, `cctv_summary` 뷰, 카메라 단위 idempotent.
- `db/ingest_sqlite.py --all` — **PostGIS 없이 즉시 동작**. 동일한 테이블/뷰 구조로, 웹맵에 표출 중인
  replay를 그대로 적재(화면=DB 일치). `GEOTRAFFIC_DB_URL` 미설정 시 LLM이 자동으로 이 SQLite를 사용.
- `llm/nl2sql.py` — Ollama Qwen이 SQL 생성 → 읽기전용 실행 → 한국어 요약. `GEOTRAFFIC_DB_URL` 있으면 PostGIS.
- 웹 채팅(`/api/ask`)은 같은 스크립트를 재사용. **"송도IC로 이동"** 같은 지도 제어는 LLM 없이 즉시 처리.

---

## 5. 데이터 소스 정리

| 소스 | 범위 | 인증 | 스트림 |
|------|------|------|--------|
| **ITS** (its.go.kr) | 전국 고속도로·국도 | 무료 키 | HLS(.m3u8) |
| **TOPIS** (서울) | 서울 510대 | **불필요** | HLS(.m3u8) — cv2에 `referer` 헤더 필요 |
| **UTIC** (경찰청/도심) | 전국 도심 | 키 + **IP 등록** | 플레이어 페이지(iframe) |
| 정밀도로지도 | 국토정보플랫폼 배포 구간 | 다운로드 | GCP 스냅 + **차로 정렬** |

> ITS는 고속·국도 위주라 **도심 자율주행 거리 CCTV는 시(TOPIS/경기/세종) API 소관**입니다.

**보유 정밀도로지도 6종** — 인천송도 시범지구 · 인천광역시도 9호선(송도-항동선) ·
고속국도 1호선(경부선, 20구간) · 100호선(서울외곽순환선, 14구간) · 120호선(경인선) ·
자율주행시범지구 판교. `A2_LINK.shp` 117개(좌표계 판본 포함), UTM52N 판본 39구간.
`export_hdmap_snap.py`가 **등록 카메라를 덮는 구간만 골라 병합**합니다.

> ⚠ **커버리지는 bbox로 판단하면 안 됩니다.** 경부선처럼 긴 노선은 bbox가 거대해서 4대가
> 커버되는 것처럼 보이지만, 실제 링크 지오메트리 거리로 재면 3대입니다.

---

## 6. 현재 등록된 카메라 (예시)

| 카메라 | 지역 | 소스 | 캘리브레이션 | HD맵 | 자세 분해 높이 |
|--------|------|------|--------------|------|----------------|
| PANGYO_2 | 경부선 판교2 | ITS | 자동 VP+IPM (+노면표시 자동정합) | **605링크** | 38.2 m ⚠ |
| YEONSU_JCT | 인천대교고속도로 연수JCT | ITS | 자동 placeholder | **265링크** | 8.3 m |
| SONGDO_L020101 | 송도 시범지구 | 경찰청 | PLACEHOLDER | **745링크** | 11.8 m |
| SONGDO_IC | 인천대교고속도로 송도IC | ITS | GCP 실측 4점 | — (1,451m 밖) | 15.3 m |
| OKRYEON_IC | 인천대교고속도로 옥련IC | ITS | 자동 VP+IPM | — (1,161m 밖) | −2.7 m ⚠ |
| SANGAM01 | 서울 상암 | TOPIS | placeholder | — | 11.6 m |

> **자세 분해 높이**는 `H = K·[r1|r2|t]` 에서 얻은 값입니다. 고속도로 CCTV 폴은 10~15m이므로
> 38.2m·−2.7m는 **그 H가 실재하는 카메라 자세에 대응하지 않는다**는 뜻입니다.
> `python tools/lane_overlay.py --loc <CAM>` 으로 눈으로 확인하세요 — 현재 5대 중 4대가 탈락합니다.
> **캘리브레이션 GCP 작업이 나머지 전부의 선행조건입니다.**

---

## 7. 좌표 설계 (핵심)

```
CCTV 픽셀 ──H──▶ sat_coords(로컬 미터, world 원점 기준)
                  └─ + world.origin_easting/northing ─▶ UTM52N(EPSG:32652)
                                                        └─ proj4 ─▶ WGS84 ─▶ MapLibre
```
`G_projection_<LOC>.json`의 **`world` 블록**(epsg·origin·지면 타원체고)이 TrafficLab 원본 대비
이 프로젝트가 추가한 지오레퍼런스 확장입니다. `px_per_meter = 1`(SAT 1단위 = 1m) 규약.

---

## 8. 라이선스 / 출처

- 엔진(`trafficlab/`)은 **TrafficLab-3D (MIT © Yuk)** 의 비-GUI 코어를 이식·개조했습니다.
  원본: https://github.com/yuk068 — 원저작권·라이선스 고지를 유지하세요.
- 위성 타일: Esri World Imagery (Esri, Maxar, Earthstar Geographics, CNES/Airbus DS) ·
  Sentinel-2 cloudless © EOX. 지도: MapLibre GL JS.
- CCTV 영상에는 개인정보(번호판·얼굴)가 포함될 수 있습니다. 저장·공개 시 마스킹 등 조치가 필요합니다.
