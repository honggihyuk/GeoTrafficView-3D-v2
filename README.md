# GeoTrafficView-3D v2

위성 지도 위에 **실측 CCTV**를 올리고, 영상에서 추출한 교통 객체를 **실세계 좌표 3D**로 표출한 뒤
**공간 DB(PostGIS)** 에 적재해 **로컬 LLM(Ollama Qwen)** 으로 자연어 질의하는 교통 디지털 트윈.

> v1(Cesium 뷰어 + 정밀도로지도 점군/벡터 렌더링)에서 **MapLibre 웹맵 스택만** 추출한 버전입니다.
> 무거운 점군·Cesium 의존성을 제거하고, 위성 타일맵 + 이중 패널(CCTV↔3D) + DB + LLM에 집중합니다.
>
> 📘 **설계·측정·근거는 [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) 참조** — 워크플로우 상세, TOPIS+HD맵 cm급 결합,
> MapLibre 3D 이동체 구현, ITS 마커 오차 실측(93~793m), 탐지 고도화, 캘리브레이션 자동화·작업 가이드

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
  export_hdmap_snap.py         정밀도로지도 → GCP 스냅용 GeoJSON
  reproject.py                 캘리브레이션 변경 시 재투영(YOLO 재실행 없이)
  lane_snap.py                 차선 정렬 — 이동체를 차로 중심선에 스냅 + heading 정렬
  bridge_to_webmap.py          결과 → 웹맵 등록(+PostGIS 적재)
  refresh_markers.py           ITS/TOPIS 마커 갱신(replay 카메라 보존·dedup)
  eval/extract_frames.py       평가용 프레임 추출
  eval/evaluate.py             GT 대비 검출(AP50/P/R)·추적(MOTA/IDF1/IDSW) 평가
  eval/check_calibration.py    캘리브레이션 타당성 자동 점검(GT 불필요)
  eval/sim_track_test.py       추적 A/B 통제 실험(정답을 아는 시뮬레이션)
  retrack.py                   검출 고정·연관만 교체(ByteTrack ↔ BEV) A/B

trafficlab/motion/bev_tracker.py   BEV(지면) 추적기 — OC-SORT(ORU/OCM/OCR) + 투영공분산 마할라노비스

webmap/label.html + src/label.js   GT 라벨러(박스·트랙ID, 이전프레임 복사)
eval/<LOC>/labels.json             정답 라벨 · metrics.json 평가 결과

db/    schema.sql · ingest_postgis.py · ingest_sqlite.py
llm/   nl2sql.py (Ollama Qwen → SQL → 답변, PostGIS/SQLite 자동 전환)
location/<LOC>/   G_projection_<LOC>.json · cctv_<LOC>.png · footage/clip.mp4
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

# ② 정밀 캘리브레이션 (http://localhost:5174/calibrate.html)
#    카메라 선택 → 🤖 자동 부트스트랩 → GCP 2~4점 클릭 → [계산 & 저장]
#    (선택) ② ROI 도로영역 · ③ 차선 방향 · 🧲 HD맵 스냅

# ③ 재투영 + (선택) 차선 정렬 + 반영 (YOLO 재실행 불필요)
python tools/reproject.py --loc PANGYO_2
python tools/lane_snap.py --loc PANGYO_2          # 이동체를 차로 중심에 정렬
python tools/bridge_to_webmap.py --replay output/lanesnap/PANGYO_2/clip.json.gz --loc PANGYO_2 --ingest

# ④ 자연어 질의
python llm/nl2sql.py "가장 혼잡한 카메라는?"      # 또는 웹맵 좌하단 💬 채팅
```

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

### 4.5 차선 정렬 (`tools/lane_snap.py`)
검출 지터·캘리브레이션 오차로 차량이 도로를 벗어나거나 차로를 가로질러 흔들리는 것을 보정.
각 객체를 **차선 네트워크에 횡방향 스냅**하고 **heading을 차로 방위에 정렬**한 뒤 3D 박스를 재생성.

| 차선 소스 | 조건 | 정렬 수준 |
|-----------|------|-----------|
| 정밀도로지도 **A2_LINK** | 카메라가 HD맵 범위 안 | **차로 단위**(진짜 차선) |
| **OSM 중심선** (폴백) | 어디서나 | 중심선 + `--lane-width`(3.5m) 양자화로 차로 근사 |

실측(경부선 판교, GCP 캘리브레이션):
- 스냅 **5,557/5,568 (99%)**, 위치 이동 평균 **0.90m**(궤적 왜곡 최소), heading 보정 평균 7.7°
- 스냅 후 heading이 **122°대(하행) / 292°대(상행) 두 방향에 집중** = 양방향 차로에 정렬

> **스냅 통계 = 캘리브레이션 품질 지표.** 스냅률 <50% 이거나 heading 보정 >45°면 도구가 경고합니다.
> (실측: GCP 카메라 87~99%·<1m vs placeholder 카메라 13%·73~84° → **후자는 GCP 보정이 먼저**입니다.
> 이 상태로 스냅하면 화면은 그럴듯해도 실측과 달라지므로 placeholder 카메라는 반영하지 않았습니다.)

### 4.6 평가 (GT 라벨링 + MOTA/IDF1) — `tools/eval/`
**측정 없이는 개선을 증명할 수 없습니다.** 소량 GT를 만들어 검출·추적을 정량 평가합니다.

```bash
# ① 라벨링할 프레임 추출 (트랙 평가를 위해 연속 구간 권장)
python tools/eval/extract_frames.py --loc PANGYO_2 --n 30 --stride 3

# ② 브라우저 라벨러에서 박스+트랙ID 라벨링
#    http://localhost:5174/label.html?loc=PANGYO_2
#    드래그=박스생성 · C=이전프레임 복사(같은 ID 유지, 트랙 라벨링 핵심)
#    1~9=클래스 · 화살표=미세이동 · Del=삭제 · Z=되돌리기 · S=저장 → eval/<loc>/labels.json

# ③ 평가 / A-B 비교
python tools/eval/evaluate.py --loc PANGYO_2                      # output/ 의 결과 전부 비교
python tools/eval/evaluate.py --loc PANGYO_2 --pred A.json.gz --pred B.json.gz
```

| 지표 | 의미 |
|------|------|
| **AP50 / P / R** | 검출 품질(IoU 0.5, class-agnostic — GT와 모델 클래스 체계가 달라도 유효) |
| **MOTA** | 종합 = 1 − (FN+FP+IDSW)/GT (음수 가능) |
| **IDF1** | ID 일관성(헝가리안 전역 매칭) — **파편화에 민감** |
| **IDSW / MT / ML** | ID 스위치 수 / 대부분 추적된 트랙 / 거의 놓친 트랙 |

> 구현 검증: GT=예측을 넣으면 **AP50 1.000·MOTA 1.000·IDF1 1.000·IDSW 0**,
> GT를 인위적으로 열화(6px 지터·ID 30% 섞기·15% 누락)하면 **AP50 0.177·MOTA −0.493·IDSW 188** —
> 지표가 정의대로 동작함을 확인했습니다.

**측정된 현재 약점**(GT 없이도 드러난 것): 트랙 수명 중앙값 0.78~0.92초, **0.5초 미만 트랙이 33~42%**
→ ID 파편화. 이미지 IoU 연관을 **지면(BEV) 연관 + OC-SORT**로 바꾸는 개선의 효과를 위 지표로 검증하세요.

### 4.7 BEV 추적 A/B (구현·측정 완료, 결론은 "캘리브레이션이 먼저")

이미지 IoU 연관(ByteTrack)을 **지면 좌표 연관**으로 바꾸는 실험. `trafficlab/motion/bev_tracker.py`는
OC-SORT의 ORU/OCM/OCR + ByteTrack 2단계 연관 + **투영 공분산 기반 마할라노비스 거리**(cf. UCMCTrack).

```bash
python tools/eval/check_calibration.py --all        # ① 기하가 말이 되는지 먼저 확인
python tools/retrack.py --loc PANGYO_2              # ② 검출 고정, 연관만 교체 → A/B 산출물
python tools/eval/sim_track_test.py --loc PANGYO_2  # ③ 정답 아는 시뮬로 통제 비교
python tools/eval/evaluate.py --loc PANGYO_2 --pred A.json.gz --pred B.json.gz   # ④ GT 있으면 최종 판정
```

**측정 결과 (실데이터 PANGYO_2)** — 연속성은 크게 개선:
| | 트랙 수 | 수명 중앙값 | 0.5초 미만 |
|---|---|---|---|
| A. ByteTrack(이미지 IoU) | 150 | 0.92초 | 33% |
| B. BEV(제안) | **98** | **2.23초** | **12%** |

**통제 시뮬 결과 — 캘리브레이션 품질이 판정을 뒤집는다** (동일 시나리오, IoU 0.3):

| 방식 | GCP 4점 캘리브 | | | GCP 10점 캘리브(개선) | | |
|---|---|---|---|---|---|---|
| | MOTA | IDF1 | IDSW | MOTA | IDF1 | IDSW |
| A. ByteTrack | **0.283** | **0.263** | **299** | 0.319 | 0.298 | **299** |
| B. BEV 등방 | 0.187 | 0.167 | 642 | 0.328 | 0.262 | 571 |
| **C. BEV 마할라노비스** | 0.243 | 0.260 | 547 | **0.345** | **0.318** | 538 |

- **4점 캘리브레이션에서는 BEV가 졌고(C 0.243 < A 0.283), 10점으로 개선하니 이겼다(C 0.345 > A 0.319).**
  즉 병목은 추적기가 아니라 **투영 정확도**였다.
- 투영 잡음(3px 지터 → 지면): 횡 0.9m·종 3.2m → **횡 0.7m·종 2.2~2.9m** 로 개선
- 실데이터: 트랙 150 → **99**, 수명 중앙값 0.92초 → **1.93초**, 0.5초 미만 33% → **11%**
- 마할라노비스(이방성) 보정은 두 조건 모두에서 등방 유클리드보다 우수(B→C).

### 4.8 DB + LLM
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
| 정밀도로지도 | 국토정보플랫폼 배포 구간 | 다운로드 | GCP 스냅용 |

> ITS는 고속·국도 위주라 **도심 자율주행 거리 CCTV는 시(TOPIS/경기/세종) API 소관**입니다.

---

## 6. 현재 등록된 카메라 (예시)

| 카메라 | 지역 | 소스 | 캘리브레이션 |
|--------|------|------|--------------|
| SONGDO_IC | 인천대교고속도로 송도IC | ITS | GCP 4점 |
| YEONSU_JCT / OKRYEON_IC | 연수JCT / 옥련IC | ITS | placeholder |
| PANGYO_2 | 경부선 판교2 | ITS | GCP 4점 (RMS 0.00m) |
| SANGAM01 | 서울 상암 | TOPIS | placeholder |

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
