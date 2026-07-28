# GeoTrafficView-3D v2 — 기술 문서

> 공개 CCTV 영상에서 교통 객체를 추출해 **실세계 좌표(UTM52N)** 로 투영하고,
> 고해상도 위성지도 위에 **3D로 표출**하며, **공간 DB**에 적재해 **로컬 LLM**으로 질의하는
> 교통 디지털 트윈. 본 문서는 실제 구현·측정 결과를 기준으로 작성되었다.

작성 기준: 코드 Python 4,272줄 + 웹 JS 1,041줄 · 카메라 5대 · detection 20,221행

---

## 1. 최종 산출물

### 1.1 무엇을 만들었는가

**"공개 CCTV 한 대 → 지도 위에서 움직이는 3D 차량 → 질의 가능한 DB"** 를 잇는 end-to-end 파이프라인.

```
[공개 CCTV API]        [컴퓨터 비전]           [기하 변환]         [표출]        [저장·질의]
 ITS / TOPIS / UTIC  →  VisDrone+SAHI      →  G-Projection    →  MapLibre   →  PostGIS
 HLS 스트림             ByteTrack/BEV추적      (GCP 캘리브)       위성맵 3D      SQLite
                        3D 리프팅              차선 정렬          이중 패널      Ollama Qwen
```

핵심 설계 결정 3가지:
1. **TrafficLab-3D(MIT © Yuk)의 비-GUI 엔진을 이식**하되, 좌표계를 위성 이미지 픽셀 → **실세계 UTM52N**으로 교체(`world` 블록 추가). 이로써 단일 카메라 데모가 **다지역 GIS 시스템**이 됐다.
2. **무거운 3D(Cesium·점군)를 버리고** MapLibre 위성 타일맵 채택 — v1 대비 239MB 제거, 도시 확장 시 부담 없음.
3. **모든 주장을 측정으로 검증** — 캘리브레이션 타당성·추적 파편화·A/B 실험 도구를 함께 구축.

### 1.2 현재 등록 상태

| 카메라 | 지역 | 소스 | GCP | 기하 타당성 | 고유객체 | 평균속도 |
|--------|------|------|-----|------------|---------|---------|
| PANGYO_2 | 경부선 판교2 | ITS | 10점 | **✓ 1.38** | 98 | 40.4 km/h |
| SONGDO_IC | 인천대교 송도IC | ITS | 4점 | ✗ 4.09 | 152 | 58.4 km/h |
| YEONSU_JCT | 인천대교 연수JCT | ITS | 10점 | ✗ 0.39 | 104 | 29.0 km/h |
| OKRYEON_IC | 인천대교 옥련IC | ITS | 7점 | ✗ 0.93 | 27 | 23.5 km/h |
| SANGAM01 | 서울 상암 | **TOPIS** | 0(자동) | ✗ 1.24 | 39 | 4.0 km/h |

지도 마커 **590개**(ITS 567 + TOPIS 23) · replay 재생 가능 5대 · DB **20,221 detection**

---

## 2. 기능 목록

### 2.1 데이터 수집
| 기능 | 구현 | 비고 |
|------|------|------|
| ITS OpenAPI CCTV | `data_ingest/fetch_its_cctv.py` | 전국 고속·국도, HLS(.m3u8), 무료 키 |
| **서울 TOPIS CCTV** | `data_ingest/fetch_topis_cctv.py` | **키·IP 인증 불필요**, 서울 510대 |
| UTIC CCTV | `data_ingest/fetch_utic_cctv.py` | 도심, 키+IP 등록 필요(iframe 재생) |
| 경찰청 CCTV 매칭 | `data_ingest/match_streams.py` | 위치만 있는 엑셀 ↔ ITS 스트림 공간매칭 |
| 스트림 캡처 | `tools/its_grab.py`, `data_ingest/grab_frame.py` | 스냅샷 + 클립(H.264) |
| 지역 원스톱 추가 | `tools/add_location.py` | 캡처→캘리브→추론→웹맵→DB |
| 도시 배치 | `tools/run_city.py` | 다중 카메라 자동화 |
| 마커 갱신 | `tools/refresh_markers.py` | replay 보존 + 중복 dedup |

### 2.2 인식·기하
| 기능 | 구현 |
|------|------|
| 객체 탐지 | `run_inference_sahi.py` — VisDrone YOLOv8 + **SAHI 슬라이싱** |
| 추적(기본) | supervision **ByteTrack** (이미지 IoU) |
| 추적(제안) | `trafficlab/motion/bev_tracker.py` — **BEV + OC-SORT(ORU/OCM/OCR) + 마할라노비스** |
| 좌표 투영 | `trafficlab/projection/g_projection.py` — 왜곡보정→호모그래피→시차→3D 리프팅 |
| 속도·방향 | `trafficlab/motion/kinematics.py` — TrackSmoother(적응형 EMA) |
| 차선 정렬 | `tools/lane_snap.py` — HD맵 A2_LINK 또는 OSM 중심선에 스냅 |
| 재투영 | `tools/reproject.py` — 캘리브 변경 시 YOLO 재실행 없이 좌표만 재계산 |
| 재추적 | `tools/retrack.py` — 검출 고정, 연관만 교체(A/B) |

### 2.3 캘리브레이션
| 기능 | 구현 |
|------|------|
| 수동 GCP | `webmap/calibrate.html` — 카메라 선택식, CCTV↔위성 대응점 |
| **자동(0클릭)** | `tools/auto_calibrate_vp.py` — 소실점+IPM+OSM 방위 |
| **하이브리드** | 자동 부트스트랩 + GCP 1~3점 저차원 보정 |
| ROI | 도로영역 폴리곤 → 추론 시 도로 밖 검출 폐기 |
| 차선 방향 | CCTV 차로선 → 실좌표 heading 프라이어 |
| HD맵 스냅 | `tools/export_hdmap_snap.py` + 🧲 — 정밀도로지도 정점에 cm급 스냅 |
| **품질 진단** | `tools/eval/check_calibration.py`, `calib_guide.py` + 브라우저 실시간 가이드 |

### 2.4 표출·저장·질의
| 기능 | 구현 |
|------|------|
| 위성 basemap | Esri World Imagery / Sentinel-2 cloudless 토글 |
| 이중 패널 | `webmap/src/player.js` — 영상 3D박스 ↔ 지도 3D박스 동기 |
| 마커 | 590개, replay 카메라 강조 + dedup |
| PostGIS | `db/ingest_postgis.py` — geom(4326)+geom_utm(32652), GIST 인덱스 |
| SQLite | `db/ingest_sqlite.py` — 서버 없이 동일 스키마 |
| LLM 질의 | `llm/nl2sql.py` + `/api/ask` — Qwen NL2SQL, 읽기전용 |
| 지도 제어 | `webmap/src/chat.js` — "송도IC로 이동" 등 |
| **평가** | GT 라벨러 + MOTA/IDF1 + 통제 시뮬 A/B |

---

## 3. 워크플로우 상세

### 3.1 전체 흐름

```
① CCTV 수집        ② 객체 탐지          ③ 기하 변환           ④ 3D 표출        ⑤ DB      ⑥ LLM
─────────────    ──────────────     ────────────────     ───────────     ──────    ──────
ITS/TOPIS API    VisDrone YOLOv8    G-Projection         MapLibre        PostGIS   Qwen
  ↓ HLS            + SAHI 타일링      픽셀→로컬미터         위성타일        /SQLite   NL2SQL
클립 캡처          + ByteTrack/BEV    + world 원점         fill-extrusion  detection  ↓
(cv2/ffmpeg)       ↓                 → UTM52N → WGS84     3D 박스         cctv      한국어 답변
                 bbox_2d, track_id   sat_coords,          + 영상 동기      뷰
                                     sat_floor_box,
                                     bbox_3d(8점)
```

### 3.2 ① CCTV 수집

**소스별 특성 (실측)**

| 소스 | 인증 | 스트림 | 커버리지 | 우리 사용 |
|------|------|--------|---------|----------|
| **ITS** (its.go.kr) | 무료 키 | HLS(.m3u8) 직접 | 고속·국도 위주 | 판교·인천 4대 |
| **TOPIS** (서울) | **불필요** | HLS 직접 | 서울 510대 | 상암 1대 |
| **UTIC** (경찰청) | 키+**IP 등록** | 플레이어 페이지 | 전국 도심 | iframe만 |

발견한 TOPIS 공개 엔드포인트:
```
POST /map/cctv/selectCctvList.do        → {rows:[{camId, camName, lat, lng}]}  (510대)
POST /map/selectCctvInfo.do             → {rows:[{hlsUrl: "...playlist.m3u8"}]}
     body: camId=<id>&cctvSourceCd=HP
```
> cv2로 열 때 `OPENCV_FFMPEG_CAPTURE_OPTIONS=referer;https://topis.seoul.go.kr/` 필요.
> 라이브 HLS는 stall이 잦아 **read-fail 상한**을 두어야 한다(초기 캡처 0프레임 → 재시도 90프레임 확보).

**중요 제약**: ITS는 고속·국도 위주라 **상암·판교·세종의 도심 자율주행 거리 CCTV는 시(TOPIS/경기/세종) API 소관**이다. 실측: 상암 DMC 반경 5km 내 ITS 스트림은 4개뿐, 모두 고속도로.

### 3.3 ② 객체 탐지

**모델 아닌 기법으로 4.3배 개선** (동일 클립 450프레임)

| 조합 | 검출 | 3D박스 | 트랙 | 특징 |
|------|------|--------|------|------|
| YOLO11n (COCO, 풀프레임) | 824 | 170 | 18 | 기준선 |
| YOLO11x (COCO, imgsz1280) | 1,830 | 1,093 | 24 | 모델 확대 |
| **VisDrone YOLOv8 + SAHI** | **7,904** | **6,288** | **152** | 기법 교체 |

- **SAHI 슬라이싱**: 384px 타일(20% 중첩)로 원거리 소형차 검출. 원논문 보고 +6.8~14.5% AP.
- **VisDrone 파인튜닝**: COCO의 오분류(train/boat) 제거, van·삼륜차 구분.
- **ROI 필터**: 도로영역 폴리곤 밖 검출 폐기 — 실측 **19% 필터**.
- **추적**: ByteTrack(기본) / BEVTracker(제안, §7).

### 3.4 ③ 기하 변환 (핵심)

```
CCTV 픽셀 (u,v)
   │ ① cv2.undistortPoints(K, D)              — 렌즈 왜곡 보정
   ▼
왜곡보정 픽셀
   │ ② cv2.perspectiveTransform(H)            — 호모그래피(GCP로 추정)
   ▼
sat_coords (로컬 미터, px_per_meter=1)
   │ ③ 시차 보정: real = C + (A−C)·(z_cam−h)/z_cam
   ▼
지면 접촉점 (미터)
   │ ④ + world.origin_easting/northing
   ▼
UTM52N (EPSG:32652)  ──proj4──▶  WGS84  ──▶  MapLibre
```

**3D 박스 생성**: `prior_dimensions.json`의 클래스별 치수(car 1.8×3.8×1.55m)로 지면 사각형을 heading만큼 회전 → `sat_floor_to_cctv_3d()`가 시차 factor `z_cam/(z_cam−h)`로 천장 4점을 만들고 `cv2.projectPoints`로 왜곡 재적용 → **CCTV 픽셀 8점**.

**v1 대비 확장**: `G_projection_<LOC>.json`에 `world` 블록 추가
```json
"world": { "epsg": 32652, "origin_easting": 331774.7, "origin_northing": 4140930.2,
           "ground_ellipsoid_h": 28.0 }
```
이 한 블록으로 TrafficLab의 "위성 PNG 픽셀" 좌표계가 **전 지구 좌표계**가 됐다.

### 3.5 ④ GIS 3D 표출 → §6
### 3.6 ⑤ DB 적재

**동일 스키마 2백엔드** (PostGIS 운영 / SQLite 즉시)

```sql
detection(ts, frame, cctv_id, track_id, class, confidence,
          speed_kmh, heading_deg, geom(4326), geom_utm(32652))
cctv(cctv_id, name, geom)
cctv_summary  -- 뷰: 카메라별 고유객체·평균속도·detection 수
```
- 카메라 단위 **idempotent**: 같은 `cctv_id` 재적재 시 해당 카메라 행만 교체
- PostGIS: GIST 공간 인덱스 + 시간 인덱스. 실측 `ST_DWithin` 반경 질의(50m 4,183 / 100m 5,974 / 200m 6,756)
- SQLite: **웹맵에 표출 중인 replay를 그대로 적재** → 화면과 DB 일치 보장

### 3.7 ⑥ LLM 질의

```
사용자 질문 ─┬─ 지도 명령 감지("~로 이동") → flyTo + 재생 (LLM 미사용, 즉시)
             └─ 그 외 → /api/ask → nl2sql.py → Ollama Qwen
                         ├ 1차: 스키마+few-shot → SQL 생성
                         ├ 안전 게이트: SELECT만, 세미콜론·DDL 차단, 읽기전용 연결
                         ├ 실행 (PostGIS 또는 SQLite 자동 선택)
                         └ 2차: 결과 → 한국어 요약
```
실측 예:
```
Q: "카메라별 차량 대수"
→ SELECT name, COUNT(DISTINCT track_id) FROM detection JOIN cctv USING(cctv_id) GROUP BY name
→ "송도IC 152대, 연수JCT 104대, 옥련IC 27대..."
```
> `track_id` DISTINCT가 핵심 — 프레임마다 중복되므로 없으면 대수가 수천으로 부풀려진다(few-shot에 명시).

---

## 4. 서울 TOPIS CCTV + 서울 정밀도로지도 = cm급 디지털트윈

### 4.1 왜 상암인가

상암은 **영상과 지도가 모두 공개된 유일한 조합**이다.
- **영상**: 서울 TOPIS CCTV — 키·IP 인증 없이 HLS 직접 접근(실증 완료, 90프레임 캡처 → 추론 39트랙)
- **지도**: 서울시가 상암 자율주행 시범지구(~20km)에 구축한 **3D 정밀도로지도를 민간 개방**(2025)

### 4.2 결합 원리 — 왜 cm급이 되는가

일반적인 GCP는 **위성 이미지에서 눈대중으로 클릭**한다(오차 1~2m). 정밀도로지도는 **실측 측량 성과**라 노면표시 정점이 cm급 좌표를 갖는다.

```
[기존] CCTV 특징점 → (사람이 위성에서 클릭) → 위경도   ← 오차 1~2m
[HD맵] CCTV 특징점 → (HD맵 정점에 자동 스냅) → 위경도   ← cm급
```

구현:
```bash
python tools/export_hdmap_snap.py --dir <서울 상암 3D 정밀도로지도 HDMap 폴더>
#  → B2_SURFACELINEMARK(정지선·차선) + A2_LINK(차로) + C1_TRAFFICLIGHT 정점 추출
#  → webmap/public/data/hdmap/hdmap.geojson
```
`calibrate.html`에서 **🧲 HD맵 스냅** 체크 → 지도 클릭이 **15m 이내 HD맵 정점으로 자동 스냅**.
현재 송도 HD맵으로 검증 완료: **20,141 정점 로드, 스냅 동작 확인**.

### 4.3 기대 효과

| 단계 | 오차 원인 | 일반 GCP | HD맵 GCP |
|------|----------|---------|----------|
| 지도쪽 좌표 | 위성 클릭 정밀도 | 1~2 m | **cm급** |
| 영상쪽 좌표 | 픽셀 클릭 정밀도 | 1~3 px | 1~3 px |
| 최종 투영 | 위 둘의 전파 | 수 m | **영상 클릭 정밀도가 지배** |

즉 HD맵 스냅은 **오차 요인 하나를 완전히 제거**한다. 남는 것은 영상 픽셀 정밀도뿐이며, 이는 원거리에서 여전히 크다(§8.3).

### 4.4 현재 상태와 남은 작업

- ✅ TOPIS 영상 파이프라인 완성(상암 SANGAM01 등록·추론·DB 적재)
- ✅ HD맵 스냅 기능 완성(송도 HD맵으로 검증)
- ⬜ **서울 상암 3D 정밀도로지도 다운로드 → `--dir`로 지정** → 상암 카메라 cm급 캘리브레이션
  > 보유한 국토정보플랫폼 HD맵은 **인천송도 시범지구**와 **경인선(120호선)** 뿐이라
  > 상암·판교·송도IC를 커버하지 않는다(실측: 송도IC는 HD맵 경계 밖, 경인선은 12.8km 밖).

---

## 5. 마커 위치 — ITS 보고 좌표 vs 실제 관측 영역 (오차 실측)

### 5.1 문제

ITS `cctvInfo` API는 CCTV의 **설치 지점 좌표**를 보고한다. 그러나 우리가 지도에 표시해야 할 것은 **그 카메라가 실제로 관측하는 도로 영역**이다. 둘은 일치하지 않는다.

### 5.2 실측 결과

GCP 캘리브레이션 원점(= 클릭한 대응점들의 중심 = 관측 영역 중심)과 ITS 보고 좌표의 거리:

| 카메라 | GCP | ITS/TOPIS 보고 좌표 | 캘리브레이션 원점 | **오차** |
|--------|-----|-------------------|-----------------|---------|
| PANGYO_2 | 10점 | 127.10044, 37.39969 | 127.09965, 37.40025 | **93.0 m** |
| OKRYEON_IC | 7점 | 126.63823, 37.42514 | 126.63652, 37.42441 | **171.1 m** |
| SONGDO_IC | 4점 | 126.64999, 37.40649 | 126.65042, 37.40482 | **190.1 m** |
| YEONSU_JCT | 10점 | 126.63310, 37.40852 | 126.62413, 37.40871 | **792.8 m** |
| SANGAM01 | 0점(자동) | 126.88290, 37.58070 | 동일 | 0 m |

**오차 범위: 93 ~ 793 m (평균 312 m)**

### 5.3 해석 (오차의 두 성분)

이 거리는 두 요인이 합쳐진 값이다:
1. **기하학적으로 당연한 성분** — 카메라는 도로를 100~300m 내다본다. 관측 영역 중심이 설치 지점에서 떨어지는 것은 정상.
2. **ITS 좌표 자체의 부정확성** — 노선·IC 단위로 대표 좌표가 부여된 경우가 있다.

YEONSU_JCT의 793m는 1번만으로 설명하기 어려워, 좌표 부정확 또는 GCP 클릭 위치 문제가 의심된다(§8.2 참조 — C01은 기하 타당성도 불합격).

### 5.4 설계 결정

**마커 위치 = 캘리브레이션된 `world.origin`** (ITS 좌표가 아님).
```python
# tools/bridge_to_webmap.py
lon, lat = Transformer(world.epsg → 4326).transform(world.origin_easting, world.origin_northing)
feature.geometry = Point(lon, lat)   # ← ITS 보고 좌표 대신
```
그리고 **dedup**: replay 카메라와 이름이 같거나 250m 이내인 원본 ITS 마커는 제거한다.
→ 캘리브레이션 후 마커가 도로로 이동하면서 원본 마커와 벌어져 "튀는" 현상을 방지.

캘리브레이션 전(placeholder) 카메라는 ITS 좌표를 그대로 쓰되, 이 오차만큼 부정확함을 인지해야 한다.

---

## 6. MapLibre + 고해상도 위성지도 위 이동체 구현

### 6.1 basemap

| 레이어 | 출처 | 특성 |
|--------|------|------|
| **Esri World Imagery** | Esri, **Maxar, Earthstar Geographics, CNES/Airbus DS** | 상용 고해상(도심 수십 cm) |
| **Sentinel-2 cloudless** | EOX | 전구·무료·구름 제거(10m) |

```js
sources: {
  esri: { type:'raster', tiles:['https://server.arcgisonline.com/ArcGIS/rest/services/
                                 World_Imagery/MapServer/tile/{z}/{y}/{x}'], tileSize:256,
          attribution:'Esri, Maxar, Earthstar Geographics, CNES/Airbus DS' },
  s2:   { type:'raster', tiles:['https://tiles.maps.eox.at/wmts/1.0.0/
                                 s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg'] }
}
```
> 둘 다 **토큰 불필요**. attribution 표기는 필수.

### 6.2 이동체를 지도에 올리는 경로

```
sat_floor_box (로컬 미터 4점, 프레임별)
   │ + world.origin_easting/northing
   ▼  UTM52N
   │ proj4('EPSG:32652' → 'EPSG:4326')
   ▼  WGS84 폴리곤
   │ GeoJSON Feature (properties: height=클래스 높이, color)
   ▼
map.getSource('boxes').setData(FeatureCollection)
   │
   ▼ fill-extrusion 레이어
   'fill-extrusion-height': ['get','height']    // car 1.55m, bus 3.5m …
   'fill-extrusion-base':   ['get','base']      // 지면
   'fill-extrusion-opacity': 0.8
```

**핵심 포인트**
- 지도에 그리는 것은 **바닥 사각형 + 높이 압출**이다. 영상 패널의 8점 `bbox_3d`와 달리, 지도는 실좌표 폴리곤이므로 **줌·회전·기울기(pitch)에 대해 정확히 변형**된다.
- 클래스별 높이/색을 `properties`로 넘겨 **데이터 기반 스타일링**(GPU에서 처리 → 수백 개도 부담 없음).
- 카메라 클릭 시 `map.flyTo({ pitch: 60, bearing: 20, zoom: 18 })` — 3D 느낌을 위해 기울여 본다.

### 6.3 영상 ↔ 지도 동기

```js
function render() {
  const fi = Math.floor(video.currentTime * fps);   // ← 영상이 마스터 클럭
  const objs = frames[fi].objects;
  // 좌: canvas에 bbox_3d 8점을 6면 폴리곤(핑크)으로
  // 우: 같은 프레임의 sat_floor_box → GeoJSON → setData()
  requestAnimationFrame(render);
}
```
`video.currentTime`을 단일 기준으로 삼아 좌우 패널이 **같은 프레임**을 그린다. 사용자가 영상을 seek하면 지도 박스도 즉시 따라간다.

### 6.4 replay 데이터 로딩

`.json.gz`를 브라우저에서 직접 해제하되, **개발 서버가 이미 gzip을 풀어주는 경우**를 구분해야 한다:
```js
const bytes = new Uint8Array(await res.arrayBuffer());
if (bytes[0]===0x1f && bytes[1]===0x8b)          // gzip 매직바이트
  → DecompressionStream('gzip')로 직접 해제
else                                              // Content-Encoding: gzip으로 이미 해제됨
  → 그대로 JSON 파싱
```

---

## 7. 객체 탐지 고도화 (기존 구현 방식에 맞춘)

### 7.1 원칙 — 모델 교체가 아닌 기법 개선

우리 실험이 보여준 것: **모델 확대(11n→11x)보다 기법 교체(VisDrone+SAHI)가 4배 더 효과적**이었다.
따라서 고도화도 "더 큰 모델"이 아니라 **파이프라인의 약한 고리**를 겨냥한다.

### 7.2 적용 완료

| 기법 | 근거 | 우리 결과 |
|------|------|----------|
| **SAHI 슬라이싱** | Akyon et al., ICIP'22 | 검출 1,830 → **7,904** |
| **도메인 모델**(VisDrone) | — | 오분류 제거, van/삼륜 구분 |
| **ROI 기하 필터** | ground-plane FP 제거 문헌 | 도로 밖 검출 **19% 폐기** |
| **차선 heading 프라이어** | TrafficLab SVG 가이드라인 | 정지·가림 차량 방향 안정화 |
| **차선 정렬(lane snap)** | HD맵/OSM 중심선 | 99% 스냅, 위치 이동 평균 0.90m |
| **BEV 추적** | OC-SORT CVPR'23 + UCMCTrack | §7.3 |

### 7.3 BEV 추적 — 연관 공간을 바꾸다

**문제 진단(실측)**: 이미지 IoU 연관은 오블리크 뷰에서 무너진다.
```
트랙 수명 중앙값 0.78~0.92초 · 0.5초 미만 트랙 33~42%  ← ID 파편화
```

**구현** (`trafficlab/motion/bev_tracker.py`)
```
검출 → 지면 투영(미터) → 등속 Kalman → 연관
  · 연관 비용 = 마할라노비스 거리 √(δᵀ(P+R)⁻¹δ)   ← R = 투영 공분산(이방성!)
  · OCM: 관측 기반 진행방향 일관성 패널티
  · ORU: 가림 후 재연결 시 가상 궤적으로 KF 재설정
  · OCR: KF 예측 대신 '마지막 관측'으로 2차 연관
  · ByteTrack식 2단계(고신뢰 → 저신뢰)
```

**왜 마할라노비스인가 (측정으로 발견)**
```
3px 지터 → 지면 오차:  횡방향 σ 0.7~0.9 m   종방향 σ 2.2~3.2 m   (3~4배 이방성)
차로 간격 3.5 m
```
등방 유클리드 거리는 종방향 잡음에 맞춰 게이트를 키워야 하고, 그러면 **옆 차로 차량과 ID가 섞인다**. 마할라노비스는 종방향만 관대하고 횡방향은 엄격하므로 **차로 구분을 보존**한다.

**A/B 결과 — 캘리브레이션 품질이 판정을 뒤집었다** (통제 시뮬, IoU 0.3)

| 방식 | GCP 4점 MOTA/IDF1 | **GCP 10점 MOTA/IDF1** |
|------|------------------|----------------------|
| A. ByteTrack | **0.283 / 0.263** | 0.319 / 0.298 |
| B. BEV 등방 | 0.187 / 0.167 | 0.328 / 0.262 |
| **C. BEV 마할라노비스** | 0.243 / 0.260 | **0.345 / 0.318** |

실데이터(판교): 트랙 150 → **99**, 수명 0.92초 → **1.93초**, 단명 트랙 33% → **11%**

> **교훈**: 4점 캘리브에서는 BEV가 졌다. 같은 코드가 10점 캘리브에서는 이겼다.
> 병목은 추적 알고리즘이 아니라 **투영 정확도**였다.

### 7.4 다음 후보 (미적용, 근거 논문 기준)

| 기법 | 논문 | 기대 |
|------|------|------|
| TTA + WBF 융합 | Solovyev et al., IVC'21 | 의사라벨 품질↑ |
| 교사-학생 카메라 특화 | Soft Teacher ICCV'21, Ekya NSDI'22 | 도메인 갭 해소(라벨 0) |
| 시간축 집계 | YOLOV AAAI'23 | 정지 카메라에 유리 |
| Roadside 3D 학습 | **BEVHeight CVPR'23** | 고정 치수 조립 → 학습 기반 |
| 다중 카메라 융합 | V2X-ViT ECCV'22 | 가림 해소 |

> 단, BEVHeight류는 **정확한 내·외부 파라미터와 3D 라벨**을 전제한다. 우리는 초점거리조차 미지이므로
> 그대로 얹으면 오히려 나빠질 수 있다(WARM-3D 같은 도메인적응 연구가 존재하는 이유).

---

## 8. 캘리브레이션 — 자동화와 작업 가이드

### 8.1 3단계 스펙트럼

| 방식 | 클릭 | 정확도 | 도구 |
|------|------|--------|------|
| 완전 자동 | **0점** | 부트스트랩(~20m) | `auto_calibrate_vp.py` |
| 하이브리드 | 1~3점 | 높음 | 🤖 자동 + GCP 보정 |
| 수동 GCP | 6~10점 | 최고 | `calibrate.html` |

### 8.2 자동 캘리브레이션 (0클릭)

```
① 도로방향 선분(Hough) → 소실점(VP) 최소자승 교점
② VP → 카메라 자세:  pitch = atan((cy−vy)/f),  yaw = atan((vx−cx)/f)
③ ray-plane IPM → 픽셀↔지면 호모그래피 (카메라 높이 입력)
④ 지오앵커: 원점 = ITS 좌표, 전방 방위 = OSM Overpass 최근접 도로
```

**실측(송도IC)**: VP (426.5, 126.7) 검출, 재투영 오차 <1px, pitch 8.9°·yaw 5.3°, OSM 방위 **324°(아암대로) 자동 취득**. GCP 대비 형상정합 RMS ≈ 21.7m → **부트스트랩 등급**.

**초점거리 자동추정의 한계(정직하게)**
- 처음 시도한 "차선 평행성 최적화"는 **축퇴**였다 — 도로선은 어떤 f에서도 VP를 지나므로 f가 불가관측. 최적화가 상한으로 발산했다.
- 올바른 방법인 **2번째 직교 소실점**(f² = −(vp₁−pp)·(vp₂−pp))으로 교체. 정지선·횡단보도 같은 수직 구조가 보이는 교차로 카메라에서 작동한다.
- 고속도로 프레임은 직교 구조가 없어 **비직교 판정 후 f=W로 폴백**(사유 출력). 가짜 추정치를 만들지 않는다.

**주의**: OSM Overpass는 User-Agent 없으면 **406**을 반환하고, 공개 서버는 rate-limit이 잦다(미러 3곳 순차 재시도 구현).

### 8.3 GCP 작업 가이드 (실측 기반)

#### 왜 가이드가 필요한가 — 잔차 0의 함정

**호모그래피 자유도 = 8, 대응점 1쌍 = 방정식 2개.**
→ **4점이면 정확히 결정되어 잔차가 수학적으로 항상 0이다.**
"RMS 0.00m"은 정확도가 아니라 **검증 불가**를 뜻한다. 이 함정 때문에 5대 중 4대의 잘못된 캘리브레이션이 오랫동안 발견되지 않았다.

#### 진단 도구 3종

**① 기하 타당성** (`check_calibration.py`) — GT 불필요, 가장 중요
```
검출된 차량의 픽셀 박스 크기  vs  그 위치에 1.8×4.2m 차량을 놓고 투영한 크기
```
실측으로 5대 중 4대를 불합격 판정:
```
YEONSU_JCT         GCP 10점  세로/가로 0.39  ✗
OKRYEON_IC         GCP  7점  세로/가로 0.93  ✗
SONGDO_IC  GCP  4점  세로/가로 4.09  ✗   ← 실제 17×12px인데 예측 17×4px
PANGYO_2        GCP 10점  세로/가로 1.38  ✓
SANGAM01        GCP  0점  세로/가로 1.24  ✗
```

**② LOO 교차검증** — 한 점을 빼고 나머지로 예측한 오차(m). 5점 이상에서만 산출.
실증: 같은 7점이라도 **세로 분포 52% → 4.47m, 세로 2%로 뭉침 → 52.88m (12배)**

**③ 브라우저 실시간 가이드** (`calibrate.html`)
- 🎯 **권장 구역**: 세로 3구간(먼거리/중간/근거리) × 가로 3구간 → 핑크 구역이 사라질 때까지 클릭
- 실시간 경고: "🔴 먼 거리 점이 없습니다 — 깊이 스케일이 틀어지는 1순위 원인"
- 📏 **투영 격자**: 저장된 H로 차로(3.5m)·깊이(10m) 격자를 영상에 되그려 육안 대조

#### 목표치와 함정

| 지표 | 목표 | 비고 |
|------|------|------|
| **기하 타당성** | **✓ 통과** | **가장 중요** |
| 점 개수 | 6~10 | 4점은 검증 불가 |
| 세로(깊이) 분포 | 50%+ | 깊이 스케일 결정 |
| 가로 분포 | 50%+ | |
| LOO | 참고용 | **수치만 보고 점을 지우지 말 것** |

> ⚠️ **LOO 기반 자동 이상치 제거는 실패했다.** LOO는 원거리 점에서 본래 크다(1px ≈ 수 m).
> 그 점들을 지우면 **깊이를 고정하는 정보가 사라져** 오히려 악화된다(실측: 10점 LOO 59.8m ✓통과 →
> 자동 제거 6점 LOO 31m이지만 차량을 764×101px로 예측하는 엉터리). 현재 `--fix`는 기하 타당성이
> 나빠지면 자동 롤백한다.

#### 찍기 좋은 지점 / 나쁜 지점

| ✅ 좋음 | ❌ 나쁨 |
|--------|--------|
| 정지선 끝점, 차선 점선의 시작/끝 | 차량·그림자(움직임) |
| 횡단보도 모서리 | 도로 한가운데(특징 없음) |
| 노면 화살표 꼭짓점 | 위성에서 식별 불가한 지점 |
| 맨홀·연석 코너 | 높이가 있는 구조물 상단(지면 아님) |

### 8.4 재작업 절차

```bash
python tools/eval/calib_guide.py --loc <LOC>       # ① 진단
# ② http://localhost:5174/calibrate.html
#    카메라 선택 → [현재 모드 지우기] → 🎯 구역 채우며 6~10점 → [계산 & 저장]
#    → 📏 격자로 육안 검증
python tools/reproject.py --loc <LOC>              # ③ 좌표 재계산(YOLO 재실행 X)
python tools/eval/check_calibration.py --loc <LOC> # ④ ✓ 확인
python tools/bridge_to_webmap.py --replay output/reprojected/<LOC>/clip.json.gz --loc <LOC> --ingest
```

**판교 재작업 실측 결과**

| 항목 | 4점 | **10점** |
|------|-----|---------|
| 세로 분포 | 42% | **69%** |
| 가로 분포 | 41% | **75%** |
| 기하 타당성 | 2.36 (경계) | **1.38 ✓** |
| 투영 잡음(횡/종) | 0.9 / 3.2 m | **0.7 / 2.2~2.9 m** |
| BEV 추적 판정 | 패배 | **승리** |

---

## 9. 평가 체계

### 9.1 왜 필요한가

**측정 장치 없이는 "좋아졌다"를 말할 수 없다.** 이 프로젝트의 여러 개선은 정량 검증 후에야 방향이 확정됐다(특히 BEV 추적은 캘리브레이션 개선 전후로 판정이 뒤집혔다).

### 9.2 도구

| 도구 | 역할 |
|------|------|
| `tools/eval/extract_frames.py` | 라벨링용 프레임 추출(트랙 평가는 연속 구간) |
| `webmap/label.html` | GT 라벨러 — 박스+트랙ID, **C키=이전 프레임 복사**(트랙 라벨링 핵심) |
| `tools/eval/evaluate.py` | AP50/P/R + **MOTA/IDF1/IDSW/MT/ML** (scipy 헝가리안) |
| `tools/eval/sim_track_test.py` | 정답을 아는 **통제 시뮬** A/B(GT 라벨 전에도 검증) |
| `tools/eval/check_calibration.py` | 캘리브레이션 타당성(GT 불필요) |

### 9.3 메트릭 구현 자체를 검증했다

| self-test | 결과 |
|-----------|------|
| GT = 예측 (완벽) | AP50 **1.000** · MOTA **1.000** · IDF1 **1.000** · IDSW **0** |
| GT 열화(6px 지터·ID 30% 섞기·15% 누락) | AP50 0.177 · MOTA −0.493 · IDSW 188 |

> MOTA가 음수로 내려가는 것은 CLEAR-MOT 정의상 정상(FP+FN+IDSW > GT).

### 9.4 소형 객체의 함정

우리 객체는 **17×12px**로 매우 작다. 이 크기에서 **IoU@0.5는 3px 지터만으로도 매칭이 깨진다.**
→ `evaluate.py --iou 0.3` 옵션 제공. 소형 객체 벤치마크에서 낮은 IoU 임계값이나 거리 기반 매칭을 쓰는 이유다.

---

## 10. 알려진 한계

| 항목 | 현황 | 대응 |
|------|------|------|
| **GT 라벨 부재** | 실데이터 정확도 미측정 | 라벨러 준비 완료, 30프레임 라벨링 필요 |
| 캘리브레이션 | 5대 중 1대만 기하 타당성 통과 | 재작업 가이드로 순차 개선 |
| 초점거리 | 대부분 f=W 가정 | 교차로 카메라는 2-VP 자동추정 가능 |
| 단일 평면 가정 | 경사·다층 도로 미지원 | TrafficLab 원본과 동일한 한계 |
| 3D 치수 | 클래스별 고정값(학습 없음) | BEVHeight류 도입 시 해소 |
| ITS 좌표 오차 | 93~793m | GCP 캘리브레이션으로 보정(§5) |
| HD맵 커버리지 | 송도지구·경인선만 보유 | 상암 HD맵 다운로드 필요 |
| 개인정보 | 번호판·얼굴 미마스킹 | 공개·저장 시 조치 필요 |

---

## 11. 출처 및 라이선스

- **엔진**: TrafficLab-3D (MIT © Yuk) 비-GUI 코어 이식·개조 — 원저작권·라이선스 고지 유지
- **지도**: MapLibre GL JS · Esri World Imagery(Esri, Maxar, Earthstar Geographics, CNES/Airbus DS) · Sentinel-2 cloudless © EOX · OpenStreetMap(Overpass)
- **데이터**: ITS 국가교통정보센터 · 서울 TOPIS · UTIC/경찰청 · 국토정보플랫폼 정밀도로지도
- **모델**: Ultralytics YOLO · VisDrone 파인튜닝(HuggingFace `mshamrai/yolov8s-visdrone`) · SAHI · supervision
- **LLM**: Ollama + Qwen3

참고 논문: SAHI(ICIP'22) · OC-SORT(CVPR'23) · UCMCTrack · ByteTrack(ECCV'22) · WBF(IVC'21) ·
Soft Teacher(ICCV'21) · Ekya(NSDI'22) · YOLOV(AAAI'23) · BEVHeight(CVPR'23) · V2X-ViT(ECCV'22)
