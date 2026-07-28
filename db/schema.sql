-- GeoTrafficView-3D PostGIS 스키마
-- 실행: psql -U postgres -d geotraffic -f schema.sql
-- 사전: CREATE DATABASE geotraffic; 후 아래 실행.

CREATE EXTENSION IF NOT EXISTS postgis;
-- 시계열 탐지 대용량 처리용(선택). 미설치 시 아래 create_hypertable 줄만 건너뛰면 됨.
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 원본 좌표계: EPSG:32652 (UTM52N). 웹 표출용은 4326으로 뷰 생성.

-- 1) CCTV (경찰청 엑셀 + ITS/UTIC 매칭 결과)
CREATE TABLE IF NOT EXISTS cctv (
    cctv_id     text PRIMARY KEY,
    name        text,
    center      text,
    stream_url  text,                       -- ITS/UTIC HLS(.m3u8) 등, 매칭 후 채움
    geom        geometry(Point, 4326)
);
CREATE INDEX IF NOT EXISTS cctv_gix ON cctv USING GIST (geom);

-- 2) 정밀도로지도 차선 링크(A2_LINK) — 대표 예시. 다른 레이어도 동일 패턴.
CREATE TABLE IF NOT EXISTS hd_link (
    id          text,
    its_link_id text,                       -- A2_LINK.ITSLinkID → ITS 교통데이터 연계 키
    road_rank   text,
    lane_no     integer,
    geom        geometry(LineStringZ, 32652)
);
CREATE INDEX IF NOT EXISTS hd_link_gix ON hd_link USING GIST (geom);

CREATE TABLE IF NOT EXISTS hd_trafficlight (
    id    text,
    type  text,
    geom  geometry(PointZ, 32652)
);
CREATE INDEX IF NOT EXISTS hd_tl_gix ON hd_trafficlight USING GIST (geom);

-- 3) 객체 탐지 결과(시계열) — TrafficLab .json.gz 를 프레임·객체 단위로 적재
CREATE TABLE IF NOT EXISTS detection (
    ts          timestamptz NOT NULL,
    cctv_id     text REFERENCES cctv(cctv_id),
    track_id    integer,
    class       text,                        -- car, bus, truck, person ...
    confidence  real,
    speed_kmh   real,
    heading_deg real,
    geom        geometry(PointZ, 32652),     -- 실세계 지상 접촉점
    bbox_3d     jsonb                          -- 8점 3D 박스(선택)
);
CREATE INDEX IF NOT EXISTS detection_gix ON detection USING GIST (geom);
CREATE INDEX IF NOT EXISTS detection_ts  ON detection (ts);
CREATE INDEX IF NOT EXISTS detection_cctv_ts ON detection (cctv_id, ts);
-- TimescaleDB 사용 시:
-- SELECT create_hypertable('detection', 'ts', if_not_exists => TRUE);

-- 4) 분석 편의 뷰: 탐지 → 가장 가까운 차선링크에 스냅(차선/구간 단위 집계)
CREATE OR REPLACE VIEW detection_on_link AS
SELECT d.*, l.id AS link_id, l.its_link_id,
       ST_Distance(d.geom, l.geom) AS dist_m
FROM detection d
CROSS JOIN LATERAL (
    SELECT id, its_link_id, geom
    FROM hd_link
    ORDER BY d.geom <-> geom
    LIMIT 1
) l;

-- LLM(NL2SQL)이 안전하게 읽도록 읽기전용 롤 예시:
-- CREATE ROLE llm_readonly LOGIN PASSWORD 'change_me';
-- GRANT CONNECT ON DATABASE geotraffic TO llm_readonly;
-- GRANT USAGE ON SCHEMA public TO llm_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO llm_readonly;
