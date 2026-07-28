"""
Phase 7(DB 고도화): 추론 결과(.json.gz) → PostGIS 적재 (다중 카메라).

- geom(4326) + geom_utm(32652) 공간 컬럼, GIST/시간 인덱스.
- 카메라 단위 idempotent: 같은 cctv_id 재적재 시 해당 카메라 행만 교체.
- TimescaleDB 있으면 detection 을 hypertable 로(없으면 시간 인덱스로 대체).

환경변수 GEOTRAFFIC_DB_URL (기본 localhost:5434 컨테이너).
사용: python db/ingest_postgis.py --src <json.gz>   (여러 카메라면 반복 호출)
"""
import argparse
import glob
import gzip
import json
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
import psycopg2
from psycopg2.extras import execute_values
from pyproj import Transformer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URL = "postgresql://postgres:geotraffic@localhost:5434/geotraffic"
BASE_TS = datetime(2026, 7, 27, 15, 0, 0)

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE TABLE IF NOT EXISTS cctv (
    cctv_id text PRIMARY KEY, name text, geom geometry(Point,4326));
CREATE TABLE IF NOT EXISTS detection (
    id bigserial PRIMARY KEY, ts timestamptz, frame int, cctv_id text,
    track_id int, class text, confidence real, speed_kmh real, heading_deg real,
    geom geometry(Point,4326), geom_utm geometry(Point,32652));
CREATE INDEX IF NOT EXISTS det_gix ON detection USING GIST(geom);
CREATE INDEX IF NOT EXISTS det_cctv_ts ON detection(cctv_id, ts);
CREATE INDEX IF NOT EXISTS det_class ON detection(class);
CREATE OR REPLACE VIEW cctv_summary AS
  SELECT d.cctv_id, c.name,
         COUNT(DISTINCT d.track_id) AS unique_objects,
         ROUND(AVG(d.speed_kmh) FILTER (WHERE d.speed_kmh>1)::numeric,1) AS avg_speed_kmh,
         COUNT(*) AS detections
  FROM detection d LEFT JOIN cctv c USING (cctv_id) GROUP BY d.cctv_id, c.name;
"""


def latest_src():
    hits = glob.glob(os.path.join(REPO, "output", "**", "*.json.gz"), recursive=True)
    return max(hits, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--url", default=os.environ.get("GEOTRAFFIC_DB_URL", DEFAULT_URL))
    args = ap.parse_args()
    src = args.src or latest_src()

    data = json.load(gzip.open(src, "rt", encoding="utf-8"))
    world = data["meta"]["world"]; fps = data["meta"].get("fps", 30)
    loc = data.get("location_code", "CAM")
    tr = Transformer.from_crs(world["epsg"], 4326, always_xy=True)

    con = psycopg2.connect(args.url); cur = con.cursor()
    cur.execute(SCHEMA)
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        cur.execute("SELECT create_hypertable('detection','ts',if_not_exists=>TRUE,migrate_data=>TRUE);")
    except Exception:
        con.rollback(); cur = con.cursor(); cur.execute(SCHEMA)  # timescale 없음 → 일반 테이블

    # 카메라 메타: 위치=G_projection world.origin(캘리브레이션 기준), 이름=_camera.json
    cam_lon, cam_lat = tr.transform(world["origin_easting"], world["origin_northing"])
    cam_name = loc
    cj = os.path.join(REPO, "location", loc, "_camera.json")
    if os.path.exists(cj):
        cam_name = json.load(open(cj, encoding="utf-8")).get("name") or loc
    cur.execute("""INSERT INTO cctv(cctv_id,name,geom) VALUES(%s,%s,ST_SetSRID(ST_MakePoint(%s,%s),4326))
                   ON CONFLICT(cctv_id) DO UPDATE SET name=EXCLUDED.name, geom=EXCLUDED.geom""",
                (loc, cam_name, cam_lon, cam_lat))

    cur.execute("DELETE FROM detection WHERE cctv_id=%s", (loc,))
    rows = []
    for fr in data["frames"]:
        i = fr["frame_index"]; ts = BASE_TS + timedelta(seconds=i / fps)
        for o in fr.get("objects", []):
            sat = o.get("sat_coords")
            if not sat:
                continue
            e = world["origin_easting"] + sat[0]; n = world["origin_northing"] + sat[1]
            lon, lat = tr.transform(e, n)
            rows.append((ts, i, loc, o.get("tracked_id"), o.get("class"), o.get("confidence"),
                         o.get("speed_kmh"), o.get("heading"), lon, lat, e, n))
    execute_values(cur,
        """INSERT INTO detection(ts,frame,cctv_id,track_id,class,confidence,speed_kmh,heading_deg,geom,geom_utm)
           VALUES %s""",
        rows,
        template="(%s,%s,%s,%s,%s,%s,%s,%s,ST_SetSRID(ST_MakePoint(%s,%s),4326),ST_SetSRID(ST_MakePoint(%s,%s),32652))")
    con.commit()
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT track_id) FROM detection WHERE cctv_id=%s", (loc,))
    c1, c2 = cur.fetchone()
    print(f"[{loc}] detection {c1}행 / 고유 {c2} 적재 → {args.url.split('@')[-1]}")
    con.close()


if __name__ == "__main__":
    main()
