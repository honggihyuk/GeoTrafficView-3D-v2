"""
추론 결과(.json.gz) → SQLite 적재 (PostGIS 없이 즉시 동작하는 경로).

- PostGIS(db/ingest_postgis.py)와 **동일한 테이블/뷰 구조**: detection, cctv, cctv_summary.
- 카메라 단위 idempotent: 같은 cctv_id 재적재 시 해당 카메라 행만 교체(다른 카메라 보존).
- 좌표: world 원점(UTM52N) + sat_coords(로컬 미터) → UTM → WGS84(pyproj).

사용:
  python db/ingest_sqlite.py --src output/reprojected/PANGYO_2/clip.json.gz
  python db/ingest_sqlite.py --all          # output/ 의 카메라별 최신 결과 전부
출력: db/geotraffic.db
"""
import argparse
import glob
import gzip
import json
import os
import sqlite3
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
from pyproj import Transformer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(REPO, "db", "geotraffic.db")
BASE_TS = datetime(2026, 7, 27, 15, 0, 0)  # 클립 기준 시각(캡처시각 미상 → 고정 기준)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cctv (
    cctv_id TEXT PRIMARY KEY, name TEXT, lon REAL, lat REAL);
CREATE TABLE IF NOT EXISTS detection (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, frame INTEGER, cctv_id TEXT, track_id INTEGER,
    class TEXT, confidence REAL, speed_kmh REAL, heading_deg REAL,
    lon REAL, lat REAL, easting REAL, northing REAL);
CREATE INDEX IF NOT EXISTS ix_det_class ON detection(class);
CREATE INDEX IF NOT EXISTS ix_det_track ON detection(track_id);
CREATE INDEX IF NOT EXISTS ix_det_cctv  ON detection(cctv_id);
CREATE VIEW IF NOT EXISTS cctv_summary AS
  SELECT d.cctv_id,
         COALESCE(c.name, d.cctv_id) AS name,
         COUNT(DISTINCT d.track_id)  AS unique_objects,
         ROUND(AVG(CASE WHEN d.speed_kmh > 1 THEN d.speed_kmh END), 1) AS avg_speed_kmh,
         COUNT(*) AS detections
  FROM detection d LEFT JOIN cctv c ON c.cctv_id = d.cctv_id
  GROUP BY d.cctv_id;
"""


def latest_per_camera():
    """카메라별 적재 대상 선택.

    1순위: webmap/public/data/replay/*.json.gz — **웹맵에 실제 표출 중인 결과**(화면=DB 일치).
    2순위: output/ 스캔(reprojected 우선, 그다음 파일 크기 = 검출량이 많은 결과).
    """
    best = {}
    for p in glob.glob(os.path.join(REPO, "webmap", "public", "data", "replay", "*.json.gz")):
        try:
            loc = json.load(gzip.open(p, "rt", encoding="utf-8")).get("location_code")
        except Exception:
            continue
        if loc:
            best[loc] = p
    for p in glob.glob(os.path.join(REPO, "output", "**", "*.json.gz"), recursive=True):
        loc = os.path.basename(os.path.dirname(p))
        if loc in best:
            continue
        score = (1 if os.sep + "reprojected" + os.sep in p else 0, os.path.getsize(p))
        prev = best.get("__score__" + loc)
        if prev is None or score > prev:
            best["__score__" + loc] = score
            best[loc] = p
    return [v for k, v in best.items() if not k.startswith("__score__")]


def ingest(src, con):
    data = json.load(gzip.open(src, "rt", encoding="utf-8"))
    world = data["meta"]["world"]
    fps = data["meta"].get("fps", 30)
    loc = data.get("location_code", "CAM")
    tr = Transformer.from_crs(world["epsg"], 4326, always_xy=True)
    cur = con.cursor()

    # 카메라 메타: 위치=world 원점(캘리브레이션 기준), 이름=_camera.json
    cam_lon, cam_lat = tr.transform(world["origin_easting"], world["origin_northing"])
    name = loc
    cj = os.path.join(REPO, "location", loc, "_camera.json")
    if os.path.exists(cj):
        name = json.load(open(cj, encoding="utf-8")).get("name") or loc
    cur.execute("INSERT INTO cctv(cctv_id,name,lon,lat) VALUES(?,?,?,?) "
                "ON CONFLICT(cctv_id) DO UPDATE SET name=excluded.name, lon=excluded.lon, lat=excluded.lat",
                (loc, name, cam_lon, cam_lat))
    cur.execute("DELETE FROM detection WHERE cctv_id=?", (loc,))

    rows = []
    for fr in data["frames"]:
        i = fr["frame_index"]
        ts = (BASE_TS + timedelta(seconds=i / fps)).isoformat(sep=" ")
        for o in fr.get("objects", []):
            sat = o.get("sat_coords")
            if not sat:
                continue
            e = world["origin_easting"] + sat[0]
            n = world["origin_northing"] + sat[1]
            lon, lat = tr.transform(e, n)
            rows.append((ts, i, loc, o.get("tracked_id"), o.get("class"), o.get("confidence"),
                         o.get("speed_kmh"), o.get("heading"), lon, lat, e, n))
    cur.executemany(
        "INSERT INTO detection (ts,frame,cctv_id,track_id,class,confidence,speed_kmh,heading_deg,lon,lat,easting,northing)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    print(f"[{loc}] detection {len(rows)}행 적재 ({os.path.relpath(src, REPO)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA)
    srcs = latest_per_camera() if args.all else [args.src or latest_per_camera()[0]]
    for s in srcs:
        ingest(s, con)
    cur = con.cursor()
    print("\n=== cctv_summary ===")
    for r in cur.execute("SELECT cctv_id, name, unique_objects, avg_speed_kmh FROM cctv_summary ORDER BY cctv_id"):
        print(f"  {r[0]:16s} 고유 {r[2]:4d}  평균 {r[3]} km/h")
    con.close()


if __name__ == "__main__":
    main()
