"""
PostGIS 적재 스크립트 (선택 단계 — DB가 준비됐을 때 실행).

사전:
  1) PostgreSQL + PostGIS 설치, DB 생성:  CREATE DATABASE geotraffic;
  2) psql -d geotraffic -f db/schema.sql
  3) pip install geopandas sqlalchemy geoalchemy2 psycopg2-binary
  4) 환경변수 GEOTRAFFIC_DB_URL 설정
     예) postgresql://postgres:pw@localhost:5432/geotraffic

실행:  python db/load_postgis.py
- HD맵 A2_LINK / C1_TRAFFICLIGHT (원본 EPSG:32652 유지) 적재
- CCTV (엑셀 → 4326) 적재
탐지(detection) 적재는 TrafficLab 추론 산출물이 나온 뒤 별도 스크립트로.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_ingest"))
from _config import load_config  # noqa: E402


def main():
    url = os.environ.get("GEOTRAFFIC_DB_URL")
    if not url:
        print("GEOTRAFFIC_DB_URL 환경변수가 없습니다. README(db) 참고.")
        return
    import geopandas as gpd
    import pandas as pd
    from sqlalchemy import create_engine

    cfg = load_config()
    eng = create_engine(url)
    hd = cfg["paths"]["hdmap_dir"]

    # A2_LINK (EPSG:32652 유지)
    g = gpd.read_file(os.path.join(hd, "A2_LINK.shp"))
    g = g.rename(columns={"ITSLinkID": "its_link_id", "RoadRank": "road_rank",
                          "LaneNo": "lane_no", "ID": "id"})
    keep = [c for c in ["id", "its_link_id", "road_rank", "lane_no", "geometry"] if c in g.columns]
    g[keep].to_postgis("hd_link", eng, if_exists="append", index=False)
    print(f"hd_link: {len(g)} 행 적재")

    # C1_TRAFFICLIGHT
    t = gpd.read_file(os.path.join(hd, "C1_TRAFFICLIGHT.shp"))
    t = t.rename(columns={"ID": "id", "Type": "type"})
    keep = [c for c in ["id", "type", "geometry"] if c in t.columns]
    t[keep].to_postgis("hd_trafficlight", eng, if_exists="append", index=False)
    print(f"hd_trafficlight: {len(t)} 행 적재")

    # CCTV (4326)
    df = pd.read_excel(cfg["paths"]["cctv_xlsx"], sheet_name=0)
    bb = cfg["aoi_wgs84_bbox"]
    m = (df.XCOORD.between(bb["min_lon"], bb["max_lon"])
         & df.YCOORD.between(bb["min_lat"], bb["max_lat"]))
    sub = df[m]
    cg = gpd.GeoDataFrame(
        {"cctv_id": sub.CCTVID.astype(str), "name": sub.CCTVNAME.astype(str),
         "center": sub.CENTERNAME.astype(str), "stream_url": None},
        geometry=gpd.points_from_xy(sub.XCOORD, sub.YCOORD), crs=4326)
    cg.to_postgis("cctv", eng, if_exists="append", index=False)
    print(f"cctv: {len(cg)} 행 적재")


if __name__ == "__main__":
    main()
