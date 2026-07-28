"""
HD맵(정밀도로지도) 노면표시/신호등 → GCP 스냅용 GeoJSON.

calibrate.html의 "HD맵 스냅"에서 로드 → GCP 지도클릭을 cm급 실좌표(HD맵 정점)로 스냅.
- B2_SURFACELINEMARK(정지선/차선, LineString) + C1_TRAFFICLIGHT + A1_NODE(Point)
- 출력: webmap/public/data/hdmap/hdmap.geojson (WGS84)

기본은 송도 시범지구 HD맵(보유). 서울 상암 3D 정밀도로지도를 받으면 --dir 로 교체.
사용: python tools/export_hdmap_snap.py [--dir <HDMap_UTM52N_...>]
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")
import geopandas as gpd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import sys
    sys.path.insert(0, os.path.join(REPO, "data_ingest"))
    from _config import load_config
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=cfg["paths"]["hdmap_dir"])
    args = ap.parse_args()

    feats = []
    for lyr, gtype in [("A2_LINK", "line"), ("B2_SURFACELINEMARK", "line"),
                       ("C1_TRAFFICLIGHT", "pt"), ("A1_NODE", "pt")]:
        p = os.path.join(args.dir, lyr + ".shp")
        if not os.path.exists(p):
            continue
        g = gpd.read_file(p, encoding="cp949").to_crs(4326)
        for geom in g.geometry:
            if geom is None:
                continue
            feats.append({"type": "Feature", "properties": {"layer": lyr},
                          "geometry": json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"]})
    out_dir = os.path.join(REPO, "webmap", "public", "data", "hdmap")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "hdmap.geojson")
    json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    # 정점 수 요약
    nv = 0
    for f in feats:
        gm = f["geometry"]
        nv += len(gm["coordinates"]) if gm["type"] == "LineString" else 1
    print(f"HD맵 스냅: 피처 {len(feats)}개 / 정점 ~{nv}개 → {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
