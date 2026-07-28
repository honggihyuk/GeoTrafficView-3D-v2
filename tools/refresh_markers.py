"""
CCTV 마커 갱신 (webmap/public/data/cameras.geojson).

- ITS(전국 고속도로/국도) · TOPIS(서울, 공개·키불필요) 마커를 bbox로 수집해 병합.
- **replay 카메라(has_replay)는 항상 보존** — 추론/캘리브레이션이 끝난 카메라는
  tools/bridge_to_webmap.py 가 등록하며, 여기서 지워지지 않는다.
- **dedup**: replay 카메라와 이름이 같거나 250m 이내인 원본 마커는 제거(중복/튐 방지).

사용:
  python tools/refresh_markers.py --its 126.40 126.80 37.20 37.60      # 인천 일대 ITS
  python tools/refresh_markers.py --topis 126.86 126.92 37.55 37.60    # 서울 상암 TOPIS
  python tools/refresh_markers.py --its ... --topis ...                # 둘 다
"""
import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "data_ingest"))
sys.path.insert(0, os.path.join(REPO, "tools"))
CAMERAS = os.path.join(REPO, "webmap", "public", "data", "cameras.geojson")


def feat(lon, lat, props):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props}


def dist_m(a, b):
    return math.hypot((a[0] - b[0]) * 88800, (a[1] - b[1]) * 111000)


def its_markers(bbox):
    from _config import load_config
    from its_grab import fetch_cameras
    cfg = load_config()                       # .env.local → os.environ 주입
    key = os.environ.get("ITS_API_KEY")
    if not key:
        print("  [ITS] ITS_API_KEY 없음(.env.local 확인) → 생략"); return []
    import its_grab
    its_grab.WIDE_BBOX = tuple(bbox)          # (minLon,maxLon,minLat,maxLat)
    cams = fetch_cameras(cfg, key)
    return [feat(c["lon"], c["lat"], {"name": c["name"], "type": c["type"], "source": "ITS",
                                      "has_replay": False}) for c in cams]


def topis_markers(bbox):
    try:
        from fetch_topis_cctv import list_cctv
    except Exception as e:
        print(f"  [TOPIS] 모듈 로드 실패({type(e).__name__}) → 생략"); return []
    lo0, lo1, la0, la1 = bbox
    out = []
    for c in list_cctv():
        if lo0 <= c["lng"] <= lo1 and la0 <= c["lat"] <= la1:
            out.append(feat(c["lng"], c["lat"], {"name": c["camName"].strip(), "source": "TOPIS",
                                                 "topis_cam_id": c["camId"], "has_replay": False}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--its", nargs=4, type=float, metavar=("minLon", "maxLon", "minLat", "maxLat"))
    ap.add_argument("--topis", nargs=4, type=float, metavar=("minLon", "maxLon", "minLat", "maxLat"))
    args = ap.parse_args()
    if not args.its and not args.topis:
        sys.exit("--its 또는 --topis bbox 를 지정하세요")

    preserved = []
    if os.path.exists(CAMERAS):
        old = json.load(open(CAMERAS, encoding="utf-8"))
        preserved = [f for f in old["features"] if f["properties"].get("has_replay")]
    print(f"보존할 replay 카메라 {len(preserved)}대: {[f['properties'].get('cctv_id') for f in preserved]}")

    raw = []
    if args.its:
        m = its_markers(args.its); print(f"  [ITS] {len(m)}개"); raw += m
    if args.topis:
        m = topis_markers(args.topis); print(f"  [TOPIS] {len(m)}개"); raw += m

    names = {f["properties"].get("name") for f in preserved}
    keep = [f for f in raw
            if f["properties"].get("name") not in names
            and all(dist_m(f["geometry"]["coordinates"], p["geometry"]["coordinates"]) >= 250
                    for p in preserved)]
    dropped = len(raw) - len(keep)

    feats = keep + preserved
    os.makedirs(os.path.dirname(CAMERAS), exist_ok=True)
    json.dump({"type": "FeatureCollection", "features": feats},
              open(CAMERAS, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"cameras.geojson: {len(feats)}개 (일반 {len(keep)} + replay {len(preserved)}, dedup으로 {dropped}개 제거)")


if __name__ == "__main__":
    main()
