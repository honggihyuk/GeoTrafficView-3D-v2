"""
서울 TOPIS CCTV 클라이언트 — 공개(키·IP 불필요), 원시 HLS(.m3u8) 스트림.

발견한 공개 엔드포인트(topis.seoul.go.kr):
  1) POST /map/cctv/selectCctvList.do            → {rows:[{camId,camName,lat,lng}], TotalRows}
  2) POST /map/selectCctvInfo.do (camId,cctvSourceCd=HP) → {rows:[{hlsUrl(.m3u8), roadNm, ...}]}

사용:
  python fetch_topis_cctv.py --bbox 126.87 126.91 37.56 37.60 [--name 상암]
출력: webmap/public/data/cctv/topis_<tag>.geojson (stream_url=hlsUrl → hls.js 직접 재생)
"""
import argparse
import json
import os
import urllib.request

from _config import load_config, data_out_dir

BASE = "https://topis.seoul.go.kr"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": BASE + "/map/openCctvMap.do",
           "X-Requested-With": "XMLHttpRequest",
           "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}


def _post(path, body=""):
    req = urllib.request.Request(BASE + path, data=body.encode(), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def list_cctv():
    return _post("/map/cctv/selectCctvList.do").get("rows", [])


def cctv_hls(cam_id):
    rows = _post("/map/selectCctvInfo.do", f"camId={cam_id}&cctvSourceCd=HP").get("rows", [])
    return rows[0].get("hlsUrl") if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("minLon", "maxLon", "minLat", "maxLat"),
                    default=[126.87, 126.91, 37.56, 37.60])
    ap.add_argument("--name", default="sangam")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    lo0, lo1, la0, la1 = args.bbox

    allc = list_cctv()
    sub = [c for c in allc if lo0 <= c["lng"] <= lo1 and la0 <= c["lat"] <= la1]
    print(f"서울 TOPIS CCTV {len(allc)}개 중 bbox 내 {len(sub)}개")

    feats = []
    for c in sub[:args.limit]:
        hls = cctv_hls(c["camId"])
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [c["lng"], c["lat"]]},
                      "properties": {"cctv_id": f"TOPIS_{c['camId']}", "name": c["camName"].strip(),
                                     "source": "TOPIS", "stream_url": hls}})
    cfg = load_config()
    out = os.path.join(data_out_dir(cfg, "cctv"), f"topis_{args.name}.geojson")
    json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"→ {out} ({len(feats)}개, stream_url 포함)")
    if feats:
        print("예시:", feats[0]["properties"]["name"], "|", feats[0]["properties"]["stream_url"])


if __name__ == "__main__":
    main()
