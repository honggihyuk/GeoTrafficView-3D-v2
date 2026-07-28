"""
UTIC(도시교통정보센터/경찰청) CCTV 스트림 페이지 URL 빌더.

- 경찰청 OpenDataCCTV.xlsx 의 CCTVID(예: L020098)가 곧 UTIC CCTV 식별자.
- UTIC 스트림은 raw .m3u8 이 아니라 HLS 플레이어 HTML 페이지를 반환:
    https://www.utic.go.kr/jsp/map/cctvStream.jsp?key=..&cctvid=..&cctvName=..&kind=N&...
  → 뷰어에서 iframe(또는 새 창)으로 재생.
- 인증: **키값 + IP 인증**. UTIC 개방데이터 신청 시 사용 IP를 등록해야 실재생됨.
  키가 없으면 URL 구조만 만들어 둔다(key=__UTIC_KEY__).

사용:
  $env:UTIC_API_KEY='발급키'   (선택; 없으면 placeholder)
  python fetch_utic_cctv.py
출력: webmap/public/data/cctv/utic_cctv.geojson (stream_page = 플레이어 URL)
"""
import json
import os
import urllib.parse
import warnings

warnings.filterwarnings("ignore")
import pandas as pd

from _config import load_config, data_out_dir


def build_url(endpoint, key, cctvid, name, kind):
    q = {
        "key": key, "cctvid": cctvid, "cctvName": name, "kind": kind,
        "cctvip": "0", "cctvch": "null", "id": "", "cctvpasswd": "null", "cctvport": "null",
    }
    return endpoint + "?" + urllib.parse.urlencode(q)


def main():
    cfg = load_config()
    key = os.environ.get("UTIC_API_KEY", "__UTIC_KEY__")
    if key == "__UTIC_KEY__":
        print("[알림] UTIC_API_KEY 없음 → URL 구조만 생성(재생하려면 IP 등록된 키 필요).")

    df = pd.read_excel(cfg["paths"]["cctv_xlsx"], sheet_name=0)
    bb = cfg["aoi_wgs84_bbox"]
    m = (df.XCOORD.between(bb["min_lon"], bb["max_lon"])
         & df.YCOORD.between(bb["min_lat"], bb["max_lat"]))
    sub = df[m]

    ep = cfg["utic"]["stream_endpoint"]; kind = cfg["utic"]["default_kind"]
    feats = []
    for _, r in sub.iterrows():
        cid = str(r.CCTVID); name = str(r.CCTVNAME)
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r.XCOORD), float(r.YCOORD)]},
            "properties": {
                "cctv_id": cid, "name": name, "center": str(r.CENTERNAME),
                "source": "UTIC", "stream_kind": "utic_jsp",
                "stream_page": build_url(ep, key, cid, name, kind),
            },
        })
    out = os.path.join(data_out_dir(cfg, "cctv"), "utic_cctv.geojson")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    print(f"AOI 내 UTIC CCTV {len(feats)}개 → {out}")
    if feats:
        print("예시 stream_page:", feats[0]["properties"]["stream_page"].replace(key, "***KEY***"))


if __name__ == "__main__":
    main()
