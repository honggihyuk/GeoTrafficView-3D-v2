"""
ITS 국가교통정보센터 OpenAPI → 송도 AOI 내 CCTV(실시간 HLS 스트림) 목록 → GeoJSON.

문서: https://www.its.go.kr/opendata/  (CCTV 정보 조회)
엔드포인트: https://openapi.its.go.kr:9443/cctvInfo
  파라미터: apiKey, type(its|ex), cctvType(1=HLS 스트리밍,2=파일,3=정지영상),
            minX,maxX(경도), minY,maxY(위도), getType(json|xml)
응답(JSON): response.data[] = { coordx, coordy, cctvname, cctvurl, cctvformat, cctvtype }

사용:
  set ITS_API_KEY=발급받은키          (Windows: setx / PowerShell $env:ITS_API_KEY=)
  python fetch_its_cctv.py            # 실제 호출
  python fetch_its_cctv.py --selftest # 키 없이 샘플 응답으로 파싱 검증

출력: webmap/public/data/cctv/its_cctv.geojson  (stream_url = cctvurl)
"""
import json
import os
import ssl
import sys
import urllib.parse
import urllib.request

from _config import load_config, data_out_dir

# ── 파싱 검증용 대표 샘플(실제 ITS 응답 스키마와 동일, 좌표만 송도로 배치) ──
SAMPLE_RESPONSE = {
    "response": {
        "datacount": 2,
        "data": [
            {"coordx": 126.6412, "coordy": 37.3921, "cctvtype": "1", "cctvformat": "HLS",
             "cctvname": "송도국제대로 A", "cctvurl": "http://cctvsec.ktict.co.kr/1/abc.m3u8"},
            {"coordx": 126.6438, "coordy": 37.3895, "cctvtype": "1", "cctvformat": "HLS",
             "cctvname": "컨벤시아대로 B", "cctvurl": "http://cctvsec.ktict.co.kr/2/def.m3u8"},
        ],
    }
}


def parse_response(obj, bbox):
    """ITS 응답 dict → GeoJSON Feature 리스트(AOI 필터 + HLS만)."""
    data = (obj.get("response") or {}).get("data") or []
    feats = []
    for d in data:
        try:
            lon = float(d["coordx"]); lat = float(d["coordy"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (bbox["min_lon"] <= lon <= bbox["max_lon"] and bbox["min_lat"] <= lat <= bbox["max_lat"]):
            continue
        url = d.get("cctvurl")
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "name": d.get("cctvname"),
                "stream_url": url,
                "format": d.get("cctvformat"),
                "source": "ITS",
            },
        })
    return feats


def fetch(cfg, api_key):
    its = cfg["its"]; bb = cfg["aoi_wgs84_bbox"]
    ctx = ssl.create_default_context()
    all_feats = []
    for typ in its["types"]:
        params = {
            "apiKey": api_key, "type": typ, "cctvType": its["cctv_type"],
            "minX": bb["min_lon"], "maxX": bb["max_lon"],
            "minY": bb["min_lat"], "maxY": bb["max_lat"],
            "getType": its["getType"],
        }
        url = its["endpoint"] + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=30) as r:
                obj = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            msg = str(e).replace(api_key, "***")  # 키 유출 방지
            print(f"  [type={typ}] 호출 실패: {type(e).__name__}: {msg}")
            continue
        feats = parse_response(obj, bb)
        print(f"  [type={typ}] AOI 내 스트림 CCTV {len(feats)}개")
        all_feats.extend(feats)
    return all_feats


def write(cfg, feats):
    out_dir = data_out_dir(cfg, "cctv")
    out = os.path.join(out_dir, "its_cctv.geojson")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    print(f"→ {out} ({len(feats)}개)")


def main():
    cfg = load_config()
    if "--selftest" in sys.argv:
        feats = parse_response(SAMPLE_RESPONSE, cfg["aoi_wgs84_bbox"])
        print(f"[selftest] 샘플 파싱 결과 {len(feats)}개 (AOI 필터 적용)")
        write(cfg, feats)
        return
    api_key = os.environ.get("ITS_API_KEY")
    if not api_key:
        print("ITS_API_KEY 환경변수가 없습니다.")
        print("  1) https://www.its.go.kr/opendata 에서 무료 인증키 발급")
        print("  2) PowerShell:  $env:ITS_API_KEY='발급키'  후 재실행")
        print("  (키 없이 파싱만 확인: python fetch_its_cctv.py --selftest)")
        return
    feats = fetch(cfg, api_key)
    write(cfg, feats)


if __name__ == "__main__":
    main()
