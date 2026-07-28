"""
경찰청 OpenDataCCTV.xlsx → 송도 AOI 필터 → GeoJSON(Point, WGS84).

- 원본 컬럼: RN, CCTVID, CCTVNAME, CENTERNAME, XCOORD(경도), YCOORD(위도)
- 전국 14,591개 중 AOI(project_config.json aoi_wgs84_bbox) 안의 CCTV만 추출.
- 주의: 이 파일은 '위치'만 제공. 실제 스트림 URL은 ITS/UTIC API로 별도 매칭 필요(stream_url=null).
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")
import pandas as pd

from _config import load_config, data_out_dir


def main():
    cfg = load_config()
    df = pd.read_excel(cfg["paths"]["cctv_xlsx"], sheet_name=0)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    bb = cfg["aoi_wgs84_bbox"]
    m = (
        df.XCOORD.between(bb["min_lon"], bb["max_lon"])
        & df.YCOORD.between(bb["min_lat"], bb["max_lat"])
    )
    sub = df[m].copy()

    features = []
    for _, r in sub.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r.XCOORD), float(r.YCOORD)]},
            "properties": {
                "cctv_id": str(r.CCTVID),
                "name": str(r.CCTVNAME),
                "center": str(r.CENTERNAME),
                "stream_url": None,  # TODO: ITS/UTIC API 매칭으로 채움
            },
        })
    fc = {"type": "FeatureCollection", "features": features}
    out_dir = data_out_dir(cfg, "cctv")
    out = os.path.join(out_dir, "cctv.geojson")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    print(f"전국 {len(df)}개 중 AOI 내 {len(sub)}개 → {out}")


if __name__ == "__main__":
    main()
