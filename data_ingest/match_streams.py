"""
경찰청 CCTV(위치만) ↔ ITS CCTV(스트림 URL 보유) 공간 매칭.

- 각 경찰청 CCTV 포인트에서 max_distance_m 이내 가장 가까운 ITS 스트림 CCTV를 찾아
  stream_url 을 채운다(동일 카메라를 서로 다른 기관이 등록한 경우 연결).
- 입력:  webmap/public/data/cctv/cctv.geojson      (export_cctv.py 산출)
          webmap/public/data/cctv/its_cctv.geojson  (fetch_its_cctv.py 산출)
- 출력:  cctv.geojson 을 갱신(stream_url, matched_its_name, match_dist_m 추가)

거리 계산은 EPSG:32652(UTM52N, 미터)로 재투영해 수행.
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")
import geopandas as gpd

from _config import load_config, data_out_dir


def main():
    cfg = load_config()
    out_dir = data_out_dir(cfg, "cctv")
    police_p = os.path.join(out_dir, "cctv.geojson")
    its_p = os.path.join(out_dir, "its_cctv.geojson")

    if not os.path.exists(its_p):
        print("its_cctv.geojson 없음 → 먼저 fetch_its_cctv.py 실행(또는 --selftest).")
        return

    police = gpd.read_file(police_p)
    its = gpd.read_file(its_p)
    if len(its) == 0:
        print("ITS 스트림 CCTV 0개 → 매칭 생략.")
        return

    epsg = cfg["crs"]["source_epsg"]
    maxd = cfg["stream_match"]["max_distance_m"]
    pm = police.to_crs(epsg=epsg)
    im = its.to_crs(epsg=epsg)[["geometry", "name", "stream_url"]].rename(
        columns={"name": "its_name", "stream_url": "its_stream"})

    joined = gpd.sjoin_nearest(pm, im, how="left", max_distance=maxd, distance_col="match_dist_m")
    # 인덱스 중복(동일 최근접 다수) 제거: 첫 매칭만
    joined = joined[~joined.index.duplicated(keep="first")]

    matched = 0
    for idx in police.index:
        row = joined.loc[idx] if idx in joined.index else None
        if row is not None and isinstance(row.get("its_stream"), str):
            police.at[idx, "stream_url"] = row["its_stream"]
            police.at[idx, "matched_its_name"] = row.get("its_name")
            police.at[idx, "match_dist_m"] = round(float(row.get("match_dist_m") or 0), 1)
            matched += 1

    police.to_file(police_p, driver="GeoJSON")
    print(f"경찰청 {len(police)}개 중 {matched}개에 ITS 스트림 매칭(≤{maxd}m) → {police_p}")


if __name__ == "__main__":
    main()
