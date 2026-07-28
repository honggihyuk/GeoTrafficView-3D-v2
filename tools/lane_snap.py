"""
차선 정렬(lane snapping) — 이동체를 도로 중심선/차로에 맞춰 정렬.

탐지·투영 결과는 검출 지터와 캘리브레이션 오차 때문에 도로를 벗어나거나 차로를 가로질러
흔들린다. 이 도구는 각 객체의 지면 접촉점을 **차선 네트워크에 횡방향 스냅**하고
**heading을 차로 방향에 정렬**해, 지도 위에서 차량이 차선을 따라 달리게 만든다.

차선 네트워크 소스(우선순위):
  1) **정밀도로지도 A2_LINK** (차로 단위 링크) — webmap/public/data/hdmap/hdmap.geojson
     (tools/export_hdmap_snap.py 로 생성). 카메라가 HD맵 범위 안일 때만 사용 → 진짜 차로 정렬.
  2) **OSM 도로 중심선**(Overpass) — HD맵 미보유 지역 폴백. 중심선 기준이므로
     `--lane-width`(기본 3.5m)로 차로 중심에 양자화해 차로 단위 정렬을 근사한다.

처리:
  sat_coords(로컬 m) → 최근접 세그먼트 투영 → 횡방향 오프셋을 차로 중심으로 양자화
  → heading = 차로 방위(관측 heading으로 진행방향 결정) → sat_floor_box·bbox_3d 재생성

사용:
  python tools/lane_snap.py --loc PANGYO_2                       # 웹맵 등록본 자동 사용
  python tools/lane_snap.py --loc PANGYO_2 --max-dist 20 --lane-width 3.5
출력: output/lanesnap/<loc>/clip.json.gz  (이후 bridge_to_webmap.py 로 반영)
"""
import argparse
import gzip
import json
import math
import os
import sys
import urllib.parse
import urllib.request
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from pyproj import Transformer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from trafficlab.projection.g_projection import GProjection


# ---------- 차선 네트워크 ----------
def lanes_from_hdmap(world, max_km=2.0):
    """정밀도로지도 A2_LINK(차로 링크) → 로컬 미터 폴리라인. 범위 밖이면 []."""
    p = os.path.join(REPO, "webmap", "public", "data", "hdmap", "hdmap.geojson")
    if not os.path.exists(p):
        return []
    tr = Transformer.from_crs(4326, world["epsg"], always_xy=True)
    oE, oN = world["origin_easting"], world["origin_northing"]
    out = []
    fc = json.load(open(p, encoding="utf-8"))
    for f in fc.get("features", []):
        if f.get("properties", {}).get("layer") != "A2_LINK":
            continue
        g = f["geometry"]
        if g["type"] != "LineString":
            continue
        pts = []
        for lon, lat, *_ in g["coordinates"]:
            e, n = tr.transform(lon, lat)
            pts.append((e - oE, n - oN))
        if pts and min(math.hypot(*q) for q in pts) < max_km * 1000:
            out.append(pts)
    return out


def lanes_from_osm(world, radius=400):
    """OSM highway 중심선 → 로컬 미터 폴리라인."""
    tr_ll = Transformer.from_crs(world["epsg"], 4326, always_xy=True)
    lon, lat = tr_ll.transform(world["origin_easting"], world["origin_northing"])
    q = (f"[out:json][timeout:25];way(around:{radius},{lat},{lon})"
         f"[highway~'^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|motorway_link|trunk_link|primary_link)$'];out geom;")
    # 공개 Overpass는 rate-limit이 잦아 미러를 순차 재시도(UA 없으면 406)
    mirrors = ["https://overpass-api.de/api/interpreter",
               "https://overpass.kumi.systems/api/interpreter",
               "https://overpass.osm.jp/api/interpreter"]
    data = None
    for m in mirrors:
        req = urllib.request.Request(m + "?" + urllib.parse.urlencode({"data": q}),
                                     headers={"User-Agent": "GeoTrafficView-3D/2.0"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            print(f"  OSM 실패({m.split('/')[2]}: {type(e).__name__}) → 다음 미러")
    if data is None:
        return []
    tr = Transformer.from_crs(4326, world["epsg"], always_xy=True)
    oE, oN = world["origin_easting"], world["origin_northing"]
    out = []
    for w in data.get("elements", []):
        g = w.get("geometry") or []
        pts = []
        for nd in g:
            e, n = tr.transform(nd["lon"], nd["lat"])
            pts.append((e - oE, n - oN))
        if len(pts) >= 2:
            out.append(pts)
    return out


def to_segments(polylines):
    segs = []
    for pts in polylines:
        for i in range(len(pts) - 1):
            a = np.array(pts[i], float); b = np.array(pts[i + 1], float)
            if np.linalg.norm(b - a) > 0.5:
                segs.append((a, b))
    return segs


# ---------- 스냅 ----------
def snap_point(p, segs, max_dist, lane_w, lanes_per_side):
    """p(로컬 m) → (스냅점, 세그먼트 방위deg, 원래 거리) / 실패 시 None."""
    p = np.array(p, float)
    best = None
    for a, b in segs:
        ab = b - a
        L2 = ab @ ab
        t = max(0.0, min(1.0, ((p - a) @ ab) / L2))
        proj = a + t * ab
        d = np.linalg.norm(p - proj)
        if best is None or d < best[0]:
            best = (d, proj, ab)
    if best is None or best[0] > max_dist:
        return None
    d, proj, ab = best
    u = ab / np.linalg.norm(ab)              # 진행방향 단위벡터
    nvec = np.array([-u[1], u[0]])           # 좌측 법선
    off = float((p - proj) @ nvec)           # 부호 있는 횡방향 오프셋
    if lane_w > 0:                           # 차로 중심으로 양자화
        k = round((abs(off) - lane_w / 2) / lane_w)
        k = max(0, min(int(k), lanes_per_side - 1))
        off_s = math.copysign(lane_w / 2 + k * lane_w, off if off != 0 else 1.0)
    else:
        off_s = 0.0
    snapped = proj + off_s * nvec
    bearing = (math.degrees(math.atan2(u[1], u[0])) + 360) % 360
    return snapped, bearing, d


def main():
    try:  # Windows 콘솔 cp949에서 한글/기호 출력 크래시 방지
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--max-dist", type=float, default=20.0, help="이 거리(m)보다 먼 객체는 스냅 안 함")
    ap.add_argument("--lane-width", type=float, default=3.5, help="0이면 중심선에 바로 스냅")
    ap.add_argument("--lanes-per-side", type=int, default=4)
    ap.add_argument("--source", choices=["auto", "hdmap", "osm"], default="auto")
    args = ap.parse_args()

    src = args.src or os.path.join(REPO, "webmap", "public", "data", "replay", f"{args.loc.lower()}.json.gz")
    if not os.path.exists(src):
        sys.exit(f"입력 replay 없음: {src}")
    data = json.load(gzip.open(src, "rt", encoding="utf-8"))
    world = data["meta"]["world"]

    gpath = os.path.join(REPO, "location", args.loc, f"G_projection_{args.loc}.json")
    g_data = json.load(open(gpath, encoding="utf-8"))
    g = GProjection(g_data, base_dir=os.path.dirname(gpath))
    priors = json.load(open(os.path.join(REPO, "prior_dimensions.json"), encoding="utf-8"))["measurements_visdrone_full"]
    priors = {k.lower(): v for k, v in priors.items()}

    # 차선 네트워크
    lanes, kind = [], None
    if args.source in ("auto", "hdmap"):
        lanes = lanes_from_hdmap(world); kind = "HD맵 A2_LINK(차로 단위)" if lanes else None
    if not lanes and args.source in ("auto", "osm"):
        lanes = lanes_from_osm(world); kind = "OSM 중심선(+차로 양자화)" if lanes else None
    if not lanes:
        sys.exit("차선 네트워크를 얻지 못했습니다(HD맵 범위 밖 + OSM 실패)")
    segs = to_segments(lanes)
    print(f"차선 소스: {kind} · 폴리라인 {len(lanes)} / 세그먼트 {len(segs)}")

    n_obj = n_snap = 0
    shifts, dh = [], []
    for fr in data["frames"]:
        for o in fr.get("objects", []):
            sat = o.get("sat_coords")
            if not sat:
                continue
            n_obj += 1
            r = snap_point(sat, segs, args.max_dist, args.lane_width, args.lanes_per_side)
            if r is None:
                continue
            snapped, bearing, d0 = r
            # 진행방향: 관측 heading에 가까운 쪽 선택(링크 디지타이즈 방향 모호성 해소)
            h_obs = o.get("heading")
            head = bearing
            if h_obs is not None:
                diff = abs((h_obs - bearing + 180) % 360 - 180)
                if diff > 90:
                    head = (bearing + 180) % 360
                dh.append(abs((h_obs - head + 180) % 360 - 180))
            shifts.append(float(np.linalg.norm(np.array(sat) - snapped)))
            o["sat_coords"] = [float(snapped[0]), float(snapped[1])]
            o["heading"] = float(head)
            o["have_heading"] = True
            o["lane_snapped"] = True
            o["lane_offset_m"] = round(float(d0), 2)

            dims = priors.get(str(o.get("class", "")).lower())
            if dims:
                w_m, l_m = dims["width"], dims["length"]
                ang = np.radians(head); c, s = np.cos(ang), np.sin(ang)
                dx, dy = (l_m * g.px_per_m) / 2, (w_m * g.px_per_m) / 2
                corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
                R = np.array([[c, -s], [s, c]])
                floor = (corners @ R.T + snapped).tolist()
                o["sat_floor_box"] = floor
                o["bbox_3d"] = g.sat_floor_to_cctv_3d(floor, float(dims["height"]))
            n_snap += 1

    out = os.path.join(REPO, "output", "lanesnap", args.loc, "clip.json.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"스냅 {n_snap}/{n_obj} 객체 ({100 * n_snap / max(n_obj,1):.0f}%) — max-dist {args.max_dist}m 내")
    if shifts:
        print(f"  위치 이동량: 평균 {np.mean(shifts):.2f} m · 중앙값 {np.median(shifts):.2f} m · "
              f"95% {np.percentile(shifts,95):.2f} m · 최대 {np.max(shifts):.2f} m  (작을수록 궤적 왜곡 적음)")
    if dh:
        print(f"  heading 보정량: 평균 {np.mean(dh):.1f}deg  (스냅 후 heading = 차로 방위와 정확히 평행)")
    # 품질 게이트: 스냅 통계는 캘리브레이션 품질의 프록시다.
    rate = 100 * n_snap / max(n_obj, 1)
    mean_dh = float(np.mean(dh)) if dh else 0.0
    if rate < 50 or mean_dh > 45:
        print("\n  ⚠ 캘리브레이션 점검 필요: "
              f"스냅률 {rate:.0f}%{' (낮음)' if rate < 50 else ''}"
              f"{f', heading 보정 {mean_dh:.0f}deg (과대)' if mean_dh > 45 else ''}")
        print("    → 객체가 도로에서 벗어나 있거나 진행방향이 어긋납니다. 이 상태로 스냅하면 "
              "보기엔 좋아도 실측과 달라집니다.\n"
              f"    → calibrate.html 에서 {args.loc} 를 GCP 2~4점으로 보정한 뒤 "
              f"reproject.py → lane_snap.py 순으로 다시 실행하세요.")
    print(f"\n→ {os.path.relpath(out, REPO)}")
    print(f"다음: python tools/bridge_to_webmap.py --replay {os.path.relpath(out, REPO)} --loc {args.loc}")


if __name__ == "__main__":
    main()
