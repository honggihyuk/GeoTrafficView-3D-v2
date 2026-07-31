"""
HD맵 GCP 후보 추천 — "영상에서 어디를 클릭할지"를 정해 준다.

**왜 필요한가**
  GCP 한 쌍은 (영상 픽셀, 실좌표)다. 지도 쪽은 🧲 HD맵 스냅이 cm급으로 해결했다. 남은
  어려움은 **영상 쪽**이다 — 어떤 특징을 골라야 하고, 그게 화면 어디쯤인지 사람이 찾아야 한다.

**어떤 점이 좋은 GCP인가 (선정 기준)**
  1) **모서리형이어야 한다.** 차선 중간의 한 점은 선을 따라 미끄러져 위치가 확정되지 않는다
     (조리개 문제). 정지선 끝점, 정지선×차선 교차점, 횡단보도 모서리처럼 두 방향이 구속되는
     점만 쓴다.
  2) **지면 위여야 한다.** 신호등·표지판은 공중에 있어 지면 호모그래피의 GCP가 될 수 없다.
     C1_TRAFFICLIGHT 를 후보에서 제외하는 이유다.
  3) **화면에 고르게 퍼져야 한다.** 한 곳에 몰리면 잔차는 0이 나오지만 바깥에서 크게 틀린다
     (TECHNICAL_DOCS §8.3 '잔차 0의 함정'). calibrate.html 과 같은 3x3 구역을 기준으로
     구역당 1~2점씩 고른다.

**현재 캘리브레이션이 나빠도 쓸 수 있다**
  예측 픽셀 위치는 지금 캘리브레이션으로 투영한 것이라 어긋나 있다. 그래도 "이 근처에
  정지선 끝점이 있다"는 안내로는 충분하다. 2~3점을 제대로 찍고 저장한 뒤 이 도구를 다시
  돌리면 예측이 좋아진다 — 반복하면 수렴한다.

사용:
  python tools/gcp_candidates.py --loc PANGYO_2
  python tools/gcp_candidates.py --loc PANGYO_2 --n 12 --radius 250
출력:
  location/<LOC>/gcp_candidates.json      (calibrate.html 이 읽는다)
  output/gcp/<LOC>/candidates.png         (번호 표시 + 확대 크롭)
"""
import argparse
import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from trafficlab.projection.g_projection import GProjection

HDMAP = os.path.join(REPO, "webmap", "public", "data", "hdmap", "hdmap.geojson")
# 지면 위 + 모서리형 특징만. C1_TRAFFICLIGHT(공중)·A2_LINK(차로 중심선, 눈에 안 보임) 제외.
USE_LAYERS = ("B2_SURFACELINEMARK", "B3_SURFACEMARK")
KIND_SCORE = {"교차점": 3.0, "폴리곤 모서리": 2.0, "선 끝점": 1.0}
SKY_FRAC = 0.15          # 상단 15%는 하늘/원경 — calibrate.html 의 zoneOf 와 같은 규약


def load_features(world, radius):
    """HD맵 → 로컬 미터 좌표의 선/폴리곤. 카메라 원점 반경 안만."""
    from pyproj import Transformer
    if not os.path.exists(HDMAP):
        sys.exit(f"HD맵 GeoJSON 없음: {HDMAP}\n  → python tools/export_hdmap_snap.py")
    tr = Transformer.from_crs(4326, world["epsg"], always_xy=True)
    oE, oN = world["origin_easting"], world["origin_northing"]
    lines, polys = [], []
    with open(HDMAP, encoding="utf-8") as f:
        fc = json.load(f)
    for feat in fc.get("features", []):
        lyr = feat.get("properties", {}).get("layer")
        if lyr not in USE_LAYERS:
            continue
        g = feat["geometry"]
        rings = []
        if g["type"] == "LineString":
            rings = [(g["coordinates"], "line")]
        elif g["type"] == "MultiLineString":
            rings = [(c, "line") for c in g["coordinates"]]
        elif g["type"] == "Polygon":
            rings = [(c, "poly") for c in g["coordinates"]]
        elif g["type"] == "MultiPolygon":
            rings = [(c, "poly") for poly in g["coordinates"] for c in poly]
        for coords, kind in rings:
            pts = []
            for lon, lat, *_ in coords:
                e, n = tr.transform(lon, lat)
                pts.append((e - oE, n - oN))
            if len(pts) < 2:
                continue
            if min(math.hypot(*q) for q in pts) > radius:
                continue
            (lines if kind == "line" else polys).append(np.asarray(pts, float))
    return lines, polys


def seg_intersection(a1, a2, b1, b2):
    """두 선분의 교점(선분 내부일 때만). 정지선×차선 = 가장 좋은 GCP."""
    d1, d2 = a2 - a1, b2 - b1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / den
    u = ((b1[0] - a1[0]) * d1[1] - (b1[1] - a1[1]) * d1[0]) / den
    if not (0.02 <= t <= 0.98 and 0.02 <= u <= 0.98):
        return None
    # 너무 얕은 각도로 만나면 교점이 미끄러진다 — 30도 이상만
    c = abs(float(d1 @ d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9))
    if c > math.cos(math.radians(30)):
        return None
    return a1 + t * d1


def extract_corners(lines, polys, max_pairs=4000):
    """모서리형 후보만 뽑는다. (점, 종류, 부모 지오메트리).

    부모 지오메트리를 함께 돌려주는 이유: 점만 표시하면 캘리브레이션 오차만큼 어긋난
    아스팔트 위 빈 곳을 가리키게 되어 조작자가 무엇을 클릭할지 알 수 없다. 그 점이 속한
    **노면표시 전체를 같이 그려야** '예측된 이 선 = 실제 저 선'으로 대응시킬 수 있다.
    """
    out = []
    for pts in lines:
        out.append((pts[0], "선 끝점", pts))
        out.append((pts[-1], "선 끝점", pts))
    for pts in polys:
        n = len(pts)
        for i in range(n):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
            v1, v2 = a - b, c - b
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 0.3 or n2 < 0.3:
                continue
            cos = float(v1 @ v2) / (n1 * n2)
            if cos > math.cos(math.radians(150)):        # 150도보다 꺾이면 모서리
                out.append((b, "폴리곤 모서리", pts))
    # 선-선 교차점 (인접한 것끼리만 — O(n²) 폭발 방지)
    segs = []
    for pts in lines:
        for i in range(len(pts) - 1):
            if np.linalg.norm(pts[i + 1] - pts[i]) > 0.5:
                segs.append((pts[i], pts[i + 1]))
    if len(segs) <= max_pairs:
        from scipy.spatial import cKDTree
        mids = np.array([(a + b) / 2 for a, b in segs])
        tree = cKDTree(mids)
        for i, (a1, a2) in enumerate(segs):
            for j in tree.query_ball_point(mids[i], 12.0):
                if j <= i:
                    continue
                p = seg_intersection(a1, a2, segs[j][0], segs[j][1])
                if p is not None:
                    out.append((p, "교차점", np.array([a1, a2, segs[j][0], segs[j][1]])))
    return out


def select_spread(cands, W, H, n_target):
    """calibrate.html 과 같은 3x3 구역에 고르게 배분 + 구역 내에서는 점수·거리 우선."""
    def zone(px):
        r = min(2, max(0, int((px[1] / H - SKY_FRAC) / 0.30)))
        c = min(2, max(0, int(px[0] / W * 3)))
        return (r, c)

    by_zone = {}
    for c in cands:
        by_zone.setdefault(zone(c["px"]), []).append(c)
    for v in by_zone.values():
        v.sort(key=lambda c: -c["score"])

    picked, guard = [], 0
    while len(picked) < n_target and guard < n_target * 4:
        guard += 1
        # 가장 적게 뽑힌 구역부터 채운다
        cnt = {z: sum(1 for p in picked if p["zone"] == list(z)) for z in by_zone}
        order = sorted(by_zone, key=lambda z: (cnt[z], -len(by_zone[z])))
        added = False
        for z in order:
            for c in by_zone[z]:
                if c in picked:
                    continue
                # 이미 뽑은 점과 화면상 너무 가까우면 건너뛴다(중복 정보)
                if any(math.hypot(c["px"][0] - p["px"][0], c["px"][1] - p["px"][1]) < W * 0.12
                       for p in picked):
                    continue
                picked.append(c)
                added = True
                break
            if added:
                break
        if not added:
            break
    return picked


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--n", type=int, default=10, help="추천 개수(호모그래피 자유도 8 → 6~10 권장)")
    ap.add_argument("--radius", type=float, default=200.0, help="카메라 원점 반경(m)")
    args = ap.parse_args()

    gp = os.path.join(REPO, "location", args.loc, f"G_projection_{args.loc}.json")
    if not os.path.exists(gp):
        sys.exit(f"캘리브레이션 파일 없음: {gp}")
    gd = json.load(open(gp, encoding="utf-8"))
    g = GProjection(gd, base_dir=os.path.dirname(gp))
    world = gd["world"]
    W, H = gd.get("undistort", {}).get("resolution", [720, 480])

    lines, polys = load_features(world, args.radius)
    print(f"[{args.loc}] HD맵 반경 {args.radius:.0f}m: 선 {len(lines)} · 폴리곤 {len(polys)}")
    if not lines and not polys:
        sys.exit("반경 안에 노면표시가 없습니다. --radius 를 키우거나 HD맵 커버리지를 확인하세요.")

    corners = extract_corners(lines, polys)
    print(f"모서리형 특징 {len(corners)}개 추출")

    # 영상으로 순투영 (역투영과 달리 어디서나 안정적)
    from pyproj import Transformer
    to_ll = Transformer.from_crs(world["epsg"], 4326, always_xy=True)
    oE, oN = world["origin_easting"], world["origin_northing"]
    def proj(pt):
        try:
            u, v = g.sat_to_cctv(float(pt[0]), float(pt[1]), h=0.0)
        except Exception:
            return None
        return (float(u), float(v)) if (np.isfinite(u) and np.isfinite(v)) else None

    cands, n_out = [], 0
    for p, kind, parent in corners:
        q = proj(p)
        if q is None:
            continue
        u, v = q
        if not (W * 0.03 <= u <= W * 0.97 and H * SKY_FRAC <= v <= H * 0.97):
            n_out += 1
            continue
        lon, lat = to_ll.transform(p[0] + oE, p[1] + oN)
        rng = float(math.hypot(*p))
        # 가까울수록 픽셀 정밀도가 좋다 → 거리에 약한 가점
        plen = float(np.sum(np.linalg.norm(np.diff(parent, axis=0), axis=1))) if len(parent) > 1 else 0.0
        # 긴 선(차선·정지선)의 끝점은 화면에서 바로 알아볼 수 있어 짧은 조각보다 훨씬 쓸모 있다
        score = KIND_SCORE[kind] + max(0.0, 1.0 - rng / args.radius) + min(1.5, plen / 8.0)
        par = [list(np.round(z, 1)) for z in
               (np.array([proj(x) for x in parent[::max(1, len(parent) // 24)]
                          if proj(x) is not None]) if len(parent) else np.empty((0, 2)))]
        cands.append({"kind": kind, "parent_px": par,
                      "px": [round(float(u), 1), round(float(v), 1)],
                      "lonlat": [round(float(lon), 8), round(float(lat), 8)],
                      "range_m": round(rng, 1), "score": round(score, 3)})
    print(f"화면 안 후보 {len(cands)}개 (화면 밖·하늘 영역 제외 {n_out})")
    if not cands:
        sys.exit("화면 안에 후보가 없습니다. 현재 캘리브레이션이 크게 어긋났을 수 있습니다 — "
                 "🤖 자동 캘리브레이션으로 부트스트랩한 뒤 다시 시도하세요.")

    for c in cands:
        r = min(2, max(0, int((c["px"][1] / H - SKY_FRAC) / 0.30)))
        col = min(2, max(0, int(c["px"][0] / W * 3)))
        c["zone"] = [r, col]
    picked = select_spread(cands, W, H, args.n)
    for i, c in enumerate(picked):
        c["no"] = i + 1

    zones = {tuple(c["zone"]) for c in picked}
    print(f"\n추천 {len(picked)}점 · 커버 구역 {len(zones)}/9")
    print(f"{'#':>2s} {'종류':>12s} {'영상 px':>14s} {'구역':>6s} {'거리':>7s}  지도 좌표")
    for c in picked:
        print(f"{c['no']:2d} {c['kind']:>12s} {str(c['px']):>14s} {str(c['zone']):>6s} "
              f"{c['range_m']:6.0f}m  {c['lonlat'][0]:.6f}, {c['lonlat'][1]:.6f}")
    if len(zones) < 5:
        print(f"\n  ⚠ 커버 구역이 {len(zones)}개뿐입니다. GCP가 몰리면 잔차는 0이 나와도 "
              f"바깥에서 크게 틀립니다 — --radius 를 키워 더 넓게 후보를 잡으세요.")

    # --- 저장 ---
    payload = {"loc": args.loc, "resolution": [W, H], "radius_m": args.radius,
               "note": "px 는 현재 캘리브레이션으로 투영한 **예상 위치**입니다. 실제 특징을 "
                       "보고 클릭한 뒤, 몇 점을 저장하고 이 도구를 다시 돌리면 예측이 좋아집니다.",
               "candidates": picked}
    out_json = os.path.join(REPO, "location", args.loc, "gcp_candidates.json")
    json.dump(payload, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # calibrate.html 이 정적으로 fetch 할 수 있도록 webmap/public 에도 둔다(서버 변경 불필요)
    web = os.path.join(REPO, "webmap", "public", "data", "gcp")
    os.makedirs(web, exist_ok=True)
    json.dump(payload, open(os.path.join(web, f"{args.loc}.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    # --- 확인용 이미지: 전체 + 후보별 확대 크롭 ---
    snap = gd.get("inputs", {}).get("cctv_path")
    img = None
    if snap:
        p = os.path.join(os.path.dirname(gp), snap)
        if os.path.exists(p):
            img = cv2.imread(p)
    if img is None:
        v = os.path.join(REPO, "webmap", "public", "data", "footage", f"{args.loc.lower()}.mp4")
        if os.path.exists(v):
            cap = cv2.VideoCapture(v); ok, img = cap.read(); cap.release()
            if not ok:
                img = None
    out_dir = os.path.join(REPO, "output", "gcp", args.loc)
    os.makedirs(out_dir, exist_ok=True)
    if img is not None:
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))
        vis = img.copy()
        for r in range(3):
            y = int(H * (SKY_FRAC + 0.30 * r))
            cv2.line(vis, (0, y), (W, y), (70, 70, 70), 1)
        for c in range(1, 3):
            cv2.line(vis, (int(W * c / 3), 0), (int(W * c / 3), H), (70, 70, 70), 1)
        # ① 컨텍스트: 화면에 걸리는 **모든** 노면표시를 흐리게 깔아 준다.
        # 캘리브레이션이 크게 어긋나면 점 하나로는 대응을 못 찾는다. 선 무리 전체를 보여 주면
        # "예측된 이 4줄 = 실제 저 4줄"로 거칠게 맞춘 뒤, 2~3점을 찍어 정밀화할 수 있다.
        for grp in (lines, polys):
            for pts in grp:
                q = [proj(x) for x in pts[::max(1, len(pts) // 30)]]
                q = np.array([z for z in q if z is not None], float)
                if len(q) >= 2:
                    cv2.polylines(vis, [q.astype(np.int32)], grp is polys, (110, 90, 40), 1, cv2.LINE_AA)

        # ② 선택된 후보의 부모 노면표시를 진하게 — 점만 있으면 아스팔트 위 빈 곳을 가리켜
        # 무엇을 클릭할지 알 수 없다. 선 모양이 있어야 실제 표시와 대응시킬 수 있다.
        for c in picked:
            par = np.asarray(c.get("parent_px") or [], float)
            if len(par) >= 2:
                cv2.polylines(vis, [par.astype(np.int32)], False, (255, 120, 0), 2, cv2.LINE_AA)
        tiles = []
        for c in picked:
            x, y = int(c["px"][0]), int(c["px"][1])
            cv2.drawMarker(vis, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.circle(vis, (x, y), 12, (0, 200, 255), 1)
            cv2.putText(vis, str(c["no"]), (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            s = 44
            x0, y0 = max(0, x - s), max(0, y - s)
            crop = img[y0:y + s, x0:x + s].copy()
            if crop.size:
                sc = 170.0 / max(crop.shape[0], crop.shape[1])
                crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
                par = np.asarray(c.get("parent_px") or [], float)
                if len(par) >= 2:
                    q = ((par - np.array([x0, y0])) * sc).astype(np.int32)
                    cv2.polylines(crop, [q], False, (255, 120, 0), 2, cv2.LINE_AA)
                cx, cy = int((x - x0) * sc), int((y - y0) * sc)
                cv2.drawMarker(crop, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
                crop = cv2.copyMakeBorder(crop, 0, max(0, 170 - crop.shape[0]),
                                          0, max(0, 170 - crop.shape[1]),
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))[:170, :170]
                cv2.putText(crop, f"{c['no']} {c['kind']}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                tiles.append(crop)
        rows = [np.hstack(tiles[i:i + 5] + [np.zeros_like(tiles[0])] * (5 - len(tiles[i:i + 5])))
                for i in range(0, len(tiles), 5)] if tiles else []
        panel = np.vstack(rows) if rows else None
        if panel is not None:
            if panel.shape[1] != vis.shape[1]:
                panel = cv2.resize(panel, (vis.shape[1], int(panel.shape[0] * vis.shape[1] / panel.shape[1])))
            vis = np.vstack([vis, panel])
        cv2.imwrite(os.path.join(out_dir, "candidates.png"), vis)
        print(f"\n→ {os.path.relpath(out_dir, REPO)}/candidates.png")
    print(f"→ {os.path.relpath(out_json, REPO)}")
    print(f"\ncalibrate.html 에서 '🧲 HD맵 GCP 후보'를 켜면 번호대로 안내됩니다.")


if __name__ == "__main__":
    main()
