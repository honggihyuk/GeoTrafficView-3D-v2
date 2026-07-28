"""
HD맵 ↔ 위성 베이스맵 정합 — Chamfer/거리변환(DT) 매칭.

**무엇을 맞추나**
  정밀도로지도는 측량 성과(cm급)이고, Esri World Imagery(Maxar)는 자체 지오레퍼런싱
  편위가 수 m 있다. 지도 위에 HD맵을 얹으면 노면표시가 영상과 어긋나 보인다.
  이 도구는 그 **어긋난 양을 측정**한다.

**왜 Chamfer/DT인가**
  대응점을 사람이 클릭하는 대신, 영상에서 뽑은 페인트 마스크의 **거리변환**을 한 번
  계산해 두고, HD맵 노면표시 정점을 워프해 가며 DT 값의 합을 최소화한다.
  정점당 조회가 O(1)이라 수만 정점 × 수천 후보를 초 단위로 훑는다.
  위상 상관(FFT)보다 느리지만, 그림자·차량·건물 때문에 페인트 추출이 지저분해도
  안정적이다(선형 피처 정합의 사실상 표준).

**무엇을 쓰나 (중요)**
  A2_LINK는 **차로 중심선**이라 그 자리에 페인트가 없다. 정합에는 실제로 칠해진
  B2_SURFACELINEMARK(차선·정지선) + B3_SURFACEMARK(횡단보도·화살표)만 쓴다.
  횡단보도·정지선은 차선의 3.5m 주기성을 깨 주므로 국소최소값 탈출에도 기여한다.

**결과를 어디에 쓰나 (더 중요)**
  HD맵은 **더 정확한 쪽**이다. 측량 성과를 상용 위성 모자이크에 맞춰 휘면
  절대정확도가 나빠진다. 그래서 이 도구는 hdmap.geojson 을 **건드리지 않고**
  hdmap_align.json 만 쓴다. 표시(렌더링) 단계에서만 적용하고, lane_snap·DB·측정에는
  절대 적용하지 말 것.
  잔차가 크면 우선 베이스맵 교체(VWorld 정사영상 등)를 검토하는 게 맞다.

사용:
  python tools/hdmap_align.py                        # HD맵 전역, z18
  python tools/hdmap_align.py --zoom 19 --local 4    # 고해상 + 4x4 국소 변위장
출력:
  webmap/public/data/hdmap/hdmap_align.json
  output/hdmap_align/qc_overlay.png · qc_mask.png    (육안 검증용)
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import urllib.request
import warnings

warnings.filterwarnings("ignore")
import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TILE = 256
R_EARTH = 6378137.0
ORIGIN = math.pi * R_EARTH            # 20037508.342789244
ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}")
UA = {"User-Agent": "GeoTrafficView-3D/2.1 (HD map alignment QC)"}

PAINT_LAYERS = ("B2_SURFACELINEMARK", "B3_SURFACEMARK")


# ---------- Web Mercator ----------
def ll_to_merc(lon, lat):
    x = np.radians(lon) * R_EARTH
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R_EARTH
    return x, y


def merc_to_ll(x, y):
    lon = np.degrees(x / R_EARTH)
    lat = np.degrees(2 * np.arctan(np.exp(y / R_EARTH)) - np.pi / 2)
    return lon, lat


def merc_res(z):
    """줌 z에서 픽셀당 머케이터 미터."""
    return 2 * ORIGIN / (TILE * 2 ** z)


def tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lr = math.radians(lat)
    y = (1.0 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2.0 * n
    return x, y


# ---------- 타일 모자이크 ----------
def fetch_tile(z, x, y, cache_dir):
    p = os.path.join(cache_dir, str(z), str(x), f"{y}.jpg")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    os.makedirs(os.path.dirname(p), exist_ok=True)
    url = ESRI.format(z=z, x=x, y=y)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            b = r.read()
        if len(b) < 100:
            return None
        with open(p, "wb") as f:
            f.write(b)
        return p
    except Exception:
        return None


def build_mosaic(bbox, z, cache_dir, max_tiles=1200):
    """bbox=(minlon,minlat,maxlon,maxlat) → (BGR 모자이크, x0_merc, y0_merc, res)."""
    x0f, y0f = tile_xy(bbox[0], bbox[3], z)      # 좌상
    x1f, y1f = tile_xy(bbox[2], bbox[1], z)      # 우하
    tx0, ty0 = int(math.floor(x0f)), int(math.floor(y0f))
    tx1, ty1 = int(math.floor(x1f)), int(math.floor(y1f))
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    n = nx * ny
    if n > max_tiles:
        sys.exit(f"타일 {n}개는 과다합니다(상한 {max_tiles}). --zoom 을 낮추거나 --bbox 로 좁히세요.")
    print(f"타일 {nx}x{ny}={n}개 (z{z}) 수집 중...")

    jobs = [(tx0 + i, ty0 + j) for j in range(ny) for i in range(nx)]
    paths = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_tile, z, x, y, cache_dir): (x, y) for x, y in jobs}
        for fu in cf.as_completed(futs):
            paths[futs[fu]] = fu.result()

    mosaic = np.zeros((ny * TILE, nx * TILE, 3), np.uint8)
    n_ok = 0
    for (x, y), p in paths.items():
        if not p:
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None or img.shape[0] != TILE or img.shape[1] != TILE:
            continue
        i, j = x - tx0, y - ty0
        mosaic[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE] = img
        n_ok += 1
    print(f"  타일 {n_ok}/{n} 확보 · 모자이크 {mosaic.shape[1]}x{mosaic.shape[0]}px")
    if n_ok < n * 0.6:
        sys.exit("타일 확보율이 낮습니다. 네트워크 또는 줌 레벨을 확인하세요.")

    res = merc_res(z)
    x0m = tx0 * TILE * res - ORIGIN
    y0m = ORIGIN - ty0 * TILE * res
    return mosaic, x0m, y0m, res


# ---------- 페인트 마스크 & 거리변환 ----------
def road_corridor(pts_px, shape, radius_px):
    """HD맵 노면표시 주변 회랑 마스크.

    AOI 전체에 임계를 걸면 공원·수면·건물 지붕이 페인트 예산을 다 먹는다(실측: 송도 AOI에서
    상위 2% 반응의 대부분이 식생이었고 목적함수가 평탄해져 최적화가 46m 밖으로 도망갔다).
    회랑은 '정답을 가정'하는 게 아니라 **탐색 반경 + 여유**로 잡는다.
    """
    m = np.zeros(shape, np.uint8)
    ok = ((pts_px[:, 0] >= 0) & (pts_px[:, 0] < shape[1])
          & (pts_px[:, 1] >= 0) & (pts_px[:, 1] < shape[0]))
    m[pts_px[ok, 1].astype(int), pts_px[ok, 0].astype(int)] = 1
    r = max(1, int(round(radius_px)))
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def paint_mask(mosaic, gm_per_px, corridor, tophat_m=3.0, keep_pct=8.0):
    """노면 페인트 후보 마스크. white top-hat = 배경보다 밝고 '가느다란' 구조만 남긴다.

    임계도, 결과도 회랑 안으로 제한한다 — 회랑 밖 페인트는 DT를 오염시킬 뿐이다.
    """
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    k = int(round(tophat_m / gm_per_px))
    k = max(3, k + 1 - k % 2)                                  # 홀수, 최소 3
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)
    valid = (gray > 0) & (corridor > 0)                        # 미확보 타일·회랑 밖 제외
    if valid.sum() < 1000:
        sys.exit("회랑 안 유효 픽셀이 너무 적습니다. --corridor 를 키우거나 AOI를 확인하세요.")
    # 페인트는 밝다 — 어두운 그림자에서 뜨는 top-hat 반응을 걸러낸다.
    sel = valid & (gray >= np.percentile(gray[valid], 40))
    thr = np.percentile(th[sel], 100.0 - keep_pct)
    mask = ((th >= max(thr, 6)) & sel).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask


def distance_field(mask):
    """각 픽셀 → 가장 가까운 페인트 픽셀까지의 거리(px)."""
    inv = np.where(mask > 0, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def sample_bilinear(field, pts, oob):
    """field(H,W) 를 pts(N,2 = x,y px)에서 이중선형 샘플. 범위 밖은 oob."""
    H, W = field.shape
    x, y = pts[:, 0], pts[:, 1]
    ok = (x >= 0) & (x <= W - 2) & (y >= 0) & (y <= H - 2)
    out = np.full(len(pts), float(oob))
    if not ok.any():
        return out
    xi, yi = x[ok], y[ok]
    x0 = np.floor(xi).astype(np.int32); y0 = np.floor(yi).astype(np.int32)
    fx, fy = xi - x0, yi - y0
    v = (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x0 + 1] * fx * (1 - fy)
         + field[y0 + 1, x0] * (1 - fx) * fy + field[y0 + 1, x0 + 1] * fx * fy)
    out[ok] = v
    return out


# ---------- HD맵 노면표시 → 픽셀 점군 ----------
def densify(pts, step):
    """폴리라인을 step 간격으로 리샘플."""
    out = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(d / step))
        for j in range(n):
            t = j / n
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    if pts:
        out.append(tuple(pts[-1]))
    return out


def hdmap_points(geojson_path, layers, step_m):
    """노면표시 정점을 머케이터 미터로. (densify는 지상 미터 기준이므로 위도 보정)"""
    with open(geojson_path, encoding="utf-8") as f:
        fc = json.load(f)
    rings = []
    for feat in fc.get("features", []):
        if feat.get("properties", {}).get("layer") not in layers:
            continue
        g = feat["geometry"]
        t, c = g["type"], g["coordinates"]
        if t == "LineString":
            rings.append(c)
        elif t == "MultiLineString":
            rings.extend(c)
        elif t == "Polygon":
            rings.extend(c)                       # 외곽선 + 내부 링
        elif t == "MultiPolygon":
            for poly in c:
                rings.extend(poly)
    if not rings:
        sys.exit(f"{'/'.join(layers)} 피처가 없습니다. export_hdmap_snap.py 를 먼저 실행하세요.")
    lat0 = np.mean([p[1] for r in rings for p in r])
    step_merc = step_m / math.cos(math.radians(lat0))
    pts = []
    for r in rings:
        mx, my = ll_to_merc(np.array([p[0] for p in r]), np.array([p[1] for p in r]))
        pts.extend(densify(list(zip(mx, my)), step_merc))
    return np.asarray(pts, float), float(lat0)


# ---------- 워프 ----------
def apply_similarity(pts, center, params):
    """params = (tx, ty, theta_rad, log_scale) — 모두 머케이터 미터/라디안."""
    tx, ty, th, ls = params
    s = math.exp(ls)
    c, sn = math.cos(th), math.sin(th)
    d = pts - center
    return np.stack([center[0] + s * (c * d[:, 0] - sn * d[:, 1]) + tx,
                     center[1] + s * (sn * d[:, 0] + c * d[:, 1]) + ty], axis=1)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", default=os.path.join(REPO, "webmap", "public", "data", "hdmap", "hdmap.geojson"))
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                    help="미지정 시 HD맵 노면표시 전체 범위")
    ap.add_argument("--loc", help="카메라 코드 — 그 주변 --radius 만 AOI로 (여러 지역이 섞인 "
                                  "hdmap.geojson에서 한 지역만 정합할 때)")
    ap.add_argument("--radius", type=float, default=600.0, help="--loc 사용 시 AOI 반경(m)")
    ap.add_argument("--zoom", type=int, default=18)
    ap.add_argument("--step", type=float, default=1.0, help="HD맵 폴리라인 리샘플 간격(m)")
    ap.add_argument("--max-points", type=int, default=40000)
    ap.add_argument("--search", type=float, default=25.0, help="거친 탐색 반경(지상 m)")
    ap.add_argument("--search-step", type=float, default=1.0, help="거친 탐색 격자 간격(지상 m)")
    ap.add_argument("--trunc", type=float, default=6.0, help="DT 절단 거리(지상 m) — 이상치 둔감화")
    ap.add_argument("--tophat", type=float, default=3.0, help="top-hat 커널 크기(지상 m)")
    ap.add_argument("--keep-pct", type=float, default=8.0,
                    help="회랑 안에서 페인트로 볼 상위 top-hat 반응 비율(%%)")
    ap.add_argument("--corridor", type=float, default=0.0,
                    help="페인트 탐색 회랑 반경(지상 m). 0이면 search+15m 자동")
    ap.add_argument("--local", type=int, default=0, help="N이면 NxN 격자로 국소 변위장까지 추정(러버시팅 대응점)")
    ap.add_argument("--no-refine", action="store_true", help="거친 격자 탐색만(회전·스케일 미추정)")
    args = ap.parse_args()

    if not os.path.exists(args.geojson):
        sys.exit(f"HD맵 GeoJSON 없음: {args.geojson}\n  → python tools/export_hdmap_snap.py")

    # --- HD맵 노면표시 점군 ---
    pts_m, lat0 = hdmap_points(args.geojson, PAINT_LAYERS, args.step)
    merc_to_ground = math.cos(math.radians(lat0))      # 머케이터 m → 지상 m

    # --loc: 여러 지역이 병합된 hdmap.geojson에서 해당 카메라 주변만 잘라낸다.
    if args.loc:
        gp = os.path.join(REPO, "location", args.loc, f"G_projection_{args.loc}.json")
        if not os.path.exists(gp):
            sys.exit(f"카메라 설정 없음: {gp}")
        from pyproj import Transformer
        w = json.load(open(gp, encoding="utf-8"))["world"]
        tr = Transformer.from_crs(w["epsg"], 4326, always_xy=True)
        clon, clat = tr.transform(w["origin_easting"], w["origin_northing"])
        cx, cy = ll_to_merc(clon, clat)
        rm = args.radius / merc_to_ground              # 지상 m → 머케이터 m
        keep = (np.abs(pts_m[:, 0] - cx) <= rm) & (np.abs(pts_m[:, 1] - cy) <= rm)
        if keep.sum() < 500:
            sys.exit(f"{args.loc} 반경 {args.radius:.0f}m 안 노면표시 정점이 {int(keep.sum())}개뿐입니다. "
                     f"--radius 를 키우거나 HD맵 커버리지를 확인하세요.")
        pts_m = pts_m[keep]
        print(f"AOI: {args.loc} 원점({clon:.5f}, {clat:.5f}) 반경 {args.radius:.0f}m")

    # --bbox 도 점군을 함께 잘라야 한다. 병합된 다지역 geojson에서 타일 AOI만 좁히면
    # 점군은 전국 범위로 남아 center/스케일 추정이 무의미해진다.
    if args.bbox:
        blon, blat = merc_to_ll(pts_m[:, 0], pts_m[:, 1])
        keep = ((blon >= args.bbox[0]) & (blon <= args.bbox[2])
                & (blat >= args.bbox[1]) & (blat <= args.bbox[3]))
        if keep.sum() < 500:
            sys.exit(f"--bbox 안 노면표시 정점이 {int(keep.sum())}개뿐입니다.")
        pts_m = pts_m[keep]

    if len(pts_m) > args.max_points:
        sel = np.linspace(0, len(pts_m) - 1, args.max_points).astype(int)
        pts_m = pts_m[sel]
    lon, lat = merc_to_ll(pts_m[:, 0], pts_m[:, 1])
    bbox = args.bbox or [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())]
    print(f"HD맵 노면표시 정점 {len(pts_m):,}개 ({'/'.join(PAINT_LAYERS)}) · "
          f"AOI {bbox[0]:.5f},{bbox[1]:.5f} ~ {bbox[2]:.5f},{bbox[3]:.5f}")

    # --- 위성 모자이크 ---
    cache = os.path.join(REPO, "output", "hdmap_align", "tiles")
    mosaic, x0m, y0m, res = build_mosaic(bbox, args.zoom, cache)
    gm_px = res * merc_to_ground                       # 픽셀당 지상 미터
    print(f"  해상도 {gm_px:.3f} m/px (지상 기준)")

    def to_px(p):
        return np.stack([(p[:, 0] - x0m) / res, (y0m - p[:, 1]) / res], axis=1)

    # --- 페인트 마스크 + DT ---
    corr_m = args.corridor if args.corridor > 0 else args.search + 15.0
    corridor = road_corridor(to_px(pts_m), mosaic.shape[:2], corr_m / gm_px)
    mask = paint_mask(mosaic, gm_px, corridor, args.tophat, args.keep_pct)
    n_paint, n_corr = int(mask.sum()), int((corridor > 0).sum())
    print(f"도로 회랑 {corr_m:.0f}m: {n_corr:,}px ({100 * n_corr / corridor.size:.1f}% of AOI)")
    print(f"페인트 마스크: {n_paint:,}px (회랑의 {100 * n_paint / max(n_corr,1):.1f}%) · 거리변환 계산")
    dt = distance_field(mask)

    trunc_px = args.trunc / gm_px
    center = pts_m.mean(axis=0)

    def residuals(params, sub=None):
        p = apply_similarity(pts_m if sub is None else pts_m[sub], center, params)
        return np.minimum(sample_bilinear(dt, to_px(p), trunc_px), trunc_px)

    def cost(params, sub=None):
        return float(residuals(params, sub).mean())

    def report(params, label):
        """평균만 보면 최소값이 진짜인지 알 수 없다. 중앙값과 인라이어율을 함께 본다."""
        r = residuals(params) * gm_px
        return (f"{label}: 평균 {r.mean():.2f} m · 중앙값 {np.median(r):.2f} m · "
                f"1m 이내 {100 * (r < 1.0).mean():.1f}% · 2m 이내 {100 * (r < 2.0).mean():.1f}%")

    base = np.array([0.0, 0.0, 0.0, 0.0])
    c0 = cost(base)
    print("\n" + report(base, "정합 전 잔차"))

    # --- ① 거친 격자 탐색 (평행이동) ---
    # 차선은 3.5m 주기라 국소최소값이 규칙적으로 생긴다. 전역 격자로 훑어 이를 피한다.
    ncoarse = min(len(pts_m), 8000)
    sub = np.linspace(0, len(pts_m) - 1, ncoarse).astype(int)
    rng = np.arange(-args.search, args.search + 1e-9, args.search_step) / merc_to_ground
    surf = np.empty((len(rng), len(rng)))
    best, bc = base.copy(), None
    for a, tx in enumerate(rng):
        for b, ty in enumerate(rng):
            c = cost(np.array([tx, ty, 0.0, 0.0]), sub)
            surf[b, a] = c                       # 행=ty(북), 열=tx(동)
            if bc is None or c < bc:
                best, bc = np.array([tx, ty, 0.0, 0.0]), c
    gx, gy = best[0] * merc_to_ground, best[1] * merc_to_ground
    print(f"① 격자 탐색({len(rng)}x{len(rng)} = {len(rng)**2:,}회): "
          f"이동 ({gx:+.2f}, {gy:+.2f}) m · 잔차 {bc*gm_px:.2f} m")
    # 최소값이 '진짜'인지: 곡면 대비(contrast) = (중앙값 - 최소값) / 중앙값.
    # 평탄한 목적함수에서는 이 값이 0에 수렴하고, 그때 추정된 편위는 노이즈다.
    contrast = float((np.median(surf) - surf.min()) / max(np.median(surf), 1e-9))
    print(f"  비용 곡면 대비 {100 * contrast:.1f}%  (최소 {surf.min()*gm_px:.2f} m / "
          f"중앙값 {np.median(surf)*gm_px:.2f} m / 최대 {surf.max()*gm_px:.2f} m)")
    if max(abs(gx), abs(gy)) > args.search - args.search_step:
        print(f"  ⚠ 최적값이 탐색 경계({args.search:.0f}m)에 붙었습니다. 진짜 편위가 더 크거나 "
              f"페인트 마스크가 부실해 목적함수가 평탄한 것입니다 — qc_mask.png 를 먼저 확인하세요.")
    if contrast < 0.05:
        print(f"  ⚠ 곡면이 평탄합니다(대비 {100*contrast:.1f}% < 5%). 추정된 편위를 신뢰하지 마세요. "
              f"--zoom 을 올리거나 그림자·고층부 영향이 적은 구간으로 --bbox 를 좁히세요.")

    # 비등방성(조리개 문제): 직선 도로만 있으면 종방향이 구속되지 않아 곡면이 길쭉해진다.
    # 저비용 영역의 2차 모멘트로 주축을 뽑아 '어느 방향 추정을 믿을 수 있는지' 알린다.
    lowc = surf <= surf.min() + 0.2 * (np.median(surf) - surf.min())
    gy_, gx_ = np.nonzero(lowc)
    aniso = {"ratio": 1.0, "weak_axis_deg": None}
    if len(gx_) >= 6:
        P = np.stack([rng[gx_] * merc_to_ground, rng[gy_] * merc_to_ground])
        ev, evec = np.linalg.eigh(np.cov(P))
        ratio = float(math.sqrt(max(ev[1], 1e-12) / max(ev[0], 1e-12)))
        wk = evec[:, 1]                                  # 분산이 큰 축 = 약하게 구속된 방향
        wdeg = (math.degrees(math.atan2(wk[0], wk[1])) + 360) % 180   # 북 기준 방위(0~180)
        aniso = {"ratio": round(ratio, 2), "weak_axis_deg": round(wdeg, 1)}
        if ratio > 2.0:
            print(f"  ⚠ 곡면이 {ratio:.1f}:1로 길쭉합니다(약축 방위 {wdeg:.0f}°). "
                  f"직선 도로만 있어 **종방향(도로 진행 방향) 편위는 구속되지 않습니다**. "
                  f"횡방향 성분만 신뢰하고, 정지선·횡단보도가 있는 교차로를 AOI에 포함시키세요.")

    # --- ② Powell 정밀화 (평행이동 + 회전 + 스케일) ---
    if args.no_refine:
        fin = best
    else:
        from scipy.optimize import minimize
        # 파라미터 스케일을 맞춘다: 회전 1e-4 rad ≈ 0.15m/1.5km, 스케일 1e-5 ≈ 1.5cm/1.5km
        scale = np.array([1.0, 1.0, 1e-4, 1e-5])
        # 경계를 건다 — 평탄한 목적함수에서 Powell이 무한정 밀려나가는 것을 막는다.
        lim = (args.search + 2 * args.search_step) / merc_to_ground
        bnds = [(-lim / scale[0], lim / scale[0]), (-lim / scale[1], lim / scale[1]),
                (math.radians(-2) / scale[2], math.radians(2) / scale[2]),
                (math.log(0.99) / scale[3], math.log(1.01) / scale[3])]
        r = minimize(lambda q: cost(q * scale), best / scale, method="Powell", bounds=bnds,
                     options={"xtol": 1e-2, "ftol": 1e-4, "maxiter": 8000})
        fin = r.x * scale
        # 경계에 붙었다 = 목적함수가 그 방향으로 구속되지 않는다는 뜻이다. 이때의 회전·
        # 스케일은 물리적 편위가 아니라 발산이므로 채택하면 안 된다(실측: YEONSU_JCT에서
        # 회전 +1.99°/스케일 0.990 으로 양쪽 경계에 동시에 붙어 18m 이동이 나왔다).
        # Powell은 경계에 '정확히' 멈추지 않으므로 구간 폭의 1% 이내면 붙은 것으로 본다.
        at_bound = any(min(abs(v - lo), abs(v - hi)) < 0.01 * (hi - lo)
                       for v, (lo, hi) in zip(r.x, bnds))
        if at_bound:
            print("  ⚠ Powell이 파라미터 경계에 도달했습니다 — 목적함수가 구속되지 않는 방향이 "
                  "있다는 뜻이라 정밀화를 기각하고 평행이동 결과만 씁니다.")
            fin = best
        elif cost(fin) > bc:                            # 정밀화가 악화시키면 되돌린다
            fin = best
    c1 = cost(fin)
    dx, dy = fin[0] * merc_to_ground, fin[1] * merc_to_ground
    print(f"② Powell 정밀화: 이동 ({dx:+.2f}, {dy:+.2f}) m · 회전 {math.degrees(fin[2]):+.4f}° · "
          f"스케일 {math.exp(fin[3]):.6f}")
    print("\n" + report(fin, "정합 후 잔차"))
    r0, r1 = residuals(base) * gm_px, residuals(fin) * gm_px
    print(f"1m 인라이어 {100*(r0<1).mean():.1f}% → {100*(r1<1).mean():.1f}% "
          f"({100*((r1<1).mean()-(r0<1).mean()):+.1f}%p)")
    print(f"총 편위량: {math.hypot(dx, dy):.2f} m  방위 {(math.degrees(math.atan2(dx, dy)) + 360) % 360:.0f}°")

    # 회전·스케일이 의미 있는 값인지: AOI 가장자리에서 만드는 변위가 잔차보다 작으면 노이즈다.
    radius = float(np.linalg.norm(pts_m - center, axis=1).max()) * merc_to_ground
    eff_rot = radius * abs(fin[2])
    eff_scl = radius * abs(math.exp(fin[3]) - 1)
    rs_meaningful = max(eff_rot, eff_scl) > c1 * gm_px
    print(f"  회전·스케일 유효성: AOI 가장자리({radius:.0f}m)에서 회전 {eff_rot:.2f} m · "
          f"스케일 {eff_scl:.2f} m vs 잔차 {c1*gm_px:.2f} m → "
          f"{'유의' if rs_meaningful else '노이즈 수준 — 평행이동만 쓰는 편이 안전'}")

    # --- ③ 국소 변위장 (러버시팅 대응점) ---
    controls = []
    if args.local > 0:
        g = args.local
        warped = apply_similarity(pts_m, center, fin)
        xs = np.linspace(pts_m[:, 0].min(), pts_m[:, 0].max(), g + 1)
        ys = np.linspace(pts_m[:, 1].min(), pts_m[:, 1].max(), g + 1)
        LR = 8.0
        lrng = np.arange(-LR, LR + 1e-9, 0.5) / merc_to_ground
        n_edge, n_thin = 0, 0
        print(f"\n③ 국소 변위장 {g}x{g} (전역 정합 위에 ±{LR:.0f}m 재탐색)")
        for i in range(g):
            for j in range(g):
                m = ((warped[:, 0] >= xs[i]) & (warped[:, 0] < xs[i + 1])
                     & (warped[:, 1] >= ys[j]) & (warped[:, 1] < ys[j + 1]))
                if m.sum() < 300:
                    n_thin += 1
                    continue
                cell = warped[m]
                lsurf = np.empty((len(lrng), len(lrng)))
                bt, bcst = (0.0, 0.0), None
                for a, tx in enumerate(lrng):
                    for b, ty in enumerate(lrng):
                        d = sample_bilinear(dt, to_px(cell + np.array([tx, ty])), trunc_px)
                        c = float(np.minimum(d, trunc_px).mean())
                        lsurf[b, a] = c
                        if bcst is None or c < bcst:
                            bt, bcst = (float(tx), float(ty)), c
                og = (bt[0] * merc_to_ground, bt[1] * merc_to_ground)
                lcon = float((np.median(lsurf) - lsurf.min()) / max(np.median(lsurf), 1e-9))
                # 경계에 붙었거나 곡면이 평탄한 셀은 버린다 — 러버시팅 대응점으로 쓰면 지도가 찢어진다.
                if max(abs(og[0]), abs(og[1])) > LR - 0.5 or lcon < 0.05:
                    n_edge += 1
                    continue
                src = [float(cell[:, 0].mean()), float(cell[:, 1].mean())]
                controls.append({"src_merc": src,
                                 "dst_merc": [src[0] + bt[0], src[1] + bt[1]],
                                 "offset_ground_m": [round(og[0], 3), round(og[1], 3)],
                                 "n_points": int(m.sum()), "residual_m": round(bcst * gm_px, 3),
                                 "contrast": round(lcon, 4)})
        print(f"  셀 {g*g}개 중 채택 {len(controls)} · 정점 부족 {n_thin} · 경계/평탄 기각 {n_edge}")
        if controls:
            off = np.array([c["offset_ground_m"] for c in controls])
            mag = np.hypot(off[:, 0], off[:, 1])
            print(f"  잔여 국소 변위 중앙값 {np.median(mag):.2f} m / 최대 {mag.max():.2f} m")
            if mag.max() < 1.0:
                print("  → 국소 왜곡이 1m 미만입니다. 전역 similarity 하나로 충분하고, "
                      "러버시팅(TPS/구간별 어파인)은 불필요합니다.")
        if len(controls) < 3:
            print("  → 유효 대응점이 3개 미만이라 러버시팅은 불가능합니다(전역 정합만 사용).")

    # --- 저장 ---
    out = {
        "method": "chamfer_dt_similarity",
        "note": "표시(렌더링) 전용 워프. hdmap.geojson·lane_snap·DB 에는 적용하지 말 것 "
                "— HD맵이 위성영상보다 정확하다.",
        "crs": "EPSG:3857",
        "center_merc": [float(center[0]), float(center[1])],
        "translation_merc": [float(fin[0]), float(fin[1])],
        "rotation_deg": float(math.degrees(fin[2])),
        "scale": float(math.exp(fin[3])),
        "apply": "p' = center + scale*R(rotation)*(p - center) + translation   (EPSG:3857, meters)",
        "shift_ground_m": {"east": round(dx, 3), "north": round(dy, 3),
                           "magnitude": round(math.hypot(dx, dy), 3)},
        "residual_ground_m": {
            "mean_before": round(c0 * gm_px, 3), "mean_after": round(c1 * gm_px, 3),
            "median_before": round(float(np.median(r0)), 3), "median_after": round(float(np.median(r1)), 3),
            "inlier_1m_before": round(float((r0 < 1).mean()), 4), "inlier_1m_after": round(float((r1 < 1).mean()), 4),
            "trunc_m": args.trunc},
        "confidence": {"cost_surface_contrast": round(contrast, 4),
                       "verdict": ("신뢰 가능" if (contrast >= 0.05 and aniso["ratio"] <= 3.0)
                                   else ("평탄 — 신뢰 불가" if contrast < 0.05
                                         else f"비등방 {aniso['ratio']:.1f}:1 — 횡방향 성분만 신뢰")),
                       "anisotropy": aniso,
                       "rotation_scale_meaningful": bool(rs_meaningful),
                       "rotation_scale_hint": ("회전·스케일 유효" if rs_meaningful else
                                               "회전·스케일은 잔차 이하 — translation_merc 만 적용 권장")},
        "imagery": {"source": "Esri World Imagery (Esri, Maxar, Earthstar Geographics, CNES/Airbus DS)",
                    "zoom": args.zoom, "ground_m_per_px": round(gm_px, 4)},
        "hdmap": {"layers": list(PAINT_LAYERS), "n_points": int(len(pts_m)), "step_m": args.step},
        "aoi_bbox_wgs84": [round(v, 6) for v in bbox],
        "local_controls": controls,
    }
    out["aoi"] = {"loc": args.loc, "radius_m": args.radius if args.loc else None}
    tag = f"_{args.loc}" if args.loc else ""
    outp = os.path.join(REPO, "webmap", "public", "data", "hdmap", f"hdmap_align{tag}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # --- QC 이미지 ---
    qc_dir = os.path.join(REPO, "output", "hdmap_align", args.loc or "ALL")
    os.makedirs(qc_dir, exist_ok=True)
    ov = mosaic.copy()
    for p, col in ((to_px(pts_m), (0, 0, 255)), (to_px(apply_similarity(pts_m, center, fin)), (0, 255, 0))):
        ok = (p[:, 0] >= 0) & (p[:, 0] < ov.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < ov.shape[0])
        ov[p[ok, 1].astype(int), p[ok, 0].astype(int)] = col
    cv2.imwrite(os.path.join(qc_dir, "qc_overlay.png"), ov)
    cv2.imwrite(os.path.join(qc_dir, "qc_mask.png"), mask * 255)

    # 비용 곡면 — 최소값이 뾰족하면 정합 성공, 평탄하면 추정값이 노이즈다.
    s = (surf - surf.min()) / max(float(np.ptp(surf)), 1e-12)
    s = cv2.applyColorMap((255 * (1 - s)).astype(np.uint8), cv2.COLORMAP_TURBO)
    s = cv2.resize(s, (512, 512), interpolation=cv2.INTER_NEAREST)
    sx = int(512 * (best[0] * merc_to_ground + args.search) / (2 * args.search))
    sy = int(512 * (best[1] * merc_to_ground + args.search) / (2 * args.search))
    cv2.drawMarker(s, (sx, 511 - sy), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(s, f"+-{args.search:.0f}m  min=({gx:+.1f},{gy:+.1f})m  contrast={100*contrast:.1f}%",
                (8, 502), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(qc_dir, "qc_cost_surface.png"), s)

    print(f"\n→ {os.path.relpath(outp, REPO)}")
    print(f"→ {os.path.relpath(qc_dir, REPO)}/qc_overlay.png  (빨강=정합 전, 초록=정합 후)")
    print(f"→ {os.path.relpath(qc_dir, REPO)}/qc_mask.png     (페인트 마스크 — 도로가 안 보이면 --keep-pct/--tophat 조정)")
    print(f"→ {os.path.relpath(qc_dir, REPO)}/qc_cost_surface.png  (뾰족하면 정합 성공 · 평탄하면 추정값 신뢰 불가)")
    if contrast >= 0.05 and math.hypot(dx, dy) < 1.0:
        print("\n  ✓ 편위가 1m 미만입니다. 표시 워프 없이 그대로 써도 됩니다.")


if __name__ == "__main__":
    main()
