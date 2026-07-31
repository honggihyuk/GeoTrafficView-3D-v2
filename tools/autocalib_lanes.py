"""
노면표시 자동 정합 — 사람이 클릭하지 않고 HD맵 GCP 대응을 찾는다.

**착상**
  영상에도 차선이 보이고 정밀도로지도에도 차선이 있다. 둘을 직접 맞추면 GCP 대응점을
  사람이 찾을 필요가 없다. 위성영상↔HD맵에 쓴 Chamfer/DT(tools/hdmap_align.py)를
  **CCTV영상↔HD맵**으로 옮기는 것이다.

**왜 배경 중앙값 프레임인가 (핵심)**
  단일 프레임은 차량이 차선을 가린다. 영상 전체의 픽셀 중앙값을 쓰면 움직이는 차가 사라지고
  **노면만 남는다**. 이게 있어야 페인트 마스크가 깨끗해진다.

**무엇을 최적화하나**
  호모그래피 8자유도를 자유롭게 풀면 국소최소값 천지다. 대신 지면 좌표에 **유사변환 보정
  C(회전·스케일·평행이동, 4자유도)** 를 걸고 H_new = C·H_old 로 합성한다. webmap 의
  correction3/mul3 과 같은 규약이라 결과를 그대로 쓸 수 있다.
  → 카메라 자세(pitch/높이)가 틀린 경우는 이걸로 못 고친다. 잔차가 안 내려가면 그 신호다.

**한계 (반드시 확인)**
  나란한 차선만 있으면 **선 방향으로는 구속되지 않는다**(조리개 문제). 파선의 끝, 정지선,
  노면 화살표처럼 종방향을 묶는 특징이 있어야 한다. 비용 곡면의 비등방성으로 이를 측정해
  경고한다.

사용:
  python tools/autocalib_lanes.py --loc PANGYO_2
  python tools/autocalib_lanes.py --loc PANGYO_2 --search 30 --apply
출력:
  output/autocalib/<loc>/qc.png · result.json
  --apply 시 location/<loc>/G_projection_<loc>.json 의 homography.H 갱신(백업 생성)
"""
import argparse
import json
import math
import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")
import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from trafficlab.projection.g_projection import GProjection

HDMAP = os.path.join(REPO, "webmap", "public", "data", "hdmap", "hdmap.geojson")
PAINT_LAYERS = ("B2_SURFACELINEMARK", "B3_SURFACEMARK")


# ---------- 영상 ----------
def median_frame(video, max_frames=200):
    """영상 전체 픽셀 중앙값 = 차량이 지워진 노면. 페인트 마스크의 전제."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"영상을 열 수 없습니다: {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    step = max(1, n // max_frames)
    fr, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            fr.append(f)
        i += 1
    cap.release()
    if len(fr) < 5:
        sys.exit(f"프레임이 {len(fr)}개뿐입니다.")
    return np.median(np.stack(fr), axis=0).astype(np.uint8), len(fr)


def road_from_detections(replay, shape, dilate=9):
    """차량 검출이 지나간 자리 = 도로 영역. **캘리브레이션과 무관한 신호**다.

    이게 결정적이다. 유효 영역을 화면 전체로 두면 가드레일 줄무늬·식생 경계가 top-hat
    반응의 대부분을 차지해, 백분위 임계가 잡동사니에 맞춰지고 진짜 차선이 잘려 나간다
    (실측 PANGYO_2: 화면 전체 기준 마스크 1,513px 중 차선은 거의 없었다 →
     도로 영역 제한 후 같은 영상에서 차선이 또렷하게 잡힌다).
    """
    import gzip as _gz
    if not os.path.exists(replay):
        return None
    d = json.load(_gz.open(replay, "rt", encoding="utf-8"))
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    for fr in d["frames"]:
        for o in fr.get("objects", []):
            b = o.get("bbox_2d")
            if not b:
                continue
            x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
            x2, y2 = min(W, int(b[2])), min(H, int(b[3]))
            if x2 > x1 and y2 > y1:
                m[y1:y2, x1:x2] = 1
    return cv2.dilate(m, np.ones((dilate, dilate), np.uint8)) > 0


def paint_mask(bgr, tophat_px=9, keep_pct=18.0, sky_frac=0.12, static=None, min_elong=1.6,
               road=None):
    """노면 페인트 후보. 밝고 + 가느다랗고 + 초록이 아니고 + 글자가 아닌 것.

    식생 제거(ExG): 오블리크 CCTV는 화면의 상당 부분이 풀·나무라 top-hat 반응이 거기서
    대량으로 뜬다.

    **박힌 안내문구 제거가 결정적이다.** "부산/경부선/판교2/407" 같은 글자는 밝고 가늘어
    top-hat이 그대로 잡는다. 실측 결과 마스크의 대부분이 글자였고, 최적화가 차선이 아니라
    **글자에 맞추려 들었다**. tools/static_mask.py 가 만든 마스크로 지우고, 남은 것도
    **길쭉함(elongation)** 으로 거른다 — 차선 조각은 길쭉하고 글자는 뭉툭하다.
    """
    H, W = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = max(3, int(tophat_px) | 1)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    b, gr, r = bgr[:, :, 0].astype(int), bgr[:, :, 1].astype(int), bgr[:, :, 2].astype(int)
    exg = 2 * gr - r - b                                   # 초과 녹색 = 식생 지표
    valid = np.zeros((H, W), bool)
    valid[int(H * sky_frac):, :] = True
    if road is not None:
        valid &= road                                      # 도로 영역으로 제한(가장 중요)
    valid &= (exg < 25) & (g >= np.percentile(g[valid], 30))
    if valid.sum() < 500:
        sys.exit("유효 픽셀이 너무 적습니다.")
    if static is not None:
        valid &= ~static
    thr = np.percentile(th[valid], 100.0 - keep_pct)
    m = ((th >= max(thr, 6)) & valid).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # 길쭉한 성분만 남긴다(차선 조각 O / 글자 X)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < 4:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == k)
        if len(xs) < 4:
            continue
        ev = np.linalg.eigvalsh(np.cov(np.stack([xs.astype(float), ys.astype(float)])))
        elong = math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
        keep[k] = elong >= min_elong
    return np.isin(lab, np.nonzero(keep)[0]).astype(np.uint8)


def sample(field, pts, oob):
    Hh, Ww = field.shape
    x, y = pts[:, 0], pts[:, 1]
    ok = (x >= 0) & (x <= Ww - 2) & (y >= 0) & (y <= Hh - 2)
    out = np.full(len(pts), float(oob))
    if not ok.any():
        return out, ok
    xi, yi = x[ok], y[ok]
    x0 = np.floor(xi).astype(np.int32); y0 = np.floor(yi).astype(np.int32)
    fx, fy = xi - x0, yi - y0
    out[ok] = (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x0 + 1] * fx * (1 - fy)
               + field[y0 + 1, x0] * (1 - fx) * fy + field[y0 + 1, x0 + 1] * fx * fy)
    return out, ok


# ---------- HD맵 ----------
def hdmap_points(world, radius, step=0.5):
    from pyproj import Transformer
    if not os.path.exists(HDMAP):
        sys.exit(f"HD맵 GeoJSON 없음: {HDMAP}")
    tr = Transformer.from_crs(4326, world["epsg"], always_xy=True)
    oE, oN = world["origin_easting"], world["origin_northing"]
    pts = []
    with open(HDMAP, encoding="utf-8") as f:
        fc = json.load(f)
    for feat in fc.get("features", []):
        if feat.get("properties", {}).get("layer") not in PAINT_LAYERS:
            continue
        g = feat["geometry"]
        rings = ([g["coordinates"]] if g["type"] == "LineString" else
                 g["coordinates"] if g["type"] in ("MultiLineString", "Polygon") else
                 [c for poly in g["coordinates"] for c in poly] if g["type"] == "MultiPolygon" else [])
        for coords in rings:
            q = []
            for lon, lat, *_ in coords:
                e, n = tr.transform(lon, lat)
                q.append((e - oE, n - oN))
            q = np.asarray(q, float)
            if len(q) < 2 or np.min(np.hypot(q[:, 0], q[:, 1])) > radius:
                continue
            for a, b in zip(q, q[1:]):                    # step 간격 리샘플
                L = float(np.linalg.norm(b - a))
                for t in np.arange(0.0, 1.0, step / max(L, 1e-6)):
                    pts.append(a + (b - a) * t)
    return np.asarray(pts, float)


def sim3(tx, ty, th, ls):
    s, c, sn = math.exp(ls), math.cos(th), math.sin(th)
    return np.array([[s * c, -s * sn, tx], [s * sn, s * c, ty], [0, 0, 1]], float)


# ---------- 카메라 자세 매개변수화 ----------
def decompose(H_g2i, K):
    """지면→영상 호모그래피를 (R, t)로 분해. H = K·[r1|r2|t] (지면 z=0 평면).

    현재 캘리브레이션을 초기값으로 삼기 위한 것이다. 오일러 각 규약을 새로 정의하지 않고
    **여기서 얻은 R0 에 증분 회전을 곱하는 방식**으로 최적화하면 규약 문제가 사라진다.
    """
    M = np.linalg.inv(K) @ H_g2i
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    if t[2] < 0:                                  # 카메라가 지면 앞에 있어야 한다
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    U, _, Vt = np.linalg.svd(np.stack([r1, r2, r3], axis=1))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R, t


def H_from_pose(R, t, f, cx, cy):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
    return K @ np.stack([R[:, 0], R[:, 1], t], axis=1)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--radius", type=float, default=200.0, help="HD맵 사용 반경(m)")
    ap.add_argument("--search", type=float, default=25.0, help="거친 탐색 반경(지면 m)")
    ap.add_argument("--search-step", type=float, default=2.0)
    ap.add_argument("--rot", type=float, default=8.0, help="거친 탐색 회전 범위(도)")
    ap.add_argument("--coarse-points", type=int, default=4000,
                    help="거친 탐색에 쓸 점 수(전량 투영은 너무 느리다). 정밀화는 전량 사용")
    ap.add_argument("--trunc", type=float, default=25.0, help="DT 절단(px)")
    ap.add_argument("--keep-pct", type=float, default=18.0,
                    help="도로 영역 안에서 페인트로 볼 상위 top-hat 반응 비율(%%)")
    ap.add_argument("--no-road-mask", action="store_true",
                    help="검출 기반 도로 영역 제한을 끈다(권장하지 않음)")
    ap.add_argument("--mode", choices=["pose", "similarity"], default="pose",
                    help="pose=카메라 자세(회전·위치·초점) 최적화 / similarity=지면 유사변환(4DOF)")
    ap.add_argument("--dang", type=float, default=12.0, help="pose: 각도 탐색 범위(도)")
    ap.add_argument("--no-refine8", action="store_true",
                    help="8자유도 정밀화 단계를 건너뛴다(자세 7자유도까지만)")
    ap.add_argument("--apply", action="store_true", help="G_projection 의 H 를 갱신(백업 생성)")
    args = ap.parse_args()

    gp = os.path.join(REPO, "location", args.loc, f"G_projection_{args.loc}.json")
    gd = json.load(open(gp, encoding="utf-8"))
    g = GProjection(gd, base_dir=os.path.dirname(gp))
    world = gd["world"]
    W, Hh = gd.get("undistort", {}).get("resolution", [720, 480])
    H = Hh
    video = os.path.join(REPO, "webmap", "public", "data", "footage", f"{args.loc.lower()}.mp4")
    if not os.path.exists(video):
        sys.exit(f"영상 없음: {video}")

    med, nfr = median_frame(video)
    if med.shape[1] != W or med.shape[0] != H:
        med = cv2.resize(med, (W, H))
    sm_path = os.path.join(REPO, "location", args.loc, "static_mask.png")
    static = None
    if os.path.exists(sm_path):
        sm = cv2.imread(sm_path, cv2.IMREAD_GRAYSCALE)
        if sm is not None:
            if sm.shape[:2] != (H, W):
                sm = cv2.resize(sm, (W, H), interpolation=cv2.INTER_NEAREST)
            static = cv2.dilate(sm, np.ones((9, 9), np.uint8)) > 127
            print(f"  정적 오탐 마스크 적용: {int(static.sum()):,}px 제외")
    road = None
    if not args.no_road_mask:
        rp = os.path.join(REPO, "webmap", "public", "data", "replay", f"{args.loc.lower()}.json.gz")
        road = road_from_detections(rp, (H, W))
        if road is not None:
            print(f"  도로 영역(검출 기반): {int(road.sum()):,}px ({100*road.mean():.1f}%)")
    mask = paint_mask(med, keep_pct=args.keep_pct, static=static, road=road)
    dt = cv2.distanceTransform(np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    print(f"[{args.loc}] 중앙값 프레임 {nfr}장 · 페인트 마스크 {int(mask.sum()):,}px "
          f"({100*mask.mean():.2f}%)")

    pts = hdmap_points(world, args.radius)
    if len(pts) < 200:
        sys.exit(f"HD맵 노면표시 점이 {len(pts)}개뿐입니다 — --radius 를 키우세요.")
    print(f"HD맵 노면표시 점 {len(pts):,}개 (반경 {args.radius:.0f}m)")

    Hinv = g.H_inv                                   # 지면(sat) → 미왜곡 영상
    Kd = (np.abs(np.asarray(g.D, float)).max() > 1e-9)
    K0 = np.asarray(g.K, float)
    f0, cx, cy = float(K0[0, 0]), float(K0[0, 2]), float(K0[1, 2])
    R0, t0 = decompose(Hinv, K0)
    C0 = -R0.T @ t0
    print(f"  현재 자세 분해: 카메라 높이 {C0[2]:.1f}m · 지면상 위치 ({C0[0]:.0f}, {C0[1]:.0f})m "
          f"· f={f0:.0f}px")

    paint_yx = np.stack(np.nonzero(mask > 0), axis=1).astype(np.float32)   # (N,2) = (y,x)
    paint_xy = paint_yx[:, ::-1].copy()
    if len(paint_xy) > 6000:                       # 역방향 항 비용 억제
        paint_xy = paint_xy[np.linspace(0, len(paint_xy) - 1, 6000).astype(int)]

    def H_g2i(par):
        """par = [dRx, dRy, dRz(rad), dtx, dty, dtz(m), log(f/f0)]  — 전부 0이면 현재 자세."""
        R = cv2.Rodrigues(np.asarray(par[:3], float))[0] @ R0
        t = t0 + np.asarray(par[3:6], float)
        return H_from_pose(R, t, f0 * math.exp(par[6]), cx, cy)

    def project_pose(par, P):
        q = cv2.perspectiveTransform(np.asarray(P, np.float64).reshape(-1, 1, 2), H_g2i(par)).reshape(-1, 2)
        if Kd:
            o = np.c_[(q[:, 0] - cx) / f0, (q[:, 1] - cy) / f0, np.ones(len(q))]
            q = cv2.projectPoints(o, np.zeros(3), np.zeros(3), K0, g.D)[0].reshape(-1, 2)
        return q

    def project_sim(par, P):
        p = np.c_[P, np.ones(len(P))] @ sim3(*par).T
        p = p[:, :2] / p[:, 2:3]
        q = cv2.perspectiveTransform(p.reshape(-1, 1, 2).astype(np.float64), Hinv).reshape(-1, 2)
        if Kd:
            o = np.c_[(q[:, 0] - cx) / f0, (q[:, 1] - cy) / f0, np.ones(len(q))]
            q = cv2.projectPoints(o, np.zeros(3), np.zeros(3), K0, g.D)[0].reshape(-1, 2)
        return q

    POSE = args.mode == "pose"
    project = project_pose if POSE else project_sim
    NP = 7 if POSE else 4

    # 평가 집합을 **초기 캘리브레이션에서 화면 안에 들어오는 점**으로 고정한다.
    # 화면 안 점만 평균하면 최적화가 점을 화면 밖으로 밀어내 비용을 낮추는 꼼수를 쓴다
    # (실측: 18,680 → 1,443개로 줄면서 '개선'된 것처럼 보였다).
    _, ok0 = sample(dt, project(np.zeros(NP), pts), args.trunc)
    pts = pts[ok0]
    if len(pts) < 200:
        sys.exit(f"초기 캘리브레이션에서 화면 안 HD맵 점이 {len(pts)}개뿐입니다.")
    print(f"  평가 집합: 초기 투영 기준 화면 안 {len(pts):,}점으로 고정 · 모드 {args.mode}")
    sub = pts[np.linspace(0, len(pts) - 1, min(args.coarse_points, len(pts))).astype(int)]

    def project_H(Hg, P):
        q = cv2.perspectiveTransform(np.asarray(P, np.float64).reshape(-1, 1, 2), Hg).reshape(-1, 2)
        if Kd:
            o = np.c_[(q[:, 0] - cx) / f0, (q[:, 1] - cy) / f0, np.ones(len(q))]
            q = cv2.projectPoints(o, np.zeros(3), np.zeros(3), K0, g.D)[0].reshape(-1, 2)
        return q

    def cost_H(Hg, P=None, ret_n=False):
        """양방향 Chamfer + 깊이 양수 제약. 자세(7DOF)와 완전 호모그래피(8DOF)가 공유한다."""
        P = pts if P is None else P
        w = np.c_[sub, np.ones(len(sub))] @ Hg[2]
        if not np.all(w > 1e-6):                          # 지평선이 화면 안으로 들어온 퇴화 해
            return (1e6, 0) if ret_n else 1e6
        q = project_H(Hg, P)
        d, ok = sample(dt, q, args.trunc)
        d[~ok] = args.trunc
        fwd = float(np.minimum(d, args.trunc).mean())
        proj_img = np.zeros(dt.shape, np.uint8)
        qi = q[ok].astype(np.int32)
        if len(qi):
            proj_img[np.clip(qi[:, 1], 0, dt.shape[0] - 1),
                     np.clip(qi[:, 0], 0, dt.shape[1] - 1)] = 255
        bwd_v, _ = sample(cv2.distanceTransform(255 - proj_img, cv2.DIST_L2, 3), paint_xy, args.trunc)
        c = 0.5 * (fwd + float(np.minimum(bwd_v, args.trunc).mean()))
        return (c, int(ok.sum())) if ret_n else c

    def physicality(Hg):
        """H 가 실제 카메라 자세로 실현 가능한 정도. K^-1H 의 앞 두 열이 정규직교여야 한다.

        8자유도로 풀면 렌즈 왜곡·지면 비평면성 같은 '자세로 설명 안 되는' 성분까지 흡수한다.
        이 값이 커지면 정합은 좋아져도 물리적 카메라와 멀어진다는 뜻이라 함께 본다.
        """
        M = np.linalg.inv(K0) @ Hg
        n1, n2 = np.linalg.norm(M[:, 0]), np.linalg.norm(M[:, 1])
        return abs(n1 - n2) / max(n1, n2, 1e-9), abs(float(M[:, 0] @ M[:, 1])) / max(n1 * n2, 1e-9)

    def depth_ok(par):
        """모든 평가점이 카메라 앞(w>0)에 있어야 한다.

        이 제약이 없으면 최적화가 **지평선을 화면 안으로 끌어들이는** 퇴화 해를 찾는다
        (실측: 차선이 화면 한가운데 소실점으로 모여 방사형으로 퍼졌다).
        """
        if not POSE:
            return True
        Hg = H_g2i(par)
        w = np.c_[sub, np.ones(len(sub))] @ Hg[2]
        return bool(np.all(w > 1e-6))

    def cost(par, ret_n=False, P=None):
        par = np.asarray(par, float)
        Hg = H_g2i(par) if POSE else None
        if Hg is None:                                    # similarity 모드
            p = np.c_[pts if P is None else P, np.ones(len(pts if P is None else P))] @ sim3(*par).T
            p = p[:, :2] / p[:, 2:3]
            q = cv2.perspectiveTransform(p.reshape(-1, 1, 2).astype(np.float64), Hinv).reshape(-1, 2)
            d, ok = sample(dt, q, args.trunc)
            d[~ok] = args.trunc
            c = float(np.minimum(d, args.trunc).mean())
            return (c, int(ok.sum())) if ret_n else c
        return cost_H(Hg, P, ret_n)

    base = np.zeros(NP)
    c0, n0 = cost(base, True)
    print(f"\n정합 전 잔차 {c0:.2f}px · 화면 안 점 {n0:,}/{len(pts):,}")

    from scipy.optimize import minimize
    if POSE:
        # ① 거친 탐색: 도로 카메라의 지배적 오차는 pitch/yaw 다. 두 축만 훑어 다중 시작점을 만든다.
        da = np.radians(np.arange(-args.dang, args.dang + 1e-9, 1.5))
        surf = np.full((len(da), len(da)), np.inf)
        starts = []
        for a, rx in enumerate(da):
            for b, ry in enumerate(da):
                q = np.zeros(NP); q[0], q[1] = rx, ry
                c = cost(q, P=sub)
                surf[b, a] = c
                starts.append((c, q.copy()))
        starts.sort(key=lambda z: z[0])
        bc, best = starts[0]
        print(f"① 자세 거친 탐색({len(da)}² = {len(da)**2:,}회): "
              f"ΔRx {math.degrees(best[0]):+.1f}° · ΔRy {math.degrees(best[1]):+.1f}° · 잔차 {bc:.2f}px")

        # ② 상위 시작점에서 각각 Powell(7자유도) — 국소최소값 대비 다중 시작
        sc = np.array([1e-2, 1e-2, 1e-2, 1.0, 1.0, 1.0, 1e-2])
        lim = np.array([math.radians(args.dang * 2)] * 3 + [60.0, 60.0, 40.0] + [math.log(1.6)])
        bnds = [(-l / k, l / k) for l, k in zip(lim, sc)]
        fin, cbest = best, bc
        for c_s, q0 in starts[:4]:
            r = minimize(lambda z: cost(z * sc), q0 / sc, method="Powell", bounds=bnds,
                         options={"xtol": 1e-3, "ftol": 1e-5, "maxiter": 40000})
            cand = r.x * sc
            cc = cost(cand)
            if cc < cbest:
                fin, cbest = cand, cc
        R = cv2.Rodrigues(fin[:3])[0] @ R0
        t = t0 + fin[3:6]
        Cc = -R.T @ t
        print(f"② Powell(7DOF, 다중시작 4): 회전 Δ({math.degrees(fin[0]):+.2f}, "
              f"{math.degrees(fin[1]):+.2f}, {math.degrees(fin[2]):+.2f})° · "
              f"이동 Δ({fin[3]:+.1f}, {fin[4]:+.1f}, {fin[5]:+.1f})m · f {f0*math.exp(fin[6]):.0f}px")
        print(f"   → 카메라 높이 {Cc[2]:.1f}m · 지면상 위치 ({Cc[0]:.0f}, {Cc[1]:.0f})m")
        H_g2i_fin = H_g2i(fin)

        # ③ 완전 호모그래피(8자유도) 정밀화 — 자세 7자유도로 설명 안 되는 성분(렌즈 왜곡,
        # 지면 비평면성)을 흡수한다. **정규화 영상좌표에서 항등원 주변 섭동**으로 매개변수화해
        # 8개 파라미터의 스케일을 맞춘다(H 원소를 직접 풀면 조건수가 나빠 발산한다).
        if not args.no_refine8:
            Tn = np.array([[2.0 / W, 0, -1.0], [0, 2.0 / Hh, -1.0], [0, 0, 1]], float)
            Tni = np.linalg.inv(Tn)
            SLOT = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]

            def H8(dv):
                Mp = np.eye(3)
                for k, (i_, j_) in enumerate(SLOT):
                    Mp[i_, j_] += dv[k]
                return Tni @ Mp @ Tn @ H_g2i_fin

            c_pose = cost_H(H_g2i_fin)
            lim8 = 0.18
            r8 = minimize(lambda z: cost_H(H8(z)), np.zeros(8), method="Powell",
                          bounds=[(-lim8, lim8)] * 8,
                          options={"xtol": 1e-4, "ftol": 1e-6, "maxiter": 60000})
            c8 = cost_H(H8(r8.x))
            edge8 = float(np.max(np.abs(r8.x))) > lim8 * 0.98
            if c8 < c_pose and not edge8:
                H_g2i_fin = H8(r8.x)
                print(f"③ 8자유도 정밀화: 잔차 {c_pose:.2f} → {c8:.2f}px "
                      f"(섭동 최대 {np.max(np.abs(r8.x)):.3f})")
            else:
                print(f"③ 8자유도 정밀화 기각 "
                      f"({'경계 도달' if edge8 else f'개선 없음 {c8:.2f}px'}) — 자세 결과를 유지합니다")
            e1, e2 = physicality(H_g2i_fin)
            print(f"   물리성: |‖r1‖−‖r2‖|/max {e1:.3f} · |r1·r2| {e2:.3f} "
                  f"({'자세로 실현 가능' if max(e1, e2) < 0.15 else '자세로 설명 안 되는 성분 포함'})")

        H_new = np.linalg.inv(H_g2i_fin)
        H_new = (H_new / H_new[2, 2]).tolist()
        rng = np.degrees(da)
    else:
        rng = np.arange(-args.search, args.search + 1e-9, args.search_step)
        rots = np.radians(np.arange(-args.rot, args.rot + 1e-9, 2.0))
        surf = np.full((len(rng), len(rng)), np.inf)
        best, bc = base.copy(), None
        for th in rots:
            for a, tx in enumerate(rng):
                for b, ty in enumerate(rng):
                    c = cost(np.array([tx, ty, th, 0.0]), P=sub)
                    if th == rots[len(rots) // 2]:
                        surf[b, a] = c
                    if bc is None or c < bc:
                        best, bc = np.array([tx, ty, th, 0.0]), c
        print(f"① 거친 탐색: 이동 ({best[0]:+.1f}, {best[1]:+.1f})m · "
              f"회전 {math.degrees(best[2]):+.1f}° · 잔차 {bc:.2f}px")
        sc = np.array([1.0, 1.0, 1e-2, 1e-3])
        lim = args.search + 2 * args.search_step
        bnds = [(-lim, lim), (-lim, lim),
                (math.radians(-args.rot * 1.5) / sc[2], math.radians(args.rot * 1.5) / sc[2]),
                (math.log(0.85) / sc[3], math.log(1.15) / sc[3])]
        r = minimize(lambda q: cost(q * sc), best / sc, method="Powell", bounds=bnds,
                     options={"xtol": 1e-2, "ftol": 1e-4, "maxiter": 20000})
        fin = r.x * sc
        if cost(fin) > bc:
            fin = best
        Cc = None
        H_new = (sim3(*fin) @ np.asarray(gd["homography"]["H"], float)).tolist()

    c1, n1 = (cost_H(H_g2i_fin, ret_n=True) if POSE else cost(fin, True))
    print(f"\n정합 후 잔차 {c1:.2f}px (개선 {100*(1-c1/max(c0,1e-9)):.0f}%) · 화면 안 점 {n1:,}")
    d1, ok1 = sample(dt, (project_H(H_g2i_fin, pts) if POSE else project(fin, pts)), args.trunc)
    d0v, _ = sample(dt, project(base, pts), args.trunc)
    for t_ in (3, 6, 12):
        print(f"  페인트 {t_:2d}px 이내: {100*np.mean(d1[ok1] < t_):5.1f}%"
              f"  (정합 전 {100*np.mean(d0v[ok1] < t_):5.1f}%)")

    fs = surf[np.isfinite(surf)]
    contrast = float((np.median(fs) - fs.min()) / max(np.median(fs), 1e-9)) if len(fs) else 0.0
    low = surf <= surf.min() + 0.2 * (np.median(fs) - surf.min()) if len(fs) else np.zeros_like(surf, bool)
    gy, gx = np.nonzero(low)
    ratio = 1.0
    if len(gx) >= 6:
        ev = np.linalg.eigvalsh(np.cov(np.stack([rng[gx], rng[gy]])))
        ratio = float(math.sqrt(max(ev[1], 1e-12) / max(ev[0], 1e-12)))
    print(f"  비용 곡면 대비 {100*contrast:.1f}% · 비등방 {ratio:.1f}:1")
    if contrast < 0.05:
        print("  ⚠ 곡면이 평탄합니다 — 결과를 신뢰하지 마세요(페인트 마스크를 먼저 확인).")
    if ratio > 3.0:
        print("  ⚠ 한 방향만 구속됩니다(조리개 문제). 파선 끝·정지선·화살표가 필요합니다.")


    # --- QC 이미지 ---
    out_dir = os.path.join(REPO, "output", "autocalib", args.loc)
    os.makedirs(out_dir, exist_ok=True)
    vis = med.copy()
    vis[mask > 0] = (0, 255, 255)
    for par, col in ((base, (0, 0, 255)), (fin, (0, 255, 0))):
        q = (project_H(H_g2i_fin, pts) if (POSE and col == (0, 255, 0)) else project(par, pts))
        okq = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
        vis[q[okq, 1].astype(int), q[okq, 0].astype(int)] = col
    cv2.putText(vis, "cyan=paint  red=before  green=after", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(out_dir, "qc.png"), np.vstack([med, vis]))

    res = {"loc": args.loc, "method": f"chamfer_dt_{args.mode}"}
    if POSE:
        res["pose"] = {"d_rot_deg": [math.degrees(v) for v in fin[:3]],
                       "d_t_m": list(fin[3:6]), "focal_px": f0 * math.exp(fin[6]),
                       "cam_height_m": float(Cc[2]),
                       "cam_sat_xy_m": [float(Cc[0]), float(Cc[1])]}
    else:
        res["params"] = {"tx_m": fin[0], "ty_m": fin[1],
                         "rot_deg": math.degrees(fin[2]), "scale": math.exp(fin[3])}
    json.dump({**res, "H_new": H_new,
               "residual_px": {"before": round(c0, 3), "after": round(c1, 3)},
               "confidence": {"contrast": round(contrast, 4), "anisotropy": round(ratio, 2),
                              "verdict": ("신뢰 가능" if contrast >= 0.05 and ratio <= 3.0
                                          else "신뢰 불가")},
               "note": ("pose 모드: H_new = inv(K·[r1|r2|t]) — 자세를 직접 최적화한 결과. "
                        "카메라 높이·위치도 함께 나오므로 parallax 값도 갱신할 수 있다."
                        if POSE else
                        "similarity 모드: H_new = correction3 · H_old (homography.js 규약)")},
              open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n→ {os.path.relpath(out_dir, REPO)}/qc.png · result.json")

    if args.apply:
        if contrast < 0.05:
            sys.exit("  곡면이 평탄해 적용을 거부합니다. QC 이미지를 먼저 확인하세요.")
        shutil.copy2(gp, gp + ".bak")
        gd["homography"]["H"] = H_new
        if POSE:
            # 자세에서 나온 카메라 높이·위치로 시차 파라미터를 갱신한다. 단 **높이와 초점거리는
            # 서로 맞바꿔지므로**, f 가 가정값(f=W 폴백)이면 높이도 그만큼 못 믿는다.
            # 물리적으로 타당한 범위(5~30m)일 때만 반영하고, 아니면 기존 값을 지킨다.
            h_est = float(Cc[2])
            if 5.0 <= h_est <= 30.0:
                gd.setdefault("parallax", {})
                gd["parallax"]["z_cam_meters"] = round(h_est, 2)
                gd["parallax"]["x_cam_coords_sat"] = round(float(Cc[0]), 2)
                gd["parallax"]["y_cam_coords_sat"] = round(float(Cc[1]), 2)
                print(f"  parallax 갱신: 높이 {h_est:.1f}m · 위치 ({Cc[0]:.0f}, {Cc[1]:.0f})m")
            else:
                print(f"  ⚠ 추정 높이 {h_est:.1f}m 가 타당 범위(5~30m) 밖입니다. "
                      f"f={f0*math.exp(fin[6]):.0f}px 가 가정값이면 높이도 그만큼 부정확합니다 "
                      f"— parallax 는 건드리지 않습니다(H 만 갱신).")
        gd.setdefault("meta", {})["note"] = (gd["meta"].get("note", "") +
                                             f" | 노면표시 자동정합({args.mode}, 잔차 {c1:.1f}px)")
        json.dump(gd, open(gp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  적용 완료 · 백업 {os.path.relpath(gp + '.bak', REPO)}")
    else:
        print("  적용하려면 --apply (QC 이미지 확인 후)")


if __name__ == "__main__":
    main()
