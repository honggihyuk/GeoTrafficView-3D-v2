"""
자동 캘리브레이션 (소실점 + IPM) — 클릭·위성 없이 CCTV 프레임만으로 호모그래피 자동 산출.

원리(교통카메라 표준 기법):
  1) 도로방향 선분들의 소실점(VP) 검출 → 카메라 pitch/yaw 추정(roll=0, f≈W 가정)
  2) 카메라 높이(가정) + ray-plane IPM → 픽셀→지면(로컬미터: 우측/전방) 매핑 → 호모그래피 H_local
  3) 지오앵커: 원점=카메라 좌표(ITS), 전방축 방위=도로 bearing(OSM/HD맵/파라미터) → H를 UTM(동/북)으로 회전
검증: GCP 수동 캘리브레이션이 있으면 형상 오차(유사변환 정합 후 RMS) 정량 비교.

사용:
  python tools/auto_calibrate_vp.py --loc SONGDO_IC --cam-height 9 [--bearing 300] [--save]
출력: location/<loc>/G_projection_<loc>_auto.json  (--save 시 실제 파일로)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from pyproj import Transformer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def detect_vp(gray):
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=60, maxLineGap=20)
    A, b, seg = [], [], []
    for l in (lines if lines is not None else []):
        x1, y1, x2, y2 = l[0]
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if 15 < ang < 85:
            a1, b1 = y2 - y1, x1 - x2
            A.append([a1, b1]); b.append(a1 * x1 + b1 * y1); seg.append((x1, y1, x2, y2))
    if len(A) < 2:
        raise SystemExit("도로방향 선분 부족 — VP 검출 실패")
    vp, *_ = np.linalg.lstsq(np.array(A, float), np.array(b, float), rcond=None)
    return float(vp[0]), float(vp[1]), seg


def estimate_focal(gray, W, H, vx, vy):
    """(a) 초점거리 f 자동추정 — **2번째(직교) 소실점** 기반.
    단일 VP로는 f 불가관측(도로선이 어떤 f에서도 평행). 도로에 수직인 구조(정지선/횡단보도/차량 후미
    /가로선)의 2번째 VP가 있으면 직교조건 f²=-(vp1-pp)·(vp2-pp) 로 산출. 없으면 None(→f=W 폴백)."""
    cx, cy = W / 2, H / 2
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=15)
    if lines is None:
        return None, "선분 없음"
    perp, A, b = [], [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        # VP1(도로 소실점)을 지나지 않는 선분만(=도로에 수직 후보)
        nx, ny = y2 - y1, x1 - x2
        d = abs(nx * vx + ny * vy - (nx * x1 + ny * y1)) / (np.hypot(nx, ny) + 1e-9)
        if d > 40:
            perp.append((x1, y1, x2, y2)); A.append([nx, ny]); b.append(nx * x1 + ny * y1)
    if len(A) < 4:
        return None, f"2번째 VP용 수직선 부족({len(A)})"
    vp2, *_ = np.linalg.lstsq(np.array(A, float), np.array(b, float), rcond=None)
    dot = (vx - cx) * (vp2[0] - cx) + (vy - cy) * (vp2[1] - cy)
    if dot >= 0:
        return None, f"VP2 비직교(dot={dot:.0f})"
    return float(np.sqrt(-dot)), f"2VP직교 vp2=({vp2[0]:.0f},{vp2[1]:.0f})"


def osm_bearing(lat, lon):
    """(b) OSM Overpass로 카메라 인근 도로(highway) 최근접 세그먼트의 방위(도, 북=0 시계) 자동 산출."""
    import json as _j, urllib.request, urllib.parse, math
    q = f"[out:json][timeout:20];way(around:100,{lat},{lon})[highway];out geom;"
    url = "https://overpass-api.de/api/interpreter?" + urllib.parse.urlencode({"data": q})
    req = urllib.request.Request(url, headers={"User-Agent": "GeoTrafficView-3D/1.0"})  # 406 방지
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = _j.loads(r.read())
    except Exception as e:
        print(f"  OSM 조회 실패({type(e).__name__}) → bearing 파라미터/기본값 사용")
        return None
    best = None; bestd = 1e18
    for w in data.get("elements", []):
        g = w.get("geometry", [])
        for i in range(len(g) - 1):
            ax, ay = g[i]["lon"], g[i]["lat"]; bx, by = g[i + 1]["lon"], g[i + 1]["lat"]
            mx, my = (ax + bx) / 2, (ay + by) / 2
            d = (mx - lon) ** 2 + (my - lat) ** 2
            if d < bestd:
                bestd = d
                brg = math.degrees(math.atan2((bx - ax) * math.cos(math.radians(lat)), by - ay))
                best = (brg + 360) % 360
    return best


def build_ipm(W, H, vx, vy, Hc, f=None):
    """VP→pitch/yaw, ray-plane IPM. f 미지정 시 W. 반환: px→(right,forward)미터 함수 + 각도/재투영VP."""
    f = float(f) if f else float(W)
    cx, cy = W / 2, H / 2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
    Kinv = np.linalg.inv(K)
    theta = np.arctan((cy - vy) / f)   # pitch(하향)
    psi = np.arctan((vx - cx) / f)     # yaw
    # 카메라축(월드: X우측, Y전방, Z상). roll=0
    z_cam = np.array([-np.sin(psi) * np.cos(theta), np.cos(psi) * np.cos(theta), -np.sin(theta)])
    x_cam = np.array([np.cos(psi), np.sin(psi), 0.0])
    y_cam = np.cross(z_cam, x_cam)
    Rc2w = np.column_stack([x_cam, y_cam, z_cam])
    C = np.array([0, 0, Hc], float)

    def px_to_ground(u, v):
        d = Rc2w @ (Kinv @ np.array([u, v, 1.0]))
        if d[2] >= -1e-6:
            return None  # 지면 아래를 안 봄(수평선 위)
        t = -C[2] / d[2]
        p = C + t * d
        return [p[0], p[1]]  # (right, forward) meters

    # VP 재투영 확인(작을수록 자세 추정 정확)
    dvp = Rc2w.T @ np.array([0.0, 1.0, 0.0])   # 월드 전방 → 카메라
    u_vp = f * dvp[0] / dvp[2] + cx; v_vp = f * dvp[1] / dvp[2] + cy
    return px_to_ground, np.degrees(theta), np.degrees(psi), (u_vp, v_vp)


def homography_from_ipm(px_to_ground, W, H, vy):
    cx = W / 2
    pts = [(cx - 120, H - 20), (cx + 120, H - 20), (cx + 120, vy + 60), (cx - 120, vy + 60)]
    src, dst = [], []
    for (u, v) in pts:
        g = px_to_ground(u, v)
        if g is None:
            raise SystemExit("IPM 지면 교차 실패(자세/VP 확인)")
        src.append([u, v]); dst.append(g)
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst)), src, dst


def applyH(Hm, x, y):
    w = Hm[2, 0] * x + Hm[2, 1] * y + Hm[2, 2]
    return np.array([(Hm[0, 0] * x + Hm[0, 1] * y + Hm[0, 2]) / w,
                     (Hm[1, 0] * x + Hm[1, 1] * y + Hm[1, 2]) / w])


def umeyama(src, dst):
    src = np.asarray(src); dst = np.asarray(dst); n = len(src)
    ms, md = src.mean(0), dst.mean(0)
    Xs, Xd = src - ms, dst - md
    U, D, Vt = np.linalg.svd((Xd.T @ Xs) / n)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    c = (D * np.diag(S)).sum() / ((Xs ** 2).sum() / n)
    t = md - c * R @ ms
    return lambda p: c * R @ np.asarray(p) + t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--cam-height", type=float, default=9.0)
    ap.add_argument("--focal", type=float, default=None, help="초점거리(px). 미지정 시 (a) 차선 평행성으로 자동추정")
    ap.add_argument("--bearing", type=float, default=None, help="도로 방위(도). 미지정 시 (b) OSM Overpass 자동")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    ldir = os.path.join(REPO, "location", args.loc)
    frame = cv2.imread(os.path.join(ldir, f"cctv_{args.loc}.png"))
    if frame is None:
        sys.exit("cctv 스냅샷 없음")
    H_, W_ = frame.shape[:2]
    cam = json.load(open(os.path.join(ldir, "_camera.json"), encoding="utf-8"))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vx, vy, seg = detect_vp(gray)
    if args.focal:
        f_est, finfo = args.focal, "param"
    else:
        f_est, finfo = estimate_focal(gray, W_, H_, vx, vy)  # (a) 2번째 직교 VP로 초점추정
        if f_est is None:
            f_est, finfo = float(W_), f"f불가관측({finfo})->f=W"
    px2g, pitch, yaw, vp_re = build_ipm(W_, H_, vx, vy, args.cam_height, f_est)
    print(f"VP=({vx:.1f},{vy:.1f}) 선분{len(seg)} · (a)f={f_est:.0f}px [{finfo}] · pitch {pitch:.1f} yaw {yaw:.1f} · VP재투영=({vp_re[0]:.1f},{vp_re[1]:.1f})")
    Hloc, src, dst = homography_from_ipm(px2g, W_, H_, vy)

    # --- GCP 정답 대비 형상 검증(있으면) ---
    gpath = os.path.join(ldir, f"G_projection_{args.loc}.json")
    gcp_bearing = None
    if os.path.exists(gpath):
        gj = json.load(open(gpath, encoding="utf-8"))
        anchors = gj.get("homography", {}).get("anchors_list", [])
        if anchors and gj.get("homography", {}).get("H"):
            Hg = np.array(gj["homography"]["H"], float)  # px→(E,N)로컬(GCP원점)
            # 검증 격자 = GCP 앵커 픽셀 bounding box 내부(캘리브레이션 유효영역에서 공정 비교)
            axs = [a["px"][0] for a in anchors]; ays = [a["px"][1] for a in anchors]
            u0, u1 = int(min(axs)), int(max(axs)); v0, v1 = int(min(ays)), int(max(ays))
            grid = [(u, v) for u in range(u0, u1 + 1, max(20, (u1 - u0) // 6))
                    for v in range(v0, v1 + 1, max(20, (v1 - v0) // 6))]
            a = [applyH(Hloc, u, v) for (u, v) in grid]   # (right,forward)
            g = [applyH(Hg, u, v) for (u, v) in grid]      # (E,N)
            align = umeyama(a, g)
            res = [np.linalg.norm(align(pa) - pg) for pa, pg in zip(a, g)]
            rms = float(np.sqrt(np.mean(np.square(res))))
            print(f"[검증] GCP 대비 형상 정합 RMS = {rms:.2f} m  (유사변환 정합 후, {len(grid)}점)")
            # GCP에서 도로 방위 추정(전방축이 UTM에서 향하는 azimuth)
            fwd = applyH(Hloc, W_ / 2, vy + 60) - applyH(Hloc, W_ / 2, H_ - 20)  # 전방 벡터(local)
            # align으로 전방벡터를 GCP(E,N)로 회전
            f0 = align(applyH(Hloc, W_ / 2, H_ - 20)); f1 = align(applyH(Hloc, W_ / 2, vy + 60))
            dE, dN = (f1 - f0)
            gcp_bearing = (np.degrees(np.arctan2(dE, dN)) + 360) % 360
            print(f"[참고] GCP 기준 도로 전방 방위 {gcp_bearing:.1f}deg (production: OSM/HD맵 자동)")

    # --- 지오앵커 + 저장 ---
    if args.bearing is not None:
        beta, bsrc = args.bearing, "param"
    else:
        b_osm = osm_bearing(cam["lat"], cam["lon"])  # (b) OSM 자동
        if b_osm is not None:
            beta, bsrc = b_osm, "OSM"
        elif gcp_bearing is not None:
            beta, bsrc = gcp_bearing, "GCP추정"
        else:
            beta, bsrc = 0.0, "기본0"
    print(f"(b) 도로 방위 {beta:.1f}deg (출처: {bsrc})")
    br = np.radians(beta)
    # (right,forward)→(E,N): E=Xr cosβ + Yf sinβ, N=-Xr sinβ + Yf cosβ  → H의 앞 두 행 회전
    Rrows = np.array([[np.cos(br), np.sin(br)], [-np.sin(br), np.cos(br)]])
    Hfin = Hloc.copy()
    Hfin[0, :] = Rrows[0, 0] * Hloc[0, :] + Rrows[0, 1] * Hloc[1, :]
    Hfin[1, :] = Rrows[1, 0] * Hloc[0, :] + Rrows[1, 1] * Hloc[1, :]

    tr = Transformer.from_crs(4326, 32652, always_xy=True)
    oE, oN = tr.transform(cam["lon"], cam["lat"])
    out = {
        "meta": {"location_code": args.loc, "note": f"자동 캘리브레이션(VP+IPM) pitch{pitch:.1f} yaw{yaw:.1f} bearing{beta:.1f} Hc{args.cam_height}"},
        "inputs": {"cctv_path": f"cctv_{args.loc}.png"},
        "undistort": {"resolution": [W_, H_], "K": [[f_est, 0, W_ / 2], [0, f_est, H_ / 2], [0, 0, 1]], "D": [0, 0, 0, 0, 0]},
        "homography": {"H": Hfin.tolist(), "fov_polygon": [], "anchors_list": []},
        "parallax": {"x_cam_coords_sat": 0, "y_cam_coords_sat": 0, "z_cam_meters": args.cam_height, "px_per_meter": 1},
        "world": {"epsg": 32652, "origin_easting": float(oE), "origin_northing": float(oN), "ground_ellipsoid_h": 28.0},
        "use_svg": False, "use_roi": False, "ref_method": "center_bottom_side", "proj_method": "down_h",
    }
    dest = os.path.join(ldir, f"G_projection_{args.loc}{'' if args.save else '_auto'}.json")
    if args.save and os.path.exists(dest):  # 기존(예: GCP) 캘리브레이션 백업 후 덮어쓰기
        import shutil
        shutil.copy(dest, dest + ".bak")
        print(f"  기존 캘리브레이션 백업 → {os.path.basename(dest)}.bak")
    json.dump(out, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"저장 → {os.path.relpath(dest, REPO)}  (원점 UTM {oE:.1f},{oN:.1f}, bearing {beta:.1f}°)")
    if not args.save:
        print("  (--save 로 실제 G_projection 교체. 이후 reproject.py → bridge_to_webmap)")


if __name__ == "__main__":
    main()
