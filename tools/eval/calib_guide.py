"""
캘리브레이션 재작업 가이드(CLI) — 기존 GCP를 진단하고 '무엇을 어떻게 다시 찍을지' 지시.

진단 항목
  1) 점 개수/잉여도 : 4점은 호모그래피 자유도(8)와 동수 → **적합 잔차가 항상 0**(검증 불가)
  2) 분포          : 화면 세로(=깊이) 커버리지가 좁으면 깊이 스케일이 틀어짐(우리 실패 모드)
  3) LOO 교차검증  : 한 점을 빼고 예측 → 진짜 정확도(m). 5점 이상에서만 산출
  4) 기하 타당성   : 검출 박스 크기 vs 캘리브레이션 예측 크기(check_calibration.py 재사용)

사용:
  python tools/eval/calib_guide.py --loc PANGYO_2
  python tools/eval/calib_guide.py --all
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_calibration import check as geom_check


def dlt(src, dst):
    A, b = [], []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -x * X, -y * X]); b.append(X)
        A.append([0, 0, 0, x, y, 1, -x * Y, -y * Y]); b.append(Y)
    h, *_ = np.linalg.lstsq(np.array(A, float), np.array(b, float), rcond=None)
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])


def apply_h(H, x, y):
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return np.array([(H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w,
                     (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w])


def loo_rms(src, dst):
    if len(src) < 5:
        return None
    errs = []
    for k in range(len(src)):
        s = [p for i, p in enumerate(src) if i != k]
        d = [p for i, p in enumerate(dst) if i != k]
        try:
            H = dlt(s, d)
        except Exception:
            continue
        errs.append(np.linalg.norm(apply_h(H, *src[k]) - np.array(dst[k])))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else None


def guide(loc):
    gp = os.path.join(REPO, "location", loc, f"G_projection_{loc}.json")
    if not os.path.exists(gp):
        return None
    gd = json.load(open(gp, encoding="utf-8"))
    W, H = gd["undistort"]["resolution"]
    anchors = [a for a in (gd.get("homography", {}).get("anchors_list") or []) if a.get("px") and a.get("lonlat")]
    print(f"\n{'=' * 62}\n[{loc}]  해상도 {W}x{H} · GCP {len(anchors)}점")

    if not anchors:
        print("  GCP 없음(placeholder/자동 캘리브레이션) → 아래 '재작업 절차'대로 6~8점 찍으세요")
    else:
        px = np.array([a["px"] for a in anchors], float)
        from pyproj import Transformer
        tr = Transformer.from_crs(4326, gd["world"]["epsg"], always_xy=True)
        utm = np.array([tr.transform(*a["lonlat"]) for a in anchors], float)
        loc_m = utm - utm.mean(axis=0)
        v = (px[:, 1].max() - px[:, 1].min()) / H
        h = (px[:, 0].max() - px[:, 0].min()) / W
        rms = loo_rms([tuple(p) for p in px], [tuple(p) for p in loc_m])
        print(f"  분포: 세로(깊이) {v * 100:.0f}% · 가로 {h * 100:.0f}% · 잉여 {len(anchors) - 4:+d}")
        print(f"  LOO 교차검증 정확도: " + ("산출 불가(5점 이상 필요)" if rms is None else f"{rms:.2f} m"))
        if len(anchors) <= 4:
            print("  ⚠ 4점 이하 → 적합 잔차가 수학적으로 항상 0. 'RMS 0.00m'은 정확도가 아님")
        if v < 0.35:
            print(f"  ⚠ 세로 분포 {v * 100:.0f}% (권장 50%+) → **깊이 스케일 오류의 직접 원인**")
        # 비어있는 구역
        rows = set()
        for p in px:
            r = int(min(2, max(0, (p[1] / H - 0.15) // 0.30)))
            rows.add(r)
        miss = [n for i, n in enumerate(["먼 거리(위쪽)", "중간", "가까운 거리(아래쪽)"]) if i not in rows]
        if miss:
            print(f"  ⚠ 비어있는 깊이 구간: {', '.join(miss)}")

    g = geom_check(loc, verbose=False)
    if g:
        verdict = "✓ 통과" if g["ok"] else "✗ 불합격"
        print(f"  기하 타당성(박스 크기 대조): 세로/가로 {g['aniso']:.2f} → {verdict}")
        if not g["ok"] and g["aniso"] > 2.5:
            print("    → 실제 차량이 예측보다 '세로로 김' = 깊이가 과도하게 압축됨")

    print("""
  재작업 절차
   1) http://localhost:5174/calibrate.html 에서 카메라 선택 (기존 GCP 자동 로드)
   2) [현재 모드 지우기]로 기존 GCP 삭제 후 다시 찍기
   3) 🎯 권장 구역 오버레이의 **핑크 구역이 사라질 때까지** 클릭
      - 반드시 '먼 거리(화면 위쪽)'와 '가까운 거리(아래쪽)' 모두 포함 → 깊이 스케일 확보
      - 좌우로도 벌리기(가로 50%+)
   4) 목표: 점 6~10개 · 세로 분포 50%+ · **기하 타당성 ✓**(가장 중요)
      ※ LOO는 원거리 점에서 본래 크게 나온다(1px≈수 m). 저조도/원거리 카메라는 수십 m도 정상.
        LOO만 보고 점을 지우면 깊이를 고정하는 원거리 점이 사라져 오히려 악화된다.
   5) 저장 후 📏 투영 격자를 켜서 차로선(3.5m)·깊이 눈금(10m)이 실제 도로와 맞는지 육안 확인
   6) 반영: python tools/reproject.py --loc {LOC}
            python tools/eval/check_calibration.py --loc {LOC}   ← ✓ 나와야 함
            python tools/bridge_to_webmap.py --replay output/reprojected/{LOC}/clip.json.gz --loc {LOC}
   ※ 찍기 좋은 지점: 정지선 끝, 차선 점선의 시작/끝, 횡단보도 모서리, 노면 화살표 꼭짓점
      (위성에서도 같은 지점을 정확히 찍을 수 있는 '점' 특징만 사용 — 차량·그림자는 금지)
""".replace("{LOC}", loc))
    return True


def per_point_loo(px, loc_m):
    """점별 Leave-One-Out 예측오차(m) 배열."""
    errs = []
    for k in range(len(px)):
        s = [tuple(p) for i, p in enumerate(px) if i != k]
        d = [tuple(p) for i, p in enumerate(loc_m) if i != k]
        try:
            H = dlt(s, d)
            errs.append(float(np.linalg.norm(apply_h(H, *px[k]) - np.array(loc_m[k]))))
        except Exception:
            errs.append(float("inf"))
    return np.array(errs)


def fix(loc, target=1.5, min_points=6):
    """이상치 GCP를 하나씩 제거하며 재적합(원본은 .bak 백업). 잘못 찍힌 점 1~2개가
    전체 호모그래피를 망치는 경우가 흔하므로, LOO가 목표 이하가 될 때까지 반복한다."""
    import shutil
    from pyproj import Transformer
    gp = os.path.join(REPO, "location", loc, f"G_projection_{loc}.json")
    gd = json.load(open(gp, encoding="utf-8"))
    A = [a for a in (gd.get("homography", {}).get("anchors_list") or []) if a.get("px") and a.get("lonlat")]
    if len(A) < min_points + 1:
        print(f"[{loc}] 점 {len(A)}개 — 제거 여지 없음(최소 {min_points}점 유지)"); return
    tr = Transformer.from_crs(4326, gd["world"]["epsg"], always_xy=True)

    def rms_of(items):
        px = [tuple(a["px"]) for a in items]
        utm = np.array([tr.transform(*a["lonlat"]) for a in items], float)
        lm = [tuple(p) for p in (utm - utm.mean(axis=0))]
        return loo_rms(px, lm), px, lm

    cur = list(A)
    r0, _, _ = rms_of(cur)
    print(f"[{loc}] 시작 {len(cur)}점 · LOO {r0:.2f} m")
    removed = []
    while len(cur) > min_points:
        r, px, lm = rms_of(cur)
        if r is not None and r <= target:
            break
        errs = per_point_loo(px, lm)
        k = int(np.argmax(errs))
        removed.append((k, cur[k], float(errs[k])))
        print(f"   제거: px({cur[k]['px'][0]:.0f},{cur[k]['px'][1]:.0f}) LOO오차 {errs[k]:.1f} m")
        cur.pop(k)
    r1, px, lm = rms_of(cur)
    print(f"   → {len(cur)}점 · LOO {r1:.2f} m  (제거 {len(removed)}개)")
    if r1 is None or (r0 is not None and r1 >= r0):
        print("   개선 없음 → 저장하지 않음"); return

    # 남은 점으로 H·world 재계산
    utm = np.array([tr.transform(*a["lonlat"]) for a in cur], float)
    oE, oN = utm.mean(axis=0)
    H = dlt([tuple(a["px"]) for a in cur], [tuple(p) for p in (utm - [oE, oN])])
    # ⚠ 안전장치: LOO만 보고 점을 지우면 **원거리 점**이 먼저 제거되는데,
    # 그 점들이야말로 깊이 스케일을 고정한다. 기하 타당성이 나빠지면 저장하지 않는다.
    import copy as _copy
    from check_calibration import check as _geom
    before_geom = _geom(loc, verbose=False)
    shutil.copy(gp, gp + ".bak")
    gd["homography"]["H"] = H.tolist()
    gd["homography"]["anchors_list"] = cur
    gd["world"]["origin_easting"] = float(oE); gd["world"]["origin_northing"] = float(oN)
    gd["meta"]["note"] = f"GCP {len(cur)}점(이상치 {len(removed)}개 제거) LOO {r1:.2f}m"
    json.dump(gd, open(gp, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    after_geom = _geom(loc, verbose=False)
    if before_geom and after_geom and (after_geom["ok"] is False and before_geom["ok"] is True
                                       or abs(np.log(max(after_geom["ratio_w"], 1e-6)))
                                          > abs(np.log(max(before_geom["ratio_w"], 1e-6))) + 0.7):
        shutil.copy(gp + ".bak", gp)
        print("   ✗ 기하 타당성이 악화되어 되돌림(원거리 점 제거는 깊이 스케일을 망가뜨립니다)")
        print("      → 자동 제거 대신 calibrate.html 에서 해당 점만 다시 찍으세요")
        return
    print(f"   저장 완료(원본 백업: {os.path.basename(gp)}.bak)")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fix", action="store_true", help="이상치 GCP 제거 후 재적합(.bak 백업)")
    ap.add_argument("--target", type=float, default=1.5, help="목표 LOO(m)")
    a = ap.parse_args()
    if a.fix:
        fix(a.loc, target=a.target)
        return
    locs = ([os.path.basename(p) for p in sorted(glob.glob(os.path.join(REPO, "location", "*")))]
            if a.all else [a.loc])
    for loc in locs:
        if loc:
            guide(loc)


if __name__ == "__main__":
    main()
