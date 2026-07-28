"""
캘리브레이션 타당성 자동 점검 — GT 없이 '기하가 말이 되는지' 검사.

원리: 캘리브레이션이 맞다면 **검출된 차량의 픽셀 박스 크기**가
      **그 위치에 실제 차량(약 1.8m × 4.2m)을 놓고 투영한 크기**와 비슷해야 한다.
      어긋나면 호모그래피의 스케일(특히 깊이 방향)이 틀린 것이다.

이 점검이 필요한 이유: GCP 4점은 호모그래피 자유도(8)와 정확히 같아 **잔차가 항상 0**이다.
  → "RMS 0.00m"은 정확도가 아니라 검증 불가를 뜻한다. 6~8점을 찍어야 잔차가 의미를 갖는다.

사용:
  python tools/eval/check_calibration.py --loc PANGYO_2
  python tools/eval/check_calibration.py --all
"""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from trafficlab.projection.g_projection import GProjection

CAR_W, CAR_L = 1.8, 4.2   # 대표 승용차 치수(m)


def check(loc, max_frames=40, verbose=True):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    gp = os.path.join(REPO, "location", loc, f"G_projection_{loc}.json")
    if not os.path.exists(gp):
        return None
    gd = json.load(open(gp, encoding="utf-8"))
    g = GProjection(gd, base_dir=os.path.dirname(gp))
    n_anchor = len(gd.get("homography", {}).get("anchors_list", []) or [])

    preds = glob.glob(os.path.join(REPO, "output", "**", loc, "*.json.gz"), recursive=True)
    if not preds:
        return None
    d = json.load(gzip.open(max(preds, key=os.path.getsize), "rt", encoding="utf-8"))

    rw, rh, pw, ph = [], [], [], []
    for fr in d["frames"][:max_frames]:
        for o in fr.get("objects", []):
            b = o.get("bbox_2d")
            if not b or str(o.get("class", "")).lower() not in ("car", "van"):
                continue
            w, h = b[2] - b[0], b[3] - b[1]
            if w < 4 or h < 4:
                continue
            # 이 검출의 지면 위치에 표준 차량을 놓고 투영 → 기대 박스
            r = g.get_ground_contact_from_box((b[0], b[1], w, h), 1.55,
                                              ref_method="center_bottom_side", proj_method="down_h")
            x, y = r["sat_coords"]
            pts = [g.sat_to_cctv(x + dx, y + dy, h=0) for dx, dy in
                   [(-CAR_W / 2, -CAR_L / 2), (CAR_W / 2, -CAR_L / 2),
                    (CAR_W / 2, CAR_L / 2), (-CAR_W / 2, CAR_L / 2)]]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ew, eh = max(xs) - min(xs), max(ys) - min(ys)
            if ew <= 0 or eh <= 0:
                continue
            rw.append(w); rh.append(h); pw.append(ew); ph.append(eh)

    if len(rw) < 20:
        return None
    rw, rh, pw, ph = map(np.array, (rw, rh, pw, ph))
    ratio_w = float(np.median(rw / pw))
    ratio_h = float(np.median(rh / ph))
    # 판정: 세로/가로 비율이 크게 어긋나면 깊이 스케일 오류
    aniso = ratio_h / max(ratio_w, 1e-9)
    ok = (0.5 <= ratio_w <= 2.0) and (0.4 <= aniso <= 2.5)
    if verbose:
        print(f"[{loc}] GCP {n_anchor}점 · 표본 {len(rw)}")
        print(f"   실제 박스 중앙값  {np.median(rw):5.1f} x {np.median(rh):5.1f} px")
        print(f"   기대(캘리브) 크기 {np.median(pw):5.1f} x {np.median(ph):5.1f} px")
        print(f"   비율  가로 {ratio_w:4.2f}배 · 세로 {ratio_h:4.2f}배 · 세로/가로 {aniso:4.2f}")
        if n_anchor and n_anchor <= 4:
            print(f"   ⚠ GCP {n_anchor}점 = 호모그래피 자유도(8)와 동수 → 잔차가 항상 0(검증 불가). 6~8점 권장")
        if not ok:
            if aniso > 2.5:
                print("   ⚠ 세로가 예상보다 과대 → **깊이 스케일 과소**(원근이 지나치게 압축됨). "
                      "GCP를 화면 상·하단으로 넓게 분산해 다시 찍으세요")
            elif aniso < 0.4:
                print("   ⚠ 세로가 예상보다 과소 → 깊이 스케일 과대")
            else:
                print("   ⚠ 전체 스케일 불일치 → GCP 위치/대응 확인 필요")
        else:
            print("   ✓ 기하 타당성 통과")
    return dict(loc=loc, anchors=n_anchor, n=len(rw), ratio_w=ratio_w, ratio_h=ratio_h,
                aniso=aniso, ok=bool(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    locs = ([os.path.basename(p) for p in sorted(glob.glob(os.path.join(REPO, "location", "*")))]
            if a.all else [a.loc])
    res = []
    for loc in locs:
        if not loc:
            continue
        r = check(loc)
        if r:
            res.append(r)
    if a.all and res:
        print("\n요약 (ok=기하 타당):")
        for r in res:
            print(f"  {r['loc']:16s} GCP {r['anchors']:2d}점  세로/가로 {r['aniso']:4.2f}  "
                  f"{'✓' if r['ok'] else '✗ 재캘리브레이션 필요'}")


if __name__ == "__main__":
    main()
