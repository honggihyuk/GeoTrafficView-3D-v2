"""
GT 라벨링 가능 구간 진단 — "이 카메라에서 어디까지 라벨해야 의미가 있는가".

**왜 필요한가**
  오블리크 CCTV는 화면 위쪽으로 갈수록 객체가 급격히 작아진다(실측 PANGYO_2:
  근거리 30x27px → 중거리 20x14px → 원거리 11x8px). 작은 객체를 무리해서 라벨하면
  **주석 오차가 객체 크기와 맞먹어**, 그 GT로 낸 MOTA/IDF1은 모델이 아니라 주석 잡음을
  측정하게 된다. 라벨링에 사람 시간을 쓰기 전에 어디가 유효한지 먼저 알아야 한다.

**판정 기준 — 실측 기반(산술 아님)**
  IoU 산술로만 따지면 8px 객체도 "±2px 오차면 IoU 0.5 통과"라는 답이 나온다. 그러나 실제
  한계는 IoU가 아니라 **육안 식별성**이다. PANGYO_2를 실제로 라벨하며 측정한 결과:
    최소변 ≳24px : 배경차분 제안 + 육안 검증이 성립. 실제로 이 구간만 GT를 만들었다.
    12~24px      : 박스는 그릴 수 있으나 배경차분이 인접 차량을 병합하고 상시 점유 차로에서는
                   아예 못 잡는다(f15~f27에서 차량이 또렷한데 제안 0개). 순수 수작업 필요.
    ≲12px        : 6배 확대해도 객체 경계와 차량/노면표시 구분이 안 된다. 라벨 불가.
  이 경계는 720x480 오블리크 CCTV 1대에서 얻은 값이므로, 다른 화질·화각에서는 다시 재야 한다.

  추가로 **배경차분 반자동 제안이 통하는지**도 본다. 상시 점유된 차로는 배경 중앙값에
  차량이 섞여 움직이는 차가 배경과 구분되지 않는다(실측: PANGYO_2 좌측 차도 f15~f27에서
  차량이 또렷이 보이는데 제안이 0개였다). 이 구간은 반자동이 불가하고 순수 수작업이다.

사용:
  python tools/eval/label_feasibility.py --loc PANGYO_2
  python tools/eval/label_feasibility.py --all
"""
import argparse
import glob
import gzip
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PANGYO_2 라벨링 실측에서 얻은 경계(위 독스트링 참고). 산술이 아니라 관측값이다.
SEMIAUTO_PX = 24.0      # 이 이상: 배경차분 제안 + 육안 검증으로 GT 생성 가능
MANUAL_PX = 12.0        # 이 이상: 수작업으로는 가능(반자동 불가)


def analyze(loc, bands=6):
    p = os.path.join(REPO, "webmap", "public", "data", "replay", f"{loc.lower()}.json.gz")
    if not os.path.exists(p):
        return None
    d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    W, H = d["meta"]["resolution"]
    sz, cy = [], []
    for f in d["frames"]:
        for o in f.get("objects", []):
            b = o.get("bbox_2d")
            if b:
                sz.append([b[2] - b[0], b[3] - b[1]])
                cy.append((b[1] + b[3]) / 2)
    if not sz:
        return None
    sz, cy = np.array(sz, float), np.array(cy, float)
    rows = []
    edges = np.linspace(0, H, bands + 1)
    for lo, hi in zip(edges, edges[1:]):
        m = (cy >= lo) & (cy < hi)
        if m.sum() < 10:
            continue
        mins = sz[m].min(axis=1)
        med = float(np.median(mins))
        rows.append({"y": [int(lo), int(hi)], "n": int(m.sum()),
                     "median_wh": [float(np.median(sz[m, 0])), float(np.median(sz[m, 1]))],
                     "median_min_side": med,
                     "p10_min_side": float(np.percentile(mins, 10)),
                     "semiauto": med >= SEMIAUTO_PX, "manual": med >= MANUAL_PX})
    return {"loc": loc, "resolution": [W, H], "n_det": int(len(sz)),
            "thresholds_px": {"semiauto": SEMIAUTO_PX, "manual": MANUAL_PX},
            "threshold_provenance": "PANGYO_2 라벨링 실측(720x480 오블리크 CCTV)",
            "bands": rows}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--bands", type=int, default=6)
    args = ap.parse_args()

    locs = ([os.path.basename(p)[:-8].upper()
             for p in sorted(glob.glob(os.path.join(REPO, "webmap", "public", "data",
                                                    "replay", "*.json.gz")))]
            if args.all else [args.loc])
    if not locs or locs == [None]:
        sys.exit("--loc 또는 --all 이 필요합니다")

    print(f"판정 경계(PANGYO_2 실측): 최소변 ≥{SEMIAUTO_PX:.0f}px 반자동+검증 가능 · "
          f"≥{MANUAL_PX:.0f}px 수작업만 가능 · 그 미만 라벨 불가\n")

    out = []
    for loc in locs:
        r = analyze(loc, args.bands)
        if not r:
            print(f"[{loc}] replay 없음 또는 검출 없음")
            continue
        out.append(r)
        print(f"[{loc}] {r['resolution'][0]}x{r['resolution'][1]} · 검출 {r['n_det']:,}개")
        print(f"  {'y 밴드':>12s} {'검출':>6s} {'박스중앙':>9s} {'최소변중앙':>9s} {'p10':>5s}  판정")
        for b in r["bands"]:
            v = ("반자동+검증 가능" if b["semiauto"] else
                 ("수작업만 가능(반자동 불가)" if b["manual"] else "라벨 불가 — 경계 식별 안 됨"))
            print(f"  {str(b['y']):>12s} {b['n']:6d} "
                  f"{b['median_wh'][0]:4.0f}x{b['median_wh'][1]:<4.0f} "
                  f"{b['median_min_side']:9.1f} {b['p10_min_side']:5.1f}  {v}")
        sa = sum(b["n"] for b in r["bands"] if b["semiauto"]) / max(r["n_det"], 1)
        mn = sum(b["n"] for b in r["bands"] if b["manual"]) / max(r["n_det"], 1)
        print(f"  → 반자동 가능 {100*sa:.0f}% · 수작업까지 포함 {100*mn:.0f}% "
              f"· 라벨 불가 {100*(1-mn):.0f}%\n")

    p = os.path.join(REPO, "output", "eval", "label_feasibility.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"→ {os.path.relpath(p, REPO)}")
    print("\n주의: 이 표는 '박스를 그릴 수 있는가'만 본다. 상시 점유된 차로는 배경차분 "
          "반자동 제안이 통하지 않아(배경 중앙값에 차량이 섞임) 순수 수작업이 되며, "
          "그 경우 비용이 몇 배로 뛴다.")


if __name__ == "__main__":
    main()
