"""
정적 오탐 마스크 — 영상에 박힌 안내문구/로고를 검출 대상에서 제외한다.

**문제**
  ITS/TOPIS 영상은 지점명·방면·거리("서울", "판교JC 1.0", "407")를 인코딩 전에 합성한다.
  YOLO는 이 글자 덩어리를 차량으로 잡는다(실측 PANGYO_2: 고정 위치 반복 검출이 전체 검출의
  11.8%, 그중 (597,312) 28x17px는 120프레임 중 83프레임에 등장). 이 오탐은
  - 정밀도를 깎고(MOTA 음수의 주원인),
  - 항상 같은 자리에 있으니 '정지 차량' 트랙을 만들어 통계를 오염시킨다.

**정지 차량과 어떻게 구분하나 (핵심)**
  둘 다 같은 자리에 계속 검출된다. 구분 근거는 **픽셀이 프레임마다 얼마나 변하는가**이다.
  박힌 문구는 합성된 평면 그래픽이라 인코더가 거의 그대로 복사해 시간축 표준편차가 낮다.
  실제 차량은 정지해 있어도 센서 잡음·조도 변화·주변 차의 그림자로 값이 흔들린다.
  다만 이 신호만으로 자동 확정하지 않는다 — **카메라당 1회 설정**이므로 후보를 제시하고
  사람이 크롭 이미지를 보고 승인하는 절차(--review)를 둔다. 정지 차량을 지워버리면
  조용히 데이터가 사라지기 때문이다.

사용:
  python tools/static_mask.py --loc PANGYO_2 --review        # 후보 제시(크롭 이미지 저장)
  python tools/static_mask.py --loc PANGYO_2 --accept 0 1 3  # 승인한 후보만 마스크로 저장
  python tools/static_mask.py --loc PANGYO_2 --apply output/mapmatch/PANGYO_2/clip.json.gz
출력:
  location/<LOC>/static_mask.png   (흰색 = 제외 영역, pipeline.py 가 자동으로 읽는다)
  output/staticmask/<LOC>/candidates.png
"""
import argparse
import glob
import gzip
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mask_path(loc):
    return os.path.join(REPO, "location", loc, "static_mask.png")


def load_mask(loc, shape=None):
    """(H,W) bool 마스크 or None. shape 주면 그 크기로 맞춘다."""
    p = mask_path(loc)
    if not os.path.exists(p):
        return None
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if shape and m.shape[:2] != tuple(shape):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def box_is_masked(bbox, mask, frac=0.6):
    """박스의 frac 이상이 마스크에 덮이면 정적 오탐으로 본다.

    중심점만 보면 글자 옆을 스치는 실제 차량까지 지워진다. 면적 비율이 안전하다.
    """
    h, w = mask.shape
    x1 = max(0, min(w - 1, int(bbox[0]))); y1 = max(0, min(h - 1, int(bbox[1])))
    x2 = max(0, min(w, int(bbox[2]))); y2 = max(0, min(h, int(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return False
    sub = mask[y1:y2, x1:x2]
    return sub.mean() >= frac


# ---------- 후보 탐색 ----------
def temporal_std(video, max_frames=200):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"영상을 열 수 없습니다: {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    step = max(1, n // max_frames)
    acc = acc2 = None
    k = 0
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            acc = g if acc is None else acc + g
            acc2 = g * g if acc2 is None else acc2 + g * g
            k += 1
        i += 1
    cap.release()
    if k < 5:
        sys.exit(f"프레임이 {k}개뿐입니다.")
    mean = acc / k
    return np.sqrt(np.maximum(acc2 / k - mean * mean, 0.0)), mean, k


def persistence(replay, shape):
    """픽셀별 '검출 박스에 덮인 프레임 비율'."""
    d = json.load(gzip.open(replay, "rt", encoding="utf-8"))
    hit = np.zeros(shape, np.float32)
    nf = 0
    for fr in d["frames"]:
        nf += 1
        seen = np.zeros(shape, bool)
        for o in fr.get("objects", []):
            b = o.get("bbox_2d")
            if not b:
                continue
            x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
            x2, y2 = min(shape[1], int(b[2])), min(shape[0], int(b[3]))
            if x2 > x1 and y2 > y1:
                seen[y1:y2, x1:x2] = True
        hit += seen
    return hit / max(nf, 1), nf


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--std", type=float, default=4.0, help="시간축 표준편차가 이보다 낮으면 정적")
    ap.add_argument("--persist", type=float, default=0.4, help="이 비율 이상 프레임에서 검출에 덮이면 후보")
    ap.add_argument("--min-area", type=int, default=150, help="후보 최소 면적(px²)")
    ap.add_argument("--review", action="store_true", help="후보만 제시하고 저장하지 않음")
    ap.add_argument("--accept", nargs="*", type=int, default=None, help="승인할 후보 번호")
    ap.add_argument("--apply", default=None, help="이 replay 에서 마스크에 덮인 검출을 제거")
    ap.add_argument("--cover", type=float, default=0.6, help="박스가 이 비율 이상 덮이면 제거")
    ap.add_argument("--fill-bbox", action="store_true", default=True,
                    help="글자 덩어리의 바운딩 박스를 채워 마스크로 쓴다(기본)")
    ap.add_argument("--no-fill-bbox", dest="fill_bbox", action="store_false")
    args = ap.parse_args()

    # ---- 적용 모드 ----
    if args.apply:
        d = json.load(gzip.open(args.apply, "rt", encoding="utf-8"))
        shape = tuple(d["meta"]["resolution"][::-1])
        m = load_mask(args.loc, shape)
        if m is None:
            sys.exit(f"마스크 없음: {mask_path(args.loc)}\n  → 먼저 --review 로 후보를 확인하고 --accept 하세요")
        n_all = n_rm = 0
        for fr in d["frames"]:
            keep = []
            for o in fr.get("objects", []):
                n_all += 1
                b = o.get("bbox_2d")
                if b and box_is_masked(b, m, args.cover):
                    n_rm += 1
                    continue
                keep.append(o)
            fr["objects"] = keep
        out = os.path.join(os.path.dirname(args.apply), "masked.json.gz")
        with gzip.open(out, "wt", encoding="utf-8") as f:
            json.dump(d, f)
        print(f"검출 {n_all}개 중 {n_rm}개 제거 ({100 * n_rm / max(n_all,1):.1f}%)")
        print(f"→ {os.path.relpath(out, REPO)}")
        return

    # ---- 후보 탐색 ----
    video = args.video or os.path.join(REPO, "webmap", "public", "data", "footage", f"{args.loc.lower()}.mp4")
    replay = args.replay or os.path.join(REPO, "webmap", "public", "data", "replay", f"{args.loc.lower()}.json.gz")
    if not os.path.exists(video):
        sys.exit(f"영상 없음: {video}")
    if not os.path.exists(replay):
        sys.exit(f"replay 없음: {replay}")

    std, mean, k = temporal_std(video)
    pers, nf = persistence(replay, std.shape)
    print(f"영상 {std.shape[1]}x{std.shape[0]} · 샘플 {k}프레임 · replay {nf}프레임")
    print(f"시간축 표준편차: 중앙값 {np.median(std):.2f} · 5%={np.percentile(std,5):.2f} · "
          f"95%={np.percentile(std,95):.2f}")

    cand = ((std < args.std) & (pers > args.persist)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(cand, 8)

    regions = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < args.min_area:
            continue
        sel = lab[y:y + h, x:x + w] == i
        regions.append({"idx": len(regions), "bbox": [int(x), int(y), int(x + w), int(y + h)],
                        "area": int(a), "std": float(std[y:y + h, x:x + w][sel].mean()),
                        "persist": float(pers[y:y + h, x:x + w][sel].mean()), "_label": i})
    regions.sort(key=lambda r: -r["area"])
    for i, r in enumerate(regions):
        r["idx"] = i

    print(f"\n정적 오탐 후보 {len(regions)}개 (std<{args.std} · 지속>{args.persist})")
    print(f"{'#':>2s} {'bbox':>24s} {'면적':>6s} {'std':>6s} {'지속':>6s}")
    for r in regions:
        print(f"{r['idx']:2d} {str(r['bbox']):>24s} {r['area']:6d} {r['std']:6.2f} {r['persist']:6.2f}")

    # 후보를 크롭해 사람이 볼 수 있게 저장 — 정지 차량을 지우는 사고를 막는 장치
    qc = os.path.join(REPO, "output", "staticmask", args.loc)
    os.makedirs(qc, exist_ok=True)
    cap = cv2.VideoCapture(video); ok, frame0 = cap.read(); cap.release()
    if ok and regions:
        tiles = []
        for r in regions:
            x1, y1, x2, y2 = r["bbox"]
            pad = 8
            c = frame0[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            c = cv2.resize(c, (240, 140), interpolation=cv2.INTER_CUBIC)
            cv2.putText(c, f"#{r['idx']} std{r['std']:.1f} p{r['persist']:.2f}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(c)
        rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4])))
                for i in range(0, len(tiles), 4)]
        cv2.imwrite(os.path.join(qc, "candidates.png"), np.vstack(rows))
        ov = frame0.copy()
        for r in regions:
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(ov, str(r["idx"]), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(qc, "overlay.png"), ov)
        print(f"\n→ {os.path.relpath(qc, REPO)}/candidates.png · overlay.png  (승인 전 반드시 확인)")

    json.dump(regions, open(os.path.join(qc, "candidates.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)

    if args.review or args.accept is None:
        print("\n크롭을 확인한 뒤 승인할 번호를 지정하세요:")
        print(f"  python tools/static_mask.py --loc {args.loc} --accept " +
              " ".join(str(r["idx"]) for r in regions))
        print("  (정지 차량을 승인하면 그 차량이 데이터에서 조용히 사라집니다)")
        return

    # ---- 승인분만 마스크로 저장 ----
    keep = set(args.accept)
    m = np.zeros(std.shape, np.uint8)
    for r in regions:
        if r["idx"] not in keep:
            continue
        if args.fill_bbox:
            # 글자 '획' 모양 그대로 두면 검출 박스의 40~60%만 덮여 --cover 기준을 못 넘긴다.
            # 글자 덩어리의 바운딩 박스를 채운다. 그래도 실제 차량 박스는 대개 더 커서
            # --cover 비율을 못 채우므로 살아남는다(반드시 GT로 확인할 것).
            x1, y1, x2, y2 = r["bbox"]
            m[y1:y2, x1:x2] = 255
        else:
            m[lab == r["_label"]] = 255
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))     # 글자 외곽 여유
    os.makedirs(os.path.dirname(mask_path(args.loc)), exist_ok=True)
    cv2.imwrite(mask_path(args.loc), m)
    print(f"\n승인 {len(keep)}개 → 마스크 픽셀 {int((m>0).sum()):,} ({100*(m>0).mean():.2f}% of frame)")
    print(f"→ {os.path.relpath(mask_path(args.loc), REPO)}")
    print(f"다음: python tools/static_mask.py --loc {args.loc} --apply <replay.json.gz>")


if __name__ == "__main__":
    main()
