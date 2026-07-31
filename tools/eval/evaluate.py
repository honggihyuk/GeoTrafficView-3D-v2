"""
평가: GT 라벨(eval/<loc>/labels.json) vs 추론 결과(.json.gz)

검출 지표: Precision / Recall / F1 / AP50 (IoU 0.5 그리디 매칭, conf 내림차순)
추적 지표: MOTA, IDF1, ID Switch, MT/ML, FP/FN  (CLEAR-MOT / IDF1 정의 직접 구현)
  - MOTA = 1 - (FN + FP + IDSW) / GT
  - IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN),  ID 매칭은 헝가리안(scipy)으로 전역 최적
  - 라벨된 프레임만 평가(부분 라벨링 지원)

사용:
  python tools/eval/evaluate.py --loc PANGYO_2
  python tools/eval/evaluate.py --loc PANGYO_2 --pred output/lanesnap/PANGYO_2/clip.json.gz
  python tools/eval/evaluate.py --loc PANGYO_2 --pred A.json.gz --pred B.json.gz   # A/B 비교
"""
import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IOU_TH = 0.5   # set_iou()로 변경 가능(소형 객체는 0.3 권장)


def set_iou(v):
    global IOU_TH
    IOU_TH = float(v)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def in_roi(bb, roi):
    """박스 중심이 ROI 안인지. GT와 예측에 **같은 기준**을 적용해야 공정하다."""
    if roi is None:
        return True
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def load_gt(loc):
    p = os.path.join(REPO, "eval", loc, "labels.json")
    if not os.path.exists(p):
        sys.exit(f"GT 없음: {p}\n  → tools/eval/extract_frames.py 후 label.html 에서 라벨링하세요")
    d = json.load(open(p, encoding="utf-8"))
    gt = defaultdict(list)
    for b in d.get("boxes", []):
        gt[int(b["frame"])].append({"id": int(b["track_id"]), "cls": b["class"], "bbox": b["bbox"]})
    return d, gt


def load_pred(path, frames, roi=None):
    d = json.load(gzip.open(path, "rt", encoding="utf-8"))
    pr = defaultdict(list)
    for fr in d["frames"]:
        i = int(fr["frame_index"])
        if i not in frames:
            continue
        for o in fr.get("objects", []):
            bb = o.get("bbox_2d")
            if not bb:
                continue
            bb = [float(v) for v in bb]
            if not in_roi(bb, roi):
                continue          # ROI 밖 예측은 FP로 세지 않는다(GT도 ROI 안만 있으므로)
            pr[i].append({"id": o.get("tracked_id"), "cls": o.get("class"),
                          "bbox": bb, "conf": float(o.get("confidence") or 0)})
    return pr


def detection_metrics(gt, pr, frames):
    """클래스 무관(class-agnostic) 검출 평가 — GT 클래스 체계와 모델 클래스가 달라도 유효."""
    recs = []
    n_gt = sum(len(gt[f]) for f in frames)
    for f in frames:
        G = gt[f]; P = sorted(pr.get(f, []), key=lambda x: -x["conf"])
        used = set()
        for p in P:
            best, bi = 0.0, -1
            for j, g in enumerate(G):
                if j in used:
                    continue
                v = iou(p["bbox"], g["bbox"])
                if v > best:
                    best, bi = v, j
            if best >= IOU_TH:
                used.add(bi); recs.append((p["conf"], 1))
            else:
                recs.append((p["conf"], 0))
    if not recs or n_gt == 0:
        return dict(P=0, R=0, F1=0, AP50=0, n_gt=n_gt, n_pred=len(recs))
    recs.sort(key=lambda x: -x[0])
    tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([1 - r[1] for r in recs])
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-9)
    ap = 0.0  # 101-point interpolation (COCO 방식)
    for t in np.linspace(0, 1, 101):
        m = prec[rec >= t]
        ap += (m.max() if m.size else 0.0) / 101
    P, R = float(prec[-1]), float(rec[-1])
    return dict(P=P, R=R, F1=2 * P * R / max(P + R, 1e-9), AP50=float(ap),
                n_gt=n_gt, n_pred=len(recs))


def mot_metrics(gt, pr, frames):
    """CLEAR-MOT(MOTA) + IDF1."""
    FP = FN = IDSW = 0
    n_gt = 0
    prev_match = {}                     # gt_id -> pred_id (직전 프레임 매칭)
    gt_len = defaultdict(int); gt_hit = defaultdict(int)
    pair = defaultdict(int)             # (gt_id, pred_id) -> 매칭 프레임 수
    n_pred_total = 0

    for f in frames:
        G = gt[f]; P = pr.get(f, [])
        n_gt += len(G); n_pred_total += len(P)
        for g in G:
            gt_len[g["id"]] += 1
        if G and P:
            C = np.zeros((len(G), len(P)))
            for i, g in enumerate(G):
                for j, p in enumerate(P):
                    v = iou(g["bbox"], p["bbox"])
                    C[i, j] = v if v >= IOU_TH else 0.0
            ri, ci = linear_sum_assignment(-C)
            matched = [(i, j) for i, j in zip(ri, ci) if C[i, j] >= IOU_TH]
        else:
            matched = []
        mg = {i for i, _ in matched}; mp = {j for _, j in matched}
        FN += len(G) - len(mg); FP += len(P) - len(mp)
        cur = {}
        for i, j in matched:
            gid, pid = G[i]["id"], P[j]["id"]
            gt_hit[gid] += 1
            pair[(gid, pid)] += 1
            cur[gid] = pid
            if gid in prev_match and pid is not None and prev_match[gid] != pid:
                IDSW += 1
        prev_match.update(cur)

    mota = 1 - (FN + FP + IDSW) / max(n_gt, 1)
    # IDF1: (gt_id, pred_id) 쌍 전역 최적 매칭
    gids = sorted({k[0] for k in pair}); pids = sorted({k[1] for k in pair})
    idtp = 0
    if gids and pids:
        M = np.zeros((len(gids), len(pids)))
        for (g, p), c in pair.items():
            M[gids.index(g), pids.index(p)] = c
        ri, ci = linear_sum_assignment(-M)
        idtp = int(M[ri, ci].sum())
    idfn = n_gt - idtp; idfp = n_pred_total - idtp
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1e-9)
    # MT/ML (GT 트랙 중 80% 이상 / 20% 미만 추적된 비율)
    mt = sum(1 for t in gt_len if gt_hit[t] / gt_len[t] >= 0.8)
    ml = sum(1 for t in gt_len if gt_hit[t] / gt_len[t] < 0.2)
    return dict(MOTA=mota, IDF1=idf1, IDSW=IDSW, FP=FP, FN=FN,
                MT=mt, ML=ml, n_gt_tracks=len(gt_len), n_gt=n_gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--pred", action="append", help="추론 .json.gz (여러 번 지정 시 A/B 비교)")
    ap.add_argument("--iou", type=float, default=0.5, help="매칭 IoU 임계값(소형 객체는 0.3 권장)")
    ap.add_argument("--roi", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                    help="이 픽셀 영역 안(박스 중심 기준)만 평가. GT가 일부 영역만 라벨된 경우 "
                         "반드시 지정해야 한다 — 안 그러면 라벨 안 된 영역의 정상 검출이 FP로 잡힌다")
    args = ap.parse_args()

    set_iou(args.iou)
    meta, gt = load_gt(args.loc)
    roi = tuple(args.roi) if args.roi else (tuple(meta["roi"]) if meta.get("roi") else None)
    if roi:
        for f in list(gt):
            gt[f] = [b for b in gt[f] if in_roi(b["bbox"], roi)]
            if not gt[f]:
                del gt[f]
    frames = sorted(gt.keys())
    if not frames:
        sys.exit("GT 박스가 없습니다")
    preds = args.pred or sorted(glob.glob(os.path.join(REPO, "output", "**", args.loc, "*.json.gz"),
                                          recursive=True))
    if not preds:
        sys.exit("비교할 예측 파일이 없습니다(--pred)")

    print(f"[{args.loc}] GT 프레임 {len(frames)} · 박스 {sum(len(gt[f]) for f in frames)} · "
          f"트랙 {len({b['id'] for f in frames for b in gt[f]})} · IoU {args.iou}"
          + (f" · ROI {[int(v) for v in roi]}" if roi else " · ROI 전체"))
    print(f"{'예측':44s} {'AP50':>6s} {'P':>6s} {'R':>6s} {'MOTA':>7s} {'IDF1':>6s} {'IDSW':>5s} {'MT':>4s} {'ML':>4s}")
    rows = []
    for p in preds:
        pr = load_pred(p, set(frames), roi)
        d = detection_metrics(gt, pr, frames)
        m = mot_metrics(gt, pr, frames)
        name = os.path.relpath(p, REPO).replace("\\", "/")
        if len(name) > 44:
            name = "…" + name[-43:]
        print(f"{name:44s} {d['AP50']:6.3f} {d['P']:6.3f} {d['R']:6.3f} "
              f"{m['MOTA']:7.3f} {m['IDF1']:6.3f} {m['IDSW']:5d} {m['MT']:4d} {m['ML']:4d}")
        rows.append({"pred": name, **d, **m})

    out = os.path.join(REPO, "eval", args.loc, "metrics.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n→ {os.path.relpath(out, REPO)}")
    print("  AP50/P/R: 검출 품질 · MOTA: 종합(FN+FP+IDSW) · IDF1: ID 일관성 · IDSW: ID 스위치(낮을수록 좋음)")


if __name__ == "__main__":
    main()
