"""
통제 실험: 이미지 IoU 추적(ByteTrack) vs 지면(BEV) 추적 — **정답을 아는 시뮬레이션**으로 A/B.

GT 라벨링 전에도 "연관 공간을 바꾸면 정말 좋아지는가?"를 검증하기 위한 장치.
실제 카메라의 캘리브레이션(G_projection)을 그대로 써서 원근을 재현하므로,
'원거리에서 박스가 작아져 IoU 연관이 무너지는' 우리 실패 모드를 그대로 모사한다.

시나리오: 차로별 차량이 등속 주행 → 지면 좌표를 카메라 픽셀로 투영(실제 H 사용)
          → 검출 노이즈(박스 지터) · 미검출(dropout) · 오탐(FP) 주입
          → (A) sv.ByteTrack(이미지 박스)  vs  (B) BEVTracker(지면 미터)
          → 동일 GT로 MOTA/IDF1/IDSW 계산(tools/eval/evaluate.py 의 지표 구현 재사용)

사용: python tools/eval/sim_track_test.py --loc PANGYO_2 [--dropout 0.2 --fp 0.05 --jitter 3]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trafficlab.projection.g_projection import GProjection
from trafficlab.motion.bev_tracker import BEVTracker, ground_cov
from evaluate import mot_metrics, set_iou


def build_scene(g, rng, n_frames, fps, lanes, veh_per_lane, jitter, dropout, fp_rate, W, H):
    """GT(프레임별 [(gt_id, bbox, ground_pos)]) + 검출(노이즈 주입) 생성."""
    gt, det = defaultdict(list), defaultdict(list)
    dt = 1.0 / fps
    vid = 0
    vehicles = []
    for li, lane_x in enumerate(lanes):
        for k in range(veh_per_lane):
            vid += 1
            speed = rng.uniform(11, 19) * (1 if li % 2 == 0 else -1)   # m/s, 차로별 반대방향
            y0 = rng.uniform(5, 45) if li % 2 == 0 else rng.uniform(5, 45)
            vehicles.append({"id": vid, "x": lane_x, "y0": y0, "v": speed,
                             "w": rng.uniform(1.7, 1.9), "l": rng.uniform(3.6, 4.6)})

    for f in range(n_frames):
        for v in vehicles:
            y = v["y0"] + v["v"] * f * dt
            if not (2 <= y <= 60):        # FOV 밖
                continue
            # 지면 → 이미지: 차량 바닥 사각형 4점을 투영해 축정렬 박스 생성
            floor = [[v["x"] + dx, y + dy] for dx, dy in
                     [(-v["w"] / 2, -v["l"] / 2), (v["w"] / 2, -v["l"] / 2),
                      (v["w"] / 2, v["l"] / 2), (-v["w"] / 2, v["l"] / 2)]]
            pts = g.sat_floor_to_cctv_3d(floor, 1.55)      # 실제 파이프라인과 동일한 3D 리프팅
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            bb = [min(xs), min(ys), max(xs), max(ys)]      # 3D 박스의 2D 외접 사각형
            if bb[2] <= bb[0] or bb[3] <= bb[1]:
                continue
            if not (0 < (bb[0] + bb[2]) / 2 < W and 0 < (bb[1] + bb[3]) / 2 < H):
                continue
            gt[f].append({"id": v["id"], "cls": "car", "bbox": bb})
            if rng.random() < dropout:                      # 미검출
                continue
            nb = [bb[0] + rng.normal(0, jitter), bb[1] + rng.normal(0, jitter),
                  bb[2] + rng.normal(0, jitter), bb[3] + rng.normal(0, jitter)]
            det[f].append({"bbox": nb, "conf": float(np.clip(rng.normal(0.75, 0.12), 0.15, 0.99)), "cls": "car"})
        # 오탐
        for _ in range(rng.poisson(fp_rate * max(1, len(gt[f])))):
            cx, cy = rng.uniform(0.1 * W, 0.9 * W), rng.uniform(0.3 * H, 0.95 * H)
            s = rng.uniform(10, 40)
            det[f].append({"bbox": [cx - s, cy - s * 0.6, cx + s, cy + s * 0.6],
                           "conf": float(rng.uniform(0.2, 0.5)), "cls": "car"})
    return gt, det


def run_bytetrack(det, n_frames, fps):
    import supervision as sv
    tracker = sv.ByteTrack(frame_rate=int(round(fps)))
    out = defaultdict(list)
    for f in range(n_frames):
        D = det.get(f, [])
        if not D:
            continue
        dets = sv.Detections(xyxy=np.array([d["bbox"] for d in D], float),
                             confidence=np.array([d["conf"] for d in D], float),
                             class_id=np.zeros(len(D), int))
        tk = tracker.update_with_detections(dets)
        for k in range(len(tk)):
            out[f].append({"id": int(tk.tracker_id[k]), "cls": "car",
                           "bbox": [float(v) for v in tk.xyxy[k]], "conf": 1.0})
    return out


def run_bev(det, n_frames, fps, g, **kw):
    tr = BEVTracker(dt=1.0 / fps, **kw)
    out = defaultdict(list)
    for f in range(n_frames):
        D = det.get(f, [])
        if not D:
            tr.update([], f); continue
        obs = []
        for d in D:                      # 검출 박스 → 지면(우리 파이프라인과 동일 경로)
            x1, y1, x2, y2 = d["bbox"]
            r = g.get_ground_contact_from_box((x1, y1, x2 - x1, y2 - y1), 1.55,
                                              ref_method="center_bottom_side", proj_method="down_h")
            cov = ground_cov(g, (x1 + x2) / 2, y2, h=0.0, sigma_px=3.0)  # 투영 불확실성(이방성)
            obs.append({"pos": r["sat_coords"], "conf": d["conf"], "cls": d["cls"], "cov": cov})
        ids = tr.update(obs, f)
        for d, tid in zip(D, ids):
            if tid is None:
                continue
            out[f].append({"id": int(tid), "cls": "car", "bbox": d["bbox"], "conf": 1.0})
    return out


def frag(pred, frames):
    life = defaultdict(list)
    for f in frames:
        for o in pred.get(f, []):
            life[o["id"]].append(f)
    if not life:
        return 0, 0.0, 0.0
    L = np.array([max(v) - min(v) + 1 for v in life.values()])
    return len(life), float(np.median(L)), float(np.mean(L < 15) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="PANGYO_2")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--jitter", type=float, default=3.0, help="박스 지터 표준편차(px)")
    ap.add_argument("--dropout", type=float, default=0.2, help="미검출 확률")
    ap.add_argument("--fp", type=float, default=0.05, help="오탐 비율")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    set_iou(args.iou)
    gp = os.path.join(REPO, "location", args.loc, f"G_projection_{args.loc}.json")
    gd = json.load(open(gp, encoding="utf-8"))
    g = GProjection(gd, base_dir=os.path.dirname(gp))
    W, H = gd["undistort"]["resolution"]
    rng = np.random.default_rng(args.seed)

    lanes = [-5.25, -1.75, 1.75, 5.25]
    gt, det = build_scene(g, rng, args.frames, args.fps, lanes, 6,
                          args.jitter, args.dropout, args.fp, W, H)
    frames = sorted(gt.keys())
    n_gt_boxes = sum(len(gt[f]) for f in frames)
    n_gt_tracks = len({b["id"] for f in frames for b in gt[f]})
    print(f"[시뮬] {args.loc} 캘리브레이션 사용 · {len(frames)}프레임 · GT 박스 {n_gt_boxes} · GT 트랙 {n_gt_tracks}")
    print(f"       노이즈: 지터 {args.jitter}px · 미검출 {args.dropout:.0%} · 오탐 {args.fp:.0%}")

    A = run_bytetrack(det, args.frames, args.fps)
    B = run_bev(det, args.frames, args.fps, g, mahalanobis=False, max_dist=6.0)
    C = run_bev(det, args.frames, args.fps, g, mahalanobis=True)

    print(f"\n{'방식':34s} {'MOTA':>7s} {'IDF1':>7s} {'IDSW':>6s} {'MT':>4s} {'ML':>4s} "
          f"{'트랙수':>6s} {'수명중앙(f)':>11s} {'0.5초미만%':>10s}")
    rows = {}
    for name, P in [("A. ByteTrack (이미지 IoU)", A), ("B. BEV 등방 유클리드", B), ("C. BEV 마할라노비스(제안)", C)]:
        m = mot_metrics(gt, P, frames)
        nt, med, sh = frag(P, frames)
        rows[name] = m
        print(f"{name:34s} {m['MOTA']:7.3f} {m['IDF1']:7.3f} {m['IDSW']:6d} {m['MT']:4d} {m['ML']:4d} "
              f"{nt:6d} {med:11.0f} {sh:10.0f}")
    a, b, c = list(rows.values())
    print(f"\n→ IDF1 {a['IDF1']:.3f} → {b['IDF1']:.3f} ({(b['IDF1']-a['IDF1'])*100:+.1f}p) · "
          f"IDSW {a['IDSW']} → {b['IDSW']} ({b['IDSW']-a['IDSW']:+d}) · "
          f"MOTA {a['MOTA']:.3f} → {b['MOTA']:.3f} ({(b['MOTA']-a['MOTA'])*100:+.1f}p)")
    print(f"  GT 트랙 {n_gt_tracks}개 대비 생성 트랙 수(적을수록 파편화 적음)")


if __name__ == "__main__":
    main()
