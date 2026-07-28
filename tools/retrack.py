"""
재추적(A/B) — 기존 추론 결과의 **검출은 그대로 두고 연관(추적)만 교체**.

검출을 다시 돌리지 않으므로 A(ByteTrack, 기존)와 B(BEV)를 공정하게 비교할 수 있다.
BEV 추적은 검출 박스를 지면(미터)으로 투영해 **투영 불확실성(이방성) 기반 마할라노비스 거리**로
연관한다(cf. UCMCTrack) + OC-SORT의 ORU/OCM/OCR.

사용:
  python tools/retrack.py --loc PANGYO_2
  python tools/eval/evaluate.py --loc PANGYO_2 --pred <기존> --pred output/retrack-bev/PANGYO_2/clip.json.gz
출력: output/retrack-bev/<loc>/clip.json.gz
"""
import argparse
import gzip
import json
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from trafficlab.projection.g_projection import GProjection
from trafficlab.motion.bev_tracker import BEVTracker, ground_cov
from trafficlab.motion.kinematics import TrackSmoother

KIN = {"heading_ema": {"alpha_min": 0.05, "alpha_max": 0.4, "speed_ref": 3.0},
       "speed_ema_alpha": 0.4, "heading_min_speed_for_update": 0.1, "heading_max_jump": 5,
       "heading_sat_coords_jitter_radius": 0.5, "heading_sat_coords_jitter_frames": 8}


def frag_stats(frames_objs, fps):
    life = defaultdict(list)
    for fi, objs in frames_objs:
        for o in objs:
            t = o.get("tracked_id")
            if t is not None:
                life[t].append(fi)
    if not life:
        return dict(tracks=0, median=0, short=0)
    L = np.array([max(v) - min(v) + 1 for v in life.values()])
    return dict(tracks=len(life), median=float(np.median(L)) / fps,
                short=float(np.mean(L < fps * 0.5) * 100))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--sigma-px", type=float, default=3.0, help="검출 박스 픽셀 잡음 가정")
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--min-hits", type=int, default=3)
    args = ap.parse_args()

    src = args.src or os.path.join(REPO, "webmap", "public", "data", "replay", f"{args.loc.lower()}.json.gz")
    if not os.path.exists(src):
        sys.exit(f"입력 replay 없음: {src}")
    data = json.load(gzip.open(src, "rt", encoding="utf-8"))
    fps = data["meta"].get("fps", 30)

    gp = os.path.join(REPO, "location", args.loc, f"G_projection_{args.loc}.json")
    g_data = json.load(open(gp, encoding="utf-8"))
    g = GProjection(g_data, base_dir=os.path.dirname(gp))
    priors = json.load(open(os.path.join(REPO, "prior_dimensions.json"), encoding="utf-8"))["measurements_visdrone_full"]
    priors = {k.lower(): v for k, v in priors.items()}

    before = [(f["frame_index"], f.get("objects", [])) for f in data["frames"]]
    tracker = BEVTracker(dt=1.0 / fps, max_age=args.max_age, min_hits=args.min_hits, mahalanobis=True)
    smoothers, last_seen = {}, {}
    n_obj = n_id = 0

    for fr in data["frames"]:
        i = fr["frame_index"]
        objs = [o for o in fr.get("objects", []) if o.get("bbox_2d")]
        dets = []
        for o in objs:
            x1, y1, x2, y2 = o["bbox_2d"]
            dims = priors.get(str(o.get("class", "")).lower())
            h_real = float(dims["height"]) if dims else 1.55
            r = g.get_ground_contact_from_box((x1, y1, x2 - x1, y2 - y1), h_real,
                                              ref_method=g_data.get("ref_method", "center_bottom_side"),
                                              proj_method=g_data.get("proj_method", "down_h"))
            dets.append({"pos": r["sat_coords"], "conf": float(o.get("confidence") or 0.5),
                         "cls": o.get("class"),
                         "cov": ground_cov(g, (x1 + x2) / 2, y2, h=0.0, sigma_px=args.sigma_px)})
        ids = tracker.update(dets, i)

        new_objs = []
        for o, d, tid in zip(objs, dets, ids):
            n_obj += 1
            o = dict(o)
            o["tracked_id"] = int(tid) if tid is not None else None
            o["sat_coords"] = list(d["pos"])
            if tid is None:
                o["have_heading"] = False; o["heading"] = None; o["speed_kmh"] = 0.0
                o["sat_floor_box"] = None; o["bbox_3d"] = None
                new_objs.append(o); continue
            n_id += 1
            if tid not in smoothers:
                smoothers[tid] = TrackSmoother(KIN); last_seen[tid] = i - 1
            dt = (i - last_seen.get(tid, i - 1)) / fps or (1.0 / fps)
            k = smoothers[tid].update(d["pos"], dt, g.px_per_m); last_seen[tid] = i
            o["heading"] = k["heading"]; o["speed_kmh"] = k["speed_kmh"]
            o["have_heading"] = k["heading"] is not None
            dims = priors.get(str(o.get("class", "")).lower())
            if k["heading"] is not None and dims:
                w_m, l_m = dims["width"], dims["length"]
                ang = np.radians(k["heading"]); c, s = np.cos(ang), np.sin(ang)
                dx, dy = (l_m * g.px_per_m) / 2, (w_m * g.px_per_m) / 2
                corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
                R = np.array([[c, -s], [s, c]])
                floor = (corners @ R.T + np.array(d["pos"])).tolist()
                o["sat_floor_box"] = floor
                o["bbox_3d"] = g.sat_floor_to_cctv_3d(floor, float(dims["height"]))
            new_objs.append(o)
        fr["objects"] = new_objs

    out = os.path.join(REPO, "output", "retrack-bev", args.loc, "clip.json.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(data, f)

    after = [(f["frame_index"], f.get("objects", [])) for f in data["frames"]]
    a, b = frag_stats(before, fps), frag_stats(after, fps)
    print(f"[{args.loc}] 검출 {n_obj}개 중 {n_id}개에 ID 부여 (미확정 {n_obj - n_id})")
    print(f"{'':22s} {'트랙수':>7s} {'수명중앙(s)':>11s} {'0.5초미만%':>10s}")
    print(f"{'A. 기존(ByteTrack)':22s} {a['tracks']:7d} {a['median']:11.2f} {a['short']:10.0f}")
    print(f"{'B. BEV(제안)':22s} {b['tracks']:7d} {b['median']:11.2f} {b['short']:10.0f}")
    print(f"\n→ {os.path.relpath(out, REPO)}")
    print(f"  정량 비교: python tools/eval/evaluate.py --loc {args.loc} --pred {os.path.relpath(src, REPO)} "
          f"--pred {os.path.relpath(out, REPO)}   (GT 라벨 필요)")


if __name__ == "__main__":
    main()
