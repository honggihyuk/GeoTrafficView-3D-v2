"""
재투영: 기존 추론 결과(.json.gz, bbox_2d 보유)를 **새 캘리브레이션**으로 다시 투영.
YOLO 재실행 없이 좌표/3D박스/속도/방향만 재계산 → 캘리브레이션 반영 즉시 확인.

사용:
  python tools/reproject.py --loc SONGDO_IC
  (source 미지정 시 output/의 해당 loc 최신 .json.gz 자동 탐색)
출력: output/reprojected/<loc>/clip.json.gz  (이후 bridge_to_webmap.py로 웹맵 반영)
"""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from trafficlab.projection.g_projection import GProjection
from trafficlab.motion.kinematics import TrackSmoother

KIN = {"heading_ema": {"alpha_min": 0.05, "alpha_max": 0.4, "speed_ref": 3.0},
       "speed_ema_alpha": 0.4, "heading_min_speed_for_update": 0.1, "heading_max_jump": 5,
       "heading_sat_coords_jitter_radius": 0.5, "heading_sat_coords_jitter_frames": 8}


def find_source(loc):
    """원본 추론 결과 우선(재투영/차선정렬/재추적 산출물은 제외 — 중첩 적용 방지)."""
    hits = glob.glob(os.path.join(REPO, "output", "**", loc, "*.json.gz"), recursive=True)
    derived = ("reprojected", "lanesnap", "retrack-bev")
    orig = [h for h in hits if not any(os.sep + d + os.sep in h for d in derived)]
    pool = orig or hits
    if not pool:
        sys.exit(f"output/에서 {loc} 결과를 찾지 못함 — run_inference 먼저 실행")
    return max(pool, key=os.path.getsize)      # 가장 검출이 많은(=최신 고품질) 결과


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--source", default=None)
    args = ap.parse_args()

    gpath = os.path.join(REPO, "location", args.loc, f"G_projection_{args.loc}.json")
    g_data = json.load(open(gpath, encoding="utf-8"))
    g = GProjection(g_data, base_dir=os.path.dirname(gpath))
    priors = json.load(open(os.path.join(REPO, "prior_dimensions.json"), encoding="utf-8"))["measurements_coco"]
    priors = {k.lower(): v for k, v in priors.items()}

    src = args.source or find_source(args.loc)
    data = json.load(gzip.open(src, "rt", encoding="utf-8"))
    fps = data["meta"].get("fps", 30)
    print(f"source: {os.path.relpath(src, REPO)}  frames={len(data['frames'])}")

    smoothers, last_seen = {}, {}
    out_frames = []
    box_cnt = 0
    for i, fr in enumerate(data["frames"]):
        objs = []
        for o in fr.get("objects", []):
            bb = o.get("bbox_2d")
            if not bb:
                continue
            cls = o.get("class", "")
            dims = priors.get(cls.lower())
            have_m = dims is not None
            h_real = float(dims["height"]) if have_m else 0.0
            x1, y1, x2, y2 = bb
            proj = g.get_ground_contact_from_box((x1, y1, x2 - x1, y2 - y1), h_real,
                     ref_method=g_data.get("ref_method", "center_bottom_side"),
                     proj_method=g_data.get("proj_method", "down_h"))
            sat = proj["sat_coords"]
            tid = o.get("tracked_id")
            heading = None; speed = 0.0
            if tid is not None:
                if tid not in smoothers:
                    smoothers[tid] = TrackSmoother(KIN); last_seen[tid] = i - 1
                dt = (i - last_seen.get(tid, i - 1)) / fps or (1.0 / fps)
                k = smoothers[tid].update(sat, dt, g.px_per_m)
                last_seen[tid] = i; heading = k["heading"]; speed = k["speed_kmh"]
            have_h = heading is not None
            floor = bbox3d = None
            if have_h and have_m:
                w_m, l_m = dims["width"], dims["length"]; px = g.px_per_m
                ang = np.radians(heading); c, s = np.cos(ang), np.sin(ang)
                dx, dy = (l_m * px) / 2, (w_m * px) / 2
                corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
                R = np.array([[c, -s], [s, c]])
                floor = (corners @ R.T + np.array(sat)).tolist()
                bbox3d = g.sat_floor_to_cctv_3d(floor, h_real); box_cnt += 1
            objs.append({**o, "sat_coords": sat, "heading": heading, "speed_kmh": speed,
                         "have_heading": have_h, "sat_floor_box": floor, "bbox_3d": bbox3d})
        out_frames.append({"frame_index": i, "objects": objs})

    out_data = {"mp4_path": data.get("mp4_path"), "location_code": args.loc,
                "meta": {"resolution": data["meta"].get("resolution"), "fps": fps, "world": g_data.get("world")},
                "mp4_frame_count": data.get("mp4_frame_count"), "animation_frame_count": len(out_frames),
                "frames": out_frames}
    out = os.path.join(REPO, "output", "reprojected", args.loc, "clip.json.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"재투영 완료: 3D박스 {box_cnt}개 → {os.path.relpath(out, REPO)}")
    print(f"다음: python tools/bridge_to_webmap.py --replay {os.path.relpath(out, REPO)} --loc {args.loc} --ingest")


if __name__ == "__main__":
    main()
