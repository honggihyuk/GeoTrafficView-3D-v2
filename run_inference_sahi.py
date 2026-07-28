"""
SAHI 슬라이싱(또는 풀프레임) 추론 + ByteTrack 추적 → .json.gz.

- slice>0: SAHI 슬라이싱(원거리 소형차 대폭 개선). slice<=0: 풀프레임(고속, 다중카메라 배치용).
- 교통 파인튜닝 모델(VisDrone YOLOv8) + supervision.ByteTrack.
- 투영/기구학/3D: 이식 GProjection·TrackSmoother. location/<loc>/G_projection 사용.

CLI:  python run_inference_sahi.py --loc SONGDO_IC --source <clip> [--slice 384|0] [--max-frame N]
함수: run_sahi(loc, source, ...) → out_path
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from trafficlab.projection.g_projection import GProjection
from trafficlab.motion.kinematics import TrackSmoother
from trafficlab.io.replay_writer import ReplayWriter

KIN = {"heading_ema": {"alpha_min": 0.05, "alpha_max": 0.4, "speed_ref": 3.0},
       "speed_ema_alpha": 0.4, "heading_min_speed_for_update": 0.1, "heading_max_jump": 5,
       "heading_sat_coords_jitter_radius": 0.5, "heading_sat_coords_jitter_frames": 8}
VISDRONE = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
            'tricycle', 'awning-tricycle', 'bus', 'motor']


def run_sahi(loc, source, model=None, measure="measurements_visdrone_full",
             slice_px=384, conf=0.2, max_frame=-1, log=print):
    import supervision as sv
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction, get_prediction
    import torch

    model = model or os.path.join(REPO, "models", "yolov8s-visdrone.pt")
    gpath = os.path.join(REPO, "location", loc, f"G_projection_{loc}.json")
    g_data = json.load(open(gpath, encoding="utf-8"))
    g = GProjection(g_data, base_dir=os.path.dirname(gpath))
    priors = json.load(open(os.path.join(REPO, "prior_dimensions.json"), encoding="utf-8"))[measure]
    priors = {k.lower(): v for k, v in priors.items()}

    # ROI(도로영역, CCTV 픽셀 폴리곤) + 차선방향 heading 가이드(로컬미터)
    roi_poly = None
    if g_data.get("use_roi") and len(g_data.get("roi_polygon", [])) >= 3:
        from shapely.geometry import Polygon
        roi_poly = Polygon(g_data["roi_polygon"])
    guides = [((.5 * (gd["sat"][0][0] + gd["sat"][1][0]), .5 * (gd["sat"][0][1] + gd["sat"][1][1])), gd["heading_deg"])
              for gd in (g_data.get("heading_guidelines") or [])]

    def nearest_heading(sat):
        if not guides:
            return None
        return min(guides, key=lambda g: (sat[0] - g[0][0]) ** 2 + (sat[1] - g[0][1]) ** 2)[1]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        det = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=model,
                                                 confidence_threshold=conf, device=device)
    except Exception:
        det = AutoDetectionModel.from_pretrained(model_type="yolov8", model_path=model,
                                                 confidence_threshold=conf, device=device)
    names = getattr(det, "category_mapping", None) or {str(i): n for i, n in enumerate(VISDRONE)}

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = sv.ByteTrack(frame_rate=int(round(fps)))

    smoothers, last_seen = {}, {}
    frames = []; box_cnt = 0; i = -1
    while True:
        ok, frame = cap.read()
        if not ok or (max_frame > 0 and i + 1 >= max_frame):
            break
        i += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if slice_px and slice_px > 0:
            res = get_sliced_prediction(rgb, det, slice_height=slice_px, slice_width=slice_px,
                                        overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
        else:
            res = get_prediction(rgb, det)
        xyxy, cf, cl = [], [], []
        for op in res.object_prediction_list:
            b = op.bbox.to_xyxy(); xyxy.append(b); cf.append(op.score.value); cl.append(op.category.id)
        objs = []
        if xyxy:
            dets = sv.Detections(xyxy=np.array(xyxy, float), confidence=np.array(cf, float),
                                 class_id=np.array(cl, int))
            tk = tracker.update_with_detections(dets)
            for k in range(len(tk)):
                tid = int(tk.tracker_id[k]); cid = int(tk.class_id[k])
                x1, y1, x2, y2 = [float(v) for v in tk.xyxy[k]]
                if roi_poly is not None:  # 지면 접촉점(하단 중앙)이 ROI 밖이면 폐기
                    from shapely.geometry import Point
                    if not roi_poly.contains(Point((x1 + x2) / 2, y2)):
                        continue
                cname = names.get(str(cid), names.get(cid, str(cid)))
                dims = priors.get(str(cname).lower()); have_m = dims is not None
                h_real = float(dims["height"]) if have_m else 0.0
                proj = g.get_ground_contact_from_box((x1, y1, x2 - x1, y2 - y1), h_real,
                         ref_method=g_data.get("ref_method", "center_bottom_side"),
                         proj_method=g_data.get("proj_method", "down_h"))
                sat = proj["sat_coords"]
                if tid not in smoothers:
                    smoothers[tid] = TrackSmoother(KIN); last_seen[tid] = i - 1
                dt = (i - last_seen.get(tid, i - 1)) / fps or (1.0 / fps)
                kr = smoothers[tid].update(sat, dt, g.px_per_m, svg_heading=nearest_heading(sat))
                last_seen[tid] = i
                heading, speed = kr["heading"], kr["speed_kmh"]
                floor = bbox3d = None
                if heading is not None and have_m:
                    w_m, l_m = dims["width"], dims["length"]; px = g.px_per_m
                    ang = np.radians(heading); c, s = np.cos(ang), np.sin(ang)
                    dx, dy = (l_m * px) / 2, (w_m * px) / 2
                    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
                    R = np.array([[c, -s], [s, c]])
                    floor = (corners @ R.T + np.array(sat)).tolist()
                    bbox3d = g.sat_floor_to_cctv_3d(floor, h_real); box_cnt += 1
                objs.append({"id": k, "tracked_id": tid, "class": cname,
                             "confidence": float(tk.confidence[k]), "bbox_2d": [x1, y1, x2, y2],
                             "sat_coords": sat, "have_heading": heading is not None, "have_measurements": have_m,
                             "heading": heading, "speed_kmh": speed, "sat_floor_box": floor, "bbox_3d": bbox3d})
        frames.append({"frame_index": i, "objects": objs})
    cap.release()

    out_data = {"mp4_path": source, "location_code": loc,
                "meta": {"resolution": [W, H], "fps": fps, "world": g_data.get("world")},
                "mp4_frame_count": i + 1, "animation_frame_count": i + 1, "frames": frames}
    mstem = os.path.splitext(os.path.basename(model))[0]
    tag = "sahi" if (slice_px and slice_px > 0) else "full"
    out_dir = os.path.join(REPO, "output", f"model-{mstem}-{tag}", loc)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "clip.json.gz")
    ReplayWriter.write(out, out_data)
    log(f"[{loc}] {tag} 프레임 {i+1}, 3D박스 {box_cnt} → {os.path.relpath(out, REPO)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--measure", default="measurements_visdrone_full")
    ap.add_argument("--slice", type=int, default=384)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--max-frame", type=int, default=-1)
    a = ap.parse_args()
    run_sahi(a.loc, a.source, a.model, a.measure, a.slice, a.conf, a.max_frame)


if __name__ == "__main__":
    main()
