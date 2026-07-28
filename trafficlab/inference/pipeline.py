"""추론 파이프라인 (TrafficLab-3D 이식 + GIS 적응).

탐지(YOLO)→추적(ByteTrack)→투영(GProjection)→기구학(TrackSmoother)→3D 리프팅 → .json.gz.
원본 대비 변경:
  - prior_dimensions.json 경로를 인자로(기본 repo 루트).
  - source: mp4/스트림(.m3u8)/프레임 디렉터리 모두 Ultralytics가 처리.
  - G-Projection의 world 블록을 replay meta 로 전달(뷰어 georeference용).
"""
import os
import json
import yaml
import cv2
import numpy as np
from pathlib import Path

from trafficlab.projection.g_projection import GProjection
from trafficlab.motion.kinematics import TrackSmoother
from trafficlab.io.replay_writer import ReplayWriter


class InferencePipeline:
    def __init__(self, location_code, footage_path, config_path, output_root, g_proj_path,
                 config_name=None, prior_dims_path="prior_dimensions.json",
                 log_fn=None, progress_fn=None, stop_flag_fn=None):
        self.loc_code = location_code
        self.footage_path = footage_path
        self.config_path = config_path
        self.config_name = config_name
        self.output_root = output_root
        self.g_proj_path = g_proj_path
        self.prior_dims_path = prior_dims_path
        self.log_fn = log_fn or (lambda msg: None)
        self.progress_fn = progress_fn or (lambda pct: None)
        self.stop_flag_fn = stop_flag_fn or (lambda: False)

    def run(self):
        from ultralytics import YOLO  # 무거운 의존성 → 지연 import

        with open(self.config_path, 'r', encoding='utf-8') as f:
            raw_cfg = yaml.safe_load(f)
        if isinstance(raw_cfg, dict) and isinstance(raw_cfg.get('configs'), dict):
            configs_map = raw_cfg['configs']
            if self.config_name and self.config_name in configs_map:
                full_config = configs_map[self.config_name]; config_name = self.config_name
            else:
                config_name = next(iter(configs_map.keys())); full_config = configs_map[config_name]
        else:
            full_config = raw_cfg or {}; config_name = full_config.get('config_name', 'default')

        with open(self.g_proj_path, 'r', encoding='utf-8') as f:
            g_data = json.load(f)
        g_proj_dir = os.path.dirname(self.g_proj_path)
        g_engine = GProjection(g_data, base_dir=g_proj_dir)
        world = g_data.get('world')  # GIS georeference (웹맵으로 전달)

        with open(self.prior_dims_path, 'r', encoding='utf-8') as f:
            all_priors = json.load(f)
        measure_set = full_config.get('prior_dimensions', 'measurements_visdrone')
        prior_dims = all_priors.get(measure_set, {})
        prior_dims_norm = {k.strip().lower(): v for k, v in prior_dims.items()}

        footage_name = os.path.basename(self.footage_path)
        model_name = Path(full_config['model']['weights']).stem
        tracker_type = full_config['tracking']['tracker_type']
        config_dir = os.path.join(self.output_root, f"model-{model_name}_tracker-{tracker_type}", config_name)
        out_subdir = os.path.join(config_dir, self.loc_code)
        os.makedirs(out_subdir, exist_ok=True)

        cap = cv2.VideoCapture(self.footage_path)
        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        use_roi = g_data.get('use_roi', False)
        roi_mask = None
        if use_roi:
            roi_rel = g_data['inputs'].get('roi_path')
            if roi_rel:
                roi_p = os.path.join(g_proj_dir, roi_rel)
                if os.path.exists(roi_p):
                    img = cv2.imread(roi_p, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        if img.shape[:2] != (real_h, real_w):
                            img = cv2.resize(img, (real_w, real_h), interpolation=cv2.INTER_NEAREST)
                        if img.ndim == 3 and img.shape[2] == 4:
                            roi_mask = (cv2.bitwise_not(img[:, :, 3]) > 128)
                        elif img.ndim == 3:
                            roi_mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 10)
        roi_method = g_data.get('roi_method', 'partial')

        self.log_fn(f"Loading Model: {full_config['model']['weights']}")
        model = YOLO(full_config['model']['weights'])

        track_smoothers = {}
        last_seen_frame = {}
        out_data = {
            "mp4_path": self.footage_path,
            "meta": {"resolution": [real_w, real_h], "fps": real_fps, "world": world},
            "location_code": self.loc_code,
            "mp4_frame_count": total_frames,
            "frames": [],
        }
        use_svg = g_data.get('use_svg', False)
        max_frame = full_config.get('frames', {}).get('max_frame', -1)
        frames_to_process = min(total_frames, max_frame) if max_frame > 0 else max(total_frames, 1)

        results = model.track(
            source=self.footage_path, device=full_config['model']['device'], persist=True,
            verbose=False, stream=True, conf=full_config['model']['conf'],
            iou=full_config['model']['iou'], imgsz=full_config['model']['imgsz'])

        i = 0
        for i, r in enumerate(results):
            if self.stop_flag_fn() or (max_frame > 0 and i >= max_frame):
                break
            frame_objects = []
            boxes = r.boxes.xyxy.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)

            for j, box in enumerate(boxes):
                if roi_mask is not None:
                    x1, y1, x2, y2 = map(int, box)
                    x1 = max(0, min(real_w - 1, x1)); y1 = max(0, min(real_h - 1, y1))
                    x2 = max(0, min(real_w - 1, x2)); y2 = max(0, min(real_h - 1, y2))
                    if x1 >= x2 or y1 >= y2:
                        continue
                    if roi_method == 'in':
                        corns = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                        if not all(roi_mask[cy, cx] for cx, cy in corns):
                            continue
                    else:
                        if np.count_nonzero(roi_mask[y1:y2, x1:x2]) == 0:
                            continue

                cls_name = r.names[int(cls_ids[j])]
                dims = prior_dims_norm.get(cls_name.strip().lower())
                have_measurements = dims is not None
                h_real = float(dims.get('height', 0.0)) if have_measurements else 0.0

                bx1, by1, bx2, by2 = box
                proj_res = g_engine.get_ground_contact_from_box(
                    (bx1, by1, bx2 - bx1, by2 - by1), h_real,
                    ref_method=g_data.get('ref_method', 'center_bottom_side'),
                    proj_method=g_data.get('proj_method', 'down_h'))
                sat_coords = proj_res['sat_coords']

                tid = int(track_ids[j]) if track_ids[j] is not None else None
                heading = None; speed = 0.0; is_def = False
                if tid is not None:
                    if tid not in track_smoothers:
                        track_smoothers[tid] = TrackSmoother(full_config['kinematics'])
                        last_seen_frame[tid] = i - 1
                    svg_h = g_engine.get_svg_heading(sat_coords) if use_svg else None
                    prev_f = last_seen_frame.get(tid, i - 1)
                    dt = (i - prev_f) / real_fps
                    if dt <= 0:
                        dt = 1.0 / real_fps
                    k_res = track_smoothers[tid].update(sat_coords, dt, g_engine.px_per_m, svg_heading=svg_h)
                    last_seen_frame[tid] = i
                    speed = k_res['speed_kmh']; heading = k_res['heading']; is_def = k_res['default_heading']

                have_heading = heading is not None
                if not have_heading:
                    speed = 0.0

                sat_floor_box = None; bbox_3d = None
                if have_heading and have_measurements:
                    w_m, l_m = dims['width'], dims['length']
                    px_m = g_engine.px_per_m
                    ang = np.radians(heading)
                    c, s = np.cos(ang), np.sin(ang)
                    dx, dy = (l_m * px_m) / 2, (w_m * px_m) / 2
                    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
                    R = np.array([[c, -s], [s, c]])
                    sat_floor_box = (corners @ R.T + sat_coords).tolist()
                    bbox_3d = g_engine.sat_floor_to_cctv_3d(sat_floor_box, h_real)

                frame_objects.append({
                    "id": j, "tracked_id": tid, "class": cls_name, "confidence": float(confs[j]),
                    "bbox_2d": [float(x) for x in box], "reference_point": proj_res['cctv_ref_point'],
                    "sat_coords": sat_coords, "have_heading": have_heading,
                    "have_measurements": have_measurements, "default_heading": is_def,
                    "heading": heading, "speed_kmh": speed,
                    "sat_floor_box": sat_floor_box, "bbox_3d": bbox_3d,
                })
            out_data['frames'].append({"frame_index": i, "objects": frame_objects})
            self.progress_fn(int((i / frames_to_process) * 100))

        cap.release()
        out_data["animation_frame_count"] = i
        out_path = os.path.join(out_subdir, f"{os.path.splitext(footage_name)[0]}.json.gz")
        ReplayWriter.write(out_path, out_data)
        self.log_fn(f"저장: {out_path}")
        return out_path
