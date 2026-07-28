"""
G-Projection 캘리브레이션 템플릿 생성 (Phase 3/4).

주의: 실 GCP 대응이 없으므로 **지오레퍼런스된 placeholder 캘리브레이션**:
  - world 원점 = 카메라 UTM52N(실제 위치)
  - K/H/parallax = 하향 도로뷰를 가정한 기하학적으로 유효한 합성값(해상도에 맞춰 스케일)
→ 좌표는 지도상 카메라 위치 근처에 떨어지며, 시각화 파이프라인 검증에 충분.
실 캘리브레이션은 cctv 프레임 + 노면표시 대응으로 H를 RANSAC 추정해 교체.

사용:
  python tools/make_calibration_template.py                     # 기본 SONGDO_L020101(1920x1080)
  python tools/make_calibration_template.py --from-camera SONGDO_IC  # _camera.json 사용
"""
import argparse
import json
import os
import sys

import numpy as np
import cv2
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trafficlab.io.trafficlab_config import default_config, save_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 해상도 무관 정규화 이미지 대응점(하향 도로뷰) ↔ 지면 미터
IMG_PTS_NORM = np.array([[0.208, 0.926], [0.792, 0.926], [0.599, 0.324], [0.401, 0.324]])
GRD_PTS_M = np.array([[-6, 5], [6, 5], [6, 45], [-6, 45]], dtype=np.float32)


def build_calibration(loc, lon, lat, img_w, img_h, cam_height=8.0, ground_h=28.0, note=""):
    tr = Transformer.from_crs(4326, 32652, always_xy=True)
    origin_e, origin_n = tr.transform(lon, lat)

    cfg = default_config(loc)
    fx = fy = float(img_w)
    cfg["undistort"]["resolution"] = [img_w, img_h]
    cfg["undistort"]["K"] = [[fx, 0, img_w / 2], [0, fy, img_h / 2], [0, 0, 1]]
    cfg["undistort"]["D"] = [0, 0, 0, 0, 0]

    img_pts = (IMG_PTS_NORM * [img_w, img_h]).astype(np.float32)
    H = cv2.getPerspectiveTransform(img_pts, GRD_PTS_M)
    cfg["homography"]["H"] = H.tolist()
    cfg["homography"]["fov_polygon"] = GRD_PTS_M.tolist()

    cfg["parallax"].update({"x_cam_coords_sat": 0.0, "y_cam_coords_sat": 0.0,
                            "z_cam_meters": cam_height, "px_per_meter": 1.0})
    cfg["world"].update({"epsg": 32652, "origin_easting": float(origin_e),
                         "origin_northing": float(origin_n), "ground_ellipsoid_h": ground_h})
    cfg["proj_method"] = "down_h"
    cfg["ref_method"] = "center_bottom_side"
    cfg["meta"]["note"] = note or "PLACEHOLDER 합성 캘리브레이션 — 실 프레임+GCP로 교체 필요"

    out_dir = os.path.join(REPO, "location", loc)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"G_projection_{loc}.json")
    save_config(out, cfg)
    print(f"[{loc}] origin UTM52N=({origin_e:.2f},{origin_n:.2f}) res={img_w}x{img_h} → {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-camera", help="location/<LOC>/_camera.json 로부터 생성")
    args = ap.parse_args()
    if args.from_camera:
        cam_p = os.path.join(REPO, "location", args.from_camera, "_camera.json")
        with open(cam_p, encoding="utf-8") as f:
            cam = json.load(f)
        build_calibration(cam["loc"], cam["lon"], cam["lat"], cam["width"], cam["height"],
                          note=f"ITS {cam.get('name','')} placeholder 캘리브레이션")
    else:
        build_calibration("SONGDO_L020101", 126.6455869, 37.39128034, 1920, 1080,
                          note="UTIC L020101 placeholder 캘리브레이션")


if __name__ == "__main__":
    main()
