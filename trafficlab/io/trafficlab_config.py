"""G-Projection JSON 스키마 (TrafficLab-3D 이식 + GIS 확장).

원본 스키마에 world 블록 추가:
  world = { epsg, origin_easting, origin_northing, ground_ellipsoid_h }
  → SAT 로컬 미터 좌표(sat_coords)를 실세계 UTM52N으로 환원:
     UTM_E = origin_easting + sat_x,  UTM_N = origin_northing + sat_y
  px_per_meter = 1.0 (SAT 1단위 = 1m) 로 두는 것이 GIS 규약.
"""
import json
from datetime import datetime
from typing import Any, Dict, Optional


def default_config(location_code: str = "PLACEHOLDER", timestamp: Optional[str] = None) -> Dict[str, Any]:
    if timestamp is None:
        timestamp = datetime.now().replace(microsecond=0).isoformat()
    return {
        "meta": {"location_code": location_code, "timestamp": timestamp},
        "inputs": {
            "cctv_path": f"cctv_{location_code}.png",
            "sat_path": f"sat_{location_code}.png",
            "layout_path": f"layout_{location_code}.svg",
            "roi_path": f"roi_{location_code}.png",
            "note": "경로는 이 json 파일 기준 상대경로",
        },
        "undistort": {
            "resolution": [1280, 720],
            "K": [[1280.0, 0.0, 640.0], [0.0, 1280.0, 360.0], [0.0, 0.0, 1.0]],
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "model": "radial_tangential",
        },
        "homography": {"H": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                       "fov_polygon": [], "anchors_list": []},
        "parallax": {
            "x_cam_coords_sat": 0.0, "y_cam_coords_sat": 0.0, "z_cam_meters": 0.0,
            "scale": {"measured_px": 0.0, "real_m": 0.0, "reference_anchors": []},
            "px_per_meter": 1.0,
        },
        # --- GIS 확장 ---
        "world": {
            "epsg": 32652,
            "origin_easting": 0.0,
            "origin_northing": 0.0,
            "ground_ellipsoid_h": 0.0,
            "note": "sat_coords(로컬 미터) → UTM52N: E=origin_easting+x, N=origin_northing+y",
        },
        "use_svg": False,
        "layout_svg": {"A": [], "association_pairs": []},
        "use_roi": False,
        "roi_method": "partial",
        "ref_method": "center_bottom_side",
        "proj_method": "down_h",
    }


def to_pretty_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, indent=4, ensure_ascii=False)


def save_config(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
