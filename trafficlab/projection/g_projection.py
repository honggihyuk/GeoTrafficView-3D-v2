"""CCTV↔지면 양방향 투영 엔진 (TrafficLab-3D 이식).

파이프라인: CCTV(왜곡) →① undistort →② homography(→SAT 평면) →③ parallax(시차).
GeoTrafficView-3D에서 SAT 평면 = 로컬 미터 평면(UTM52N 원점 기준), px_per_meter=1.0.
SVG 방향 가이드는 선택(use_svg=False면 미사용).
"""
import os
import numpy as np
import cv2

try:
    from trafficlab.projection.svg_parser import SVGParser  # 선택
except Exception:  # svg_parser 미이식/미사용 시에도 동작
    SVGParser = None


class GProjection:
    def __init__(self, config_data: dict, base_dir: str = "."):
        self.config = config_data
        self.base_dir = base_dir
        self._init_matrices()
        self._init_parallax()
        self._init_svg()

    def _init_matrices(self):
        und = self.config.get('undistort', {})
        self.K = np.array(und.get('K'), dtype=np.float64)
        self.D = np.array(und.get('D'), dtype=np.float64)
        self.P = self.K.copy()

        hom = self.config.get('homography', {})
        self.H = np.array(hom.get('H'), dtype=np.float64)
        self.H_inv = np.linalg.inv(self.H)

    def _init_parallax(self):
        par = self.config.get('parallax', {})
        self.z_cam = par.get('z_cam_meters', 10.0)
        x_sat = par.get('x_cam_coords_sat', 0.0)
        y_sat = par.get('y_cam_coords_sat', 0.0)
        self.cam_sat = np.array([x_sat, y_sat], dtype=np.float64)
        self.px_per_m = par.get('px_per_meter', 1.0)
        if self.px_per_m <= 0.001:
            self.px_per_m = 1.0

    def _init_svg(self):
        self.svg_parser = None
        if self.config.get('use_svg', False) and SVGParser is not None:
            rel_path = self.config.get('inputs', {}).get('layout_path')
            if rel_path:
                full_path = os.path.join(self.base_dir, rel_path)
                align_A = self.config.get('layout_svg', {}).get('A')
                self.svg_parser = SVGParser(full_path, align_A)

    def get_svg_heading(self, sat_pt):
        if self.svg_parser and self.svg_parser.valid:
            return self.svg_parser.get_nearest_heading(sat_pt)
        return None

    # --- TRANSFORMATIONS ---
    def cctv_to_undistorted(self, u, v):
        src = np.array([[[u, v]]], dtype=np.float64)
        dst = cv2.undistortPoints(src, self.K, self.D, P=self.P)
        return tuple(dst[0, 0])

    def undistorted_to_flat_sat(self, u_u, v_u):
        src = np.array([[[u_u, v_u]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.H)
        return tuple(dst[0, 0])

    def flat_sat_to_undistorted(self, x, y):
        src = np.array([[[x, y]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.H_inv)
        return tuple(dst[0, 0])

    def undistorted_to_cctv(self, u_u, v_u):
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x_norm = (u_u - cx) / fx
        y_norm = (v_u - cy) / fy
        obj_pts = np.array([[[x_norm, y_norm, 1.0]]], dtype=np.float64)
        img_pts, _ = cv2.projectPoints(obj_pts, (0, 0, 0), (0, 0, 0), self.K, self.D)
        return tuple(img_pts[0, 0])

    # --- PARALLAX ---
    def parallax_correct_ground_to_real(self, apparent_sat_pt, h):
        if self.z_cam == 0:
            return apparent_sat_pt
        A = np.array(apparent_sat_pt); C = self.cam_sat
        factor = (self.z_cam - h) / self.z_cam
        return tuple(C + (A - C) * factor)

    def parallax_project_real_to_ground(self, real_sat_pt, h):
        if abs(self.z_cam - h) < 0.01:
            return real_sat_pt
        R = np.array(real_sat_pt); C = self.cam_sat
        factor = self.z_cam / (self.z_cam - h)
        return tuple(C + (R - C) * factor)

    def cctv_to_sat(self, u, v, h=0.0):
        u_u, v_u = self.cctv_to_undistorted(u, v)
        apparent_pt = self.undistorted_to_flat_sat(u_u, v_u)
        if h != 0:
            return self.parallax_correct_ground_to_real(apparent_pt, h)
        return apparent_pt

    def sat_to_cctv(self, x, y, h=0.0):
        if h != 0:
            flat_pt = self.parallax_project_real_to_ground((x, y), h)
        else:
            flat_pt = (x, y)
        u_u, v_u = self.flat_sat_to_undistorted(flat_pt[0], flat_pt[1])
        return self.undistorted_to_cctv(u_u, v_u)

    def get_ground_contact_from_box(self, rect, h_meters, ref_method="center_bottom_side", proj_method="down_h"):
        if hasattr(rect, 'x'):
            rx, ry, rw, rh = rect.x(), rect.y(), rect.width(), rect.height()
        else:
            rx, ry, rw, rh = rect
        cx = rx + rw / 2
        cy = ry + rh if ref_method == "center_bottom_side" else ry + rh / 2

        apparent_sat = self.cctv_to_sat(cx, cy, h=0)
        final_sat = apparent_sat
        if proj_method == "down_h":
            final_sat = self.parallax_correct_ground_to_real(apparent_sat, h_meters)
        elif proj_method == "down_h_2":
            final_sat = self.parallax_correct_ground_to_real(apparent_sat, h_meters / 2.0)

        gc_cctv = self.sat_to_cctv(final_sat[0], final_sat[1], h=0)
        return {"sat_coords": final_sat, "cctv_ref_point": (cx, cy), "cctv_ground_point": gc_cctv}

    def sat_floor_to_cctv_3d(self, sat_poly, obj_height_m):
        """SAT 바닥 4점 → CCTV 픽셀 3D 박스 8점(바닥4+천장4)."""
        undistorted_ground_pts = []
        for p in sat_poly:
            u_u, v_u = self.flat_sat_to_undistorted(p[0], p[1])
            undistorted_ground_pts.append([u_u, v_u])

        undistorted_top_pts = []
        if abs(self.z_cam - obj_height_m) < 0.01:
            factor = 1.0
        else:
            factor = self.z_cam / (self.z_cam - obj_height_m)
        c_sat = self.cam_sat
        for p in sat_poly:
            p_arr = np.array(p)
            apparent_pt = c_sat + (p_arr - c_sat) * factor
            u_u, v_u = self.flat_sat_to_undistorted(apparent_pt[0], apparent_pt[1])
            undistorted_top_pts.append([u_u, v_u])

        all_undist = np.array(undistorted_ground_pts + undistorted_top_pts, dtype=np.float64)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        obj_pts = np.array([[(pt[0] - cx) / fx, (pt[1] - cy) / fy, 1.0] for pt in all_undist], dtype=np.float64)
        distorted, _ = cv2.projectPoints(obj_pts, np.zeros(3), np.zeros(3), self.K, self.D)
        return distorted.reshape(-1, 2).tolist()
