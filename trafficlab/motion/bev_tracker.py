"""
BEV(지면 좌표) 다중객체 추적기 — OC-SORT 아이디어를 지면 평면에 적용.

왜 지면인가:
  오블리크 CCTV는 원근 때문에 같은 차량도 화면 아래=크고 위=작다. 이미지 IoU 연관은
  원거리에서 박스가 작고 겹쳐 급격히 불안정해진다(우리 실측: 트랙 42%가 0.5초 미만).
  캘리브레이션이 있으면 검출을 지면(미터)으로 투영해 **거리·속도로 연관**할 수 있고,
  이때 원근 왜곡이 사라져 원거리/근거리가 동일 기준이 된다. (cf. UCMCTrack)

OC-SORT(CVPR'23)에서 가져온 요소:
  - ORU (Observation-centric Re-Update): 가림 후 재연결 시 마지막 관측↔새 관측을 잇는
    가상 궤적으로 KF 상태를 재설정 → 가림 구간의 오차 누적 제거
  - OCM (Observation-centric Momentum): 관측 기반 진행방향 일관성을 연관 비용에 추가
  - OCR (Observation-centric Recovery): KF 예측이 아닌 '마지막 관측' 위치로 2차 연관 시도
  + ByteTrack식 2단계 연관(고신뢰 → 저신뢰)

좌표계: 로컬 미터(= G_projection 의 sat_coords, px_per_meter=1 규약).
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


class _Track:
    __slots__ = ("id", "x", "P", "hits", "age", "tsu", "last_obs", "last_obs_f",
                 "obs_hist", "conf", "cls")

    def __init__(self, tid, z, frame, dt, conf, cls):
        self.id = tid
        self.x = np.array([z[0], z[1], 0.0, 0.0], float)   # [x, y, vx, vy] (m, m/s)
        self.P = np.diag([1.0, 1.0, 25.0, 25.0])
        self.hits = 1
        self.age = 0
        self.tsu = 0                                        # time since update (frames)
        self.last_obs = np.array(z, float)
        self.last_obs_f = frame
        self.obs_hist = [(frame, np.array(z, float))]
        self.conf = conf
        self.cls = cls

    # --- Kalman (등속 모델) ---
    def predict(self, dt, q=1.0):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.x = F @ self.x
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]], float)
        self.P = F @ self.P @ F.T + G @ (q * np.eye(2)) @ G.T
        self.age += 1
        self.tsu += 1
        return self.x[:2]

    def update(self, z, frame, dt, conf, cls, R=None):
        """R: 측정 공분산(2x2). 투영 불확실성(이방성)을 그대로 반영."""
        z = np.array(z, float)
        if R is None:
            R = 0.5 * np.eye(2)
        # ORU: 오래 끊겼다가 재연결되면 가상 궤적으로 상태 재설정(오차 누적 차단)
        gap = frame - self.last_obs_f
        if gap >= 3:
            v = (z - self.last_obs) / max(gap * dt, 1e-6)
            self.x = np.array([z[0], z[1], v[0], v[1]], float)
            self.P = np.block([[R, np.zeros((2, 2))], [np.zeros((2, 2)), 9.0 * np.eye(2)]])
        else:
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ (z - H @ self.x)
            self.P = (np.eye(4) - K @ H) @ self.P
        self.last_obs = z
        self.last_obs_f = frame
        self.obs_hist.append((frame, z))
        if len(self.obs_hist) > 30:
            self.obs_hist.pop(0)
        self.hits += 1
        self.tsu = 0
        self.conf = conf
        self.cls = cls

    def direction(self, delta=3):
        """OCM용: 최근 관측 기반 진행방향 단위벡터(없으면 None)."""
        if len(self.obs_hist) < 2:
            return None
        f0, p0 = self.obs_hist[max(0, len(self.obs_hist) - 1 - delta)]
        f1, p1 = self.obs_hist[-1]
        d = p1 - p0
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else None


class BEVTracker:
    """지면 좌표 기반 추적기.

    update(dets, frame) 의 dets: [{"pos": (x,y) 미터, "conf": float, "cls": str}, ...]
    반환: dets 와 같은 길이의 track_id 리스트(미할당은 None)
    """

    def __init__(self, dt=1 / 30, max_dist=15.0, max_age=30, min_hits=3,
                 high_thresh=0.5, low_thresh=0.1, ocm_weight=2.0, class_aware=True,
                 mahalanobis=True, chi2_gate=9.21):
        self.dt = dt
        self.mahalanobis = mahalanobis    # 투영 공분산 기반 정규화 거리(권장)
        self.chi2_gate = chi2_gate        # 2 DOF 카이제곱 99% = 9.21
        self.max_dist = max_dist          # 물리적 상한(m)
        self.max_age = max_age            # 미관측 허용 프레임
        self.min_hits = min_hits
        self.high = high_thresh
        self.low = low_thresh
        self.ocm_w = ocm_weight           # 방향 불일치 패널티(m 환산)
        self.class_aware = class_aware
        self.tracks = []
        self._next = 1

    # --- 연관 비용: (마할라노비스 | 유클리드) 거리 + OCM 방향 패널티 (+ 클래스 패널티) ---
    def _cost(self, tr, det, pred_pos):
        delta = np.array(det["pos"], float) - np.asarray(pred_pos, float)
        eu = float(np.linalg.norm(delta))
        if eu > self.max_dist:            # 물리적 상한(터무니없는 연관 차단)
            return None
        cov = det.get("cov")
        if self.mahalanobis and cov is not None:
            # 투영 불확실성은 이방성(종방향 σ가 횡방향의 3~4배) → 정규화 거리로 차로 구분 보존
            S = tr.P[:2, :2] + np.asarray(cov, float)
            try:
                m2 = float(delta @ np.linalg.inv(S) @ delta)
            except np.linalg.LinAlgError:
                return None
            if m2 > self.chi2_gate:       # 2자유도 카이제곱 게이트
                return None
            c = math.sqrt(max(m2, 0.0))
        else:
            c = eu
        u = tr.direction()
        if u is not None:
            v = np.array(det["pos"]) - tr.last_obs
            n = np.linalg.norm(v)
            if n > 0.3:                                   # 정지 객체는 방향 무의미
                cos = float(np.clip(u @ (v / n), -1, 1))
                c += self.ocm_w * (1 - cos) / 2           # 0(동일방향)~ocm_w(반대)
        if self.class_aware and tr.cls and det.get("cls") and tr.cls != det["cls"]:
            c += 1.0
        return c

    def _associate(self, tracks, dets, positions):
        """(matches, unmatched_tracks, unmatched_dets) — positions: 트랙별 기준 위치."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        C = np.full((len(tracks), len(dets)), np.inf)
        for i, tr in enumerate(tracks):
            for j, d in enumerate(dets):
                c = self._cost(tr, d, positions[i])
                if c is not None:
                    C[i, j] = c
        big = 1e6
        Cf = np.where(np.isinf(C), big, C)
        ri, ci = linear_sum_assignment(Cf)
        matches, ut, ud = [], [], []
        mt, md = set(), set()
        for i, j in zip(ri, ci):
            if np.isinf(C[i, j]):
                continue
            matches.append((i, j)); mt.add(i); md.add(j)
        ut = [i for i in range(len(tracks)) if i not in mt]
        ud = [j for j in range(len(dets)) if j not in md]
        return matches, ut, ud

    def update(self, dets, frame):
        # 1) 예측
        preds = [tr.predict(self.dt) for tr in self.tracks]
        assigned = [None] * len(dets)

        hi = [j for j, d in enumerate(dets) if d.get("conf", 1.0) >= self.high]
        lo = [j for j, d in enumerate(dets) if self.low <= d.get("conf", 1.0) < self.high]

        # 2) 1단계: 고신뢰 검출 ↔ 전체 트랙 (KF 예측 위치)
        tr_idx = list(range(len(self.tracks)))
        m1, ut1, ud1 = self._associate([self.tracks[i] for i in tr_idx],
                                       [dets[j] for j in hi], [preds[i] for i in tr_idx])
        for a, b in m1:
            i, j = tr_idx[a], hi[b]
            self.tracks[i].update(dets[j]["pos"], frame, self.dt, dets[j].get("conf", 1), dets[j].get("cls"), dets[j].get("cov"))
            assigned[j] = self.tracks[i].id

        # 3) 2단계: 남은 트랙 ↔ 저신뢰 검출 (ByteTrack 아이디어)
        rem = [tr_idx[a] for a in ut1]
        if rem and lo:
            m2, ut2, _ = self._associate([self.tracks[i] for i in rem],
                                         [dets[j] for j in lo], [preds[i] for i in rem])
            for a, b in m2:
                i, j = rem[a], lo[b]
                self.tracks[i].update(dets[j]["pos"], frame, self.dt, dets[j].get("conf", 1), dets[j].get("cls"), dets[j].get("cov"))
                assigned[j] = self.tracks[i].id
            rem = [rem[a] for a in ut2]

        # 4) OCR: 아직 못 붙은 트랙을 '마지막 관측' 위치 기준으로 재시도
        left_dets = [j for j in range(len(dets)) if assigned[j] is None and dets[j].get("conf", 1) >= self.low]
        if rem and left_dets:
            m3, _, _ = self._associate([self.tracks[i] for i in rem],
                                       [dets[j] for j in left_dets],
                                       [self.tracks[i].last_obs for i in rem])
            for a, b in m3:
                i, j = rem[a], left_dets[b]
                self.tracks[i].update(dets[j]["pos"], frame, self.dt, dets[j].get("conf", 1), dets[j].get("cls"), dets[j].get("cov"))
                assigned[j] = self.tracks[i].id

        # 5) 신규 트랙(고신뢰 미할당 검출만)
        for j, d in enumerate(dets):
            if assigned[j] is None and d.get("conf", 1.0) >= self.high:
                t = _Track(self._next, d["pos"], frame, self.dt, d.get("conf", 1), d.get("cls"))
                self._next += 1
                self.tracks.append(t)
                assigned[j] = t.id

        # 6) 오래된 트랙 제거
        self.tracks = [t for t in self.tracks if t.tsu <= self.max_age]

        # 7) min_hits 미만(미확정) 트랙 ID는 출력하지 않음 → 오탐이 ID를 낭비하지 않게
        confirmed = {t.id for t in self.tracks if t.hits >= self.min_hits or t.age <= self.min_hits}
        return [tid if (tid in confirmed) else None for tid in assigned]


def ground_cov(gproj, u, v, h=0.0, sigma_px=3.0):
    """픽셀 잡음(sigma_px)을 투영 야코비안으로 전파해 지면 공분산(2x2, m^2) 산출.

    오블리크 CCTV에서는 종방향(깊이) 불확실성이 횡방향의 수 배라, 이 공분산을 쓰면
    '차로 구분은 유지하면서 깊이 오차는 관용'하는 연관이 가능하다.
    """
    import numpy as _np
    p0 = _np.array(gproj.cctv_to_sat(u, v, h=h), float)
    eps = 1.0
    ju = (_np.array(gproj.cctv_to_sat(u + eps, v, h=h), float) - p0) / eps
    jv = (_np.array(gproj.cctv_to_sat(u, v + eps, h=h), float) - p0) / eps
    J = _np.stack([ju, jv], axis=1)
    return J @ (sigma_px ** 2 * _np.eye(2)) @ J.T
