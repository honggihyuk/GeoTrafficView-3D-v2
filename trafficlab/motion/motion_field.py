"""
운동 필드 — "이 차로의 이 지점에서 차들은 평균적으로 얼마나 빠른가"를 관측에서 학습한다.

**무엇을 학습하나**
  방향은 학습할 게 없다. 차로 접선이 곧 방향이고, 그건 정밀도로지도가 이미 정확히 준다.
  학습이 필요한 건 **종방향 속도 v(link, s)** 뿐이다. 여기에 분기 선택 확률(turn_counts)과
  차간 헤드웨이를 더하면 "차로를 이탈하지 않고 자연스럽게 움직이는" 데 필요한 게 다 나온다.

**왜 Frenet 종방향 속도인가**
  2D 평면 속도(|Δp|/Δt)는 횡방향 검출 지터를 그대로 먹는다. 맵매칭이 이미 각 관측을
  (link, s, d)로 바꿔 놨으므로 **ds/dt** 를 쓰면 횡방향 잡음이 구조적으로 빠진다.
  같은 데이터에서 훨씬 안정적인 추정치가 나온다.

**희소 구간 처리 (시각 품질이 실측보다 중요하다는 전제)**
  관측이 없는 구간을 비워 두면 차가 그 자리에서 멈추거나 튄다. 그래서 계단식으로 메운다:
    ① (link, s) 관측 중앙값  ② 같은 링크 내 s축 보간+평활  ③ 링크 전체 중앙값
    ④ 도로등급(road_rank) 중앙값  ⑤ 전역 기본값
  각 bin이 어느 단계에서 왔는지 src 로 기록해, 어디가 관측이고 어디가 추정인지 남긴다.
"""
import gzip
import json
import math
import os
from collections import defaultdict

import numpy as np

BIN_M = 10.0            # s축 격자 간격(m)
MAX_KMH = 160.0         # 이보다 빠른 관측은 투영 잡음으로 보고 버린다
MIN_KMH = 0.0
DEFAULT_KMH = 50.0


def _robust(v):
    """중앙값 + 사분위. 표본이 적을 때 평균보다 정직하다."""
    a = np.asarray(v, float)
    return (float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75)))


def build(replay_path, bin_m=BIN_M, min_n=3, smooth=2, clamp=(15.0, 120.0)):
    """맵매칭 결과 → 운동 필드. 반환 dict 는 그대로 JSON 직렬화 가능."""
    d = json.load(gzip.open(replay_path, "rt", encoding="utf-8"))
    fps = float(d["meta"].get("fps", 30))
    links_meta = d["meta"].get("lane_links", {})       # id -> [bear0, bear1, length, lane_no, directed]

    # --- 트랙별 (t, link, s) 시퀀스 ---
    tr = defaultdict(list)
    for fr in d["frames"]:
        for o in fr.get("objects", []):
            tid, lid, fre = o.get("tracked_id"), o.get("link_id"), o.get("frenet")
            if tid is None or lid is None or not fre:
                continue
            tr[tid].append((fr["frame_index"] / fps, str(lid), float(fre[0]), float(fre[1]),
                            o.get("class")))
    for v in tr.values():
        v.sort()

    # --- ds/dt 누적. **트랙 단위로 먼저 안정화한 뒤** (link, s_bin) 에 넣는다 ---
    # 프레임쌍 ds/dt 를 그대로 쌓으면 검출 지터가 만든 '거의 0' 표본이 중앙값을 끌어내린다
    # (실측 PANGYO_2: 경부고속도로인데 중앙값 31km/h, 표본의 17%가 물리적 이상치).
    # 한 차량은 몇 초 사이 속도가 크게 변하지 않으므로, 트랙×bin 단위로 한 번 중앙값을
    # 취해 대표값 1개만 기여시킨다. 이러면 "이 지점을 지나는 차들의 대표 속도"가 된다.
    per_track_bin = defaultdict(list)   # (tid, lid, bin) -> [kmh]
    per_track = defaultdict(list)       # tid -> [kmh]
    lat = defaultdict(list)
    n_pair = n_drop = 0
    for tid, rows in tr.items():
        for (t0, l0, s0, d0, _), (t1, l1, s1, d1, _) in zip(rows, rows[1:]):
            lat[l0].append(abs(d0))
            dt = t1 - t0
            if dt <= 1e-6 or l0 != l1:
                continue                      # 링크가 바뀐 구간은 s 가 불연속이라 제외
            n_pair += 1
            kmh = (s1 - s0) / dt * 3.6
            if not (MIN_KMH <= kmh <= MAX_KMH):
                n_drop += 1
                continue
            per_track_bin[(tid, l0, int(s0 // bin_m))].append(kmh)
            per_track[tid].append(kmh)

    track_med = {t: float(np.median(v)) for t, v in per_track.items() if len(v) >= 2}
    obs = defaultdict(list)          # (lid, bin) -> [트랙별 대표 kmh]
    for (tid, lid, b), v in per_track_bin.items():
        # bin 내 표본이 얕으면 트랙 전체 중앙값으로 대체(트랙 하나가 여러 표를 던지지 않게)
        obs[(lid, b)].append(float(np.median(v)) if len(v) >= 3 else track_med.get(tid, float(np.median(v))))

    # --- 링크별 프로파일 ---
    link_med = {}
    for lid in {k[0] for k in obs}:
        vs = [x for (l, b), v in obs.items() if l == lid for x in v]
        if len(vs) >= min_n:
            link_med[lid] = float(np.median(vs))
    global_med = float(np.median([x for v in obs.values() for x in v])) if obs else DEFAULT_KMH

    out_links, clamped_bins = {}, {}
    for lid, meta in (links_meta.items() if links_meta else
                      {l: [0, 0, 200.0, None, True] for l in {k[0] for k in obs}}.items()):
        length = float(meta[2]) if len(meta) > 2 else 200.0
        nb = max(1, int(math.ceil(length / bin_m)))
        v = np.full(nb, np.nan)
        n = np.zeros(nb, int)
        src = ["none"] * nb
        for b in range(nb):
            o = obs.get((lid, b))
            if o and len(o) >= min_n:
                v[b], _, _ = _robust(o)
                n[b] = len(o)
                src[b] = "observed"
        # ② 링크 내 s축 보간(관측 사이를 잇는다) — 빈 구간에서 차가 멈추지 않게
        idx = np.where(~np.isnan(v))[0]
        if len(idx) >= 2:
            v = np.interp(np.arange(nb), idx, v[idx])
            for b in range(nb):
                if src[b] == "none":
                    src[b] = "interp"
        # ③④⑤ 링크 → 전역 폴백
        if np.isnan(v).any():
            fb = link_med.get(lid, global_med)
            for b in range(nb):
                if np.isnan(v[b]):
                    v[b] = fb
                    src[b] = "link_median" if lid in link_med else "global"
        # 평활 — 시각적으로 급가속이 보이지 않게(이동평균)
        if smooth > 0 and nb > 2:
            k = 2 * smooth + 1
            pad = np.pad(v, smooth, mode="edge")
            v = np.convolve(pad, np.ones(k) / k, mode="valid")
        # 타당성 클램프 — 캘리브레이션이 어긋난 링크는 0km/h나 130km/h 같은 값을 만든다
        # (실측 PANGYO_2 링크 중앙값: 0, 7, …, 99, 129). 화면에서 차가 멈춰 서 있거나
        # 순간이동하는 것보다는 클램프가 낫다는 판단이다. **측정치가 아니라 표시값**임을
        # 잊지 않도록 clamped 수를 기록한다.
        if clamp:
            n_cl = int(((v < clamp[0]) | (v > clamp[1])).sum())
            if n_cl:
                clamped_bins[lid] = n_cl
            v = np.clip(v, clamp[0], clamp[1])
        out_links[lid] = {"length": round(length, 2), "bin_m": bin_m,
                          "v_kmh": [round(float(x), 2) for x in v],
                          "n": [int(x) for x in n], "src": src,
                          "lat_med": round(float(np.median(lat[lid])), 2) if lat.get(lid) else None}

    n_obs_bin = sum(1 for L in out_links.values() for s in L["src"] if s == "observed")
    n_bin = sum(len(L["src"]) for L in out_links.values())
    return {"bin_m": bin_m, "global_kmh": round(global_med, 2),
            "link_median_kmh": {k: round(v, 2) for k, v in link_med.items()},
            "links": out_links,
            "clamp_kmh": list(clamp) if clamp else None,
            "stats": {"tracks": len(tr), "pairs": n_pair, "dropped": n_drop,
                      "bins": n_bin, "bins_observed": n_obs_bin,
                      "coverage": round(n_obs_bin / max(n_bin, 1), 4),
                      "bins_clamped": int(sum(clamped_bins.values())),
                      "links_clamped": len(clamped_bins)}}


class MotionField:
    """런타임 조회. 합성 이동 엔진이 (link, s) 에서 속도를 읽는다."""

    def __init__(self, data):
        self.d = data
        self.bin_m = data.get("bin_m", BIN_M)
        self.default = data.get("global_kmh", DEFAULT_KMH)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def speed_kmh(self, link_id, s):
        """(link, s) 에서 기대 속도. bin 사이를 선형보간해 계단이 보이지 않게 한다."""
        L = self.d["links"].get(str(link_id))
        if not L:
            return self.default
        v = L["v_kmh"]
        if not v:
            return self.default
        x = max(0.0, min(float(s) / self.bin_m - 0.5, len(v) - 1.0))
        i = int(x)
        j = min(i + 1, len(v) - 1)
        f = x - i
        return v[i] * (1 - f) + v[j] * f

    def source(self, link_id, s):
        L = self.d["links"].get(str(link_id))
        if not L or not L["src"]:
            return "global"
        i = max(0, min(int(float(s) / self.bin_m), len(L["src"]) - 1))
        return L["src"][i]
