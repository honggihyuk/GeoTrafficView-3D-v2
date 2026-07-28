"""
차선 그래프 + Frenet 좌표 — 맵매칭·차로구속 추적의 공용 기반.

**왜 Frenet인가**
  차량은 도로를 '따라' 움직이지 평면을 자유롭게 움직이지 않는다. 위치를 (x, y) 대신
  (s, d) = (차로를 따라 간 거리, 횡방향 오프셋)으로 두면
    - heading이 차로 접선에서 **유도**된다 → EMA·점프클램프 같은 후처리가 불필요해진다
    - 프로세스 잡음을 종방향에 크게·횡방향에 작게 줄 수 있다 → 연관이 안정된다
    - 가림 구간 예측이 직선이 아니라 **차로 곡선을 따라간다** → 교차로에서 엉뚱한 데서
      다시 나타나지 않는다

**좌표계**: 로컬 미터(= replay meta.world 원점 기준, px_per_meter=1 규약).

차선 소스:
  1) 정밀도로지도 A2_LINK — 링크 1개 = 차로 1개, 지오메트리 진행 방향 = 주행 방향
     (tools/export_hdmap_snap.py 가 A1_NODE 좌표로 검증·정규화해 둔다)
  2) OSM 도로 중심선 — HD맵 미보유 지역 폴백. oneway 태그가 있을 때만 방향을 신뢰한다.
"""
import heapq
import json
import math
import os
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HDMAP_GEOJSON = os.path.join(REPO, "webmap", "public", "data", "hdmap", "hdmap.geojson")

SAMPLE_STEP = 2.0      # 링크 리샘플 간격(m) — KDTree 후보 탐색 해상도
OSM_MIRRORS = ("https://overpass-api.de/api/interpreter",
               "https://overpass.kumi.systems/api/interpreter",
               "https://overpass.osm.jp/api/interpreter")


# ---------------- 차선 네트워크 로더 ----------------
def lanes_from_hdmap(world, max_km=2.0, require_within=150.0, verbose=True):
    """정밀도로지도 A2_LINK → [{pts, directed, lane_no, link_id, from_node, to_node}].

    require_within: 카메라 원점에서 이 거리(m) 안에 링크가 하나도 없으면 **커버리지 밖**으로
    보고 []를 반환한다. 2km 안에 링크가 있다는 이유만으로 HD맵을 택하면, 정작 카메라가
    보는 도로는 HD맵에 없어 스냅률 0%로 실패한다(송도지구 HD맵 ↔ 송도IC 카메라가 실제 사례).
    """
    if not os.path.exists(HDMAP_GEOJSON):
        return []
    tr = Transformer.from_crs(4326, world["epsg"], always_xy=True)
    oE, oN = world["origin_easting"], world["origin_northing"]
    out, nearest = [], float("inf")
    with open(HDMAP_GEOJSON, encoding="utf-8") as f:
        fc = json.load(f)
    for feat in fc.get("features", []):
        pr = feat.get("properties", {})
        if pr.get("layer") != "A2_LINK":
            continue
        g = feat["geometry"]
        if g["type"] != "LineString":
            continue
        pts = []
        for lon, lat, *_ in g["coordinates"]:
            e, n = tr.transform(lon, lat)
            pts.append((e - oE, n - oN))
        if len(pts) < 2:
            continue
        d = min(math.hypot(*q) for q in pts)
        nearest = min(nearest, d)
        if d < max_km * 1000:
            # 지오메트리 진행 방향 = 주행 방향 (export 단계에서 From→To 로 정규화됨)
            out.append({"pts": pts, "directed": True, "lane_no": pr.get("lane_no"),
                        "link_id": pr.get("id"), "from_node": pr.get("from_node"),
                        "to_node": pr.get("to_node")})
    if out and nearest > require_within:
        if verbose:
            print(f"  HD맵 커버리지 밖 — 최근접 A2_LINK가 원점에서 {nearest:.0f}m "
                  f"(기준 {require_within:.0f}m). OSM 폴백으로 전환합니다.")
        return []
    return out


def lanes_from_osm(world, radius=400, verbose=True):
    """OSM highway 중심선 → 같은 스키마. oneway=yes 인 way만 방향을 신뢰한다.

    OSM way는 노드 ID를 주므로 시종점 노드로 토폴로지를 만든다.
    """
    tr_ll = Transformer.from_crs(world["epsg"], 4326, always_xy=True)
    lon, lat = tr_ll.transform(world["origin_easting"], world["origin_northing"])
    q = (f"[out:json][timeout:25];way(around:{radius},{lat},{lon})"
         f"[highway~'^(motorway|trunk|primary|secondary|tertiary|residential|unclassified"
         f"|motorway_link|trunk_link|primary_link)$'];out geom;")
    data = None
    for m in OSM_MIRRORS:
        req = urllib.request.Request(m + "?" + urllib.parse.urlencode({"data": q}),
                                     headers={"User-Agent": "GeoTrafficView-3D/2.1"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            if verbose:
                print(f"  OSM 실패({m.split('/')[2]}: {type(e).__name__}) → 다음 미러")
    if data is None:
        return []
    tr = Transformer.from_crs(4326, world["epsg"], always_xy=True)
    oE, oN = world["origin_easting"], world["origin_northing"]
    out = []
    for w in data.get("elements", []):
        g = w.get("geometry") or []
        nd = w.get("nodes") or []
        pts = []
        for p in g:
            e, n = tr.transform(p["lon"], p["lat"])
            pts.append((e - oE, n - oN))
        if len(pts) < 2:
            continue
        oneway = str(w.get("tags", {}).get("oneway", "")).lower() in ("yes", "true", "1")
        out.append({"pts": pts, "directed": oneway, "lane_no": None, "link_id": w.get("id"),
                    "from_node": nd[0] if nd else None, "to_node": nd[-1] if nd else None})
    return out


# ---------------- 그래프 ----------------
class _Link:
    __slots__ = ("idx", "id", "pts", "s", "tan", "length", "directed", "lane_no",
                 "from_node", "to_node")

    def __init__(self, idx, raw):
        pts = np.asarray(raw["pts"], float)
        # 중복점 제거 후 균일 리샘플 — 접선/Frenet 계산을 안정화한다.
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6])
        pts = pts[keep]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        L = float(s[-1])
        n = max(2, int(math.ceil(L / SAMPLE_STEP)) + 1)
        su = np.linspace(0.0, L, n)
        self.pts = np.stack([np.interp(su, s, pts[:, 0]), np.interp(su, s, pts[:, 1])], axis=1)
        self.s = su
        self.length = L
        d = np.gradient(self.pts, axis=0)
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        self.tan = d / np.maximum(nrm, 1e-9)
        self.idx = idx
        self.id = raw.get("link_id")
        self.directed = bool(raw.get("directed"))
        self.lane_no = raw.get("lane_no")
        self.from_node = raw.get("from_node")
        self.to_node = raw.get("to_node")

    def at(self, s):
        """s(m) → (위치, 접선). 링크 밖 s는 양끝으로 클램프."""
        s = float(np.clip(s, 0.0, self.length))
        p = np.array([np.interp(s, self.s, self.pts[:, 0]), np.interp(s, self.s, self.pts[:, 1])])
        i = int(np.clip(np.searchsorted(self.s, s), 1, len(self.s) - 1))
        t = self.tan[i]
        return p, t / max(np.linalg.norm(t), 1e-9)


class LaneGraph:
    """차선 폴리라인 집합 + 위상 + Frenet 연산."""

    def __init__(self, lanes, junction_tol=3.0):
        self.links = [_Link(i, r) for i, r in enumerate(lanes) if len(r["pts"]) >= 2]
        if not self.links:
            raise ValueError("링크 없음")
        # 전체 샘플점 KDTree (후보 탐색용)
        pts, owner, sidx = [], [], []
        for lk in self.links:
            pts.append(lk.pts)
            owner.append(np.full(len(lk.pts), lk.idx))
            sidx.append(np.arange(len(lk.pts)))
        self.P = np.concatenate(pts)
        self.owner = np.concatenate(owner)
        self.sidx = np.concatenate(sidx)
        self.tree = cKDTree(self.P)
        self._build_topology(junction_tol)
        self._gap_cache = {}

    # --- 위상 ---
    def _build_topology(self, tol):
        """후속 링크: 노드 ID가 있으면 ID로, 없으면 끝점 좌표 근접으로 잇는다."""
        by_from = defaultdict(list)
        have_nodes = 0
        for lk in self.links:
            if lk.from_node is not None:
                by_from[lk.from_node].append(lk.idx)
                have_nodes += 1
        self.succ = defaultdict(list)
        if have_nodes >= len(self.links) * 0.5:
            for lk in self.links:
                if lk.to_node is not None:
                    self.succ[lk.idx] = list(by_from.get(lk.to_node, []))
            self.topology_src = "node_id"
        else:
            ends = np.array([lk.pts[-1] for lk in self.links])
            starts = np.array([lk.pts[0] for lk in self.links])
            t = cKDTree(starts)
            for lk in self.links:
                self.succ[lk.idx] = [j for j in t.query_ball_point(ends[lk.idx], tol) if j != lk.idx]
            self.topology_src = f"endpoint<{tol:g}m"
        # 양방향 링크는 역방향 진행도 허용해야 하므로 반대편 연결도 추가
        for lk in self.links:
            if not lk.directed:
                for j in list(self.succ[lk.idx]):
                    if lk.idx not in self.succ[j]:
                        self.succ[j].append(lk.idx)

    def n_succ(self):
        return sum(len(v) for v in self.succ.values())

    # --- Frenet ---
    def project(self, p, max_dist=25.0, k=6):
        """점(로컬 m) → 링크별 최근접 후보 [(link_idx, s, d, pos, tan)] 최대 k개.

        d는 부호 있는 횡방향 오프셋(좌측 +). 링크당 1개로 접어서 반환한다 —
        교차로에서 직진·좌회전 링크가 물리적으로 겹치므로 후보를 여러 개 남겨야 한다.
        """
        p = np.asarray(p, float)
        idx = self.tree.query_ball_point(p, max_dist + SAMPLE_STEP)
        if not idx:
            return []
        idx = np.asarray(idx, int)
        best = {}
        for i in idx:
            li = int(self.owner[i])
            d2 = float(np.sum((self.P[i] - p) ** 2))
            if li not in best or d2 < best[li][0]:
                best[li] = (d2, int(self.sidx[i]))
        out = []
        for li, (_, si) in best.items():
            lk = self.links[li]
            # 인접 샘플 구간에 정밀 투영
            lo, hi = max(0, si - 1), min(len(lk.pts) - 1, si + 1)
            bs, bd2, bpos = None, None, None
            for j in range(lo, hi):
                a, b = lk.pts[j], lk.pts[j + 1]
                ab = b - a
                L2 = float(ab @ ab)
                if L2 < 1e-12:
                    continue
                t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
                pos = a + t * ab
                d2 = float(np.sum((pos - p) ** 2))
                if bd2 is None or d2 < bd2:
                    bd2, bpos = d2, pos
                    bs = lk.s[j] + t * (lk.s[j + 1] - lk.s[j])
            if bs is None:
                continue
            dist = math.sqrt(bd2)
            if dist > max_dist:
                continue
            _, tan = lk.at(bs)
            nvec = np.array([-tan[1], tan[0]])          # 좌측 법선
            out.append((li, float(bs), float((p - bpos) @ nvec), bpos, tan))
        out.sort(key=lambda c: abs(c[2]))
        return out[:k]

    def project_onto(self, li, p):
        """**특정 링크**에 투영 → (s, d). 차로 고정(lane lock)에서 쓴다.

        project() 는 후보 중에서 고르지만, 차로를 하나로 묶은 뒤에는 '이 링크 위 어디인가'만
        필요하다. 후보 선택이 없으므로 나란한 차로로 새는 경로가 원천적으로 사라진다.
        """
        p = np.asarray(p, float)
        lk = self.links[li]
        seg = lk.pts
        ab = seg[1:] - seg[:-1]
        L2 = np.einsum("ij,ij->i", ab, ab)
        t = np.clip(np.einsum("ij,ij->i", p - seg[:-1], ab) / np.maximum(L2, 1e-12), 0.0, 1.0)
        proj = seg[:-1] + t[:, None] * ab
        d = np.linalg.norm(proj - p, axis=1)
        i = int(np.argmin(d))
        s = float(lk.s[i] + t[i] * (lk.s[i + 1] - lk.s[i]))
        _, tan = lk.at(s)
        nvec = np.array([-tan[1], tan[0]])
        return s, float((p - proj[i]) @ nvec)

    def to_xy(self, li, s, d=0.0):
        p, tan = self.links[li].at(s)
        return p + d * np.array([-tan[1], tan[0]])

    def tangent(self, li, s):
        return self.links[li].at(s)[1]

    def heading_deg(self, li, s):
        t = self.tangent(li, s)
        return (math.degrees(math.atan2(t[1], t[0])) + 360.0) % 360.0

    # --- 경로 ---
    def link_gap(self, a, b, limit=300.0):
        """링크 a의 끝 → 링크 b의 시작 최단거리(m). 도달 불가면 inf.

        Viterbi 전이확률에 쓰이므로 (a,b) 쌍을 캐시한다. 홉 제한 BFS라 비용이 작다.
        """
        if a == b:
            return 0.0
        key = (a, b)
        if key in self._gap_cache:
            return self._gap_cache[key]
        # a의 '끝'에서 출발하는 Dijkstra를 한 번 돌려 a발 전체를 캐시한다.
        # 거리 정의: a의 끝 → 후속 링크의 시작 = 0, 그 뒤로는 지나온 링크 길이 누적.
        dist = {a: 0.0}
        pq = [(0.0, a)]
        seen = set()
        while pq:
            dcur, cur = heapq.heappop(pq)
            if cur in seen:
                continue
            seen.add(cur)
            step = 0.0 if cur == a else self.links[cur].length
            for nx in self.succ.get(cur, ()):
                nd = dcur + step
                if nd > limit or nx in seen:
                    continue
                if nd < dist.get(nx, float("inf")):
                    dist[nx] = nd
                    heapq.heappush(pq, (nd, nx))
        for lk in self.links:
            self._gap_cache[(a, lk.idx)] = dist.get(lk.idx, float("inf"))
        self._gap_cache[(a, a)] = 0.0
        return self._gap_cache[key]

    def route_distance(self, st_a, st_b, limit=300.0):
        """(link,s) → (link,s) 도로를 따라간 거리(m). 도달 불가면 inf."""
        la, sa = st_a
        lb, sb = st_b
        if la == lb:
            if sb >= sa - 1.0:                     # 같은 링크에서 전진(1m 후퇴는 잡음 허용)
                return abs(sb - sa)
            return float("inf")
        gap = self.link_gap(la, lb, limit)
        if not np.isfinite(gap):
            return float("inf")
        return (self.links[la].length - sa) + gap + sb

    def advance(self, li, s, ds, prefer_tan=None):
        """(link,s)에서 도로를 따라 ds(m) 전진. 분기에서는 접선이 가장 비슷한 쪽을 고른다.

        가림 구간 coasting이 직선이 아니라 차로 곡선을 따르게 하는 핵심.
        ds<0(링크 진행 방향의 반대로 이동)은 선행 링크를 거슬러 올라가지 않고 링크 안에서만
        후퇴한다 — 방향성 링크에서는 애초에 일어나면 안 되는 경우이고, 양방향 도로에서는
        어차피 반대편 링크가 따로 잡히기 때문이다.
        """
        if ds < 0:
            return li, max(0.0, s + ds)
        for _ in range(8):                          # 링크 8개까지 넘어감(무한루프 방지)
            lk = self.links[li]
            if s + ds <= lk.length:
                return li, s + ds
            ds -= (lk.length - s)
            nxt = self.succ.get(li, [])
            if not nxt:
                return li, lk.length
            if len(nxt) == 1 or prefer_tan is None:
                li = nxt[0]
            else:
                li = max(nxt, key=lambda j: float(prefer_tan @ self.links[j].at(0.0)[1]))
            s = 0.0
        return li, s
