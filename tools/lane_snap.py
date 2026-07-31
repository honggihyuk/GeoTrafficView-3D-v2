"""
차선 정렬(lane snapping) — 이동체를 차로에 맞춰 정렬하고 **주행 방향을 확정**한다.

탐지·투영 결과는 검출 지터와 캘리브레이션 오차 때문에 도로를 벗어나거나 차로를 가로질러
흔들린다. 이 도구는 각 객체의 지면 접촉점을 **차선 네트워크에 횡방향 스냅**하고
**heading을 차로 방향에 정렬**해, 지도 위에서 차량이 차선을 따라 달리게 만든다.

**heading의 근거 (v2.1에서 바뀐 부분)**
  기존: heading = 세그먼트 방위, ±180 모호성은 **프레임별 관측 heading**으로 해소.
        → 정지 차량은 관측 heading이 없어 정면/후면을 정할 수 없었다.
  현재: A2_LINK는 **방향성 링크**(지오메트리 진행 = FromNode→ToNode = 주행 방향,
        export_hdmap_snap.py 가 A1_NODE 좌표로 검증·정규화)이므로 **링크 방향이 곧 정면**이다.
        관측 heading은 이제 뒤집기가 아니라 **트랙 단위 역주행 판정**에만 쓴다.
        → 신호 대기 중인 정지 차량도 정면/후면이 확정된다.
  OSM 폴백은 방향 정보가 약해(oneway 태그만) 양방향 링크에 한해 트랙 단위 다수결을 유지한다.

차선 네트워크 소스(우선순위):
  1) **정밀도로지도 A2_LINK** — webmap/public/data/hdmap/hdmap.geojson
     (tools/export_hdmap_snap.py 로 생성). 링크 자체가 **차로 1개**이므로 횡방향 양자화 없이
     링크 중심선에 바로 스냅한다(--lane-width 기본 0).
  2) **OSM 도로 중심선**(Overpass) — HD맵 미보유 지역 폴백. 중심선 기준이므로
     `--lane-width`(기본 3.5m)로 차로 중심에 양자화해 차로 단위 정렬을 근사한다.

사용:
  python tools/lane_snap.py --loc PANGYO_2                       # 웹맵 등록본 자동 사용
  python tools/lane_snap.py --loc SONGDO_IC --max-dist 20
출력: output/lanesnap/<loc>/clip.json.gz  (이후 bridge_to_webmap.py 로 반영)
"""
import argparse
import gzip
import json
import math
import os
import sys
import urllib.parse
import urllib.request
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from trafficlab.projection.g_projection import GProjection

MAX_SEG_LEN = 5.0   # 세그먼트 최대 길이(m) — KDTree 중점 후보 탐색이 유효하려면 짧아야 한다


# ---------- 차선 네트워크 ----------
def lanes_from_hdmap(world, max_km=2.0, require_within=150.0):
    """정밀도로지도 A2_LINK(방향성 차로 링크) → [{pts, directed, lane_no, link_id}].

    require_within: 카메라 원점에서 이 거리(m) 안에 링크가 하나도 없으면 **커버리지 밖**으로
    보고 []를 반환한다. 2km 안에 링크가 있다는 이유만으로 HD맵을 택하면, 정작 카메라가
    보는 도로는 HD맵에 없어 스냅률 0%로 실패한다(송도지구 HD맵 ↔ 송도IC 카메라가 실제 사례).
    """
    p = os.path.join(REPO, "webmap", "public", "data", "hdmap", "hdmap.geojson")
    if not os.path.exists(p):
        return []
    tr = Transformer.from_crs(4326, world["epsg"], always_xy=True)
    oE, oN = world["origin_easting"], world["origin_northing"]
    out = []
    nearest = float("inf")
    with open(p, encoding="utf-8") as f:
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
        if not pts:
            continue
        d = min(math.hypot(*q) for q in pts)
        nearest = min(nearest, d)
        if d < max_km * 1000:
            # 지오메트리 진행 방향 = 주행 방향 (export 단계에서 From→To 로 정규화됨)
            out.append({"pts": pts, "directed": True,
                        "lane_no": pr.get("lane_no"), "link_id": pr.get("id")})
    if out and nearest > require_within:
        print(f"  HD맵 커버리지 밖 — 최근접 A2_LINK가 원점에서 {nearest:.0f}m "
              f"(기준 {require_within:.0f}m). OSM 폴백으로 전환합니다.")
        return []
    return out


def lanes_from_osm(world, radius=400):
    """OSM highway 중심선 → [{pts, directed, ...}]. oneway=yes 인 way만 방향 신뢰."""
    tr_ll = Transformer.from_crs(world["epsg"], 4326, always_xy=True)
    lon, lat = tr_ll.transform(world["origin_easting"], world["origin_northing"])
    q = (f"[out:json][timeout:25];way(around:{radius},{lat},{lon})"
         f"[highway~'^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|motorway_link|trunk_link|primary_link)$'];out geom;")
    # 공개 Overpass는 rate-limit이 잦아 미러를 순차 재시도(UA 없으면 406)
    mirrors = ["https://overpass-api.de/api/interpreter",
               "https://overpass.kumi.systems/api/interpreter",
               "https://overpass.osm.jp/api/interpreter"]
    data = None
    for m in mirrors:
        req = urllib.request.Request(m + "?" + urllib.parse.urlencode({"data": q}),
                                     headers={"User-Agent": "GeoTrafficView-3D/2.1"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            print(f"  OSM 실패({m.split('/')[2]}: {type(e).__name__}) → 다음 미러")
    if data is None:
        return []
    tr = Transformer.from_crs(4326, world["epsg"], always_xy=True)
    oE, oN = world["origin_easting"], world["origin_northing"]
    out = []
    for w in data.get("elements", []):
        g = w.get("geometry") or []
        pts = []
        for nd in g:
            e, n = tr.transform(nd["lon"], nd["lat"])
            pts.append((e - oE, n - oN))
        if len(pts) >= 2:
            oneway = str(w.get("tags", {}).get("oneway", "")).lower() in ("yes", "true", "1")
            out.append({"pts": pts, "directed": oneway, "lane_no": None, "link_id": w.get("id")})
    return out


class LaneIndex:
    """세그먼트 배열 + 중점 KDTree. 최근접 세그먼트를 O(log n)에 찾는다."""

    def __init__(self, lanes):
        A, B, D, L = [], [], [], []
        for ln in lanes:
            pts = np.asarray(ln["pts"], float)
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                seg_len = float(np.linalg.norm(b - a))
                if seg_len < 1e-3:
                    continue
                # 긴 구간은 쪼갠다 — 중점 KDTree 후보 탐색의 전제
                k = max(1, int(math.ceil(seg_len / MAX_SEG_LEN)))
                for j in range(k):
                    p0 = a + (b - a) * (j / k)
                    p1 = a + (b - a) * ((j + 1) / k)
                    A.append(p0); B.append(p1)
                    D.append(ln["directed"]); L.append(ln["link_id"])
        if not A:
            raise ValueError("세그먼트 없음")
        self.A = np.asarray(A, float)
        self.B = np.asarray(B, float)
        self.directed = np.asarray(D, bool)
        self.link_id = L
        self.mid = (self.A + self.B) / 2
        self.tree = cKDTree(self.mid)

    def __len__(self):
        return len(self.A)

    def candidates(self, p, max_dist, k=6):
        """p(로컬 m) → 링크별 최근접 후보 [(거리, 세그idx, 투영점)] 최대 k개(거리 오름차순).

        **왜 최근접 1개가 아닌가**: 교차로 내부에는 직진·좌회전·우회전 링크가 물리적으로
        교차한다. 실측(송도 HD맵) 결과 링크 위 점의 16%가 1m 이내 거리에서 방위가 140° 이상
        다른 링크에 붙었다. 거리만으로는 구분 불가이므로, 진행 방향 일관성을 함께 봐야 한다.
        후보는 **링크당 1개**로 접어서 반환한다(같은 링크의 연속 세그먼트가 자리를 다 먹지 않게).
        """
        p = np.asarray(p, float)
        idx = self.tree.query_ball_point(p, max_dist + MAX_SEG_LEN)
        if not idx:
            return []
        idx = np.asarray(idx, int)
        a, b = self.A[idx], self.B[idx]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(p - proj, axis=1)
        keep = d <= max_dist
        if not keep.any():
            return []
        idx, proj, d = idx[keep], proj[keep], d[keep]
        out, seen = [], set()
        for i in np.argsort(d):
            lid = self.link_id[idx[i]]
            key = lid if lid is not None else int(idx[i])
            if key in seen:
                continue
            seen.add(key)
            out.append((float(d[i]), int(idx[i]), proj[i]))
            if len(out) >= k:
                break
        return out

    def nearest(self, p, max_dist, k=6):
        """거리만 기준인 최근접 1개. (방향 무시 — 진단·비교용)"""
        c = self.candidates(p, max_dist, k=1)
        return c[0] if c else None

    def unit(self, si):
        u = self.B[si] - self.A[si]
        return u / np.linalg.norm(u)


def _bearing(v):
    return (math.degrees(math.atan2(v[1], v[0])) + 360.0) % 360.0


def _angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def build_plan(lanes, kind, is_hd, data, args):
    """차선망 하나에 대해 '어떻게 스냅할지'를 계산만 한다(객체는 건드리지 않음).

    계산과 반영을 분리해 둔 이유: HD맵과 OSM 두 후보를 **같은 기준으로 채점한 뒤**
    좋은 쪽만 반영하기 위해서다. HD맵이 카메라가 보는 도로를 담고 있지 않고 직교하는
    교차 도로만 담고 있는 경우가 실제로 있다(실측: YEONSU_JCT — 방향차 89.5°).
    """
    lane_w = args.lane_width if args.lane_width is not None else (0.0 if is_hd else 3.5)
    index = LaneIndex(lanes)

    # ---------- 1패스: 후보 수집 ----------
    hits, n_obj = [], 0
    for fr in data["frames"]:
        for o in fr.get("objects", []):
            sat = o.get("sat_coords")
            if not sat:
                continue
            n_obj += 1
            cand = index.candidates(sat, args.max_dist)
            if not cand:
                continue
            obs = o.get("heading") if o.get("have_heading") else None
            spd = float(o.get("speed_kmh") or 0.0)
            hits.append({"o": o, "sat": np.asarray(sat, float), "cand": cand,
                         "tid": o.get("tracked_id"),
                         "obs": obs if (obs is not None and spd >= 5.0) else None,
                         "spd": spd})
    if not hits:
        return None

    # ---------- 2패스: 트랙 단위 기준 방향 ----------
    # 트랙의 '움직이던 프레임' 관측 heading을 원형평균해 트랙 기준 방향을 만든다.
    # 이 기준은 ① 교차 링크 중 올바른 것을 고르는 데, ② 정지 프레임에 방향 맥락을
    # 물려주는 데 쓰인다. (정지 차량이 좌회전 링크에 잘못 붙는 것을 막는 핵심)
    by_tid = defaultdict(list)
    for h in hits:
        by_tid[h["tid"]].append(h)

    track_ref = {}
    for tid, hs in by_tid.items():
        v = np.zeros(2)
        for h in hs:
            if h["obs"] is None:
                continue
            rad = math.radians(h["obs"])
            v += np.array([math.cos(rad), math.sin(rad)])
        n = np.linalg.norm(v)
        track_ref[tid] = (v / n) if n > 1e-6 else None

    def ref_dir(h):
        """이 관측에 적용할 기준 방향: 자기 관측 우선, 없으면 트랙 기준."""
        if h["obs"] is not None:
            rad = math.radians(h["obs"])
            return np.array([math.cos(rad), math.sin(rad)])
        return track_ref.get(h["tid"])

    # ---------- 3패스: 방향 일관성을 넣어 링크 선택 ----------
    W = args.dir_weight
    n_redirected = 0
    for h in hits:
        r = ref_dir(h)
        best_i, best_cost = 0, None
        for i, (d0, si, proj) in enumerate(h["cand"]):
            pen = 0.0
            if r is not None:
                cos = float(r @ index.unit(si))
                # 방향성 링크는 역방향까지 벌점, 양방향(OSM)은 '축'만 맞으면 된다.
                pen = W * (1 - cos) / 2 if index.directed[si] else W * (1 - abs(cos)) / 2
            c = d0 + pen
            if best_cost is None or c < best_cost:
                best_i, best_cost = i, c
        if best_i != 0:
            n_redirected += 1
        h["d0"], h["si"], h["proj"] = h["cand"][best_i]
        h["u"] = index.unit(h["si"])
        h["directed"] = bool(index.directed[h["si"]])

        # 횡방향 오프셋
        nvec = np.array([-h["u"][1], h["u"][0]])           # 좌측 법선
        off = float((h["sat"] - h["proj"]) @ nvec)
        if lane_w > 0:                                     # OSM: 차로 중심으로 양자화
            k = round((abs(off) - lane_w / 2) / lane_w)
            k = max(0, min(int(k), args.lanes_per_side - 1))
            off_s = math.copysign(lane_w / 2 + k * lane_w, off if off != 0 else 1.0)
        else:                                              # HD맵: 링크가 곧 차로 중심
            off_s = 0.0
        h["snap"] = h["proj"] + off_s * nvec

    # ---------- 4패스: 진행 방향 확정 ----------
    # 방향성 링크 → 링크 방향이 곧 정면(정지해도 유효). 관측은 역주행 판정에만.
    # 비방향 링크(OSM 양방향) → 트랙 전체 관측의 다수결로 부호 결정.
    track_sign, wrong_way, n_directed_tracks = {}, set(), 0
    for tid, hs in by_tid.items():
        s, n_used = 0.0, 0
        for h in hs:
            if h["obs"] is None:
                continue
            rad = math.radians(h["obs"])
            s += float(np.array([math.cos(rad), math.sin(rad)]) @ h["u"])
            n_used += 1
        mean_cos = s / n_used if n_used else None
        track_sign[tid] = -1.0 if (mean_cos is not None and mean_cos < 0) else 1.0
        # 역주행은 **방향성 링크 위에서만** 의미가 있다. OSM 양방향 도로에서 링크
        # 디지타이즈 방향의 반대로 달리는 건 지극히 정상이다.
        if np.mean([h["directed"] for h in hs]) > 0.5:
            n_directed_tracks += 1
            if mean_cos is not None and mean_cos < args.wrongway_cos:
                wrong_way.add(tid)

    # heading 확정 + 품질 지표
    dh = []
    for h in hits:
        if h["directed"]:
            head = _bearing(h["u"])
            if h["tid"] in wrong_way:
                head = (head + 180.0) % 360.0              # 역주행: 실제 진행방향을 따른다
        else:
            head = _bearing(h["u"] * track_sign.get(h["tid"], 1.0))
        h["head"] = head
        if h["obs"] is not None:
            dh.append(_angdiff(h["obs"], head))

    return {"kind": kind, "is_hd": is_hd, "index": index, "lane_w": lane_w, "lanes": lanes,
            "hits": hits, "by_tid": by_tid, "wrong_way": wrong_way, "dh": dh,
            "n_obj": n_obj, "n_snap": len(hits), "n_redirected": n_redirected,
            "n_directed_tracks": n_directed_tracks,
            "snap_rate": 100.0 * len(hits) / max(n_obj, 1),
            "median_dh": float(np.median(dh)) if dh else 0.0}


def plan_ok(p, min_rate=50.0, max_dh=45.0):
    return p is not None and p["snap_rate"] >= min_rate and p["median_dh"] <= max_dh


def main():
    try:  # Windows 콘솔 cp949에서 한글/기호 출력 크래시 방지
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--max-dist", type=float, default=20.0, help="이 거리(m)보다 먼 객체는 스냅 안 함")
    ap.add_argument("--lane-width", type=float, default=None,
                    help="횡방향 양자화 폭(m). 미지정 시 HD맵=0(링크가 곧 차로), OSM=3.5")
    ap.add_argument("--lanes-per-side", type=int, default=4)
    ap.add_argument("--source", choices=["auto", "hdmap", "osm"], default="auto")
    ap.add_argument("--hdmap-radius", type=float, default=150.0,
                    help="원점에서 이 거리(m) 안에 A2_LINK가 없으면 HD맵 커버리지 밖으로 보고 OSM 폴백")
    ap.add_argument("--dir-weight", type=float, default=8.0,
                    help="링크 선택 시 진행방향 불일치 벌점(m 환산). 교차로에서 교차하는 "
                         "직진/좌회전 링크를 거리만으로 못 가릴 때 이 값이 결정한다. 0이면 순수 최근접")
    ap.add_argument("--wrongway-cos", type=float, default=-0.3,
                    help="트랙 평균 cos(관측, 링크방향)이 이 값보다 작으면 역주행으로 표시")
    args = ap.parse_args()

    src = args.src or os.path.join(REPO, "webmap", "public", "data", "replay", f"{args.loc.lower()}.json.gz")
    if not os.path.exists(src):
        sys.exit(f"입력 replay 없음: {src}")
    data = json.load(gzip.open(src, "rt", encoding="utf-8"))
    world = data["meta"]["world"]

    gpath = os.path.join(REPO, "location", args.loc, f"G_projection_{args.loc}.json")
    g_data = json.load(open(gpath, encoding="utf-8"))
    g = GProjection(g_data, base_dir=os.path.dirname(gpath))
    priors = json.load(open(os.path.join(REPO, "prior_dimensions.json"), encoding="utf-8"))["measurements_visdrone_full"]
    priors = {k.lower(): v for k, v in priors.items()}

    # --- 차선 네트워크 후보를 채점해서 고른다 ---
    def describe(p):
        idx = p["index"]
        return (f"{p['kind']} · 폴리라인 {len(p['lanes'])} / 세그먼트 {len(idx)} "
                f"(방향성 {100 * int(idx.directed.sum()) / len(idx):.0f}%) · 양자화폭 {p['lane_w']:g}m\n"
                f"     스냅률 {p['snap_rate']:.0f}% · 방향차 중앙값 {p['median_dh']:.1f}°")

    plan = None
    if args.source in ("auto", "hdmap"):
        # --source hdmap 은 사용자가 명시적으로 강제한 것이므로 커버리지 게이트를 끈다.
        lanes = lanes_from_hdmap(world, require_within=float("inf") if args.source == "hdmap"
                                 else args.hdmap_radius)
        if lanes:
            plan = build_plan(lanes, "HD맵 A2_LINK(방향성 차로 링크)", True, data, args)
            if plan:
                print(f"후보 ① {describe(plan)}")

    # HD맵이 카메라가 보는 도로를 담고 있지 않으면(교차 도로만 있는 등) OSM으로 폴백한다.
    if args.source in ("auto", "osm") and not plan_ok(plan):
        if plan is not None:
            print(f"  → HD맵 품질 미달(스냅률 {plan['snap_rate']:.0f}% / 방향차 "
                  f"{plan['median_dh']:.0f}°). OSM 후보와 비교합니다.")
        osm = lanes_from_osm(world)
        if osm:
            p2 = build_plan(osm, "OSM 중심선(+차로 양자화)", False, data, args)
            if p2:
                print(f"후보 ② {describe(p2)}")
                # 게이트 통과 여부 우선, 둘 다 같으면 스냅률로 결정
                if plan is None or (plan_ok(p2) and not plan_ok(plan)) or \
                        (plan_ok(p2) == plan_ok(plan) and p2["snap_rate"] > plan["snap_rate"]):
                    plan = p2

    if plan is None:
        sys.exit("차선 네트워크를 얻지 못했습니다(HD맵 범위 밖 + OSM 실패).")

    index, lane_w, is_hd = plan["index"], plan["lane_w"], plan["is_hd"]
    hits, by_tid, wrong_way = plan["hits"], plan["by_tid"], plan["wrong_way"]
    dh, n_obj = plan["dh"], plan["n_obj"]
    n_redirected, n_directed_tracks = plan["n_redirected"], plan["n_directed_tracks"]
    W = args.dir_weight
    print(f"\n채택: {plan['kind']}")

    # ---------- 5패스: 반영 ----------
    # 여기서 처음으로 객체를 건드린다. head/snap 은 build_plan 이 이미 확정해 둔 값.
    n_snap = 0
    shifts, no_obs_fixed = [], 0
    for h in hits:
        o, tid, head, directed = h["o"], h["tid"], h["head"], h["directed"]
        if h["obs"] is None and o.get("heading") is None:
            no_obs_fixed += 1                        # 관측 heading이 없던 객체를 구제한 수
        snapped = h["snap"]
        shifts.append(float(np.linalg.norm(np.asarray(o["sat_coords"], float) - snapped)))
        o["sat_coords"] = [float(snapped[0]), float(snapped[1])]
        o["heading"] = float(head)
        o["have_heading"] = True
        o["heading_src"] = "hdmap_link" if directed else "lane_vote"
        o["lane_snapped"] = True
        o["lane_offset_m"] = round(float(h["d0"]), 2)
        if index.link_id[h["si"]] is not None:
            o["link_id"] = index.link_id[h["si"]]
        if tid in wrong_way:
            o["wrong_way"] = True

        dims = priors.get(str(o.get("class", "")).lower())
        if dims:
            w_m, l_m = dims["width"], dims["length"]
            ang = np.radians(head); c, s = np.cos(ang), np.sin(ang)
            dx, dy = (l_m * g.px_per_m) / 2, (w_m * g.px_per_m) / 2
            corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
            R = np.array([[c, -s], [s, c]])
            floor = (corners @ R.T + snapped).tolist()
            o["sat_floor_box"] = floor
            o["bbox_3d"] = g.sat_floor_to_cctv_3d(floor, float(dims["height"]))
        n_snap += 1

    out = os.path.join(REPO, "output", "lanesnap", args.loc, "clip.json.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"스냅 {n_snap}/{n_obj} 객체 ({100 * n_snap / max(n_obj,1):.0f}%) — max-dist {args.max_dist}m 내")
    if shifts:
        print(f"  위치 이동량: 평균 {np.mean(shifts):.2f} m · 중앙값 {np.median(shifts):.2f} m · "
              f"95% {np.percentile(shifts,95):.2f} m · 최대 {np.max(shifts):.2f} m  (작을수록 궤적 왜곡 적음)")
    if dh:
        print(f"  링크방향 vs 관측 heading 차이: 평균 {np.mean(dh):.1f}° · 중앙값 {np.median(dh):.1f}°  "
              f"(움직이는 객체 {len(dh)}건 기준 — 작을수록 캘리브레이션·링크방향이 일치)")
    print(f"  heading 신규 부여: {no_obs_fixed}건 (관측 heading이 없던 정지/저속 객체 — "
          f"{'HD맵 링크 방향' if is_hd else '트랙 다수결'}으로 정면 확정)")
    print(f"  방향 일관성으로 링크 재선택: {n_redirected}건 "
          f"(최근접이 아닌 링크를 골랐다 — 교차로 교차링크 오스냅 방지, --dir-weight {W:g}m)")
    if wrong_way:
        print(f"  역주행 의심 트랙: {len(wrong_way)}/{n_directed_tracks}개(방향성 링크 위 트랙 기준) "
              f"{sorted(wrong_way)[:10]}")

    # 품질 게이트: 스냅 통계는 캘리브레이션 품질의 프록시다.
    rate = 100 * n_snap / max(n_obj, 1)
    mean_dh = float(np.mean(dh)) if dh else 0.0
    ww_rate = 100 * len(wrong_way) / max(n_directed_tracks, 1)
    if rate < 50 or mean_dh > 45 or ww_rate > 30:
        print("\n  ⚠ 캘리브레이션 점검 필요: "
              f"스냅률 {rate:.0f}%{' (낮음)' if rate < 50 else ''}"
              f"{f', 방향 차이 {mean_dh:.0f}° (과대)' if mean_dh > 45 else ''}"
              f"{f', 역주행 판정 {ww_rate:.0f}% (과다)' if ww_rate > 30 else ''}")
        print("    → 객체가 도로에서 벗어나 있거나 진행방향이 어긋납니다. 이 상태로 스냅하면 "
              "보기엔 좋아도 실측과 달라집니다.")
        if ww_rate > 30:
            print("    → 역주행이 이렇게 많을 수는 없습니다. 실제 역주행이 아니라 "
                  "캘리브레이션 회전 오차이거나 잘못된 링크에 스냅된 것입니다.")
        print(f"    → calibrate.html 에서 {args.loc} 를 GCP 2~4점으로 보정한 뒤 "
              f"reproject.py → lane_snap.py 순으로 다시 실행하세요.")
    print(f"\n→ {os.path.relpath(out, REPO)}")
    print(f"다음: python tools/bridge_to_webmap.py --replay {os.path.relpath(out, REPO)} --loc {args.loc}")


if __name__ == "__main__":
    main()
