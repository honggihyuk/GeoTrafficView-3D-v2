"""
맵매칭 + Frenet 재구성 — 트랙을 차로 시퀀스에 얹고 위치·heading을 기하에서 재생성한다.

lane_snap 과의 차이:
  lane_snap : 프레임마다 독립으로 링크를 고른다(그리디). 빠르지만 나란한 차로 사이를 깜빡인다.
  mapmatch  : 트랙 전체를 Viterbi로 한 번에 푼다. 링크 시퀀스가 일관되고, 그 시퀀스가 곧
              '이 차가 지나간 경로'다. 위치는 Frenet(s,d)로, heading은 **차로 접선**으로
              재생성하므로 EMA·점프클램프 같은 후처리가 필요 없다.

정량 비교(그리디 vs Viterbi)를 항상 함께 출력한다:
  링크 플리커  트랙당 초당 링크 전환 횟수 — 깜빡임의 직접 지표
  heading 지터 인접 프레임 |Δheading| 중앙값 — 화면에서 차가 떠는 정도
  횡잔차       |d| 중앙값 — 차로 중심에서 벗어난 정도

사용:
  python tools/mapmatch.py --loc PANGYO_2
  python tools/mapmatch.py --loc PANGYO_2 --sigma 2.0 --beta 4.0
출력: output/mapmatch/<loc>/clip.json.gz
"""
import argparse
import gzip
import json
import math
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from trafficlab.projection.g_projection import GProjection
from trafficlab.motion.lane_graph import LaneGraph, lanes_from_hdmap, lanes_from_osm
from trafficlab.motion.map_matching import (viterbi_match, greedy_match, smooth_lateral,
                                            lane_lock)


def _angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def metrics(graph, tracks, states_by_tid, fps):
    """플리커 / heading 지터 / 횡잔차."""
    flick, jit, lat, n_state = [], [], [], 0
    for tid, rows in tracks.items():
        st = states_by_tid.get(tid)
        if not st:
            continue
        seq = [(rows[i][0], s) for i, s in enumerate(st) if s is not None]
        if len(seq) < 2:
            continue
        n_state += len(seq)
        span = (seq[-1][0] - seq[0][0]) / fps
        sw = sum(1 for a, b in zip(seq, seq[1:]) if a[1][0] != b[1][0])
        if span > 0.2:
            flick.append(sw / span)
        hs = [graph.heading_deg(s[0], s[1]) for _, s in seq]
        jit.extend(_angdiff(a, b) for a, b in zip(hs, hs[1:]))
        lat.extend(abs(s[2]) for _, s in seq)
    # heading 지터는 직선 도로에서 중앙값이 0으로 깔린다 — 꼬리(p90)를 함께 봐야 의미가 있다.
    return {"n": n_state,
            "flicker": float(np.mean(flick)) if flick else 0.0,
            "jitter": float(np.median(jit)) if jit else 0.0,
            "jit90": float(np.percentile(jit, 90)) if jit else 0.0,
            "lat": float(np.median(lat)) if lat else 0.0}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--source", choices=["auto", "hdmap", "osm"], default="auto")
    ap.add_argument("--hdmap-radius", type=float, default=150.0)
    ap.add_argument("--max-dist", type=float, default=20.0)
    ap.add_argument("--sigma", type=float, default=2.0, help="방출 σ — 횡거리 허용 오차(m)")
    ap.add_argument("--beta", type=float, default=4.0, help="전이 β — 경로거리 불일치 허용(m)")
    ap.add_argument("--k", type=int, default=6, help="관측당 후보 링크 수")
    ap.add_argument("--lat-smooth", type=int, default=5, help="횡오프셋 중앙값 필터 창(0이면 끔)")
    ap.add_argument("--lane-lock", action="store_true", default=True,
                    help="트랙을 하나의 연결된 차로 사슬에 고정하고 s 를 단조화한다(기본). "
                         "나란한 차로로 건너뛰는 것과 앞뒤 떨림을 구조적으로 없앤다")
    ap.add_argument("--no-lane-lock", dest="lane_lock", action="store_false")
    ap.add_argument("--center-lane", action="store_true",
                    help="HD맵처럼 링크=차로일 때 d를 0으로 두어 차로 중심에 정확히 올린다")
    args = ap.parse_args()

    src = args.src or os.path.join(REPO, "webmap", "public", "data", "replay", f"{args.loc.lower()}.json.gz")
    if not os.path.exists(src):
        sys.exit(f"입력 replay 없음: {src}")
    data = json.load(gzip.open(src, "rt", encoding="utf-8"))
    world = data["meta"]["world"]
    fps = float(data["meta"].get("fps", 30))

    gpath = os.path.join(REPO, "location", args.loc, f"G_projection_{args.loc}.json")
    g = GProjection(json.load(open(gpath, encoding="utf-8")), base_dir=os.path.dirname(gpath))
    priors = json.load(open(os.path.join(REPO, "prior_dimensions.json"), encoding="utf-8"))["measurements_visdrone_full"]
    priors = {k.lower(): v for k, v in priors.items()}

    # --- 트랙 구성 ---
    tracks = defaultdict(list)      # tid -> [(frame_index, obj)]
    for fr in data["frames"]:
        fi = fr["frame_index"]
        for o in fr.get("objects", []):
            if o.get("sat_coords") and o.get("tracked_id") is not None:
                tracks[o["tracked_id"]].append((fi, o))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    print(f"트랙 {len(tracks)}개 · 관측 {sum(len(v) for v in tracks.values())}개")

    # --- 매칭(차선망 후보별) ---
    n_obs = sum(len(v) for v in tracks.values())

    def run(lanes, kind, is_hd):
        graph = LaneGraph(lanes)
        vit, grd = {}, {}
        for tid, rows in tracks.items():
            obs = [(fi / fps, np.asarray(o["sat_coords"], float)) for fi, o in rows]
            dirs = []
            for _, o in rows:
                h = o.get("heading") if o.get("have_heading") else None
                spd = float(o.get("speed_kmh") or 0.0)
                if h is None or spd < 5.0:
                    dirs.append(None)
                else:
                    r = math.radians(h)
                    dirs.append(np.array([math.cos(r), math.sin(r)]))
            # 트랙 기준 방향(정지 프레임에 맥락을 물려준다)
            v = np.zeros(2)
            for dv in dirs:
                if dv is not None:
                    v += dv
            tref = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else None
            dirs = [d if d is not None else tref for d in dirs]

            grd[tid] = greedy_match(graph, obs, max_dist=args.max_dist, k=args.k, dir_ref=dirs)
            st = viterbi_match(graph, obs, sigma_lat=args.sigma, beta=args.beta,
                               max_dist=args.max_dist, k=args.k, dir_ref=dirs)
            if args.lane_lock:
                st, _chain = lane_lock(graph, st)
            if args.lat_smooth > 1:
                st = smooth_lateral(st, args.lat_smooth)
            vit[tid] = st
        rate = 100.0 * sum(1 for st in vit.values() for s in st if s is not None) / max(n_obs, 1)
        print(f"후보: {kind} · 링크 {len(graph.links)} · 위상 {graph.topology_src} "
              f"(후속연결 {graph.n_succ()}개) → 매칭률 {rate:.0f}%")
        return {"graph": graph, "vit": vit, "grd": grd, "rate": rate, "kind": kind, "is_hd": is_hd}

    sel = None
    if args.source in ("auto", "hdmap"):
        lanes = lanes_from_hdmap(world, require_within=float("inf") if args.source == "hdmap"
                                 else args.hdmap_radius)
        if lanes:
            sel = run(lanes, "HD맵 A2_LINK", True)

    # HD맵이 카메라가 보는 도로를 담고 있지 않으면(교차 도로만 있는 등) OSM으로 폴백한다.
    if args.source in ("auto", "osm") and (sel is None or sel["rate"] < 50.0):
        if sel is not None:
            print(f"  → HD맵 매칭률 {sel['rate']:.0f}% (미달). OSM 후보와 비교합니다.")
        osm = lanes_from_osm(world)
        if osm:
            alt = run(osm, "OSM 중심선", False)
            if sel is None or alt["rate"] > sel["rate"]:
                sel = alt
    if sel is None:
        sys.exit("차선 네트워크를 얻지 못했습니다.")

    graph, vit, grd, is_hd = sel["graph"], sel["vit"], sel["grd"], sel["is_hd"]
    print(f"\n채택: {sel['kind']}")
    mg, mv = metrics(graph, tracks, grd, fps), metrics(graph, tracks, vit, fps)
    print(f"\n{'':10s} {'매칭관측':>8s} {'링크플리커/s':>12s} {'heading지터(중앙/p90)':>22s} {'횡잔차':>8s}")
    for nm, m in (("그리디", mg), ("Viterbi", mv)):
        print(f"{nm:10s} {m['n']:8d} {m['flicker']:12.2f} "
              f"{m['jitter']:12.2f}° /{m['jit90']:7.2f}° {m['lat']:7.2f}m")
    if mg["flicker"] > 0:
        print(f"{'개선':10s} {'':8s} {100*(1-mv['flicker']/mg['flicker']):11.0f}% "
              f"{'':12s}  {100*(1-mv['jit90']/max(mg['jit90'],1e-9)):6.0f}%")

    # --- Frenet 재구성 → 객체에 반영 ---
    n_apply, n_head_new, shifts = 0, 0, []
    for tid, rows in tracks.items():
        st = vit.get(tid) or []
        for (fi, o), s in zip(rows, st):
            if s is None:
                continue
            li, sv, d = s
            if args.center_lane and is_hd:
                d = 0.0                                # 링크가 곧 차로 중심
            pos = graph.to_xy(li, sv, d)
            head = graph.heading_deg(li, sv)
            lk = graph.links[li]
            if not lk.directed:
                # 양방향 링크는 관측으로 부호를 정한다(HD맵 방향성 링크는 그대로가 정면).
                h = o.get("heading") if o.get("have_heading") else None
                if h is not None and _angdiff(h, head) > 90:
                    head = (head + 180.0) % 360.0
            if o.get("heading") is None:
                n_head_new += 1
            shifts.append(float(np.linalg.norm(np.asarray(o["sat_coords"], float) - pos)))
            o["sat_coords"] = [float(pos[0]), float(pos[1])]
            o["heading"] = float(head)
            o["have_heading"] = True
            o["heading_src"] = "mapmatch_link" if lk.directed else "mapmatch_vote"
            o["lane_snapped"] = True
            o["frenet"] = [round(float(sv), 2), round(float(d), 2)]
            if lk.id is not None:
                o["link_id"] = lk.id
            if lk.lane_no is not None:
                o["lane_no"] = lk.lane_no

            dims = priors.get(str(o.get("class", "")).lower())
            if dims:
                w_m, l_m = dims["width"], dims["length"]
                ang = np.radians(head); c, sn = np.cos(ang), np.sin(ang)
                dx, dy = (l_m * g.px_per_m) / 2, (w_m * g.px_per_m) / 2
                corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
                R = np.array([[c, -sn], [sn, c]])
                floor = (corners @ R.T + pos).tolist()
                o["sat_floor_box"] = floor
                o["bbox_3d"] = g.sat_floor_to_cctv_3d(floor, float(dims["height"]))
            n_apply += 1

    n_obj = sum(len(f.get("objects", [])) for f in data["frames"])
    print(f"\n반영 {n_apply}/{n_obj} 객체 ({100*n_apply/max(n_obj,1):.0f}%) · "
          f"heading 신규 부여 {n_head_new}건")
    if shifts:
        print(f"  위치 이동량: 중앙값 {np.median(shifts):.2f} m · 95% {np.percentile(shifts,95):.2f} m")

    # 경로(링크 시퀀스) 요약 — 회전교통량 카운팅의 입력이 된다
    seqs = []
    for tid, st in vit.items():
        seq = []
        for s in st:
            if s is None:
                continue
            if not seq or seq[-1] != s[0]:
                seq.append(s[0])
        if len(seq) >= 2:
            seqs.append(seq)
    print(f"  링크 2개 이상을 지난 트랙: {len(seqs)}/{len(tracks)}개 "
          f"(전이 {sum(len(s)-1 for s in seqs)}건 — 회전 카운팅의 원재료)")

    # 링크 방위를 결과에 캐시한다 — 하류(turn_counts)가 OSM Overpass를 다시 부르지 않아도
    # 회전 종류(직진/좌/우)를 판정할 수 있게. 공개 Overpass는 자주 죽는다.
    data["meta"]["lane_links"] = {
        str(lk.id): [round(graph.heading_deg(lk.idx, 0.0), 2),
                     round(graph.heading_deg(lk.idx, lk.length), 2),
                     round(lk.length, 2), lk.lane_no, bool(lk.directed)]
        for lk in graph.links if lk.id is not None}
    data["meta"]["lane_source"] = sel["kind"]

    out = os.path.join(REPO, "output", "mapmatch", args.loc, "clip.json.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"\n→ {os.path.relpath(out, REPO)}")
    print(f"다음: python tools/bridge_to_webmap.py --replay {os.path.relpath(out, REPO)} --loc {args.loc}")


if __name__ == "__main__":
    main()
