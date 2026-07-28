"""
재추적(A/B/C) — 기존 추론 결과의 **검출은 그대로 두고 연관(추적)만 교체**.

검출을 다시 돌리지 않으므로 세 방식을 공정하게 비교할 수 있다.
  A. 기존(ByteTrack)   : 이미지 IoU 연관
  B. BEV               : 검출을 지면(미터)으로 투영해 **투영 불확실성(이방성) 기반
                         마할라노비스 거리**로 연관(cf. UCMCTrack) + OC-SORT의 ORU/OCM/OCR
  C. BEV + 차로 구속    : B에 차선 그래프를 얹어 **예측이 차로 곡선을 따라간다**.
                         가림 구간 coasting이 직선으로 도로를 벗어나지 않으므로 재등장 시
                         재연관된다. 프로세스 잡음도 차로 프레임에서 종/횡 비대칭으로 준다.

사용:
  python tools/retrack.py --loc PANGYO_2              # A vs B
  python tools/retrack.py --loc PANGYO_2 --lane       # A vs B vs C (C를 저장)
  python tools/eval/evaluate.py --loc PANGYO_2 --pred <기존> --pred output/retrack-bev/PANGYO_2/clip.json.gz
출력: output/retrack-bev/<loc>/clip.json.gz
"""
import argparse
import copy
import gzip
import json
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from trafficlab.projection.g_projection import GProjection
from trafficlab.motion.bev_tracker import BEVTracker, ground_cov
from trafficlab.motion.kinematics import TrackSmoother
from trafficlab.motion.lane_graph import LaneGraph, lanes_from_hdmap, lanes_from_osm

KIN = {"heading_ema": {"alpha_min": 0.05, "alpha_max": 0.4, "speed_ref": 3.0},
       "speed_ema_alpha": 0.4, "heading_min_speed_for_update": 0.1, "heading_max_jump": 5,
       "heading_sat_coords_jitter_radius": 0.5, "heading_sat_coords_jitter_frames": 8}


def frag_stats(frames_objs, fps, impossible_kmh=180.0):
    """수명 통계 + **물리적 타당성 프록시**.

    max_age를 늘리면 수명 중앙값은 반드시 올라간다 — 하지만 그게 올바른 재연관인지
    서로 다른 차를 이어붙인 것인지는 수명만 봐서 알 수 없다. GT 라벨이 없으므로
    재연관 구간의 **함의 속도**로 대신 본다: 두 관측을 잇는 데 180km/h가 필요했다면
    그 연관은 틀린 것이다.
    """
    life = defaultdict(list)
    for fi, objs in frames_objs:
        for o in objs:
            t = o.get("tracked_id")
            if t is not None:
                life[t].append((fi, o.get("sat_coords")))
    if not life:
        return dict(tracks=0, median=0.0, p90=0.0, short=0.0, cov=0.0, bad=0, vmax=0.0)
    L, speeds, bad = [], [], 0
    for v in life.values():
        v.sort(key=lambda r: r[0])
        L.append(v[-1][0] - v[0][0] + 1)
        for (f0, p0), (f1, p1) in zip(v, v[1:]):
            if not p0 or not p1 or f1 <= f0:
                continue
            kmh = np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / ((f1 - f0) / fps) * 3.6
            speeds.append(kmh)
            if kmh > impossible_kmh:
                bad += 1
    L = np.array(L, float)
    seen = sum(len(v) for v in life.values())
    total = sum(len(objs) for _, objs in frames_objs)
    return dict(tracks=len(life), median=float(np.median(L)) / fps,
                p90=float(np.percentile(L, 90)) / fps,
                short=float(np.mean(L < fps * 0.5) * 100),
                cov=100.0 * seen / max(total, 1),
                bad=bad, vmax=float(np.percentile(speeds, 99)) if speeds else 0.0)


def run_tracker(data, g, g_data, priors, args, graph=None):
    """추적만 다시 돌려 (프레임별 객체 리스트, ID 부여 수)를 반환. 입력 data는 건드리지 않는다."""
    fps = data["meta"].get("fps", 30)
    tracker = BEVTracker(dt=1.0 / fps, max_age=args.max_age, min_hits=args.min_hits,
                         mahalanobis=True, lane_graph=graph)
    smoothers, last_seen = {}, {}
    out, n_id, n_drop = [], 0, 0

    for fr in data["frames"]:
        i = fr["frame_index"]
        cand = [o for o in fr.get("objects", []) if o.get("bbox_2d")]
        objs, dets, rejected = [], [], []
        for o in cand:
            x1, y1, x2, y2 = o["bbox_2d"]
            dims = priors.get(str(o.get("class", "")).lower())
            h_real = float(dims["height"]) if dims else 1.55
            r = g.get_ground_contact_from_box((x1, y1, x2 - x1, y2 - y1), h_real,
                                              ref_method=g_data.get("ref_method", "center_bottom_side"),
                                              proj_method=g_data.get("proj_method", "down_h"))
            pos = r["sat_coords"]
            cov = ground_cov(g, (x1 + x2) / 2, y2, h=0.0, sigma_px=args.sigma_px)
            # 지평선 근처 검출은 지면 광선이 지면과 거의 평행해져 투영 거리가 발산한다
            # (실측: SONGDO_IC 검출의 13.7%가 300m 밖, 최대 33km). 이런 관측은 위치 정보가
            # 없는 것이나 마찬가지라 연관에 넣으면 트랙만 오염시킨다.
            sig = float(np.sqrt(max(np.linalg.eigvalsh(np.asarray(cov, float)).max(), 0.0)))
            if np.hypot(*pos) > args.max_range or sig > args.max_sigma or not np.all(np.isfinite(pos)):
                n_drop += 1
                o = dict(o)
                o["tracked_id"] = None
                o["have_heading"] = False; o["heading"] = None; o["speed_kmh"] = 0.0
                o["sat_coords"] = list(pos); o["sat_floor_box"] = None; o["bbox_3d"] = None
                o["proj_rejected"] = True
                rejected.append(o)
                continue
            objs.append(o)
            dets.append({"pos": pos, "conf": float(o.get("confidence") or 0.5),
                         "cls": o.get("class"), "cov": cov})
        ids = tracker.update(dets, i)

        new_objs = []
        for o, d, tid in zip(objs, dets, ids):
            o = dict(o)
            o["tracked_id"] = int(tid) if tid is not None else None
            o["sat_coords"] = list(d["pos"])
            if tid is None:
                o["have_heading"] = False; o["heading"] = None; o["speed_kmh"] = 0.0
                o["sat_floor_box"] = None; o["bbox_3d"] = None
                new_objs.append(o); continue
            n_id += 1
            if tid not in smoothers:
                smoothers[tid] = TrackSmoother(KIN); last_seen[tid] = i - 1
            dt = (i - last_seen.get(tid, i - 1)) / fps or (1.0 / fps)
            k = smoothers[tid].update(d["pos"], dt, g.px_per_m); last_seen[tid] = i
            o["heading"] = k["heading"]; o["speed_kmh"] = k["speed_kmh"]
            o["have_heading"] = k["heading"] is not None
            dims = priors.get(str(o.get("class", "")).lower())
            if k["heading"] is not None and dims:
                w_m, l_m = dims["width"], dims["length"]
                ang = np.radians(k["heading"]); c, s = np.cos(ang), np.sin(ang)
                dx, dy = (l_m * g.px_per_m) / 2, (w_m * g.px_per_m) / 2
                corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
                R = np.array([[c, -s], [s, c]])
                floor = (corners @ R.T + np.array(d["pos"])).tolist()
                o["sat_floor_box"] = floor
                o["bbox_3d"] = g.sat_floor_to_cctv_3d(floor, float(dims["height"]))
            new_objs.append(o)
        out.append((i, new_objs + rejected))
    return out, n_id, n_drop


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--sigma-px", type=float, default=3.0, help="검출 박스 픽셀 잡음 가정")
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--lane", action="store_true", help="차선 그래프로 예측을 구속(C안)")
    ap.add_argument("--source", choices=["auto", "hdmap", "osm"], default="auto")
    ap.add_argument("--hdmap-radius", type=float, default=150.0)
    ap.add_argument("--max-range", type=float, default=250.0,
                    help="지면 투영 거리가 이보다 크면 검출을 버린다(지평선 근처 발산 차단)")
    ap.add_argument("--max-sigma", type=float, default=25.0,
                    help="투영 불확실성 1σ가 이보다 크면 검출을 버린다(m)")
    args = ap.parse_args()

    src = args.src or os.path.join(REPO, "webmap", "public", "data", "replay", f"{args.loc.lower()}.json.gz")
    if not os.path.exists(src):
        sys.exit(f"입력 replay 없음: {src}")
    data = json.load(gzip.open(src, "rt", encoding="utf-8"))
    fps = data["meta"].get("fps", 30)

    gp = os.path.join(REPO, "location", args.loc, f"G_projection_{args.loc}.json")
    g_data = json.load(open(gp, encoding="utf-8"))
    g = GProjection(g_data, base_dir=os.path.dirname(gp))
    priors = json.load(open(os.path.join(REPO, "prior_dimensions.json"), encoding="utf-8"))["measurements_visdrone_full"]
    priors = {k.lower(): v for k, v in priors.items()}

    graph, kind = None, None
    if args.lane:
        world = data["meta"]["world"]
        lanes = []
        if args.source in ("auto", "hdmap"):
            lanes = lanes_from_hdmap(world, require_within=float("inf") if args.source == "hdmap"
                                     else args.hdmap_radius)
            kind = "HD맵 A2_LINK" if lanes else None
        if not lanes and args.source in ("auto", "osm"):
            lanes = lanes_from_osm(world)
            kind = "OSM 중심선" if lanes else None
        if not lanes:
            sys.exit("--lane 을 켰지만 차선 네트워크를 얻지 못했습니다.")
        graph = LaneGraph(lanes)
        print(f"차선망: {kind} · 링크 {len(graph.links)} · 위상 {graph.topology_src}")

    before = [(f["frame_index"], f.get("objects", [])) for f in data["frames"]]
    n_obj = sum(len(o) for _, o in before)

    rows = [("A. 기존(ByteTrack)", frag_stats(before, fps))]
    bev, n_id_b, n_drop = run_tracker(data, g, g_data, priors, args, graph=None)
    rows.append(("B. BEV", frag_stats(bev, fps)))
    chosen, n_id = bev, n_id_b
    if graph is not None:
        lane, n_id_c, _ = run_tracker(data, g, g_data, priors, args, graph=graph)
        rows.append(("C. BEV+차로구속", frag_stats(lane, fps)))
        chosen, n_id = lane, n_id_c

    print(f"\n[{args.loc}] 검출 {n_obj}개 · 최종안 ID 부여 {n_id}개 (미확정 {n_obj - n_id})")
    print(f"{'':20s} {'트랙수':>6s} {'수명중앙':>9s} {'수명p90':>8s} {'0.5초미만%':>10s} "
          f"{'ID부여율%':>9s} {'속도p99':>11s} {'>180km/h':>9s}")
    for name, st in rows:
        print(f"{name:20s} {st['tracks']:6d} {st['median']:8.2f}s {st['p90']:7.2f}s "
              f"{st['short']:10.0f} {st['cov']:9.0f} {st['vmax']:8.0f}km/h {st['bad']:9d}")
    print("  속도p99/>180km/h: 재연관이 물리적으로 가능한지 보는 프록시(GT 라벨이 0개라 대체 지표). "
          "수명이 늘어도 이 값이 함께 늘면 서로 다른 차를 이어붙인 것이다.")

    for fr, (i, objs) in zip(data["frames"], chosen):
        fr["objects"] = objs
    out = os.path.join(REPO, "output", "retrack-bev", args.loc, "clip.json.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"\n→ {os.path.relpath(out, REPO)}  ({'C. BEV+차로구속' if graph is not None else 'B. BEV'} 저장)")
    print(f"  정량 비교: python tools/eval/evaluate.py --loc {args.loc} --pred {os.path.relpath(src, REPO)} "
          f"--pred {os.path.relpath(out, REPO)}   (GT 라벨 필요)")


if __name__ == "__main__":
    main()
