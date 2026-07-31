"""
링크 전이 카운팅 — 맵매칭된 궤적에서 P(다음 링크 | 현재 링크)와 방향별 회전교통량.

**왜 신경망이 아니라 카운팅인가**
  분기에서 차가 어디로 갈지는 결국 경험적 회전비율이다. 맵매칭이 끝난 궤적에서는
  세기만 하면 되고(학습 수초), 결과가 그대로 **교통공학 산출물(방향별 회전교통량)** 이며,
  나중에 VectorNet/LaneGCN류를 붙일 때 이 표가 사전분포이자 기준선이 된다.
  지금 데이터 규모(관측 수십 초)에서는 신경망이 검증조차 불가능하다.

모델:
  P(next | cur) = (n(cur→next) + α) / (n(cur) + α·K)      # Dirichlet(α) 스무딩
  95% 신뢰구간은 Wilson score interval (표본이 작을 때 정규근사보다 정직하다)

입력: tools/mapmatch.py 결과(객체에 link_id 가 실려 있어야 한다)
사용:
  python tools/turn_counts.py --loc PANGYO_2
  python tools/turn_counts.py --loc SONGDO_IC --src output/mapmatch/SONGDO_IC/clip.json.gz
출력: output/turncounts/<loc>/turns.json
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
from trafficlab.motion.lane_graph import lanes_from_hdmap, lanes_from_osm, LaneGraph

TURN_BINS = [(-180, -135, "유턴"), (-135, -35, "우회전"), (-35, 35, "직진"),
             (35, 135, "좌회전"), (135, 180, "유턴")]


def wilson(k, n, z=1.96):
    """이항 비율의 Wilson 95% 신뢰구간 — n이 작을 때 정규근사보다 정직하다."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def turn_label(a_deg, b_deg):
    d = (b_deg - a_deg + 180) % 360 - 180
    for lo, hi, name in TURN_BINS:
        if lo <= d < hi:
            return name, d
    return "직진", d


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--src", default=None, help="기본: output/mapmatch/<loc>/clip.json.gz")
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet 스무딩 α")
    ap.add_argument("--source", choices=["auto", "hdmap", "osm"], default="auto")
    ap.add_argument("--hdmap-radius", type=float, default=150.0)
    ap.add_argument("--min-count", type=int, default=5,
                    help="이 표본 수 미만인 전이는 '근거 부족'으로 표시")
    args = ap.parse_args()

    src = args.src or os.path.join(REPO, "output", "mapmatch", args.loc, "clip.json.gz")
    if not os.path.exists(src):
        sys.exit(f"맵매칭 결과 없음: {src}\n  → python tools/mapmatch.py --loc {args.loc}")
    data = json.load(gzip.open(src, "rt", encoding="utf-8"))
    world = data["meta"]["world"]

    # 링크 방위를 알아야 회전 종류(직진/좌/우)를 판정할 수 있다.
    # 1순위: mapmatch 가 결과에 캐시해 둔 방위(공개 Overpass는 자주 죽으므로 재조회에 의존하지 않는다)
    bearing, src_kind = {}, None
    cached = data["meta"].get("lane_links")
    if cached:
        bearing = {k: (v[0], v[1]) for k, v in cached.items()}
        src_kind = data["meta"].get("lane_source", "캐시")
    else:
        lanes = []
        if args.source in ("auto", "hdmap"):
            lanes = lanes_from_hdmap(world, require_within=float("inf") if args.source == "hdmap"
                                     else args.hdmap_radius, verbose=False)
        if not lanes and args.source in ("auto", "osm"):
            lanes = lanes_from_osm(world, verbose=False)
        if lanes:
            graph = LaneGraph(lanes)
            src_kind = "재조회"
            bearing = {str(lk.id): (graph.heading_deg(lk.idx, 0.0),
                                    graph.heading_deg(lk.idx, lk.length))
                       for lk in graph.links if lk.id is not None}

    # --- 트랙별 링크 시퀀스 ---
    tracks = defaultdict(list)
    for fr in data["frames"]:
        for o in fr.get("objects", []):
            tid, lid = o.get("tracked_id"), o.get("link_id")
            if tid is None or lid is None:
                continue
            tracks[tid].append((fr["frame_index"], str(lid), o.get("class")))
    seqs = {}
    for tid, rows in tracks.items():
        rows.sort()
        s = []
        for _, lid, cls in rows:
            if not s or s[-1] != lid:
                s.append(lid)
        seqs[tid] = (s, rows[0][2])

    trans = defaultdict(int)              # (cur, next) -> n
    out_of = defaultdict(int)             # cur -> n
    by_class = defaultdict(lambda: defaultdict(int))
    for tid, (s, cls) in seqs.items():
        for a, b in zip(s, s[1:]):
            trans[(a, b)] += 1
            out_of[a] += 1
            by_class[(a, b)][cls or "unknown"] += 1

    n_tr = sum(trans.values())
    multi = sum(1 for s, _ in seqs.values() if len(s) >= 2)
    print(f"트랙 {len(seqs)}개 · 링크 2개 이상 통과 {multi}개 · 전이 관측 {n_tr}건")
    if bearing:
        print(f"링크 방위 {len(bearing)}개 확보 ({src_kind})")
    else:
        print("  ⚠ 링크 방위를 못 구해 회전 종류(직진/좌/우)를 판정할 수 없습니다 — "
              "mapmatch.py 를 다시 돌려 방위 캐시를 만드세요(전이 카운트 자체는 유효).")

    rows = []
    for (a, b), n in sorted(trans.items(), key=lambda x: -x[1]):
        tot = out_of[a]
        K = max(1, sum(1 for (x, _) in trans if x == a))
        p = (n + args.alpha) / (tot + args.alpha * K)
        lo, hi = wilson(n, tot)
        lab, ang = ("미상", None)
        if a in bearing and b in bearing:
            lab, ang = turn_label(bearing[a][1], bearing[b][0])
        rows.append({"from": a, "to": b, "count": n, "from_total": tot,
                     "p_smoothed": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "turn": lab, "delta_deg": round(ang, 1) if ang is not None else None,
                     "by_class": dict(by_class[(a, b)]),
                     "sufficient": tot >= args.min_count})

    print(f"\n{'from':>14s} → {'to':<14s} {'n':>4s}/{'tot':<4s} {'P':>6s} {'95% CI':>15s}  회전")
    for r in rows[:20]:
        mark = "" if r["sufficient"] else "  ← 근거 부족"
        print(f"{r['from'][-14:]:>14s} → {r['to'][-14:]:<14s} {r['count']:4d}/{r['from_total']:<4d} "
              f"{r['p_smoothed']:6.3f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]   {r['turn']}{mark}")

    # 회전 종류별 집계 = 방향별 회전교통량
    agg = defaultdict(int)
    for r in rows:
        agg[r["turn"]] += r["count"]
    print("\n회전 종류별 전이 수: " + " · ".join(f"{k} {v}" for k, v in sorted(agg.items(), key=lambda x: -x[1])))

    n_suff = sum(1 for r in rows if r["sufficient"])
    print(f"\n표본 충분(≥{args.min_count})한 분기: {n_suff}/{len(rows)}개")
    if n_suff < len(rows) * 0.5:
        print(f"  ⚠ 대부분의 전이가 표본 부족입니다. 이 확률표는 아직 **예측에 쓸 수 없습니다** — "
              f"구조는 맞지만 수치가 우연에 지배됩니다.")
        print(f"    회전비율이 ±10%로 수렴하려면 분기당 100건 이상이 필요합니다. "
              f"현재 관측 {len(data['frames'])/float(data['meta'].get('fps',30)):.0f}초 "
              f"→ 카메라당 수 시간 녹화가 필요합니다.")

    out = os.path.join(REPO, "output", "turncounts", args.loc, "turns.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"loc": args.loc, "n_tracks": len(seqs), "n_transitions": n_tr,
               "alpha": args.alpha, "min_count": args.min_count,
               "note": "P(next|cur), Dirichlet 스무딩. sufficient=false 인 행은 예측에 쓰지 말 것.",
               "transitions": rows,
               "turn_totals": dict(agg)},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
