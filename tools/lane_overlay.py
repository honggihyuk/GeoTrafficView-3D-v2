"""
차선 영상투영 오버레이 — "차선 배정 방식"의 전제 조건을 눈으로 검증한다.

**왜 필요한가**
  검출을 지면으로 역투영(image→ground)하면 지평선 근처에서 발산한다(실측: SONGDO_IC
  검출의 13.7%가 300m 밖, 최대 33km). 반대 방향인 **순투영(ground→image)** 은 어디서나
  안정적이다. 카메라가 고정이므로 차선 중심선을 한 번 영상으로 투영해 두고 검출을
  **픽셀 공간에서** 차선에 배정하면, 발산하는 계산을 통째로 우회할 수 있다.

  다만 이 방식은 캘리브레이션 오차를 **그대로 결과로 만든다**. 추적을 하면 운동 증거가
  차선 배정을 사후에 교정해 주지만, 단일 프레임 배정에는 그런 보정이 없다. 투영된 차선이
  실제 도로에 얹히지 않으면 모든 차량이 틀린 차로에 놓인다.
  → 이 오버레이가 통과하지 못하면 그 카메라에서는 이 방식을 쓰면 안 된다.

**함께 출력하는 것**
  종방향 민감도: 영상 1px이 도로를 따라 몇 m인가. 차량 길이(4.4m)보다 크면 그 행에서는
  앞뒤 차를 구분할 수 없다는 뜻이다.

사용:
  python tools/lane_overlay.py --loc PANGYO_2
  python tools/lane_overlay.py --loc PANGYO_2 --frame 30 --source osm
출력: output/laneoverlay/<loc>/overlay.png · sensitivity.json
"""
import argparse
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from trafficlab.projection.g_projection import GProjection
from trafficlab.motion.lane_graph import lanes_from_hdmap, lanes_from_osm, LaneGraph

CAR_LEN = 4.4     # m — 1px가 이보다 크면 앞뒤 차 구분 불가


def frame_image(loc, idx):
    p = os.path.join(REPO, "webmap", "public", "eval", loc, "frames", f"f{idx:06d}.jpg")
    if os.path.exists(p):
        return cv2.imread(p)
    v = os.path.join(REPO, "webmap", "public", "data", "footage", f"{loc.lower()}.mp4")
    if not os.path.exists(v):
        return None
    cap = cv2.VideoCapture(v)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--source", choices=["auto", "hdmap", "osm"], default="auto")
    ap.add_argument("--hdmap-radius", type=float, default=150.0)
    ap.add_argument("--step", type=int, default=2, help="링크 정점 샘플 간격")
    args = ap.parse_args()

    gp = os.path.join(REPO, "location", args.loc, f"G_projection_{args.loc}.json")
    if not os.path.exists(gp):
        sys.exit(f"캘리브레이션 없음: {gp}")
    gd = json.load(open(gp, encoding="utf-8"))
    g = GProjection(gd, base_dir=os.path.dirname(gp))
    world = gd["world"]

    lanes, kind = [], None
    if args.source in ("auto", "hdmap"):
        lanes = lanes_from_hdmap(world, require_within=float("inf") if args.source == "hdmap"
                                 else args.hdmap_radius, verbose=False)
        kind = "HD맵 A2_LINK" if lanes else None
    if not lanes and args.source in ("auto", "osm"):
        lanes = lanes_from_osm(world, verbose=False)
        kind = "OSM 중심선" if lanes else None
    if not lanes:
        sys.exit("차선 네트워크를 얻지 못했습니다.")
    graph = LaneGraph(lanes)

    img = frame_image(args.loc, args.frame)
    if img is None:
        sys.exit(f"프레임 이미지를 찾지 못했습니다({args.loc} f{args.frame})")
    H, W = img.shape[:2]
    vis = img.copy()

    n_on = 0
    for lk in graph.links:
        pts = []
        for p in lk.pts[::max(1, args.step)]:
            u, v = g.sat_to_cctv(float(p[0]), float(p[1]), h=0.0)
            if np.isfinite(u) and np.isfinite(v):
                pts.append((int(round(u)), int(round(v))))
        seg = [q for q in pts if -W <= q[0] <= 2 * W and -H <= q[1] <= 2 * H]
        if len(seg) < 2:
            continue
        vis_any = any(0 <= q[0] < W and 0 <= q[1] < H for q in seg)
        if not vis_any:
            continue
        n_on += 1
        col = (0, 255, 255) if lk.directed else (255, 160, 0)
        for a, b in zip(seg, seg[1:]):
            cv2.line(vis, a, b, col, 1, cv2.LINE_AA)
        # 진행 방향 화살표 — 정면/후면 판정의 기준이 되는 방향이다
        mid = seg[len(seg) // 2]
        nxt = seg[min(len(seg) // 2 + 3, len(seg) - 1)]
        if 0 <= mid[0] < W and 0 <= mid[1] < H and (nxt != mid):
            cv2.arrowedLine(vis, mid, nxt, (0, 80, 255), 1, cv2.LINE_AA, tipLength=0.5)

    # 종방향 민감도
    rows, cx = [], W // 2
    for y in range(int(H * 0.2), H, max(1, H // 14)):
        try:
            p0 = np.array(g.cctv_to_sat(cx, y, h=0.0), float)
            p1 = np.array(g.cctv_to_sat(cx, y + 1, h=0.0), float)
        except Exception:
            continue
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            continue
        dm = float(np.linalg.norm(p1 - p0))
        rows.append({"y": y, "m_per_px": round(dm, 3), "usable": dm < CAR_LEN})

    out_dir = os.path.join(REPO, "output", "laneoverlay", args.loc)
    os.makedirs(out_dir, exist_ok=True)
    cv2.putText(vis, f"{args.loc} f{args.frame} · {kind} · 노랑=방향성 차로 / 파랑=양방향",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "overlay.png"), np.vstack([img, vis]))
    json.dump({"loc": args.loc, "source": kind, "links_visible": n_on,
               "car_len_m": CAR_LEN, "sensitivity": rows},
              open(os.path.join(out_dir, "sensitivity.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"[{args.loc}] {kind} · 화면에 걸리는 링크 {n_on}/{len(graph.links)}개")
    print(f"\n{'y(px)':>6s} {'1px당 종방향(m)':>16s}  판정(차량길이 {CAR_LEN}m 기준)")
    for r in rows:
        print(f"{r['y']:6d} {r['m_per_px']:16.2f}  "
              f"{'양호' if r['m_per_px'] < 1.0 else ('주의' if r['usable'] else '사용 불가 — 앞뒤차 구분 불가')}")
    bad = [r for r in rows if not r["usable"]]
    if bad:
        print(f"\n  ⚠ y<{max(r['y'] for r in bad)}px 구간은 1px이 차량 길이보다 커서 "
              f"차선 위 위치를 확정할 수 없습니다.")
    print(f"\n→ {os.path.relpath(out_dir, REPO)}/overlay.png")
    print("  ★ 반드시 육안 확인: 노란 선이 실제 차로 위에 얹혀야 합니다. 한 차로 폭(3.5m)이라도")
    print("    어긋나면 단일 프레임 차선 배정이 전부 틀립니다(추적과 달리 사후 교정이 없습니다).")


if __name__ == "__main__":
    main()
