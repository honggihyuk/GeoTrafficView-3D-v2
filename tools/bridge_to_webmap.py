"""
브리지: 원본 TrafficLab-3D 산출물 → webmap 자동 반영.

두 경로:
  (A) replay 또는 G_projection 에 world 블록이 있으면(우리 GCP/템플릿 캘리브레이션):
      sat_coords 는 이미 실좌표 로컬미터 → meta.world 주입 후 그대로 등록.
  (B) 원본 앱 산출물처럼 sat_coords 가 위성 이미지 '픽셀'이고 world 가 없으면:
      --anchors (sat_px↔lonlat ≥2) 로 유사변환(회전·스케일·평행이동) 추정 →
      sat_coords/sat_floor_box 를 실좌표 로컬미터로 재계산 + meta.world 생성.

공통 후처리:
  - webmap/public/data/replay/<loc>.json.gz 작성
  - clip → H.264 트랜스코딩(webmap/public/data/footage/<loc>.mp4), snapshot 복사
  - cameras.geojson 에 카메라 upsert(마커=world 원점, has_replay=true, replay/clip/snapshot 경로)
  - (--ingest) PostGIS 적재

사용:
  python tools/bridge_to_webmap.py --replay output/.../clip.json.gz --loc YEONSU_JCT --clip location/YEONSU_JCT/footage/clip.mp4
  python tools/bridge_to_webmap.py --replay <원본.json.gz> --loc CAM_X --anchors anchors.json --epsg 32652
anchors.json 예:  [{"sat_px":[512,300],"lonlat":[126.65,37.405]}, ...]
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from pyproj import Transformer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBDATA = os.path.join(REPO, "webmap", "public", "data")


def load_replay(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def affine_2d(src, dst):
    """2D 어파인 최소자승(≥3점). dst = M@[x,y,1]. 반환 M(2x3).
    위성 이미지(탑다운)의 px→실좌표는 어파인으로 충분(균일/비균일 스케일·회전 모두 수용)."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    A = np.zeros((2 * len(src), 6)); b = np.zeros(2 * len(src))
    for i, ((x, y), (X, Y)) in enumerate(zip(src, dst)):
        A[2 * i] = [x, y, 1, 0, 0, 0]; b[2 * i] = X
        A[2 * i + 1] = [0, 0, 0, x, y, 1]; b[2 * i + 1] = Y
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]]])


def resolve_world(replay, loc, anchors, epsg):
    """world 블록과 (필요시) 픽셀→UTM 변환함수 반환."""
    w = (replay.get("meta") or {}).get("world")
    if w and w.get("origin_easting"):
        return w, None  # 이미 로컬미터(변환 불필요)
    # G_projection 의 world
    gp = os.path.join(REPO, "location", loc, f"G_projection_{loc}.json")
    if os.path.exists(gp):
        gw = json.load(open(gp, encoding="utf-8")).get("world")
        if gw and gw.get("origin_easting"):
            return gw, None
    # anchors 기반 (원본 SAT-픽셀)
    if not anchors:
        sys.exit("world 블록이 없고 --anchors 도 없음 → 지오레퍼런스 불가")
    if len(anchors) < 3:
        sys.exit("어파인 지오레퍼런스는 앵커 3개 이상 필요")
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    src = [a["sat_px"] for a in anchors]
    dst = [list(tr.transform(a["lonlat"][0], a["lonlat"][1])) for a in anchors]
    M = affine_2d(src, dst)
    utm = np.array(dst); origin = utm.mean(0)
    world = {"epsg": epsg, "origin_easting": float(origin[0]),
             "origin_northing": float(origin[1]), "ground_ellipsoid_h": 28.0}

    def px_to_local(x, y):
        p = M @ np.array([x, y, 1.0])  # → UTM
        return [float(p[0] - origin[0]), float(p[1] - origin[1])]
    return world, px_to_local


def upsert_camera(feature):
    """카메라 upsert + 중복 제거: 같은 cctv_id, 또는 같은 이름/근접(≤250m)한 '원본(무-replay)' 마커 제거."""
    cg = os.path.join(WEBDATA, "cameras.geojson")
    fc = json.load(open(cg, encoding="utf-8")) if os.path.exists(cg) else {"type": "FeatureCollection", "features": []}
    cid = feature["properties"]["cctv_id"]
    name = feature["properties"].get("name")
    rc = feature["geometry"]["coordinates"]

    def dist(a, b):
        import math
        return math.hypot((a[0] - b[0]) * 88800, (a[1] - b[1]) * 111000)

    def keep(f):
        p = f["properties"]
        if p.get("cctv_id") == cid:
            return False  # 같은 카메라(재등록) → 교체
        if not p.get("has_replay"):  # 원본 ITS 마커 중 이 카메라와 겹치는 것 제거
            if name and p.get("name") == name:
                return False
            if dist(rc, f["geometry"]["coordinates"]) < 250:
                return False
        return True

    fc["features"] = [f for f in fc["features"] if keep(f)]
    fc["features"].append(feature)
    json.dump(fc, open(cg, "w", encoding="utf-8"), ensure_ascii=False)
    return len(fc["features"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--loc", default=None)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--anchors", default=None)
    ap.add_argument("--epsg", type=int, default=32652)
    ap.add_argument("--name", default=None)
    ap.add_argument("--ingest", action="store_true")
    args = ap.parse_args()

    replay = load_replay(args.replay)
    loc = args.loc or replay.get("location_code", "CAM")
    anchors = json.load(open(args.anchors, encoding="utf-8")) if args.anchors else None
    world, px_to_local = resolve_world(replay, loc, anchors, args.epsg)

    # 픽셀→로컬미터 변환이 필요하면 sat_coords/sat_floor_box 재계산
    if px_to_local is not None:
        for fr in replay["frames"]:
            for o in fr.get("objects", []):
                if o.get("sat_coords"):
                    o["sat_coords"] = px_to_local(*o["sat_coords"])
                if o.get("sat_floor_box"):
                    o["sat_floor_box"] = [px_to_local(x, y) for x, y in o["sat_floor_box"]]
        print(f"[{loc}] 원본 SAT-픽셀 → 실좌표 변환(앵커 {len(anchors)}개)")
    replay.setdefault("meta", {})["world"] = world

    os.makedirs(os.path.join(WEBDATA, "replay"), exist_ok=True)
    os.makedirs(os.path.join(WEBDATA, "footage"), exist_ok=True)
    loc_l = loc.lower()
    rp = os.path.join(WEBDATA, "replay", f"{loc_l}.json.gz")
    with gzip.open(rp, "wt", encoding="utf-8") as f:
        json.dump(replay, f)

    # clip 트랜스코딩(H.264)
    clip_url = None
    clip = args.clip or os.path.join(REPO, "location", loc, "footage", "clip.mp4")
    if os.path.exists(clip):
        import imageio_ffmpeg
        dst = os.path.join(WEBDATA, "footage", f"{loc_l}.mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", clip,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], check=True)
        clip_url = f"data/footage/{loc_l}.mp4"

    # snapshot
    snap_url = None
    snap = os.path.join(REPO, "location", loc, f"cctv_{loc}.png")
    if os.path.exists(snap):
        shutil.copy(snap, os.path.join(WEBDATA, f"cctv_{loc}.png"))
        snap_url = f"data/cctv_{loc}.png"

    # 이름
    name = args.name
    cj = os.path.join(REPO, "location", loc, "_camera.json")
    if not name and os.path.exists(cj):
        name = json.load(open(cj, encoding="utf-8")).get("name")

    lon, lat = Transformer.from_crs(world["epsg"], 4326, always_xy=True).transform(
        world["origin_easting"], world["origin_northing"])
    res = replay["meta"].get("resolution", [1280, 720])
    feat = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"cctv_id": loc, "name": name or loc, "has_replay": True,
                           "replay_url": f"data/replay/{loc_l}.json.gz", "clip_url": clip_url,
                           "snapshot_url": snap_url, "world": world,
                           "width": res[0], "height": res[1], "fps": replay["meta"].get("fps", 30)}}
    total = upsert_camera(feat)
    print(f"[{loc}] webmap 등록 완료 · replay={os.path.relpath(rp, REPO)} · 마커@{lon:.5f},{lat:.5f} · cameras={total}")

    if args.ingest:
        subprocess.run([sys.executable, os.path.join(REPO, "db", "ingest_postgis.py"), "--src", args.replay], check=True)


if __name__ == "__main__":
    main()
