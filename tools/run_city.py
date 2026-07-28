"""
도시 단위 다중 카메라 자동화 배치.

각 ITS 카메라에 대해:  스트림 클립 캡처 → 자동(placeholder) 캘리브레이션 → 추론 → PostGIS 적재.
※ 정밀 위치는 카메라별로 calibrate.html에서 GCP를 찍어 교체(placeholder는 ITS 좌표 기반 근사).

사용:
  python tools/run_city.py --n 2 --start 1        # 송도 인근 2번째~ 카메라 자동 처리
  python tools/run_city.py --n 10 --start 0 --sahi # 10대, SAHI(정밀·느림)
"""
import argparse
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "data_ingest"))
sys.path.insert(0, os.path.join(REPO, "tools"))
from _config import load_config
from its_grab import fetch_cameras
from make_calibration_template import build_calibration
from run_inference_sahi import run_sahi


def grab(url, loc, clip_sec):
    out_dir = os.path.join(REPO, "location", loc)
    os.makedirs(os.path.join(out_dir, "footage"), exist_ok=True)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        cap.release(); return None
    h, w = frame.shape[:2]
    cv2.imwrite(os.path.join(out_dir, f"cctv_{loc}.png"), frame)
    clip = os.path.join(out_dir, "footage", "clip.mp4")
    vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    vw.write(frame)
    for _ in range(int(fps * clip_sec) - 1):
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr)
    vw.release(); cap.release()
    return {"w": w, "h": h, "fps": fps, "clip": clip, "dir": out_dir}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--start", type=int, default=1, help="가까운 순 시작 인덱스(0=최근접, 이미 처리한 카메라 건너뛰기용)")
    ap.add_argument("--clip-sec", type=float, default=5)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--sahi", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    key = os.environ.get("ITS_API_KEY")
    if not key:
        sys.exit("ITS_API_KEY 없음(.env.local)")
    c = cfg["center_wgs84"]
    cams = fetch_cameras(cfg, key)
    cams.sort(key=lambda k: (k["lon"] - c["lon"]) ** 2 + (k["lat"] - c["lat"]) ** 2)
    picked = cams[args.start:args.start + args.n]
    print(f"대상 카메라 {len(picked)}대 (전체 {len(cams)}대 중 {args.start}~)")

    done = []
    for idx, cam in enumerate(picked):
        loc = f"ITS_C{args.start + idx:02d}"
        print(f"\n[{loc}] {cam['name']} ({cam['lon']:.5f},{cam['lat']:.5f})")
        g = grab(cam["url"], loc, args.clip_sec)
        if not g:
            print("  스트림 실패 → 건너뜀"); continue
        json.dump({"loc": loc, "lon": cam["lon"], "lat": cam["lat"], "name": cam["name"],
                   "width": g["w"], "height": g["h"], "fps": g["fps"]},
                  open(os.path.join(g["dir"], "_camera.json"), "w", encoding="utf-8"), ensure_ascii=False)
        build_calibration(loc, cam["lon"], cam["lat"], g["w"], g["h"],
                          note=f"ITS {cam['name']} 자동 placeholder")
        out = run_sahi(loc, g["clip"], slice_px=(384 if args.sahi else 0), max_frame=args.frames)
        subprocess.run([sys.executable, os.path.join(REPO, "db", "ingest_postgis.py"), "--src", out], check=True)
        done.append(loc)

    print(f"\n완료: {done}")
    print("도시 요약 → python -c \"...\" 또는 llm/nl2sql.py 로 질의")


if __name__ == "__main__":
    main()
