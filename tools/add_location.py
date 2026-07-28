"""
임의 지역에 CCTV 지점 추가(도시 확장 일반화) — 중심좌표 근처 ITS 스트림 CCTV 1대를
스트림캡처 → 자동 placeholder 캘리브레이션 → SAHI 추론 → webmap 등록 + PostGIS 적재.

사용:
  python tools/add_location.py --loc PANGYO_2 --center-lon 127.1004 --center-lat 37.3997
"""
import argparse
import json
import os
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request
import warnings

warnings.filterwarnings("ignore")
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "data_ingest"))
sys.path.insert(0, os.path.join(REPO, "tools"))
from _config import load_config
from make_calibration_template import build_calibration
from run_inference_sahi import run_sahi


def fetch_bbox(cfg, key, bb):
    ctx = ssl.create_default_context(); out = []
    for typ in cfg["its"]["types"]:
        p = {"apiKey": key, "type": typ, "cctvType": 1, "minX": bb[0], "maxX": bb[1],
             "minY": bb[2], "maxY": bb[3], "getType": "json"}
        u = cfg["its"]["endpoint"] + "?" + urllib.parse.urlencode(p)
        try:
            with urllib.request.urlopen(u, context=ctx, timeout=30) as r:
                d = (json.loads(r.read().decode()).get("response") or {}).get("data") or []
                d = [d] if isinstance(d, dict) else d
        except Exception:
            d = []
        for c in d:
            if isinstance(c, dict) and c.get("cctvurl"):
                out.append({"lon": float(c["coordx"]), "lat": float(c["coordy"]),
                            "name": c.get("cctvname"), "url": c["cctvurl"], "type": typ})
    return out


def grab(url, loc, clip_sec):
    d = os.path.join(REPO, "location", loc); os.makedirs(os.path.join(d, "footage"), exist_ok=True)
    # TOPIS(eseoul) 등 Referer 요구 HLS를 위해 ffmpeg 헤더 옵션 주입
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "referer;https://topis.seoul.go.kr/|user_agent;Mozilla/5.0|timeout;5000000")  # 5s 소켓 타임아웃
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    fps = min(cap.get(cv2.CAP_PROP_FPS) or 15.0, 30.0)
    ok, fr = cap.read()
    if not ok:
        cap.release(); return None
    h, w = fr.shape[:2]
    cv2.imwrite(os.path.join(d, f"cctv_{loc}.png"), fr)
    clip = os.path.join(d, "footage", "clip.mp4")
    vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)); vw.write(fr)
    target = int(fps * clip_sec); got = 1; fails = 0
    while got < target and fails < 30:  # 연속 실패 30회면 중단(stall 방지)
        ok, f2 = cap.read()
        if not ok:
            fails += 1; continue
        fails = 0; vw.write(f2); got += 1
    vw.release(); cap.release()
    print(f"  캡처: {got}프레임 (목표 {target})")
    return {"w": w, "h": h, "fps": fps, "clip": clip, "dir": d} if got >= 10 else None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--center-lon", type=float)
    ap.add_argument("--center-lat", type=float)
    ap.add_argument("--url", help="직접 스트림 URL(TOPIS hlsUrl 등). 지정 시 ITS 조회 생략")
    ap.add_argument("--lon", type=float); ap.add_argument("--lat", type=float); ap.add_argument("--name", default=None)
    ap.add_argument("--span", type=float, default=0.06)
    ap.add_argument("--clip-sec", type=float, default=5)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--sahi", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    if a.url:  # 직접 스트림(TOPIS 등)
        cam = {"lon": a.lon, "lat": a.lat, "name": a.name or a.loc, "url": a.url, "type": "direct"}
        print(f"선택(직접): {cam['name']} ({cam['lon']:.4f},{cam['lat']:.4f})")
    else:
        key = os.environ.get("ITS_API_KEY")
        if not key:
            sys.exit("ITS_API_KEY 없음")
        cx, cy = a.center_lon, a.center_lat
        cams = fetch_bbox(cfg, key, [cx - a.span, cx + a.span, cy - a.span, cy + a.span])
        if not cams:
            sys.exit("인근 ITS 스트림 CCTV 없음")
        cams.sort(key=lambda c: (c["lon"] - cx) ** 2 + (c["lat"] - cy) ** 2)
        cam = cams[0]
        print(f"선택: {cam['name']} ({cam['lon']:.4f},{cam['lat']:.4f}) [{cam['type']}] · 후보 {len(cams)}대")

    g = grab(cam["url"], a.loc, a.clip_sec)
    if not g:
        sys.exit("스트림 캡처 실패")
    json.dump({"loc": a.loc, "lon": cam["lon"], "lat": cam["lat"], "name": cam["name"],
               "width": g["w"], "height": g["h"], "fps": g["fps"]},
              open(os.path.join(g["dir"], "_camera.json"), "w", encoding="utf-8"), ensure_ascii=False)
    build_calibration(a.loc, cam["lon"], cam["lat"], g["w"], g["h"], note=f"ITS {cam['name']} placeholder")
    out = run_sahi(a.loc, g["clip"], slice_px=(384 if a.sahi else 0), max_frame=a.frames)
    subprocess.run([sys.executable, os.path.join(REPO, "tools", "bridge_to_webmap.py"),
                    "--replay", out, "--loc", a.loc, "--ingest"], check=True)
    print(f"완료: {a.loc} — webmap 등록 + PostGIS 적재")


if __name__ == "__main__":
    main()
