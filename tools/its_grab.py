"""
Phase 4-①: ITS 실시간 CCTV에서 추론용 클립/스냅샷 캡처.

- ITS OpenAPI(cctvType=1, HLS)로 인천 인근 CCTV + cctvurl(.m3u8) 조회.
- 송도 중심에 가장 가까운 카메라 선택 → cv2로 열어 스냅샷 + N초 클립 저장.
- 카메라 메타(loc/lon/lat/resolution)를 json 으로 저장(다음 단계 캘리브레이션에서 사용).
- ITS 키는 .env.local 에서 로드, 로그에서 마스킹.

출력:
  location/<LOC>/footage/clip.mp4, location/<LOC>/cctv_<LOC>.png,
  location/<LOC>/_camera.json
"""
import json
import os
import ssl
import sys
import urllib.parse
import urllib.request

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_ingest"))
from _config import load_config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_SEC = 15
WIDE_BBOX = (126.40, 126.80, 37.20, 37.60)  # 인천 연안


def fetch_cameras(cfg, key):
    ctx = ssl.create_default_context()
    cams = []
    for typ in cfg["its"]["types"]:
        params = {"apiKey": key, "type": typ, "cctvType": 1,
                  "minX": WIDE_BBOX[0], "maxX": WIDE_BBOX[1],
                  "minY": WIDE_BBOX[2], "maxY": WIDE_BBOX[3], "getType": "json"}
        url = cfg["its"]["endpoint"] + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=30) as r:
                data = (json.loads(r.read().decode("utf-8")).get("response") or {}).get("data") or []
        except Exception as e:
            print(f"  [type={typ}] 실패: {type(e).__name__}"); continue
        for d in data:
            if d.get("cctvurl"):
                cams.append({"type": typ, "lon": float(d["coordx"]), "lat": float(d["coordy"]),
                             "name": d.get("cctvname"), "url": d["cctvurl"]})
    return cams


def main():
    cfg = load_config()
    key = os.environ.get("ITS_API_KEY")
    if not key:
        sys.exit("ITS_API_KEY 없음(.env.local 확인)")
    c = cfg["center_wgs84"]
    cams = fetch_cameras(cfg, key)
    if not cams:
        sys.exit("ITS 스트림 CCTV를 찾지 못함")
    cams.sort(key=lambda k: (k["lon"] - c["lon"]) ** 2 + (k["lat"] - c["lat"]) ** 2)
    cam = cams[0]
    print(f"카메라 {len(cams)}개 중 최근접 선택: {cam['name']}  ({cam['lon']:.5f},{cam['lat']:.5f}) type={cam['type']}")

    loc = "SONGDO_IC"
    out_dir = os.path.join(REPO, "location", loc)
    os.makedirs(os.path.join(out_dir, "footage"), exist_ok=True)

    cap = cv2.VideoCapture(cam["url"])
    if not cap.isOpened():
        sys.exit("스트림 열기 실패(ffmpeg 백엔드/URL 확인)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    ok, frame = cap.read()
    if not ok:
        sys.exit("첫 프레임 읽기 실패")
    h, w = frame.shape[:2]
    print(f"스트림 열림: {w}x{h} @ {fps:.1f}fps")

    cv2.imwrite(os.path.join(out_dir, f"cctv_{loc}.png"), frame)
    clip = os.path.join(out_dir, "footage", "clip.mp4")
    vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    vw.write(frame)
    n = int(fps * CLIP_SEC); got = 1
    for _ in range(n - 1):
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr); got += 1
    vw.release(); cap.release()
    print(f"클립 저장: {clip}  ({got}프레임)")

    with open(os.path.join(out_dir, "_camera.json"), "w", encoding="utf-8") as f:
        json.dump({"loc": loc, "lon": cam["lon"], "lat": cam["lat"], "name": cam["name"],
                   "width": w, "height": h, "fps": fps}, f, ensure_ascii=False, indent=2)
    print(f"메타 저장: {out_dir}\\_camera.json")


if __name__ == "__main__":
    main()
