"""
CCTV 스트림 → 프레임/클립 캡처 (캘리브레이션·추론 입력 준비).

- ITS 카메라: fetch_its_cctv.py 의 cctvurl(.m3u8)을 --url 로 바로 사용 가능.
- UTIC 카메라: cctvStream.jsp 는 HLS 플레이어 HTML 페이지이므로, 인증된(등록 IP) 세션에서
  실제 .m3u8 스트림 URL을 추출해 --url 로 전달해야 한다(브라우저 개발자도구 네트워크 탭 등).
- OpenCV(cv2.VideoCapture)로 열리는 모든 소스(mp4/HLS/RTSP) 지원.

사용:
  python grab_frame.py --loc SONGDO_L020101 --url <stream_or_m3u8> --snapshot --clip-sec 20
출력:
  location/<LOC>/cctv_<LOC>.png        (캘리브레이션용 스냅샷)
  location/<LOC>/footage/clip.mp4      (추론용 클립, --clip-sec>0)
"""
import argparse
import os
import sys

import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--url", required=True, help="mp4/.m3u8/RTSP 등 cv2로 열리는 소스")
    ap.add_argument("--snapshot", action="store_true", help="cctv_<LOC>.png 저장")
    ap.add_argument("--clip-sec", type=float, default=0.0, help="clip.mp4 로 녹화할 초")
    args = ap.parse_args()

    out_dir = os.path.join(REPO, "location", args.loc)
    os.makedirs(os.path.join(out_dir, "footage"), exist_ok=True)

    cap = cv2.VideoCapture(args.url)
    if not cap.isOpened():
        sys.exit(f"스트림 열기 실패: {args.url}\n  (UTIC은 등록 IP·키 필요; jsp가 아닌 실제 .m3u8 URL 필요)")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"열림: {w}x{h} @ {fps:.1f}fps")

    ok, frame = cap.read()
    if not ok:
        sys.exit("첫 프레임 읽기 실패")
    if args.snapshot:
        snap = os.path.join(out_dir, f"cctv_{args.loc}.png")
        cv2.imwrite(snap, frame)
        print(f"스냅샷 → {snap}  (K/H/해상도 캘리브레이션에 이 이미지 사용)")

    if args.clip_sec > 0:
        clip = os.path.join(out_dir, "footage", "clip.mp4")
        vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n = int(fps * args.clip_sec)
        vw.write(frame)
        for _ in range(n - 1):
            ok, fr = cap.read()
            if not ok:
                break
            vw.write(fr)
        vw.release()
        print(f"클립 → {clip} ({args.clip_sec}s)")
    cap.release()


if __name__ == "__main__":
    main()
