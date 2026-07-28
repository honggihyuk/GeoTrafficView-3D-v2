"""
평가용 프레임 추출 — 라벨링할 프레임을 클립에서 균등 샘플링.

트랙 평가(MOTA/IDF1)를 하려면 **연속 프레임**이 필요하므로 기본은 '연속 구간' 모드다.
  --mode seq   : start 프레임부터 stride 간격으로 n장 (트랙 평가용, 권장)
  --mode even  : 클립 전체에서 균등 n장 (검출 평가만 할 때)

사용:
  python tools/eval/extract_frames.py --loc PANGYO_2 --n 40 --stride 3
출력:
  webmap/public/eval/<loc>/frames/f######.jpg   (브라우저 라벨러가 로드)
  webmap/public/eval/<loc>/manifest.json
"""
import argparse
import json
import os
import sys

import cv2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--video", default=None, help="기본: location/<loc>/footage/clip.mp4")
    ap.add_argument("--n", type=int, default=40, help="추출 프레임 수")
    ap.add_argument("--stride", type=int, default=3, help="seq 모드 프레임 간격")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--mode", choices=["seq", "even"], default="seq")
    args = ap.parse_args()

    video = args.video or os.path.join(REPO, "location", args.loc, "footage", "clip.mp4")
    if not os.path.exists(video):
        sys.exit(f"영상 없음: {video}")
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.mode == "seq":
        idxs = [args.start + i * args.stride for i in range(args.n)]
        idxs = [i for i in idxs if i < total]
    else:
        step = max(1, total // args.n)
        idxs = list(range(0, total, step))[:args.n]

    out_dir = os.path.join(REPO, "webmap", "public", "eval", args.loc, "frames")
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        name = f"f{i:06d}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append({"index": i, "file": name})
    cap.release()

    manifest = {"loc": args.loc, "video": os.path.relpath(video, REPO).replace("\\", "/"),
                "fps": fps, "width": W, "height": H, "total_frames": total,
                "mode": args.mode, "stride": args.stride if args.mode == "seq" else None,
                "frames": saved}
    mp = os.path.join(REPO, "webmap", "public", "eval", args.loc, "manifest.json")
    json.dump(manifest, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[{args.loc}] {len(saved)}프레임 추출 ({W}x{H}, {fps:.0f}fps, 원본 {total}프레임)")
    print(f"  → {os.path.relpath(out_dir, REPO)}")
    print(f"  라벨러: http://localhost:5174/label.html?loc={args.loc}")


if __name__ == "__main__":
    main()
