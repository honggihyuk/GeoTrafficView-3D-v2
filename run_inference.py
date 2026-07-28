"""
추론 실행 진입점 (TrafficLab InferencePipeline 래퍼).

준비물:
  1) location/<LOC>/G_projection_<LOC>.json   (tools/make_calibration_template.py 로 생성 후 실캘리브레이션으로 교체)
  2) 영상 소스: mp4 / HLS(.m3u8) / 프레임 디렉터리  (tools/grab_frame.py 로 UTIC·ITS 프레임 확보)
  3) 모델 가중치: inference_config.yaml 의 model.weights (예: models/yolo11s.pt)
  4) pip install ultralytics torch  (GPU 권장)

사용:
  python run_inference.py --loc SONGDO_L020101 --source location/SONGDO_L020101/footage/clip.mp4 \
                          --config slight_smoothing
→ output/model-<>_tracker-bytetrack/<config>/<LOC>/<clip>.json.gz
그 파일을 tools/bridge_to_webmap.py 로 웹맵(5174)에 등록.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from trafficlab.inference.pipeline import InferencePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True, help="location code (예: SONGDO_L020101)")
    ap.add_argument("--source", required=True, help="mp4 / .m3u8 / 프레임 디렉터리 경로")
    ap.add_argument("--config", default=None, help="inference_config.yaml 의 config 키")
    args = ap.parse_args()

    g_proj = os.path.join(REPO, "location", args.loc, f"G_projection_{args.loc}.json")
    if not os.path.exists(g_proj):
        sys.exit(f"G-Projection 없음: {g_proj}\n  → python tools/make_calibration_template.py 로 먼저 생성")

    pipe = InferencePipeline(
        location_code=args.loc,
        footage_path=args.source,
        config_path=os.path.join(REPO, "inference_config.yaml"),
        output_root=os.path.join(REPO, "output"),
        g_proj_path=g_proj,
        config_name=args.config,
        prior_dims_path=os.path.join(REPO, "prior_dimensions.json"),
        log_fn=print,
        progress_fn=lambda p: print(f"  진행 {p}%", end="\r"),
    )
    out = pipe.run()
    print(f"\n완료: {out}")


if __name__ == "__main__":
    main()
