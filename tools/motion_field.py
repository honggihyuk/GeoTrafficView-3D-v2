"""
운동 필드 학습 CLI — 맵매칭 궤적에서 v(link, s) 를 뽑아 저장한다.

"그 위치에서 객체들이 평균적으로 어떻게 움직이는가"를 데이터로 만든다. 방향은 차로 접선이
주므로 학습 대상이 아니고, 학습이 필요한 것은 종방향 속도뿐이다(자세한 근거는
trafficlab/motion/motion_field.py 독스트링).

사용:
  python tools/motion_field.py --loc PANGYO_2
  python tools/motion_field.py --loc PANGYO_2 --bin 5 --min-n 2
출력: output/motionfield/<loc>/field.json · profile.png
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from trafficlab.motion.motion_field import build, MotionField


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True)
    ap.add_argument("--src", default=None, help="기본: output/mapmatch/<loc>/clip.json.gz")
    ap.add_argument("--bin", type=float, default=10.0, help="s축 격자(m)")
    ap.add_argument("--min-n", type=int, default=3, help="bin 당 최소 관측 수")
    ap.add_argument("--smooth", type=int, default=2, help="이동평균 반폭(bin). 0이면 평활 없음")
    ap.add_argument("--clamp", nargs=2, type=float, default=[15.0, 120.0], metavar=("MIN", "MAX"),
                    help="표시용 속도 상·하한(km/h). 캘리브레이션이 어긋난 링크의 0/130km/h 방지")
    args = ap.parse_args()

    src = args.src or os.path.join(REPO, "output", "mapmatch", args.loc, "clip.json.gz")
    if not os.path.exists(src):
        sys.exit(f"맵매칭 결과 없음: {src}\n  → python tools/mapmatch.py --loc {args.loc}")

    f = build(src, bin_m=args.bin, min_n=args.min_n, smooth=args.smooth,
              clamp=tuple(args.clamp) if args.clamp else None)
    st = f["stats"]
    print(f"[{args.loc}] 트랙 {st['tracks']} · 속도 표본 {st['pairs']}쌍 "
          f"(이상치 제외 {st['dropped']})")
    print(f"전역 중앙 속도 {f['global_kmh']:.1f} km/h · 링크별 중앙값 확보 {len(f['link_median_kmh'])}개")
    print(f"격자 {st['bins']:,}칸 중 관측으로 채운 칸 {st['bins_observed']:,} "
          f"({100*st['coverage']:.1f}%)")
    if st.get("bins_clamped"):
        print(f"  표시용 클램프 적용: {st['bins_clamped']:,}칸 / 링크 {st['links_clamped']}개 "
              f"({args.clamp[0]:.0f}~{args.clamp[1]:.0f} km/h) — 이 칸들은 관측이 아니라 "
              f"보기 좋게 만든 값입니다")

    # 출처 분포 — 어디까지가 관측이고 어디부터가 추정인지
    from collections import Counter
    c = Counter(s for L in f["links"].values() for s in L["src"])
    print("  칸 출처: " + " · ".join(f"{k} {v:,}" for k, v in c.most_common()))

    # 속도 분포와 평활성(인접 칸 변화율) — 시각적 자연스러움의 직접 지표
    obs_v = [v for L in f["links"].values()
             for v, s in zip(L["v_kmh"], L["src"]) if s == "observed"]
    if obs_v:
        print(f"  관측 칸 속도: 중앙 {np.median(obs_v):.1f} · "
              f"5% {np.percentile(obs_v,5):.1f} · 95% {np.percentile(obs_v,95):.1f} km/h")
    jumps = []
    for L in f["links"].values():
        v = np.asarray(L["v_kmh"], float)
        if len(v) >= 2:
            jumps.extend(np.abs(np.diff(v)) / max(L["bin_m"], 1e-6))
    if jumps:
        j = np.asarray(jumps)
        # km/h per m → 가속도 환산(m/s²) = (dv/ds)*v, v는 중앙값으로 근사
        vmid = float(np.median(obs_v)) if obs_v else f["global_kmh"]
        acc = j / 3.6 * (vmid / 3.6)
        print(f"  속도 변화율: 중앙 {np.median(j):.2f} km/h/m · 95% {np.percentile(j,95):.2f}"
              f"  → 가속도 환산 95% {np.percentile(acc,95):.2f} m/s²"
              f" ({'자연스러움' if np.percentile(acc,95) < 2.5 else '급변 — --smooth 를 키우세요'})")

    out_dir = os.path.join(REPO, "output", "motionfield", args.loc)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "field.json")
    json.dump(f, open(p, "w", encoding="utf-8"), ensure_ascii=False)

    # 조회 검증 — 실제 런타임 경로로 몇 점 찍어 본다
    mf = MotionField(f)
    sample = [(lid, s) for lid in list(f["links"])[:3] for s in (0.0, 25.0, 60.0)]
    print("\n조회 검증 (런타임 경로):")
    for lid, s in sample[:9]:
        print(f"  link {lid[-12:]:>12s} s={s:5.1f}m → {mf.speed_kmh(lid, s):6.1f} km/h "
              f"[{mf.source(lid, s)}]")

    print(f"\n→ {os.path.relpath(p, REPO)}  ({os.path.getsize(p)/1e3:.0f} KB)")
    if st["coverage"] < 0.05:
        print(f"\n  ⚠ 관측 커버리지가 {100*st['coverage']:.1f}% 뿐입니다. 대부분의 칸이 "
              f"링크·전역 중앙값으로 채워졌다는 뜻이라, '그 지점의 평균 속도'가 아니라 "
              f"'이 카메라의 평균 속도'가 적용됩니다. 시각적으로는 자연스럽지만 지점별 "
              f"차이는 재현되지 않습니다 — 녹화 시간을 늘리면 그대로 개선됩니다.")


if __name__ == "__main__":
    main()
