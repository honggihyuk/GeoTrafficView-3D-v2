"""
보조 데이터셋 → 국내 CCTV 택소노미 YOLO 변환 (datasets/class_map_kr.yaml 기준).

- UA-DETRAC(XML, 시퀀스별) / MIO-TCD Localization(CSV, 플랫)을 지원.
- 리맵 실패는 조용히 넘기지 않고 즉시 중단한다. 학습 오염의 대부분이
  '매핑 누락을 drop으로 흘려보낸' 데서 생기기 때문.
- __conflict__ 로 표시된 소스는 빌드를 거부한다(v2의 aihub_164 등).

CLI:
  python tools/build_kr_dataset.py --source ua_detrac --root D:/data/UA-DETRAC \\
      --out datasets/kr/stageA --split train --taxonomy v1 --stride 10 --mask-drops
  python tools/build_kr_dataset.py --source mio_tcd --root D:/data/MIO-TCD-Localization \\
      --out datasets/kr/stageA --split train --taxonomy v1 --mask-drops
  python tools/build_kr_dataset.py --source aihub_165 --taxonomy v2 --check

주의: UA-DETRAC의 차종은 car/bus/van/others 4종뿐이라 truck이 하나도 나오지 않는다.
      Stage A의 truck은 MIO-TCD(single_unit_truck·articulated_truck)에서만 온다.
"""
import argparse
import csv
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_MAP = os.path.join(REPO, "datasets", "class_map_kr.yaml")
DROP, CONFLICT = "__drop__", "__conflict__"

# 파서가 자체 사유로 버린 박스의 집계. 리맵 폐기와 구분해 보고한다.
# (커버리지를 줄이는 모든 동작은 로그에 남긴다 — 조용한 누락이 곧 오염이다)
EXTRA_DROPS = Counter()


# ── 리맵 명세 ────────────────────────────────────────────────────────────────
def load_taxonomy(source, taxonomy, force=False, path=CLASS_MAP):
    """(names, name→idx, src_class→name) 반환. 명세 위반 시 즉시 예외."""
    spec = yaml.safe_load(open(path, encoding="utf-8"))
    key = "target_taxonomy" if taxonomy == "v1" else "target_taxonomy_v2"
    names = [spec[key][i] for i in sorted(spec[key])]

    src = spec["sources"].get(source)
    if src is None:
        raise SystemExit(f"[{source}] class_map_kr.yaml 에 없는 소스. "
                         f"가능: {', '.join(spec['sources'])}")
    if src.get("role") == "excluded" and not force:
        raise SystemExit(f"[{source}] role=excluded — 학습 믹스에서 제외된 소스다.\n"
                         f"  사유: {' '.join((src.get('note') or '').split())}\n"
                         f"  그래도 빌드하려면 --force")

    mkey = "map" if taxonomy == "v1" else "map_v2"
    cmap = src.get(mkey)
    if cmap is None:
        raise SystemExit(f"[{source}] {taxonomy} 매핑({mkey})이 정의돼 있지 않다.")

    bad = sorted(k for k, v in cmap.items() if v == CONFLICT)
    if bad:
        raise SystemExit(
            f"[{source}] {taxonomy}에서 사용할 수 없는 소스다.\n"
            f"  충돌 클래스: {', '.join(bad)}\n"
            f"  class_map_kr.yaml 의 trailer_prerequisite 섹션을 참조할 것.")

    unknown = sorted({v for v in cmap.values() if v != DROP} - set(names))
    if unknown:
        raise SystemExit(f"[{source}] 택소노미({taxonomy})에 없는 대상 클래스: {unknown}")

    return names, {n: i for i, n in enumerate(names)}, cmap


def resolve(cmap, src_class, source):
    """매핑에 없는 클래스는 drop이 아니라 에러. 조용한 누락을 막는다."""
    if src_class not in cmap:
        raise SystemExit(f"[{source}] 매핑되지 않은 클래스 '{src_class}'.\n"
                         f"  class_map_kr.yaml 에 명시할 것(폐기하려면 {DROP}).")
    return cmap[src_class]


# ── 소스 파서 ────────────────────────────────────────────────────────────────
# 각 파서는 (image_path, out_stem, width, height,
#            [(src_class, x1, y1, x2, y2), ...],   # 라벨 후보
#            [(x1, y1, x2, y2), ...])              # 주석 미보장 영역(--mask-drops 대상)
# 를 순서대로 내보낸다. 좌표는 픽셀 절대값.

def _boxes_center_in(regions, x1, y1, x2, y2):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return any(rx1 <= cx <= rx2 and ry1 <= cy <= ry2 for rx1, ry1, rx2, ry2 in regions)


def iter_ua_detrac(root, split, stride):
    """UA-DETRAC: 시퀀스별 XML + Insight-MVT_Annotation_* 이미지 디렉터리."""
    tag = "Train" if split == "train" else "Test"
    ann_dir = _first_existing(root, [f"DETRAC-{tag}-Annotations-XML",
                                     f"DETRAC-{tag}-Annotations-XML-v3",
                                     f"{tag}-Annotations-XML"])
    img_root = _first_existing(root, [f"Insight-MVT_Annotation_{tag}"])

    for xml_name in sorted(os.listdir(ann_dir)):
        if not xml_name.endswith(".xml"):
            continue
        seq = os.path.splitext(xml_name)[0]
        seq_dir = os.path.join(img_root, seq)
        if not os.path.isdir(seq_dir):
            print(f"  ! 이미지 없음, 건너뜀: {seq}")
            continue
        rootel = ET.parse(os.path.join(ann_dir, xml_name)).getroot()

        # ignored_region: 중심이 이 안에 들어가는 박스는 폐기(주석 미보장 영역)
        ignored = []
        for b in rootel.findall("./ignored_region/box"):
            x, y = float(b.get("left")), float(b.get("top"))
            ignored.append((x, y, x + float(b.get("width")), y + float(b.get("height"))))

        for fi, frame in enumerate(rootel.findall("frame")):
            if fi % stride:
                continue
            num = int(frame.get("num"))
            img = os.path.join(seq_dir, f"img{num:05d}.jpg")
            if not os.path.isfile(img):
                continue
            boxes = []
            for tgt in frame.findall("./target_list/target"):
                b, at = tgt.find("box"), tgt.find("attribute")
                if b is None or at is None:
                    continue
                x, y = float(b.get("left")), float(b.get("top"))
                x2, y2 = x + float(b.get("width")), y + float(b.get("height"))
                if _boxes_center_in(ignored, x, y, x2, y2):
                    EXTRA_DROPS["ignored_region"] += 1
                    continue
                boxes.append((at.get("vehicle_type"), x, y, x2, y2))
            # 시퀀스 내 프레임은 해상도가 동일 → 디렉터리 단위 캐시
            w, h = _image_size(img, cache_dir=True)
            yield img, f"{seq}_img{num:05d}", w, h, boxes, ignored


def iter_mio_tcd(root, split, stride):
    """MIO-TCD Localization: gt_<split>.csv (헤더 없음) + <split>/ 플랫 이미지."""
    gt = _first_existing(root, [f"gt_{split}.csv", os.path.join(split, f"gt_{split}.csv")],
                         want_dir=False)
    img_dir = _first_existing(root, [split])

    rows = {}
    with open(gt, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) < 6 or not r[0].strip():
                continue
            if r[0].strip().lower() in ("id", "image", "filename"):  # 헤더 방어
                continue
            stem = r[0].strip()
            rows.setdefault(stem, []).append(
                (r[1].strip(), float(r[2]), float(r[3]), float(r[4]), float(r[5])))

    for i, stem in enumerate(sorted(rows)):
        if i % stride:
            continue
        img = os.path.join(img_dir, f"{stem}.jpg")
        if not os.path.isfile(img):
            continue
        w, h = _image_size(img)
        yield img, stem, w, h, rows[stem], []


PARSERS = {"ua_detrac": iter_ua_detrac, "mio_tcd": iter_mio_tcd}
# AI Hub 164/165는 승인 후 실물 XML/JSON 스키마를 확인하고 추가한다.
# 문서만 보고 추측해 넣으면 조용히 어긋난 라벨을 만든다.


def _first_existing(root, candidates, want_dir=True):
    for c in candidates:
        p = os.path.join(root, c)
        if (os.path.isdir(p) if want_dir else os.path.isfile(p)):
            return p
    raise SystemExit(f"{root} 아래에서 찾지 못함: {candidates}")


_SIZE_CACHE = {}


def _image_size(path, cache_dir=False):
    """cache_dir=True는 '이 디렉터리의 이미지는 모두 같은 해상도'가 보장될 때만.
    MIO-TCD처럼 한 폴더에 크기가 뒤섞인 소스에 쓰면 전부 틀어진다."""
    from PIL import Image  # 헤더만 읽음
    if not cache_dir:
        with Image.open(path) as im:
            return im.size
    key = os.path.dirname(path)
    if key not in _SIZE_CACHE:
        with Image.open(path) as im:
            _SIZE_CACHE[key] = im.size
    return _SIZE_CACHE[key]


_PLACE_FN = []   # 처음 성공한 방식을 기억. Windows에서 심볼릭이 매번 실패하면
                 # 78만 장 × 예외 발생이 되므로 재시도하지 않는다.


def _place(src, dst, mode, log=print):
    if os.path.exists(dst):
        return
    if mode == "copy":
        shutil.copy2(src, dst); return
    if _PLACE_FN:
        _PLACE_FN[0](src, dst); return
    for fn in (os.symlink, os.link, shutil.copy2):
        try:
            fn(src, dst)
        except (OSError, NotImplementedError, AttributeError):
            continue
        _PLACE_FN.append(fn)
        log(f"  이미지 배치 방식: {fn.__name__}")
        return
    raise SystemExit(f"이미지를 배치할 수 없다: {src} → {dst}")


# YOLO의 레터박스 패딩색. 모델이 이미 '내용 없음'으로 학습해 온 값이라
# 임의의 검정/흰색보다 인공적 경계가 덜 생긴다.
PAD = (114, 114, 114)


def _paint(src, dst, regions):
    """주석 미보장/폐기 영역을 덮어 배경 음성으로 학습되는 것을 막는다."""
    from PIL import Image, ImageDraw
    with Image.open(src) as im:
        im = im.convert("RGB")
        d = ImageDraw.Draw(im)
        for x1, y1, x2, y2 in regions:
            d.rectangle([x1, y1, x2, y2], fill=PAD)
        im.save(dst, quality=95)


# ── 빌드 ─────────────────────────────────────────────────────────────────────
def check(source, taxonomy, force, log=print):
    """빌드 없이 매핑만 검증. 파서가 없는 소스(aihub_164 등)도 점검할 수 있다."""
    names, idx, cmap = load_taxonomy(source, taxonomy, force)
    log(f"[{source}] {taxonomy} 매핑 유효. 택소노미 nc={len(names)} {names}")
    for k in sorted(cmap):
        v = cmap[k]
        log(f"  {k:24} → {v}" + ("" if v == DROP else f"  (idx {idx[v]})"))
    if source not in PARSERS:
        log(f"  ! 파서 미구현 — 빌드 불가. 지원: {', '.join(sorted(PARSERS))}")
    return names


def build(source, root, out, split, taxonomy, stride, link, force,
          mask_drops=False, log=print):
    names, idx, cmap = load_taxonomy(source, taxonomy, force)
    if source not in PARSERS:
        raise SystemExit(f"[{source}] 파서 미구현. 지원: {', '.join(sorted(PARSERS))}\n"
                         f"  매핑만 점검하려면 --check")
    img_out = os.path.join(out, "images", split)
    lbl_out = os.path.join(out, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    # 반드시 쓰기 '전'에 검사한다. 뒤로 미루면 클래스 인덱스가 어긋난 라벨이
    # 이미 디스크에 깔린 뒤에 중단된다.
    _write_data_yaml(out, names, log)

    EXTRA_DROPS.clear()
    kept = Counter(); dropped = Counter(); n_img = n_empty = n_bad = 0
    n_painted = n_paint_box = 0
    MIN_PX = 3   # 클리핑 후 이보다 작은 박스는 학습에 해롭다

    for img, stem, w, h, boxes, unlabeled in PARSERS[source](root, split, stride):
        lines = []
        paint = list(unlabeled) if mask_drops else []
        for src_class, x1, y1, x2, y2 in boxes:
            tgt = resolve(cmap, src_class, source)
            if tgt == DROP:
                dropped[src_class] += 1
                # 폐기한 박스는 '라벨 없는 객체'로 남아 배경 음성이 된다.
                # UA-DETRAC others(트럭 다수)가 대표적 — truck을 배경으로 가르친다.
                if mask_drops:
                    paint.append((x1, y1, x2, y2))
                continue
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            if x2 - x1 < MIN_PX or y2 - y1 < MIN_PX:   # 잘린 뒤 소멸한 박스
                n_bad += 1
                continue
            lines.append(f"{idx[tgt]} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                         f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
            kept[tgt] += 1

        # 박스가 하나도 없는 프레임도 배경(negative) 샘플로 유효하다 → 빈 txt 유지
        if not lines:
            n_empty += 1
        name = f"{source}_{stem}"
        dst = os.path.join(img_out, name + os.path.splitext(img)[1])
        if paint:
            _paint(img, dst, paint)
            n_painted += 1
            n_paint_box += len(paint)
        else:
            _place(img, dst, link, log)
        with open(os.path.join(lbl_out, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        if n_img % 5000 == 0:
            log(f"  … {n_img} 장")

    log(f"[{source}/{split}] 이미지 {n_img} (빈 프레임 {n_empty}) → {os.path.relpath(out, REPO)}")
    log(f"  유지 : {dict(sorted(kept.items()))}")
    if dropped:
        log(f"  폐기 : {dict(sorted(dropped.items()))}  (명세상 {DROP})")
    if EXTRA_DROPS:
        log(f"  제외 : {dict(sorted(EXTRA_DROPS.items()))}  (파서 자체 사유)")
    if mask_drops:
        log(f"  마스킹: 이미지 {n_painted} / 영역 {n_paint_box} (배경 음성화 차단)")
    elif dropped or EXTRA_DROPS:
        log(f"  ! 폐기·제외 영역이 라벨 없이 남아 배경 음성이 된다. "
            f"해당 클래스를 학습할 계획이면 --mask-drops 를 쓸 것.")
    if n_bad:
        log(f"  무효 : {n_bad} (클리핑 후 {MIN_PX}px 미만)")
    return out


def _write_data_yaml(out, names, log):
    """여러 소스를 같은 out에 누적하므로, 택소노미가 어긋나면 덮어쓰지 않고 중단."""
    p = os.path.join(out, "data.yaml")
    if os.path.isfile(p):
        prev = yaml.safe_load(open(p, encoding="utf-8")).get("names")
        if prev != names:
            raise SystemExit(f"{p} 의 기존 택소노미({prev})와 불일치({names}).\n"
                             f"  서로 다른 taxonomy를 한 디렉터리에 섞고 있다. --out 을 분리할 것.")
        return
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": os.path.abspath(out), "train": "images/train",
                        "val": "images/val", "nc": len(names), "names": names},
                       f, allow_unicode=True, sort_keys=False)
    log(f"  data.yaml 생성 (nc={len(names)}, names={names})")


def main():
    ap = argparse.ArgumentParser()
    # choices로 막지 않는다 — 파서 없는 소스도 --check 로 매핑을 점검할 수 있어야 한다.
    ap.add_argument("--source", required=True, help="class_map_kr.yaml 의 sources 키")
    ap.add_argument("--root", help="원본 데이터셋 루트 (--check 시 불필요)")
    ap.add_argument("--check", action="store_true", help="매핑만 검증하고 종료")
    ap.add_argument("--out", default=os.path.join("datasets", "kr", "stageA"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--taxonomy", default="v1", choices=["v1", "v2"])
    ap.add_argument("--stride", type=int, default=1,
                    help="프레임 간격. UA-DETRAC은 연속 영상이라 10 전후 권장")
    ap.add_argument("--link", default="auto", choices=["auto", "copy"],
                    help="auto=심볼릭→하드링크→복사 순 폴백")
    ap.add_argument("--force", action="store_true", help="role=excluded 소스도 빌드")
    ap.add_argument("--mask-drops", action="store_true",
                    help="폐기 박스·주석 미보장 영역을 회색으로 덮는다. "
                         "UA-DETRAC others(트럭 다수)가 truck의 배경 음성이 되는 것을 막음")
    a = ap.parse_args()
    if a.check:
        check(a.source, a.taxonomy, a.force)
        return
    if not a.root:
        ap.error("--root 는 필수 (매핑만 볼 거라면 --check)")
    build(a.source, a.root, os.path.join(REPO, a.out) if not os.path.isabs(a.out) else a.out,
          a.split, a.taxonomy, max(1, a.stride), a.link, a.force, a.mask_drops)


if __name__ == "__main__":
    main()
