"""
카메라(로케이션) 코드 일괄 변경 — 파일·JSON·웹맵·DB를 한 번에 정합성 있게 갱신.

로케이션 코드는 다음 모든 곳에 나타나므로 수동 변경은 위험하다:
  location/<LOC>/ (폴더·G_projection_<LOC>.json·cctv_<LOC>.png·_camera.json)
  output/**/<LOC>/ · eval/<LOC>/ · webmap/public/eval/<LOC>/
  webmap/public/data/{replay/<loc>.json.gz, footage/<loc>.mp4, cctv_<LOC>.png}
  webmap/public/data/cameras.geojson (cctv_id·replay_url·clip_url·snapshot_url)
  replay 내부 location_code · db/geotraffic.db (cctv, detection)

사용:
  python tools/rename_location.py --map OLD=NEW [--map OLD2=NEW2] [--dry-run]
"""
import argparse
import glob
import gzip
import json
import os
import re
import shutil
import sqlite3
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(REPO, "webmap", "public", "data")


def mv(src, dst, dry, log):
    if not os.path.exists(src) or src == dst:
        return
    if os.path.exists(dst):
        log.append(f"    ! 대상 존재, 건너뜀: {os.path.relpath(dst, REPO)}")
        return
    log.append(f"    {os.path.relpath(src, REPO)} → {os.path.relpath(dst, REPO)}")
    if not dry:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)


def rename(old, new, dry):
    log = [f"[{old} → {new}]"]

    # 1) location/ 폴더와 내부 파일
    ld, nd = os.path.join(REPO, "location", old), os.path.join(REPO, "location", new)
    mv(ld, nd, dry, log)
    base = nd if not dry else ld
    for pat, repl in ((f"G_projection_{old}.json", f"G_projection_{new}.json"),
                      (f"G_projection_{old}.json.bak", f"G_projection_{new}.json.bak"),
                      (f"cctv_{old}.png", f"cctv_{new}.png")):
        mv(os.path.join(base, pat), os.path.join(base, repl), dry, log)

    # 2) JSON 내부 필드
    if not dry:
        gp = os.path.join(nd, f"G_projection_{new}.json")
        if os.path.exists(gp):
            g = json.load(open(gp, encoding="utf-8"))
            g.setdefault("meta", {})["location_code"] = new
            g.setdefault("inputs", {})["cctv_path"] = f"cctv_{new}.png"
            json.dump(g, open(gp, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        cj = os.path.join(nd, "_camera.json")
        if os.path.exists(cj):
            c = json.load(open(cj, encoding="utf-8")); c["loc"] = new
            json.dump(c, open(cj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log.append("    JSON 필드(location_code·cctv_path·loc) 갱신")

    # 3) output/**/<LOC>, eval/<LOC>, webmap/public/eval/<LOC>
    for d in glob.glob(os.path.join(REPO, "output", "**", old), recursive=True):
        if os.path.isdir(d):
            mv(d, os.path.join(os.path.dirname(d), new), dry, log)
    for root in (os.path.join(REPO, "eval"), os.path.join(REPO, "webmap", "public", "eval")):
        mv(os.path.join(root, old), os.path.join(root, new), dry, log)

    # 4) 웹맵 산출물
    mv(os.path.join(WEB, "replay", f"{old.lower()}.json.gz"),
       os.path.join(WEB, "replay", f"{new.lower()}.json.gz"), dry, log)
    mv(os.path.join(WEB, "footage", f"{old.lower()}.mp4"),
       os.path.join(WEB, "footage", f"{new.lower()}.mp4"), dry, log)
    mv(os.path.join(WEB, f"cctv_{old}.png"), os.path.join(WEB, f"cctv_{new}.png"), dry, log)

    # 5) replay 내부 location_code
    if not dry:
        for p in glob.glob(os.path.join(REPO, "output", "**", "*.json.gz"), recursive=True) + \
                 glob.glob(os.path.join(WEB, "replay", "*.json.gz")):
            try:
                d = json.load(gzip.open(p, "rt", encoding="utf-8"))
            except Exception:
                continue
            if d.get("location_code") == old:
                d["location_code"] = new
                with gzip.open(p, "wt", encoding="utf-8") as f:
                    json.dump(d, f)
    log.append("    replay location_code 갱신")

    # 6) cameras.geojson
    cg = os.path.join(WEB, "cameras.geojson")
    if os.path.exists(cg) and not dry:
        fc = json.load(open(cg, encoding="utf-8"))
        for f in fc["features"]:
            p = f["properties"]
            if p.get("cctv_id") == old:
                p["cctv_id"] = new
                for k in ("replay_url", "clip_url", "snapshot_url"):
                    if p.get(k):
                        p[k] = p[k].replace(old.lower(), new.lower()).replace(old, new)
        json.dump(fc, open(cg, "w", encoding="utf-8"), ensure_ascii=False)
    log.append("    cameras.geojson 갱신")

    # 7) SQLite
    db = os.path.join(REPO, "db", "geotraffic.db")
    if os.path.exists(db) and not dry:
        con = sqlite3.connect(db)
        con.execute("UPDATE cctv SET cctv_id=? WHERE cctv_id=?", (new, old))
        con.execute("UPDATE detection SET cctv_id=? WHERE cctv_id=?", (new, old))
        con.commit(); con.close()
    log.append("    SQLite(cctv·detection) 갱신")
    return "\n".join(log)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="append", required=True, metavar="OLD=NEW")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pairs = []
    for m in a.map:
        old, _, new = m.partition("=")
        if not re.fullmatch(r"[A-Za-z0-9_]+", new or ""):
            sys.exit(f"잘못된 이름: {m} (영숫자·밑줄만)")
        pairs.append((old, new))
    if a.dry_run:
        print("=== DRY RUN (변경 없음) ===")
    for old, new in pairs:
        print(rename(old, new, a.dry_run))
    print("\n완료. 확인: python tools/eval/check_calibration.py --all")


if __name__ == "__main__":
    main()
