"""
HD맵(정밀도로지도) → GCP 스냅 / 차로 정렬 / 영상정합용 GeoJSON.

용도 3가지를 한 파일로 낸다:
  ① calibrate.html "HD맵 스냅" — GCP 지도클릭을 cm급 실좌표(HD맵 정점)로 스냅
  ② tools/lane_snap.py — A2_LINK 차로 링크 + **주행 방향**으로 이동체 정렬
  ③ tools/hdmap_align.py — B2/B3 노면표시로 위성영상과 Chamfer/DT 정합

**여러 데이터셋 병합 (v2.1)**
  보유 정밀도로지도는 노선/지구별로 쪼개져 있고(경부선 20구간, 외곽순환선 14구간 …),
  카메라마다 덮는 데이터셋이 다르다. 그래서 이 도구는 **국토정보플랫폼 루트를 훑어**
  등록 카메라를 덮는 섹션만 골라 병합한다. 카메라 반경 밖 피처는 버리므로
  경부선 전체를 지정해도 결과 파일은 작다.

**주행 방향(핵심)**
  A2_LINK는 차로 단위 방향성 링크다. 지오메트리 진행 방향 = FromNodeID→ToNodeID = 주행 방향.
  이 규약을 A1_NODE 좌표로 **섹션별로 검증하고 필요하면 뒤집어** 저장한다(properties.dir_fixed).
  덕분에 lane_snap이 **정지 차량의 정면/후면까지** 결정할 수 있다(관측 heading 불필요).

출력: webmap/public/data/hdmap/hdmap.geojson (WGS84)
  properties.layer / .id / .src(출처 데이터셋)
  A2_LINK  : lane_no, from_node, to_node, r_link, l_link, road_rank, link_type, length, dir_fixed
  A1_NODE  : node_type
  B2_SURFACELINEMARK / B3_SURFACEMARK : type, kind, link_id
  C1_TRAFFICLIGHT : type, link_id

사용:
  python tools/export_hdmap_snap.py                       # 등록 카메라 전부 자동 탐색
  python tools/export_hdmap_snap.py --loc PANGYO_2        # 특정 카메라만
  python tools/export_hdmap_snap.py --radius 1500         # 유지 반경 확대
  python tools/export_hdmap_snap.py --dir <HDMap_...>     # 자동탐색 없이 명시 지정
"""
import argparse
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from shapely.geometry import mapping, MultiPoint, Point

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 레이어 → (보존할 원본 컬럼 → geojson 키). 이름/제작사 등 한글 텍스트 필드는
# dbf 인코딩이 배포본마다 달라 깨지므로 아예 읽지 않는다(ID·코드값은 전부 ASCII).
LAYERS = {
    "A2_LINK": {"ID": "id", "LaneNo": "lane_no", "FromNodeID": "from_node", "ToNodeID": "to_node",
                "R_LinkID": "r_link", "L_LinkID": "l_link", "RoadRank": "road_rank",
                "LinkType": "link_type", "Length": "length"},
    "A1_NODE": {"ID": "id", "NodeType": "node_type"},
    "B2_SURFACELINEMARK": {"ID": "id", "Type": "type", "Kind": "kind", "R_LinkID": "r_link", "L_LinkID": "l_link"},
    "B3_SURFACEMARK": {"ID": "id", "Type": "type", "Kind": "kind", "LinkID": "link_id"},
    "C1_TRAFFICLIGHT": {"ID": "id", "Type": "type", "LinkID": "link_id"},
}

# 같은 구간이 좌표계별로 중복 배포된다. 프로젝트 기준(EPSG:32652)과 같은 판본을 우선한다.
VARIANT_ORDER = ("UTM52N_타원체고", "UTM52N", "UTMK_타원체고", "UTMK")


def _clean(v):
    """dbf의 NaN/numpy 스칼라를 JSON 직렬화 가능한 값으로."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, float):
        return None if v != v else round(v, 3)
    return v


def _round_coords(obj, nd=7):
    """좌표를 소수 7자리(≈1cm)로 반올림하고 z(타원체고)를 버린다 — 파일 크기 절반."""
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(float(c), nd) for c in obj[:2]]
        return [_round_coords(o, nd) for o in obj]
    return obj


# ---------- 카메라 원점 ----------
def load_targets(codes=None):
    """location/*/G_projection_<CODE>.json 의 world 원점 → [(code, lon, lat)]."""
    from pyproj import Transformer
    out = []
    for p in sorted(glob.glob(os.path.join(REPO, "location", "*", "G_projection_*.json"))):
        code = os.path.basename(os.path.dirname(p))
        if codes and code not in codes:
            continue
        if os.path.basename(p) != f"G_projection_{code}.json":
            continue                                     # _auto.json 등 부산물 제외
        try:
            w = json.load(open(p, encoding="utf-8"))["world"]
        except Exception:
            continue
        tr = Transformer.from_crs(w["epsg"], 4326, always_xy=True)
        lon, lat = tr.transform(w["origin_easting"], w["origin_northing"])
        out.append((code, float(lon), float(lat)))
    return out


# ---------- HD맵 섹션 탐색 ----------
def discover_sections(root):
    """root 아래 A2_LINK.shp 를 훑어 구간별로 좌표계 판본 1개씩 고른다."""
    best = {}
    for shp in glob.glob(os.path.join(root, "**", "A2_LINK.shp"), recursive=True):
        hd = os.path.dirname(shp)                        # .../HDMap_UTM52N_타원체고
        sec = os.path.dirname(hd)                        # .../SEC001_...
        rank = next((i for i, v in enumerate(VARIANT_ORDER) if v in os.path.basename(hd)),
                    len(VARIANT_ORDER))
        if sec not in best or rank < best[sec][0]:
            best[sec] = (rank, hd)
    return [hd for _, hd in sorted(best.values(), key=lambda x: x[1])]


def section_bbox_ll(shp):
    """.shp 헤더의 bbox만 읽어 WGS84로. (전체 로드 없이 후보 압축)"""
    import fiona
    from pyproj import Transformer
    with fiona.open(shp) as src:
        b, crs = src.bounds, src.crs
    tr = Transformer.from_crs(crs, 4326, always_xy=True)
    lo1, la1 = tr.transform(b[0], b[1])
    lo2, la2 = tr.transform(b[2], b[3])
    return (min(lo1, lo2), min(la1, la2), max(lo1, lo2), max(la1, la2))


def bbox_hit(bb, targets, margin_deg):
    return any(bb[0] - margin_deg <= lon <= bb[2] + margin_deg
               and bb[1] - margin_deg <= lat <= bb[3] + margin_deg
               for _, lon, lat in targets)


# ---------- 방향 정규화 ----------
def link_flip_flags(links, nodes):
    """A2_LINK 지오메트리 방향이 FromNode→ToNode 인지 판정. 반환 (flip 리스트, 통계).

    이것이 '부호 자기검증'이다 — 관측 트랙 없이 HD맵 자체 토폴로지만으로 성립한다.
    """
    npos = {r.ID: np.array([r.geometry.x, r.geometry.y])
            for r in nodes.itertuples() if r.geometry is not None} if nodes is not None else {}
    flip, fixed, checked, missing = [], 0, 0, 0
    for r in links.itertuples():
        fn = npos.get(getattr(r, "FromNodeID", None))
        if fn is None or r.geometry is None:
            missing += 1
            flip.append(False)
            continue
        cs = np.asarray(r.geometry.coords, dtype=float)[:, :2]
        checked += 1
        if np.linalg.norm(cs[0] - fn) > np.linalg.norm(cs[-1] - fn):
            fixed += 1
            flip.append(True)
        else:
            flip.append(False)
    return flip, fixed, checked, missing


def main():
    try:  # Windows 콘솔 cp949에서 한글 출력 크래시 방지
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    sys.path.insert(0, os.path.join(REPO, "data_ingest"))
    from _config import load_config
    cfg = load_config()
    default_root = cfg["paths"].get("hdmap_root") or os.path.dirname(
        os.path.dirname(os.path.dirname(cfg["paths"]["hdmap_dir"])))

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root, help="정밀도로지도 루트(하위를 재귀 탐색)")
    ap.add_argument("--dir", nargs="+", default=None, help="자동탐색 대신 HDMap 폴더를 직접 지정")
    ap.add_argument("--loc", nargs="+", default=None, help="대상 카메라 코드(미지정 시 등록 전체)")
    ap.add_argument("--radius", type=float, default=800.0, help="카메라 원점 기준 유지 반경(m)")
    ap.add_argument("--encoding", default="cp949", help="shapefile dbf 인코딩")
    args = ap.parse_args()

    import geopandas as gpd

    targets = load_targets(set(args.loc) if args.loc else None)
    if not targets:
        sys.exit("대상 카메라를 찾지 못했습니다(location/*/G_projection_<CODE>.json 확인).")
    print(f"대상 카메라 {len(targets)}대: " + ", ".join(c for c, _, _ in targets))

    if args.dir:
        sections = list(args.dir)
        print(f"명시 지정 섹션 {len(sections)}개")
    else:
        if not os.path.isdir(args.root):
            sys.exit(f"정밀도로지도 루트 없음: {args.root}")
        allsec = discover_sections(args.root)
        margin = args.radius / 111000.0 + 0.005
        sections = []
        for hd in allsec:
            try:
                if bbox_hit(section_bbox_ll(os.path.join(hd, "A2_LINK.shp")), targets, margin):
                    sections.append(hd)
            except Exception:
                continue
        print(f"섹션 탐색: 전체 {len(allsec)}개 중 카메라 주변 후보 {len(sections)}개")

    feats, seen = [], set()
    counts = {}
    per_cam = {c: 0 for c, _, _ in targets}
    used = []

    for hd in sections:
        name = os.path.relpath(hd, args.root) if not args.dir else hd
        short = name.split(os.sep)[0].replace("(B110)정밀도로지도_", "")

        links = gpd.read_file(os.path.join(hd, "A2_LINK.shp"), encoding=args.encoding)
        npath = os.path.join(hd, "A1_NODE.shp")
        nodes = gpd.read_file(npath, encoding=args.encoding) if os.path.exists(npath) else None

        # 카메라 반경 안에 링크가 있는지 실제 지오메트리로 확인(bbox는 긴 노선에서 무의미)
        from pyproj import Transformer
        tr = Transformer.from_crs(4326, links.crs, always_xy=True)
        pts = {c: Point(*tr.transform(lon, lat)) for c, lon, lat in targets}
        cam_pts = MultiPoint(list(pts.values()))
        near = links.distance(cam_pts) <= args.radius
        if not near.any():
            continue

        flip, n_fix, n_chk, n_mis = link_flip_flags(links, nodes)
        for c, p in pts.items():
            per_cam[c] += int((links.distance(p) <= args.radius).sum())
        used.append((short, os.sep.join(name.split(os.sep)[1:]) or ".", int(near.sum()), n_fix, n_chk, n_mis))

        for lyr, colmap in LAYERS.items():
            if lyr == "A2_LINK":
                g, fl = links, flip
            elif lyr == "A1_NODE":
                if nodes is None:
                    continue
                g, fl = nodes, None
            else:
                p = os.path.join(hd, lyr + ".shp")
                if not os.path.exists(p):
                    continue
                g, fl = gpd.read_file(p, encoding=args.encoding), None
            g = g.reset_index(drop=True)
            m = g.distance(cam_pts) <= args.radius       # 카메라 반경 밖 피처는 버린다
            g4 = g[m].to_crs(4326)
            idxs = list(g[m].index)
            cols = [c for c in colmap if c in g.columns]
            n = 0
            for row, src_i in zip(g4.itertuples(), idxs):
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                gj = mapping(geom)
                coords = _round_coords(gj["coordinates"])
                if lyr == "A2_LINK" and fl is not None and fl[src_i]:
                    coords = coords[::-1]
                props = {"layer": lyr, "src": short}
                for c in cols:
                    props[colmap[c]] = _clean(getattr(row, c, None))
                key = (lyr, props.get("id"))
                if props.get("id") is not None and key in seen:
                    continue                                              # 섹션 중복 제거
                seen.add(key)
                if lyr == "A2_LINK" and fl is not None:
                    props["dir_fixed"] = bool(fl[src_i])
                feats.append({"type": "Feature", "properties": props,
                              "geometry": {"type": gj["type"], "coordinates": coords}})
                n += 1
            counts[lyr] = counts.get(lyr, 0) + n

    if not feats:
        sys.exit(f"카메라 반경 {args.radius:.0f}m 안에 HD맵 피처가 없습니다. "
                 f"--radius 를 키우거나 해당 지역 정밀도로지도를 확보하세요.")

    out_dir = os.path.join(REPO, "webmap", "public", "data", "hdmap")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "hdmap.geojson")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)

    def nvert(c):
        return 1 if (c and isinstance(c[0], (int, float))) else sum(nvert(x) for x in c)

    nv = sum(nvert(f["geometry"]["coordinates"]) for f in feats)

    print("\n사용 섹션:")
    for short, sec, n_near, n_fix, n_chk, n_mis in used:
        print(f"  {short[:40]:40s} {sec[:34]:34s} 반경내 링크 {n_near:5d} · "
              f"방향검증 {n_chk}개/역방향 {n_fix}개/노드누락 {n_mis}개")
    print("\n카메라별 HD맵 커버리지:")
    for c, _, _ in targets:
        n = per_cam[c]
        print(f"  {c:16s} 반경 {args.radius:.0f}m 내 링크 {n:5d}  "
              f"{'★ 차로 정렬 가능' if n >= 5 else '— 미커버(OSM 폴백)'}")
    print("\n레이어별 피처: " + " · ".join(f"{k} {v}" for k, v in counts.items()) + f"  (정점 {nv:,}개)")
    print(f"→ {os.path.relpath(out, REPO)}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    print("다음: python tools/lane_snap.py --loc <카메라>     (차로·주행방향 정렬)")
    print("      python tools/hdmap_align.py --zoom 19        (위성영상 정합)")


if __name__ == "__main__":
    main()
