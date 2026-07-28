// 지도 위 3D 이동체 — deck.gl SimpleMeshLayer + 트랙 보간.
//
// 기존 fill-extrusion 박스와의 차이:
//   1) 실제 3D 형상 + **전조등/미등을 별도 레이어**로 그려 정면/후면이 눈에 보인다.
//      (박스는 방향이 모양에만 반영돼 어느 쪽이 앞인지 알 수 없었다)
//   2) 30fps 키프레임을 rAF(60fps+)에서 **보간**한다. 기존에는 프레임을 그대로 찍어
//      지터가 그대로 보였다. 위치는 Catmull-Rom, heading은 최단호 각도 보간.
//   3) 외부 glTF 에셋을 쓰지 않는다 — 메시를 코드로 생성하므로 라이선스·다운로드가 없다.
//
// 좌표 규약(여기서 틀리면 차가 옆으로 간다):
//   sat_coords heading = atan2(dy,dx) → **동쪽 0°, 반시계**
//   deck.gl orientation = [pitch, yaw, roll], yaw = z축 반시계 회전
//   메시를 +X 정면 / +Z 상방으로 만들었으므로 yaw = heading 그대로, roll = 0.

import { MapboxOverlay } from '@deck.gl/mapbox';
import { SimpleMeshLayer } from '@deck.gl/mesh-layers';

const CLASS_DIM = {                      // [길이, 폭, 높이] m — 메시는 단위크기라 이 값이 스케일
  car: [4.4, 1.8, 1.5], van: [5.2, 1.9, 2.0], truck: [8.0, 2.5, 3.1],
  bus: [11.0, 2.5, 3.4], motorcycle: [2.1, 0.8, 1.5], motor: [2.1, 0.8, 1.5],
  bicycle: [1.8, 0.7, 1.5], person: [0.6, 0.6, 1.7], pedestrian: [0.6, 0.6, 1.7],
  tricycle: [2.8, 1.2, 1.75], 'awning-tricycle': [3.0, 1.4, 1.9], people: [0.6, 0.6, 1.7],
};
const CLASS_COLOR = {
  car: [34, 211, 238], van: [56, 189, 248], truck: [234, 179, 8], bus: [245, 158, 11],
  motorcycle: [244, 114, 182], motor: [244, 114, 182], bicycle: [167, 139, 250],
  person: [132, 204, 22], pedestrian: [132, 204, 22], people: [163, 230, 53],
  tricycle: [251, 146, 60], 'awning-tricycle': [253, 186, 116],
};

// ---------- 절차적 메시 ----------
function box(P, N, I, [x0, x1, y0, y1, z0, z1]) {
  const v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
             [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
  const faces = [[[0, 3, 2, 1], [0, 0, -1]], [[4, 5, 6, 7], [0, 0, 1]],
                 [[0, 1, 5, 4], [0, -1, 0]], [[2, 3, 7, 6], [0, 1, 0]],
                 [[1, 2, 6, 5], [1, 0, 0]], [[3, 0, 4, 7], [-1, 0, 0]]];
  for (const [q, n] of faces) {
    const b = P.length / 3;
    for (const k of q) { P.push(...v[k]); N.push(...n); }
    I.push(b, b + 1, b + 2, b, b + 2, b + 3);
  }
}

function mesh(boxes) {
  const P = [], N = [], I = [];
  boxes.forEach((b) => box(P, N, I, b));
  return {
    positions: { value: new Float32Array(P), size: 3 },
    normals: { value: new Float32Array(N), size: 3 },
    indices: { value: new Uint32Array(I), size: 1 },
  };
}

// 단위 차량(길이·폭·높이 1). +X 정면. 캐빈이 뒤로 치우쳐 실루엣만으로도 앞뒤가 보인다.
const MESH_BODY = mesh([[-0.5, 0.5, -0.5, 0.5, 0.0, 0.42],
                        [-0.30, 0.14, -0.42, 0.42, 0.42, 0.72]]);
const MESH_HEAD = mesh([[0.44, 0.51, -0.44, -0.22, 0.14, 0.28],
                        [0.44, 0.51, 0.22, 0.44, 0.14, 0.28]]);
const MESH_TAIL = mesh([[-0.51, -0.44, -0.44, -0.20, 0.16, 0.32],
                        [-0.51, -0.44, 0.20, 0.44, 0.16, 0.32]]);

// ---------- 트랙 타임라인 ----------
const MAX_MS = 200 / 3.6;          // 200 km/h — 이보다 빠른 '이동'은 관측이 아니라 투영 발산이다

/** 물리적으로 불가능한 점프에서 트랙을 **끊는다**(이어 붙이지 않는다).
 *
 * 튄 샘플 몇 개를 버리고 이어 붙이면(재앵커) 결국 어딘가에서 순간이동이 남는다.
 * 렌더링에서는 차가 사라졌다 다시 나타나는 편이 수 km를 날아가는 것보다 정직하다.
 * 반환: 연속 구간들의 배열.
 */
function splitOnJumps(a) {
  if (a.length < 2) return [a];
  const segs = [];
  let cur = [a[0]];
  for (let i = 1; i < a.length; i++) {
    const p = cur[cur.length - 1], c = a[i];
    const dt = Math.max(c.t - p.t, 1e-3);
    if (Math.hypot(c.x - p.x, c.y - p.y) / dt > MAX_MS) {
      if (cur.length >= 2) segs.push(cur);
      cur = [c];
    } else {
      cur.push(c);
    }
  }
  if (cur.length >= 2) segs.push(cur);
  return segs;
}

/** 프레임 배열 → tid별 시간순 샘플. 보간의 입력.
 *
 * 지면 투영은 지평선 근처에서 발산한다(실측: 송도IC 검출의 13.7%가 300m 밖, 최대 33km).
 * 그 좌표를 그대로 그리면 차가 수 km를 순간이동한다. 여기서 걸러 낸다 —
 * 근본 해결은 파이프라인(tools/retrack.py --max-range)이지만 웹맵은 게이트를 거치지 않은
 * replay 도 받으므로 렌더러도 방어한다. */
export function buildTracks(frames, fps) {
  const m = new Map();
  frames.forEach((fr, i) => {
    const t = (fr.frame_index ?? i) / fps;
    for (const o of fr.objects || []) {
      if (o.tracked_id == null || !o.sat_coords) continue;
      const [x, y] = o.sat_coords;
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (!m.has(o.tracked_id)) m.set(o.tracked_id, []);
      m.get(o.tracked_id).push({
        t, x, y,
        h: o.have_heading && o.heading != null ? o.heading : null,
        cls: o.class,
      });
    }
  });
  const out = new Map();
  for (const [k, a] of m) {
    a.sort((p, q) => p.t - q.t);
    const segs = splitOnJumps(a);
    segs.forEach((s, j) => out.set(segs.length > 1 ? `${k}#${j}` : k, fillHeadings(s)));
  }
  return out;
}

const MOVE_M = 0.6;      // 이보다 적게 움직이면 방향을 정할 수 없다고 본다

/** heading 이 비어 있는 샘플을 **트랙 자신의 이동**에서 채운다.
 *
 * 파이프라인은 저속·단트랙에 have_heading=false 를 남긴다(실측: OKRYEON_IC 11%,
 * SANGAM01 3.8% 만 heading 보유). 그대로 두면 렌더러가 전부 걸러내 지도가 비어 보인다.
 * 여기서 채우는 값은 **그 트랙이 실제로 지나간 경로**에서 나온 것이라 지어낸 정보가 아니다.
 * 끝까지 안 움직인 트랙만 null 로 남긴다(그건 정말 방향을 모르는 경우다). */
function fillHeadings(a) {
  if (a.length < 2) return a;
  const deg = (dx, dy) => (Math.atan2(dy, dx) * 180 / Math.PI + 360) % 360;
  for (let i = 0; i < a.length; i++) {
    if (a[i].h != null) continue;
    // 앞뒤로 창을 넓혀 가며 충분한 변위를 찾는다
    for (let w = 1; w < a.length; w++) {
      const lo = Math.max(0, i - w), hi = Math.min(a.length - 1, i + w);
      const dx = a[hi].x - a[lo].x, dy = a[hi].y - a[lo].y;
      if (Math.hypot(dx, dy) >= MOVE_M) { a[i].h = deg(dx, dy); break; }
      if (lo === 0 && hi === a.length - 1) break;
    }
  }
  // 그래도 빈 곳은 트랙 전체의 평균 방향으로(원형 평균)
  let sx = 0, sy = 0, n = 0;
  for (const p of a) if (p.h != null) { sx += Math.cos(p.h * Math.PI / 180); sy += Math.sin(p.h * Math.PI / 180); n++; }
  if (n) {
    const mean = deg(sx, sy);
    for (const p of a) if (p.h == null) p.h = mean;
  }
  return a;
}

const lerpAngle = (a, b, u) => {
  const d = ((b - a + 540) % 360) - 180;      // 최단호
  return (a + d * u + 360) % 360;
};

/** Catmull-Rom(균일) — 키프레임을 지나면서 부드럽다. */
const cr = (p0, p1, p2, p3, u) => {
  const u2 = u * u, u3 = u2 * u;
  return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
    + (-p0 + 3 * p1 - 3 * p2 + p3) * u3);
};

/** 시각 t(초)에 살아있는 트랙들의 보간 상태. */
export function sampleTracks(tracks, t) {
  const out = [];
  for (const [tid, a] of tracks) {
    if (!a.length || t < a[0].t || t > a[a.length - 1].t) continue;
    let i = 1;
    while (i < a.length && a[i].t < t) i++;
    const p1 = a[i - 1], p2 = a[Math.min(i, a.length - 1)];
    const span = p2.t - p1.t;
    const u = span > 1e-6 ? (t - p1.t) / span : 0;
    let x, y;
    // 균일 Catmull-Rom은 **샘플 간격이 균일할 때만** 성립한다. 트랙에 프레임 결손이 있어
    // 간격이 들쭉날쭉하면 크게 오버슈트한다(실측: 120Hz 샘플에서 8.5m 튐 = 3687km/h 상당).
    // 간격이 고르지 않으면 선형으로 떨어뜨린다.
    const uniform = a.length >= 4 && i >= 2 && i <= a.length - 2
      && (() => {
        const d0 = a[i - 1].t - a[i - 2].t, d1 = span, d2 = a[i + 1].t - a[i].t;
        const mn = Math.min(d0, d1, d2), mx = Math.max(d0, d1, d2);
        return mn > 1e-6 && mx / mn < 1.2;
      })();
    if (uniform) {
      const p0 = a[i - 2], p3 = a[i + 1];
      x = cr(p0.x, p1.x, p2.x, p3.x, u);
      y = cr(p0.y, p1.y, p2.y, p3.y, u);
    } else {
      x = p1.x + (p2.x - p1.x) * u;
      y = p1.y + (p2.y - p1.y) * u;
    }
    let h = p1.h;
    if (p1.h != null && p2.h != null) h = lerpAngle(p1.h, p2.h, u);
    else if (h == null) h = p2.h;
    out.push({ tid, x, y, heading: h, cls: p1.cls || p2.cls });
  }
  return out;
}

// ---------- 렌더러 ----------
export class VehicleRenderer {
  /** @param toLL (x,y 로컬미터) → [lon,lat] */
  constructor(map, toLL) {
    this.map = map;
    this.toLL = toLL;
    this.overlay = new MapboxOverlay({ interleaved: true, layers: [] });
    map.addControl(this.overlay);
  }

  /** 카메라를 바꿀 때 **반드시** 호출한다.
   *
   * sat_coords 는 카메라별 world 원점 기준 로컬 미터라, toLL 은 카메라마다 다르다.
   * 렌더러를 재사용하면서 이걸 갱신하지 않으면 두 번째 카메라부터 이전 원점으로
   * 좌표가 계산돼 지도 밖에 그려진다(= 이동체가 안 보인다). */
  setProjector(toLL) {
    this.toLL = toLL;
    this.clear();
  }

  update(states) {
    // heading이 없는(정지·미확정) 객체는 그리지 않는다 — 임의 방향으로 세워 두면
    // 없는 정보를 있는 것처럼 보여주게 된다. lane_snap/mapmatch를 거치면 거의 사라진다.
    const data = states.filter((s) => s.heading != null).map((s) => ({
      position: this.toLL(s.x, s.y),
      yaw: s.heading,
      scale: CLASS_DIM[s.cls] || CLASS_DIM.car,
      color: CLASS_COLOR[s.cls] || CLASS_COLOR.car,
      tid: s.tid, cls: s.cls,
    }));
    const common = {
      data,
      getPosition: (d) => d.position,
      getOrientation: (d) => [0, d.yaw, 0],   // [pitch, yaw, roll] — 메시가 +X정면/+Z상방
      getScale: (d) => d.scale,
      sizeScale: 1,
      parameters: { depthTest: true },
      updateTriggers: { getPosition: data, getOrientation: data },
    };
    this.overlay.setProps({
      layers: [
        new SimpleMeshLayer({ ...common, id: 'veh-body', mesh: MESH_BODY,
                              getColor: (d) => d.color, pickable: true,
                              material: { ambient: 0.45, diffuse: 0.6, shininess: 40 } }),
        new SimpleMeshLayer({ ...common, id: 'veh-head', mesh: MESH_HEAD,
                              getColor: () => [255, 240, 200], material: false }),
        new SimpleMeshLayer({ ...common, id: 'veh-tail', mesh: MESH_TAIL,
                              getColor: () => [230, 40, 40], material: false }),
      ],
    });
  }

  clear() { this.overlay.setProps({ layers: [] }); }

  remove() {
    try { this.map.removeControl(this.overlay); } catch (e) { /* 이미 제거됨 */ }
  }
}
