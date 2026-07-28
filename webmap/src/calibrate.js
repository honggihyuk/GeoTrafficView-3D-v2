import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import proj4 from 'proj4';
import { computeHomography, applyH, mul3, translate3, correction3, looCvRms, gcpSpread } from './homography.js';

proj4.defs('EPSG:32652', '+proj=utm +zone=52 +datum=WGS84 +units=m +no_defs +type=crs');
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
const EPSG = 32652;
const SVGNS = 'http://www.w3.org/2000/svg';

const statusEl = document.getElementById('status');
const img = document.getElementById('img');
const left = document.getElementById('left');
document.getElementById('dots').remove(); // SVG 오버레이로 대체
const ov = document.createElementNS(SVGNS, 'svg');
ov.setAttribute('id', 'ov');
ov.style.cssText = 'position:absolute;inset:0;pointer-events:none;width:100%;height:100%';
left.appendChild(ov);

let cam = null, existing = null, Hcur = null;
let pairs = [], roi = [], lanes = [], pending = null, laneTmp = [];
let mode = 'gcp';
const mapMarkers = [];

const map = new maplibregl.Map({
  container: 'map',
  style: { version: 8, sources: { esri: { type: 'raster', tiles: [ESRI], tileSize: 256, attribution: 'Esri, Maxar, CNES/Airbus DS' } }, layers: [{ id: 'esri', type: 'raster', source: 'esri' }] },
  center: [126.65, 37.4065], zoom: 17,
});
map.addControl(new maplibregl.NavigationControl());
const setStatus = (t) => (statusEl.textContent = t);

// HD맵 스냅: 정밀도로지도 정점을 로드해 GCP 지도클릭을 cm급 실좌표로 스냅
let hdVerts = [], hdLoaded = false, snapOn = false;
// HD맵 GCP 후보(tools/gcp_candidates.py 산출물) — 번호순으로 안내한다
let cands = [], candIdx = 0, candOn = false;
async function loadHdmap() {
  if (hdLoaded) return;
  hdLoaded = true;
  let fc;
  try {
    fc = await (await fetch('data/hdmap/hdmap.geojson')).json();
  } catch (e) { hdLoaded = false; setStatus('HD맵 스냅 데이터 없음 — export_hdmap_snap.py 실행 필요'); return; }
  for (const f of fc.features) {
    const g = f.geometry;
    if (g.type === 'LineString') g.coordinates.forEach((c) => hdVerts.push(c));
    else if (g.type === 'Point') hdVerts.push(g.coordinates);
    // B3_SURFACEMARK(횡단보도·화살표)는 Polygon이다. 횡단보도 모서리는 영상에서도
    // 또렷해 GCP 스냅 대상으로 최상급이라 정점을 함께 싣는다.
    else if (g.type === 'Polygon') g.coordinates.forEach((r) => r.forEach((c) => hdVerts.push(c)));
  }
  setStatus(`HD맵 스냅 로드(${hdVerts.length} 정점) · GCP 지도클릭이 정밀도로지도로 스냅됩니다`);
  // 지도 표시는 best-effort(스타일 로드 후)
  const addLayers = () => {
    try {
      if (!map.getSource('hdmap')) {
        map.addSource('hdmap', { type: 'geojson', data: fc });
        map.addLayer({ id: 'hdmap-line', type: 'line', source: 'hdmap',
          filter: ['==', ['geometry-type'], 'LineString'], paint: { 'line-color': '#22d3ee', 'line-width': 1.5 } });
        map.addLayer({ id: 'hdmap-poly', type: 'line', source: 'hdmap',
          filter: ['==', ['geometry-type'], 'Polygon'], paint: { 'line-color': '#a3e635', 'line-width': 1 } });
        map.addLayer({ id: 'hdmap-pt', type: 'circle', source: 'hdmap',
          filter: ['==', ['geometry-type'], 'Point'], paint: { 'circle-radius': 3, 'circle-color': '#f59e0b' } });
      }
    } catch (e) { /* 스타일 미로드 시 idle에서 재시도 */ map.once('idle', addLayers); }
  };
  if (map.isStyleLoaded()) addLayers(); else map.once('load', addLayers);
}
function nearestVert(ll) {
  let best = null, bd = Infinity;
  const clat = Math.cos(ll[1] * Math.PI / 180);
  for (const v of hdVerts) {
    const dx = (v[0] - ll[0]) * clat * 111320, dy = (v[1] - ll[1]) * 111320;
    const d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = v; }
  }
  return Math.sqrt(bd) < 15 ? best : null;   // 15m 이내면 스냅
}
document.getElementById('hdcand').addEventListener('change', (e) => {
  candOn = e.target.checked;
  if (candOn && !cands.length) setStatus('이 카메라의 GCP 후보가 없습니다 — python tools/gcp_candidates.py --loc <코드>');
  redraw(); updateCounts();
});

document.getElementById('hdsnap').addEventListener('change', (e) => {
  snapOn = e.target.checked;
  if (snapOn) loadHdmap();
  ['hdmap-line', 'hdmap-poly', 'hdmap-pt'].forEach((id) => map.getLayer && map.getLayer(id) &&
    map.setLayoutProperty(id, 'visibility', snapOn ? 'visible' : 'none'));
});

init();
async function init() {
  const fc = await (await fetch('data/cameras.geojson')).json();
  const cams = fc.features.filter((f) => f.properties.snapshot_url);
  const sel = document.getElementById('camSel');
  cams.forEach((f, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = f.properties.cctv_id;
    sel.appendChild(o);
  });
  window._cams = cams;
  sel.addEventListener('change', () => loadCamera(cams[+sel.value]));
  document.querySelectorAll('input[name=mode]').forEach((r) =>
    r.addEventListener('change', () => { mode = r.value; setStatus(modeHint()); }));
  if (cams.length) loadCamera(cams[0]);
  else setStatus('스냅샷 보유 카메라 없음 — grab_frame으로 먼저 캡처하세요');
}

function modeHint() {
  return { gcp: '① 좌 CCTV 특징점 → 우 지도 같은 지점 (4쌍↑)',
           roi: '② 좌 CCTV에서 도로 영역 외곽을 순서대로 클릭(3점↑)',
           lane: '③ 좌 CCTV에서 차로 진행방향 2점(시작→끝) 클릭' }[mode];
}

async function loadCamera(f) {
  cam = f.properties;
  pairs = []; roi = []; lanes = []; pending = null; laneTmp = [];
  mapMarkers.splice(0).forEach((m) => m.remove());
  img.src = cam.snapshot_url;
  const [lon, lat] = f.geometry.coordinates;
  map.setCenter([lon, lat]); map.setZoom(17.5);
  existing = await (await fetch(`/api/gproj?loc=${cam.cctv_id}`)).json().catch(() => null);
  Hcur = existing?.homography?.H || null;
  if (existing?.homography?.anchors_list) {
    pairs = existing.homography.anchors_list.filter((a) => a.px && a.lonlat)
      .map((a) => ({ px: a.px, ll: a.lonlat }));
    pairs.forEach((p) => addMapMarker(p.ll, pairs.indexOf(p) + 1));
  }
  if (existing?.roi_polygon) roi = existing.roi_polygon.slice();
  if (existing?.heading_guidelines) lanes = existing.heading_guidelines.map((g) => ({ sat: g.sat, heading: g.heading_deg }));
  // HD맵 GCP 후보 — tools/gcp_candidates.py 산출물. 없으면 조용히 넘어간다.
  cands = []; candIdx = 0;
  try {
    const r = await fetch(`data/gcp/${cam.cctv_id}.json`);
    if (r.ok) cands = (await r.json()).candidates || [];
  } catch (e) { /* 후보 없음 */ }
  redraw(); updateCounts();
  setStatus(`${cam.cctv_id} 로드 · ${modeHint()}` + (existing ? ' (기존 캘리브레이션 있음)' : ''));
}

// contain-fit 변환
function fit() {
  const r = img.getBoundingClientRect();
  const nat = cam.width / cam.height, box = r.width / r.height;
  let dw = r.width, dh = r.height, ox = 0, oy = 0;
  if (box > nat) { dw = r.height * nat; ox = (r.width - dw) / 2; } else { dh = r.width / nat; oy = (r.height - dh) / 2; }
  return { ox, oy, dw, dh };
}
const toNative = (e) => { const { ox, oy, dw, dh } = fit(); const r = img.getBoundingClientRect();
  return [(e.clientX - r.left - ox) / dw * cam.width, (e.clientY - r.top - oy) / dh * cam.height]; };
const disp = (u, v) => { const { ox, oy, dw, dh } = fit(); return [ox + u / cam.width * dw, oy + v / cam.height * dh]; };

img.addEventListener('click', (e) => {
  if (!cam) return;
  const p = toNative(e);
  if (mode === 'gcp') {
    // 후보 안내 모드: 지도 쪽 좌표를 HD맵 후보에서 그대로 가져오므로 **영상만 클릭**하면 된다.
    // 지도를 눈대중으로 클릭하는 단계가 사라져 지도 쪽 오차(1~2m)가 통째로 없어진다.
    if (candOn && candIdx < cands.length) {
      const c = cands[candIdx];
      pairs.push({ px: p, ll: c.lonlat });
      addMapMarker(c.lonlat, pairs.length);
      candIdx++;
      setStatus(`GCP ${pairs.length}쌍 · 후보 #${c.no}(${c.kind}) 완료` +
        (candIdx < cands.length ? ` → 다음 #${cands[candIdx].no} (${cands[candIdx].kind})` : ' · 후보 소진'));
      redraw(); updateCounts();
      return;
    }
    if (pending) { setStatus('우측 지도에서 같은 지점 클릭'); return; }
    pending = p; setStatus(`GCP #${pairs.length + 1}: 우측 지도에서 같은 지점 클릭`);
  } else if (mode === 'roi') {
    roi.push(p);
  } else if (mode === 'lane') {
    laneTmp.push(p);
    if (laneTmp.length === 2) { lanes.push({ px: [laneTmp[0], laneTmp[1]] }); laneTmp = []; }
  }
  redraw(); updateCounts();
});

map.on('click', (e) => {
  if (!cam || mode !== 'gcp' || !pending) return;
  let ll = [e.lngLat.lng, e.lngLat.lat];
  let snapped = false;
  if (snapOn && hdVerts.length) { const s = nearestVert(ll); if (s) { ll = s; snapped = true; } }
  pairs.push({ px: pending, ll });
  addMapMarker(ll, pairs.length);
  pending = null; redraw(); updateCounts();
  setStatus(`GCP ${pairs.length}쌍${snapped ? ' · 🧲HD맵 정점 스냅(cm)' : ''}`);
});

function addMapMarker(ll, n) {
  const el = document.createElement('div');
  el.style.cssText = 'width:16px;height:16px;border-radius:50%;background:#ec4899;color:#fff;font-size:10px;text-align:center;line-height:16px;border:2px solid #fff';
  el.textContent = n;
  mapMarkers.push(new maplibregl.Marker({ element: el }).setLngLat(ll).addTo(map));
}

function mk(tag, attrs) { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }

// ── 재작업 가이드 ────────────────────────────────────────────────
// 깊이 스케일이 틀어지는 주원인은 "GCP가 화면 세로(=깊이) 방향으로 좁게 뭉치는 것".
// 권장 구역 = 세로 3구간(원거리/중간/근거리) × 가로 3구간. 각 구역에 최소 1점.
const ZONES = [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => ({ r, c })));
const zoneOf = (px) => ({
  r: Math.min(2, Math.floor((px[1] / cam.height - 0.15) / 0.30)),   // 상단 15% 제외(하늘)
  c: Math.min(2, Math.floor(px[0] / cam.width * 3)),
});
function drawZones() {
  if (!document.getElementById('showZones').checked) return;
  const covered = new Set(pairs.map((p) => { const z = zoneOf(p.px); return `${z.r},${z.c}`; }));
  for (const z of ZONES) {
    if (z.r < 0) continue;
    const x0 = z.c / 3 * cam.width, x1 = (z.c + 1) / 3 * cam.width;
    const y0 = (0.15 + z.r * 0.30) * cam.height, y1 = (0.15 + (z.r + 1) * 0.30) * cam.height;
    const [sx0, sy0] = disp(x0, y0), [sx1, sy1] = disp(x1, y1);
    const has = covered.has(`${z.r},${z.c}`);
    ov.appendChild(mk('rect', {
      x: sx0 + 2, y: sy0 + 2, width: sx1 - sx0 - 4, height: sy1 - sy0 - 4,
      fill: has ? 'rgba(126,231,135,.07)' : 'rgba(236,72,153,.10)',
      stroke: has ? '#7ee787' : '#ec4899', 'stroke-width': 1, 'stroke-dasharray': has ? '' : '5 4',
    }));
    const t = mk('text', { x: (sx0 + sx1) / 2, y: (sy0 + sy1) / 2, 'text-anchor': 'middle',
                           'font-size': 11, fill: has ? '#7ee787' : '#ec4899' });
    t.textContent = has ? '✓' : (['먼 거리', '중간', '가까운 거리'][z.r] || '');
    ov.appendChild(t);
  }
}
/** 계산된 H로 지면 격자(차로 3.5m · 깊이 10m)를 영상에 되그려 육안 검증 */
function drawGrid() {
  if (!document.getElementById('showGrid').checked || !Hcur) return;
  const inv = (() => { try { return invert3(Hcur); } catch { return null; } })();
  if (!inv) return;
  const toPx = (X, Y) => { const p = applyH(inv, X, Y); return disp(p[0], p[1]); };
  const lanes = [-7, -3.5, 0, 3.5, 7], depths = [0, 10, 20, 30, 40, 50];
  for (const L of lanes) {
    const pts = depths.map((D) => toPx(L, D)).filter((p) => isFinite(p[0]) && isFinite(p[1]));
    if (pts.length > 1) ov.appendChild(mk('polyline', {
      points: pts.map((p) => p.join(',')).join(' '), fill: 'none',
      stroke: L === 0 ? '#22d3ee' : '#22d3ee88', 'stroke-width': L === 0 ? 2 : 1 }));
  }
  for (const D of depths) {
    const pts = lanes.map((L) => toPx(L, D)).filter((p) => isFinite(p[0]) && isFinite(p[1]));
    if (pts.length > 1) {
      ov.appendChild(mk('polyline', { points: pts.map((p) => p.join(',')).join(' '),
        fill: 'none', stroke: '#f59e0b88', 'stroke-width': 1 }));
      const t = mk('text', { x: pts[0][0] - 4, y: pts[0][1], 'text-anchor': 'end',
                             'font-size': 10, fill: '#f59e0b' });
      t.textContent = `${D}m`; ov.appendChild(t);
    }
  }
}
function invert3(M) {
  const [a, b, c] = M[0], [d, e, f] = M[1], [g, h, i] = M[2];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!isFinite(det) || Math.abs(det) < 1e-12) throw new Error('singular');
  return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
          [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
          [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
}
/** 실시간 품질 진단 + 다음 행동 지시 */
function updateGuide() {
  const el = document.getElementById('guide');
  if (!cam) return;
  const sp = gcpSpread(pairs.map((p) => p.px), cam.width, cam.height);
  let loo = null;
  if (pairs.length >= 5) {
    const utm = pairs.map((p) => proj4('EPSG:4326', `EPSG:${EPSG}`, p.ll));
    const oE = utm.reduce((s, u) => s + u[0], 0) / utm.length, oN = utm.reduce((s, u) => s + u[1], 0) / utm.length;
    loo = looCvRms(pairs.map((p) => p.px), utm.map((u) => [u[0] - oE, u[1] - oN]));
  }
  const covered = new Set(pairs.map((p) => { const z = zoneOf(p.px); return `${z.r},${z.c}`; }));
  const rows = [0, 1, 2].map((r) => [...covered].some((k) => k.startsWith(`${r},`)));
  const msgs = [];
  if (sp.n < 6) msgs.push(`점 ${sp.n}/6 이상 필요 <b>(4점은 잔차가 항상 0 = 검증 불가)</b>`);
  if (!rows[0]) msgs.push('🔴 <b>먼 거리(화면 위쪽)</b> 점이 없습니다 — 깊이 스케일이 틀어지는 1순위 원인');
  if (!rows[2]) msgs.push('🔴 <b>가까운 거리(화면 아래쪽)</b> 점이 없습니다');
  if (sp.vSpan < 0.35 && sp.n >= 2) msgs.push(`🔴 세로 분포 ${(sp.vSpan * 100) | 0}% — 50% 이상으로 넓히세요`);
  if (sp.hSpan < 0.35 && sp.n >= 2) msgs.push(`🟠 가로 분포 ${(sp.hSpan * 100) | 0}% — 좌우로 넓히세요`);
  const looTxt = loo === null
    ? '<span style="color:#9aa7b4">— (5점 이상부터 산출)</span>'
    : `<b style="color:${loo < 1 ? '#7ee787' : loo < 3 ? '#eab308' : '#ff7b72'}">${loo.toFixed(2)} m</b>`;
  el.innerHTML =
    `<b>정확도(LOO 교차검증)</b>: ${looTxt}<br/>` +
    `점 ${sp.n} · 잉여 ${sp.redundancy >= 0 ? '+' + sp.redundancy : sp.redundancy} · ` +
    `세로 ${(sp.vSpan * 100) | 0}% · 가로 ${(sp.hSpan * 100) | 0}%` +
    (msgs.length ? '<hr style="border:0;border-top:1px solid #2b333d;margin:4px 0">' + msgs.join('<br/>')
                 : '<br/><span style="color:#7ee787">✓ 분포 양호 — 저장 후 📏 격자로 육안 확인</span>');
}

function redraw() {
  ov.innerHTML = '';
  drawZones();
  drawGrid();
  // ROI 폴리곤
  if (roi.length) {
    const pts = roi.map(([u, v]) => disp(u, v).join(',')).join(' ');
    ov.appendChild(mk('polygon', { points: pts, fill: 'rgba(34,211,238,.18)', stroke: '#22d3ee', 'stroke-width': 2 }));
    roi.forEach(([u, v]) => { const [x, y] = disp(u, v); ov.appendChild(mk('circle', { cx: x, cy: y, r: 3, fill: '#22d3ee' })); });
  }
  // 차선 방향
  lanes.forEach((L) => {
    if (!L.px) return;
    const [a, b] = L.px.map(([u, v]) => disp(u, v));
    ov.appendChild(mk('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: '#f59e0b', 'stroke-width': 3, 'marker-end': 'url(#arr)' }));
    ov.appendChild(mk('circle', { cx: b[0], cy: b[1], r: 4, fill: '#f59e0b' }));
  });
  // GCP 점
  pairs.forEach((p, i) => { const [x, y] = disp(...p.px); ov.appendChild(dot(x, y, i + 1, '#ec4899')); });
  if (candOn) {
    // 예측 위치는 현재 캘리브레이션으로 투영한 값이라 어긋나 있다. '이 근처의 저 표시'를
    // 찾으라는 안내이지 그 픽셀을 그대로 찍으라는 뜻이 아니다(전체 그림은 output/gcp/*.png).
    cands.forEach((c, i) => {
      const [x, y] = disp(c.px[0], c.px[1]);
      const d = dot(x, y, c.no, i === candIdx ? '#22d3ee' : '#64748b');
      if (i < candIdx) d.style.opacity = '0.3';
      if (i === candIdx) d.style.boxShadow = '0 0 0 3px rgba(34,211,238,0.45)';
      ov.appendChild(d);
    });
  }
  if (pending) { const [x, y] = disp(...pending); ov.appendChild(dot(x, y, '?', '#22d3ee')); }
}
function dot(x, y, n, c) {
  const g = mk('g', {});
  g.appendChild(mk('circle', { cx: x, cy: y, r: 8, fill: c, stroke: '#fff', 'stroke-width': 2 }));
  const t = mk('text', { x, y: y + 3, 'text-anchor': 'middle', 'font-size': 10, fill: '#fff' }); t.textContent = n;
  g.appendChild(t); return g;
}
const updateCounts = () => {
  document.getElementById('counts').textContent =
    `GCP ${pairs.length} · ROI ${roi.length} · 차선 ${lanes.length}`;
  updateGuide();
};
['showZones','showGrid'].forEach((id) =>
  document.getElementById(id).addEventListener('change', () => { redraw(); }));
window.addEventListener('resize', () => cam && redraw());

document.getElementById('undo').onclick = () => {
  if (mode === 'gcp') { if (pending) pending = null; else if (pairs.length) { pairs.pop(); mapMarkers.pop()?.remove(); } }
  else if (mode === 'roi') roi.pop();
  else if (mode === 'lane') { if (laneTmp.length) laneTmp = []; else lanes.pop(); }
  redraw(); updateCounts();
};
document.getElementById('clear').onclick = () => {
  if (mode === 'gcp') { pairs = []; pending = null; mapMarkers.splice(0).forEach((m) => m.remove()); }
  else if (mode === 'roi') roi = [];
  else if (mode === 'lane') { lanes = []; laneTmp = []; }
  redraw(); updateCounts();
};

// 🤖 자동 캘리브레이션(부트스트랩): 서버에서 VP+IPM+OSM 실행 → 결과를 기존 캘리브레이션으로 로드
document.getElementById('autocalib').onclick = async () => {
  if (!cam) return;
  const btn = document.getElementById('autocalib');
  btn.disabled = true; setStatus('자동 캘리브레이션 실행 중… (VP·OSM, ~30초)');
  try {
    const zc = parseFloat(document.getElementById('zcam').value) || 9;
    const res = await fetch('/api/autocalib', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ loc: cam.cctv_id, camHeight: zc }) });
    const g = await res.json();
    if (g.error) { setStatus('자동 실패: ' + g.error); btn.disabled = false; return; }
    existing = g; Hcur = g.homography?.H || null;
    document.getElementById('result').textContent =
      `🤖 자동 부트스트랩 로드됨\n${(g.meta && g.meta.note) || ''}\n→ 이제 GCP 1~3점만 찍어 [계산 & 저장]하면 하이브리드 정밀보정됩니다.`;
    setStatus('자동 부트스트랩 적용 · GCP 소수점으로 보정하세요');
  } catch (e) { setStatus('자동 요청 실패: ' + e); }
  btn.disabled = false;
};

document.getElementById('save').onclick = async () => {
  try {
    await doSave();
  } catch (e) {                    // 예외로 버튼이 '죽은 것처럼' 보이지 않게 항상 표면화
    console.error(e);
    setStatus('저장 실패: ' + (e.message || e));
    document.getElementById('result').textContent = '오류: ' + (e.stack || e.message || e);
  }
};

async function doSave() {
  if (!cam) return;
  const W = cam.width, H = cam.height, fx = W;
  const zcam = parseFloat(document.getElementById('zcam').value) || 10;
  const gh = parseFloat(document.getElementById('gh').value) || 0;
  const g = existing && existing.homography ? JSON.parse(JSON.stringify(existing)) : {
    meta: { location_code: cam.cctv_id }, inputs: { cctv_path: `cctv_${cam.cctv_id}.png` },
    undistort: { resolution: [W, H], K: [[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], D: [0, 0, 0, 0, 0] },
    homography: { H: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] },
    parallax: { x_cam_coords_sat: 0, y_cam_coords_sat: 0, z_cam_meters: zcam, px_per_meter: 1 },
    world: {}, ref_method: 'center_bottom_side', proj_method: 'down_h',
  };
  g.parallax.z_cam_meters = zcam;
  let msg = [];

  // ① GCP → 호모그래피 + world
  if (pairs.length >= 4) {
    const utm = pairs.map((p) => proj4('EPSG:4326', `EPSG:${EPSG}`, p.ll));
    const oE = utm.reduce((s, u) => s + u[0], 0) / utm.length, oN = utm.reduce((s, u) => s + u[1], 0) / utm.length;
    const src = pairs.map((p) => p.px), dst = utm.map((u) => [u[0] - oE, u[1] - oN]);
    const Hn = computeHomography(src, dst);
    let rms = Math.sqrt(src.reduce((a, s, i) => { const q = applyH(Hn, s[0], s[1]); return a + (q[0] - dst[i][0]) ** 2 + (q[1] - dst[i][1]) ** 2; }, 0) / src.length);
    g.undistort = { resolution: [W, H], K: [[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], D: [0, 0, 0, 0, 0] };
    g.homography = { H: Hn, fov_polygon: [], anchors_list: pairs.map((p) => ({ px: p.px, lonlat: p.ll })) };
    g.world = { epsg: EPSG, origin_easting: oE, origin_northing: oN, ground_ellipsoid_h: gh };
    Hcur = Hn;
    msg.push(`GCP ${pairs.length}쌍(완전) · RMS ${rms.toFixed(2)}m`);
  } else if (pairs.length >= 1 && g.homography && g.homography.H && g.world && g.world.origin_easting) {
    // 하이브리드: 자동/기존 H(형상) + N점 저차원 보정(1=평행이동, 2↑=유사변환)
    const Hutm = mul3(translate3(g.world.origin_easting, g.world.origin_northing), g.homography.H); // px→UTM
    const srcU = pairs.map((p) => applyH(Hutm, p.px[0], p.px[1]));
    const dstU = pairs.map((p) => proj4('EPSG:4326', `EPSG:${EPSG}`, p.ll));
    const A = correction3(srcU, dstU);
    const oNew = [dstU.reduce((s, u) => s + u[0], 0) / dstU.length, dstU.reduce((s, u) => s + u[1], 0) / dstU.length];
    const Hfin = mul3(mul3(translate3(-oNew[0], -oNew[1]), A), Hutm); // px→local(rel oNew)
    let rms = Math.sqrt(dstU.reduce((a, d, i) => {
      const q = applyH(mul3(translate3(oNew[0], oNew[1]), Hfin), pairs[i].px[0], pairs[i].px[1]);
      return a + (q[0] - d[0]) ** 2 + (q[1] - d[1]) ** 2;
    }, 0) / dstU.length);
    g.homography = { H: Hfin, fov_polygon: [], anchors_list: pairs.map((p) => ({ px: p.px, lonlat: p.ll })) };
    g.world = { epsg: EPSG, origin_easting: oNew[0], origin_northing: oNew[1], ground_ellipsoid_h: gh };
    Hcur = Hfin;
    msg.push(`하이브리드 ${pairs.length}점 보정(${pairs.length === 1 ? '평행이동' : '유사변환'}) · GCP잔차 ${rms.toFixed(2)}m`);
  } else if (pairs.length > 0) {
    msg.push(`GCP ${pairs.length}쌍 — 자동/기존 H 없어 보정 불가(먼저 🤖자동 또는 4쌍↑)`);
  }

  // ② ROI (CCTV 픽셀 폴리곤)
  if (roi.length >= 3) { g.roi_polygon = roi; g.use_roi = true; g.roi_method = 'in'; msg.push(`ROI ${roi.length}점`); }

  // ③ 차선 방향 → world heading
  // lanes 는 두 형태가 섞일 수 있다:
  //   · 이번에 그린 것      → { px: [[u,v],[u,v]] }        (H로 실좌표 변환 필요)
  //   · 기존 파일에서 로드   → { sat: [[x,y],[x,y]], heading } (이미 실좌표 — 그대로 유지)
  if (lanes.length) {
    const out = [];
    let newCnt = 0, keepCnt = 0, skipped = 0;
    for (const L of lanes) {
      if (L.px && L.px.length === 2) {
        if (!Hcur) { skipped++; continue; }         // H 없으면 픽셀→실좌표 변환 불가
        const a = applyH(Hcur, L.px[0][0], L.px[0][1]);
        const b = applyH(Hcur, L.px[1][0], L.px[1][1]);
        const heading = ((Math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / Math.PI) + 360) % 360;
        out.push({ sat: [a, b], heading_deg: heading }); newCnt++;
      } else if (L.sat && L.sat.length === 2) {
        out.push({ sat: L.sat, heading_deg: L.heading ?? L.heading_deg ?? 0 }); keepCnt++;
      }
    }
    if (out.length) {
      g.heading_guidelines = out;
      g.use_svg = true;
      msg.push(`차선 ${out.length}개(신규 ${newCnt}·유지 ${keepCnt})`);
    }
    if (skipped) msg.push(`차선 ${skipped}개 건너뜀(먼저 GCP 캘리브레이션 필요)`);
  }

  const res = await fetch('/api/save-gproj', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ loc: cam.cctv_id, gproj: g, gcps: pairs }) });
  const j = await res.json();
  existing = g;
  document.getElementById('result').textContent =
    (j.ok ? `저장됨: ${j.path}\n` : `실패: ${j.error}\n`) + msg.join(' · ') +
    `\n→ python tools/reproject.py --loc ${cam.cctv_id} (또는 run_inference_sahi 재실행) 후 bridge_to_webmap`;
  setStatus('저장 완료 ✓');
};

// 디버그/검증 훅
window._calib = { get pairs() { return pairs; }, set pairs(v) { pairs = v; },
                  redraw, updateCounts, get Hcur() { return Hcur; } };
