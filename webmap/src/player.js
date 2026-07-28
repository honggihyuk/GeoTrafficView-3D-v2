// 이중 패널 플레이어: 좌 = CCTV 영상 + bbox_3d(핑크), 우 = 지도 위 fill-extrusion 3D 박스.
// video.currentTime 을 마스터 클럭으로 두 패널을 동일 프레임으로 동기화.
import proj4 from 'proj4';

proj4.defs('EPSG:32652', '+proj=utm +zone=52 +datum=WGS84 +units=m +no_defs +type=crs');

const CLASS_H = { car: 1.55, bus: 3.5, truck: 3.1, van: 2.0, person: 1.7, pedestrian: 1.7,
                  people: 1.7, bicycle: 1.6, motorcycle: 1.6, motor: 1.6,
                  tricycle: 1.75, 'awning-tricycle': 1.9 };
const CLASS_C = { car: '#22d3ee', bus: '#f59e0b', truck: '#eab308', van: '#38bdf8',
                  person: '#84cc16', pedestrian: '#84cc16', people: '#a3e635',
                  bicycle: '#a78bfa', motorcycle: '#f472b6', motor: '#f472b6',
                  tricycle: '#fb923c', 'awning-tricycle': '#fdba74' };

let raf = null;

async function loadGz(url) {
  const res = await fetch(url);
  const buf = await res.arrayBuffer();
  const b = new Uint8Array(buf);
  if (b[0] === 0x1f && b[1] === 0x8b) {
    const s = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    return JSON.parse(await new Response(s).text());
  }
  return JSON.parse(new TextDecoder('utf-8').decode(buf));
}

export async function openPlayer(map, props) {
  const world = typeof props.world === 'string' ? JSON.parse(props.world) : props.world;
  const fps = Number(props.fps) || 30;
  const data = await loadGz(props.replay_url);
  const frames = data.frames;
  const toLL = (x, y) => proj4('EPSG:32652', 'EPSG:4326',
    [world.origin_easting + x, world.origin_northing + y]); // [lon,lat]

  const panel = document.getElementById('detect');
  const video = document.getElementById('vid');
  const canvas = document.getElementById('cv');
  const ctx = canvas.getContext('2d');
  document.getElementById('detTitle').textContent = props.cctv_id || props.name || 'CCTV';
  canvas.width = Number(props.width) || 720;
  canvas.height = Number(props.height) || 480;
  panel.style.display = 'block';

  video.src = props.clip_url;
  video.loop = true; video.muted = true;
  video.play().catch(() => {});

  const [lon, lat] = toLL(0, 0);
  map.flyTo({ center: [lon, lat], zoom: 18, pitch: 60, bearing: 20, duration: 1500 });

  function drawBox3d(pts) {
    const faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];
    ctx.fillStyle = 'rgba(236,72,153,0.22)';
    ctx.strokeStyle = 'rgba(236,72,153,0.95)';
    ctx.lineWidth = 2;
    for (const f of faces) {
      ctx.beginPath();
      ctx.moveTo(pts[f[0]][0], pts[f[0]][1]);
      for (let i = 1; i < f.length; i++) ctx.lineTo(pts[f[i]][0], pts[f[i]][1]);
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
  }

  function render() {
    const fi = Math.min(frames.length - 1, Math.floor((video.currentTime || 0) * fps));
    const objs = (frames[fi] && frames[fi].objects) || [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const feats = [];
    for (const o of objs) {
      if (o.bbox_3d && o.bbox_3d.length === 8) drawBox3d(o.bbox_3d);
      if (o.sat_floor_box) {
        const ring = o.sat_floor_box.map(([x, y]) => toLL(x, y));
        ring.push(ring[0]);
        const h = CLASS_H[o.class] || 1.6;
        feats.push({
          type: 'Feature', geometry: { type: 'Polygon', coordinates: [ring] },
          properties: { base: 0, height: h, color: CLASS_C[o.class] || '#22d3ee' },
        });
      }
    }
    const src = map.getSource('boxes');
    if (src) src.setData({ type: 'FeatureCollection', features: feats });
    raf = requestAnimationFrame(render);
  }
  if (raf) cancelAnimationFrame(raf);
  render();

  document.getElementById('detClose').onclick = () => {
    panel.style.display = 'none';
    video.pause();
    if (raf) cancelAnimationFrame(raf);
    const src = map.getSource('boxes');
    if (src) src.setData({ type: 'FeatureCollection', features: [] });
  };
}
