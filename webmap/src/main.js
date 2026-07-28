import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { openPlayer } from './player.js';
import { initChat } from './chat.js';

// 위성 래스터 타일 (토큰 불필요, attribution 필수)
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
const S2 = 'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg';

const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      esri: { type: 'raster', tiles: [ESRI], tileSize: 256,
              attribution: 'Esri, Maxar, Earthstar Geographics, CNES/Airbus DS' },
      s2: { type: 'raster', tiles: [S2], tileSize: 256,
            attribution: 'Sentinel-2 cloudless © EOX' },
    },
    layers: [
      { id: 'esri', type: 'raster', source: 'esri' },
      { id: 's2', type: 'raster', source: 's2', layout: { visibility: 'none' } },
    ],
  },
  center: [126.650, 37.4065], zoom: 15.5, pitch: 0, bearing: 0, maxPitch: 80,
});
map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'bottom-right');
window.map = map;
window.openPlayer = openPlayer; // 디버그/검증용
initChat(map, openPlayer); // LLM 채팅 패널 (+지도 제어)

map.on('load', async () => {
  // 3D 박스 소스(빈 상태) + fill-extrusion
  map.addSource('boxes', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addLayer({
    id: 'boxes', type: 'fill-extrusion', source: 'boxes',
    paint: {
      'fill-extrusion-color': ['get', 'color'],
      'fill-extrusion-height': ['get', 'height'],
      'fill-extrusion-base': ['get', 'base'],
      'fill-extrusion-opacity': 0.8,
    },
  });

  // ITS CCTV 마커
  const cams = await (await fetch('data/cameras.geojson')).json();
  map.addSource('cams', { type: 'geojson', data: cams });
  map.addLayer({
    id: 'cams', type: 'circle', source: 'cams',
    paint: {
      'circle-radius': ['case', ['get', 'has_replay'], 8, 5],
      'circle-color': ['case', ['get', 'has_replay'], '#22d3ee', '#3b82f6'],
      'circle-stroke-width': 2, 'circle-stroke-color': '#fff',
    },
  });
  map.addLayer({
    id: 'cam-labels', type: 'symbol', source: 'cams',
    filter: ['==', ['get', 'has_replay'], true],
    layout: { 'text-field': ['get', 'name'], 'text-size': 11, 'text-offset': [0, 1.4], 'text-anchor': 'top' },
    paint: { 'text-color': '#fff', 'text-halo-color': '#000', 'text-halo-width': 1.5 },
  });

  document.getElementById('camCount').textContent = `ITS CCTV ${cams.features.length}개`;

  map.on('click', 'cams', (e) => {
    const p = e.features[0].properties;
    if (p.has_replay) {
      openPlayer(map, p);
    } else {
      new maplibregl.Popup({ offset: 10 })
        .setLngLat(e.lngLat)
        .setHTML(`<b>${p.name || 'CCTV'}</b><br/>추론 미실행 — 라이브 재생은 ITS/UTIC 키·IP 필요`)
        .addTo(map);
    }
  });
  map.on('mouseenter', 'cams', () => (map.getCanvas().style.cursor = 'pointer'));
  map.on('mouseleave', 'cams', () => (map.getCanvas().style.cursor = ''));
});

// Basemap 토글
document.getElementById('bm-esri').onchange = (e) =>
  map.setLayoutProperty('esri', 'visibility', e.target.checked ? 'visible' : 'none');
document.getElementById('bm-s2').onchange = (e) =>
  map.setLayoutProperty('s2', 'visibility', e.target.checked ? 'visible' : 'none');
