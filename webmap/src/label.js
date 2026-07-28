// GT 라벨러: 프레임별 2D 박스 + 트랙 ID 라벨링 → eval/<loc>/labels.json
// 트랙 평가(MOTA/IDF1)를 위해 프레임 간 동일 track_id 유지가 핵심 → 'C'(이전 프레임 복사) 사용.
const CLASSES = ['car', 'truck', 'bus', 'van', 'motor', 'pedestrian', 'bicycle'];
const COLORS = ['#22d3ee', '#eab308', '#f59e0b', '#38bdf8', '#f472b6', '#84cc16', '#a78bfa'];

const q = new URLSearchParams(location.search);
const LOC = q.get('loc') || '';
const img = document.getElementById('img');
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const statusEl = document.getElementById('status');
const setStatus = (t) => (statusEl.textContent = t);

let man = null;          // manifest
let boxes = [];          // [{frame, track_id, class, bbox:[x1,y1,x2,y2]}]
let fi = 0;              // 현재 프레임 인덱스(manifest.frames 배열 인덱스)
let curClass = 0, curId = 1, selected = -1;
let undoStack = [];
let drag = null;         // {x0,y0,x1,y1}
let scale = 1, offX = 0, offY = 0;

init();
async function init() {
  if (!LOC) return setStatus('?loc=<LOC> 필요');
  man = await (await fetch(`eval/${LOC}/manifest.json`)).json();
  const saved = await (await fetch(`/api/labels?loc=${LOC}`)).json().catch(() => null);
  if (saved && saved.boxes) { boxes = saved.boxes; curId = Math.max(0, ...boxes.map(b => b.track_id)) + 1; }
  renderClasses();
  await showFrame(0);
  setStatus(`${LOC} · ${man.frames.length}프레임 로드${saved ? ` · 기존 라벨 ${boxes.length}개` : ''}`);
}

function renderClasses() {
  const c = document.getElementById('classes');
  c.innerHTML = '';
  CLASSES.forEach((name, i) => {
    const s = document.createElement('span');
    s.className = 'cls' + (i === curClass ? ' on' : '');
    s.style.borderColor = COLORS[i];
    s.textContent = `${i + 1} ${name}`;
    s.onclick = () => { curClass = i; renderClasses(); draw(); };
    c.appendChild(s);
  });
}

const curFrame = () => man.frames[fi].index;
const frameBoxes = () => boxes.filter(b => b.frame === curFrame());

async function showFrame(i) {
  fi = Math.max(0, Math.min(man.frames.length - 1, i));
  selected = -1;
  await new Promise((res) => {
    img.onload = res;
    img.src = `eval/${LOC}/frames/${man.frames[fi].file}`;
  });
  layout();
  document.getElementById('frameInfo').textContent =
    `프레임 ${curFrame()} (${fi + 1}/${man.frames.length})`;
  draw();
}

function layout() {
  const st = document.getElementById('stage');
  // 레이아웃이 0인 환경(숨김 패널·작은 iframe)에서도 좌표계가 깨지지 않도록 원본 크기로 폴백
  const sw = st.clientWidth || man.width, sh = st.clientHeight || man.height;
  scale = Math.min(sw / man.width, sh / man.height) || 1;
  const dw = man.width * scale, dh = man.height * scale;
  offX = (sw - dw) / 2; offY = (sh - dh) / 2;
  img.style.left = offX + 'px'; img.style.top = offY + 'px';
  img.width = dw; img.height = dh;
  cv.width = sw; cv.height = sh; cv.style.width = sw + 'px'; cv.style.height = sh + 'px';
}
const toImg = (e) => {
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left - offX) / scale, (e.clientY - r.top - offY) / scale];
};
const toScr = (x, y) => [offX + x * scale, offY + y * scale];

function draw() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  const fb = frameBoxes();
  fb.forEach((b, i) => {
    const [x1, y1] = toScr(b.bbox[0], b.bbox[1]);
    const [x2, y2] = toScr(b.bbox[2], b.bbox[3]);
    const col = COLORS[CLASSES.indexOf(b.class)] || '#fff';
    ctx.lineWidth = i === selected ? 3 : 2;
    ctx.strokeStyle = col;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = col;
    ctx.font = '11px sans-serif';
    const tag = `#${b.track_id} ${b.class}`;
    const w = ctx.measureText(tag).width + 6;
    ctx.fillRect(x1, y1 - 14, w, 14);
    ctx.fillStyle = '#04121a';
    ctx.fillText(tag, x1 + 3, y1 - 3);
  });
  if (drag) {
    ctx.strokeStyle = '#fff'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
    const [x1, y1] = toScr(Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1));
    const [x2, y2] = toScr(Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1));
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.setLineDash([]);
  }
  // 사이드 목록
  const list = document.getElementById('boxlist');
  list.innerHTML = '';
  fb.forEach((b, i) => {
    const d = document.createElement('div');
    d.className = i === selected ? 'sel' : '';
    d.textContent = `#${b.track_id} ${b.class} [${b.bbox.map(v => Math.round(v)).join(',')}]`;
    d.style.color = COLORS[CLASSES.indexOf(b.class)] || '#fff';
    d.onclick = () => { selected = i; draw(); };
    list.appendChild(d);
  });
  document.getElementById('cnt').textContent = fb.length;
  document.getElementById('curId').textContent = curId;
  const labeled = new Set(boxes.map(b => b.frame)).size;
  document.getElementById('progress').textContent =
    `라벨된 프레임 ${labeled}/${man.frames.length} · 총 박스 ${boxes.length} · 트랙 ${new Set(boxes.map(b => b.track_id)).size}`;
}

// ---- 마우스 ----
cv.addEventListener('mousedown', (e) => {
  const [x, y] = toImg(e);
  const fb = frameBoxes();
  const hit = fb.findIndex(b => x >= b.bbox[0] && x <= b.bbox[2] && y >= b.bbox[1] && y <= b.bbox[3]);
  if (hit >= 0 && e.shiftKey === false) { selected = hit; draw(); return; }
  drag = { x0: x, y0: y, x1: x, y1: y };
});
cv.addEventListener('mousemove', (e) => {
  if (!drag) return;
  const [x, y] = toImg(e); drag.x1 = x; drag.y1 = y; draw();
});
cv.addEventListener('mouseup', () => {
  if (!drag) return;
  const b = [Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1),
             Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)];
  drag = null;
  if (b[2] - b[0] < 4 || b[3] - b[1] < 4) return draw();
  push();
  boxes.push({ frame: curFrame(), track_id: curId, class: CLASSES[curClass],
               bbox: b.map(v => Math.round(v * 10) / 10) });
  curId += 1;
  draw();
});

// ---- 편집 ----
function push() { undoStack.push(JSON.stringify(boxes)); if (undoStack.length > 60) undoStack.shift(); }
function undo() { if (undoStack.length) { boxes = JSON.parse(undoStack.pop()); selected = -1; draw(); } }
function delSel() {
  const fb = frameBoxes();
  if (selected < 0 || selected >= fb.length) return;
  push();
  const t = fb[selected];
  boxes = boxes.filter(b => b !== t);
  selected = -1; draw();
}
/** 이전 프레임 박스를 그대로 복사(같은 track_id 유지) → 트랙 라벨링 속도의 핵심 */
function copyPrev() {
  if (fi === 0) return setStatus('첫 프레임입니다');
  const prev = man.frames[fi - 1].index, cur = curFrame();
  const src = boxes.filter(b => b.frame === prev);
  if (!src.length) return setStatus('이전 프레임에 박스가 없습니다');
  push();
  boxes = boxes.filter(b => b.frame !== cur);   // 현재 프레임 교체
  src.forEach(b => boxes.push({ ...b, frame: cur, bbox: [...b.bbox] }));
  setStatus(`이전 프레임 ${src.length}개 복사 — 위치만 조정하세요`);
  draw();
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (k === 'a') showFrame(fi - 1);
  else if (k === 'd') showFrame(fi + 1);
  else if (k === 'c') copyPrev();
  else if (k === 's') { e.preventDefault(); save(); }
  else if (k === 'z') undo();
  else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); delSel(); }
  else if (/^[1-9]$/.test(e.key)) {
    const i = +e.key - 1;
    if (i < CLASSES.length) {
      const fb = frameBoxes();
      if (selected >= 0 && selected < fb.length) { push(); fb[selected].class = CLASSES[i]; }
      curClass = i; renderClasses(); draw();
    }
  }
});

// 선택 박스 이동/크기조정(화살표 = 이동, Shift+화살표 = 크기)
document.addEventListener('keydown', (e) => {
  if (!e.key.startsWith('Arrow')) return;
  const fb = frameBoxes();
  if (selected < 0 || selected >= fb.length) return;
  e.preventDefault();
  const b = fb[selected].bbox, s = e.shiftKey ? 0 : 1, d = 2;
  const dx = e.key === 'ArrowLeft' ? -d : e.key === 'ArrowRight' ? d : 0;
  const dy = e.key === 'ArrowUp' ? -d : e.key === 'ArrowDown' ? d : 0;
  b[0] += dx * s; b[1] += dy * s; b[2] += dx * s + (1 - s) * dx; b[3] += dy * s + (1 - s) * dy;
  draw();
});

document.getElementById('prev').onclick = () => showFrame(fi - 1);
document.getElementById('next').onclick = () => showFrame(fi + 1);
document.getElementById('copy').onclick = copyPrev;
document.getElementById('save').onclick = save;
document.getElementById('idUp').onclick = () => { curId++; draw(); };
document.getElementById('idDown').onclick = () => { curId = Math.max(1, curId - 1); draw(); };
document.getElementById('idNew').onclick = () => { curId = Math.max(0, ...boxes.map(b => b.track_id), 0) + 1; draw(); };
window.addEventListener('resize', () => { layout(); draw(); });

async function save() {
  const payload = { loc: LOC, video: man.video, fps: man.fps, width: man.width, height: man.height,
                    frames: man.frames, classes: CLASSES, boxes };
  const r = await fetch('/api/labels', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                         body: JSON.stringify(payload) });
  const j = await r.json();
  setStatus(j.ok ? `저장됨: ${j.path} (박스 ${j.boxes}개)` : `저장 실패: ${j.error}`);
}
window._labeler = { get boxes() { return boxes; }, set boxes(v) { boxes = v; }, save, showFrame, draw };
