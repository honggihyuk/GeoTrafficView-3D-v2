// LLM 채팅 패널: 질문 → 지도 명령(즉시) 또는 /api/ask (Ollama Qwen + PostGIS NL2SQL).
const CHIPS = [
  '송도IC로 이동',
  '가장 혼잡한 카메라는?',
  '카메라별 차량 대수',
  '버스 몇 대 지나갔어?',
];
// 지도 이동/표시 의도 (한/영). '가장' 등 오탐 방지 위해 '가' 단독은 제외.
const MOVE_RE = /(로\s*이동|이동해|이동|로\s*가|가줘|가자|보여|포커스|줌인|줌|확대|\b(go|move|show|fly|zoom|focus)\b)/i;

let _cams = null;
async function getCams() {
  if (!_cams) _cams = await (await fetch('data/cameras.geojson')).json();
  return _cams;
}

export function initChat(map, openPlayer) {
  const msgs = document.getElementById('chatMsgs');
  const input = document.getElementById('chatQ');
  const send = document.getElementById('chatSend');
  const chips = document.getElementById('chatChips');
  const hd = document.getElementById('chatHd');
  const panel = document.getElementById('chat');

  hd.addEventListener('click', () => {
    panel.classList.toggle('min');
    document.getElementById('chatMin').textContent = panel.classList.contains('min') ? '▸' : '▾';
  });

  CHIPS.forEach((q) => {
    const c = document.createElement('span');
    c.className = 'chip'; c.textContent = q;
    c.addEventListener('click', () => ask(q));
    chips.appendChild(c);
  });

  function bubble(cls, html) {
    const d = document.createElement('div');
    d.className = 'msg ' + cls; d.innerHTML = html;
    msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight;
    return d;
  }
  const esc = (s) => String(s).replace(/[&<>]/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[m]));

  function table(cols, rows) {
    if (!cols || !rows || !rows.length) return '';
    const head = '<tr>' + cols.map((c) => `<th>${esc(c)}</th>`).join('') + '</tr>';
    const body = rows.slice(0, 10).map((r) => '<tr>' + r.map((v) => `<td>${esc(v)}</td>`).join('') + '</tr>').join('');
    return `<table>${head}${body}</table>` + (rows.length > 10 ? `<div style="color:#9aa7b4">…${rows.length}행</div>` : '');
  }

  // 지도 명령 시도: 카메라 이름 + 이동 의도 → flyTo (+ 재생). LLM 건너뜀.
  async function tryMapCommand(q) {
    if (!map || !MOVE_RE.test(q)) return false;
    const cams = await getCams();
    let best = null;
    for (const f of cams.features) {
      const nm = f.properties.name || '';
      const short = nm.split(']').pop().trim();       // "[..] 송도IC" → "송도IC"
      if (short && q.replace(/\s/g, '').includes(short.replace(/\s/g, ''))) { best = f; break; }
    }
    if (!best) { bubble('a', '🗺️ 해당 이름의 카메라를 찾지 못했습니다.'); return true; }
    const [lon, lat] = best.geometry.coordinates;
    map.flyTo({ center: [lon, lat], zoom: 17.5, pitch: 60, bearing: 20, duration: 1500 });
    let extra = '';
    if (best.properties.has_replay && openPlayer) { openPlayer(map, best.properties); extra = ' · 영상·3D 박스 재생 시작'; }
    bubble('a', `🗺️ ${esc(best.properties.name)}(으)로 이동${extra}`);
    return true;
  }

  async function ask(q) {
    if (!q || send.disabled) return;
    input.value = '';
    bubble('u', esc(q));
    if (await tryMapCommand(q)) return;   // 지도 명령이면 LLM 호출 생략
    const wait = bubble('a', '🤔 생각 중… (Qwen이 SQL 생성·실행)');
    send.disabled = true;
    try {
      const res = await fetch('/api/ask', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      const j = await res.json();
      if (j.error) {
        wait.innerHTML = `<span class="err">오류: ${esc(j.error)}</span>` + (j.sql ? `<div class="sql">${esc(j.sql)}</div>` : '');
      } else {
        wait.innerHTML = esc(j.answer || '(응답 없음)') +
          (j.sql ? `<div class="sql">${esc(j.sql)}</div>` : '') +
          table(j.columns, j.rows);
      }
    } catch (e) {
      wait.innerHTML = `<span class="err">요청 실패: ${esc(e.message)}</span>`;
    } finally {
      send.disabled = false; msgs.scrollTop = msgs.scrollHeight;
    }
  }

  send.addEventListener('click', () => ask(input.value.trim()));
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') ask(input.value.trim()); });
  bubble('a', '교통 데이터를 묻거나(예: "버스 몇 대?"), 지도를 제어하세요(예: "송도IC로 이동"). 아래 예시 클릭 가능.');
}
