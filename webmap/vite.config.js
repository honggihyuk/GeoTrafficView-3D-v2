import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';

// LLM 채팅: 질문 → llm/nl2sql.py(--json) → Ollama Qwen + (PostGIS | SQLite) → 답변
// GEOTRAFFIC_DB_URL 이 설정돼 있으면 PostGIS, 없으면 nl2sql.py 가 db/geotraffic.db(SQLite)로 폴백.
function llmAskPlugin() {
  const REPO = path.resolve(process.cwd(), '..');
  const DB_URL = process.env.GEOTRAFFIC_DB_URL || '';
  return {
    name: 'llm-ask',
    configureServer(server) {
      server.middlewares.use('/api/ask', (req, res, next) => {
        if (req.method !== 'POST') return next();
        let body = '';
        req.on('data', (c) => (body += c));
        req.on('end', () => {
          let question, model;
          try { ({ question, model } = JSON.parse(body)); } catch { question = ''; }
          if (!question) { res.statusCode = 400; return res.end(JSON.stringify({ error: 'no question' })); }
          const args = [path.join(REPO, 'llm', 'nl2sql.py'), '--json'];
          if (model) args.push('--model', model);
          args.push(String(question));
          const py = spawn('python', args, {
            cwd: REPO,
            env: { ...process.env, GEOTRAFFIC_DB_URL: DB_URL, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
          });
          let out = '', err = '';
          const timer = setTimeout(() => py.kill(), 150000);
          py.stdout.setEncoding('utf-8');
          py.stderr.setEncoding('utf-8');
          py.stdout.on('data', (d) => (out += d));
          py.stderr.on('data', (d) => (err += d));
          py.on('close', () => {
            clearTimeout(timer);
            res.setHeader('Content-Type', 'application/json');
            const line = out.trim().split('\n').filter(Boolean).pop() || '';
            try { res.end(JSON.stringify(JSON.parse(line))); }
            catch { res.statusCode = 500; res.end(JSON.stringify({ error: 'LLM 응답 파싱 실패', stderr: err.slice(-500), raw: out.slice(-500) })); }
          });
        });
      });
    },
  };
}

// 캘리브레이션 결과(G-Projection JSON + GCP 쌍)를 repo/location/<loc>/ 에 저장하는 미들웨어
function saveGprojPlugin() {
  return {
    name: 'save-gproj',
    configureServer(server) {
      // 기존 G_projection 읽기 (카메라 선택 시 현재 캘리브레이션/ROI 로드)
      server.middlewares.use('/api/gproj', (req, res, next) => {
        if (req.method !== 'GET') return next();
        try {
          const loc = new URL(req.url, 'http://x').searchParams.get('loc') || '';
          if (!/^[A-Za-z0-9_]+$/.test(loc)) throw new Error('bad loc');
          const p = path.resolve(process.cwd(), '..', 'location', loc, `G_projection_${loc}.json`);
          res.setHeader('Content-Type', 'application/json');
          res.end(fs.existsSync(p) ? fs.readFileSync(p, 'utf-8') : 'null');
        } catch (e) {
          res.statusCode = 400; res.end(JSON.stringify({ error: String(e) }));
        }
      });
      // 자동 캘리브레이션(VP+IPM+OSM) 실행 → 결과 G_projection 반환 (하이브리드 부트스트랩)
      server.middlewares.use('/api/autocalib', (req, res, next) => {
        if (req.method !== 'POST') return next();
        let body = '';
        req.on('data', (c) => (body += c));
        req.on('end', () => {
          let loc, camHeight;
          try { ({ loc, camHeight } = JSON.parse(body)); } catch { loc = ''; }
          if (!/^[A-Za-z0-9_]+$/.test(loc || '')) { res.statusCode = 400; return res.end(JSON.stringify({ error: 'bad loc' })); }
          const REPO = path.resolve(process.cwd(), '..');
          const args = [path.join(REPO, 'tools', 'auto_calibrate_vp.py'), '--loc', loc];
          if (camHeight) args.push('--cam-height', String(camHeight));
          const py = spawn('python', args, { cwd: REPO, env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' } });
          let err = '';
          const timer = setTimeout(() => py.kill(), 120000);
          py.stderr.on('data', (d) => (err += d));
          py.on('close', (code) => {
            clearTimeout(timer);
            res.setHeader('Content-Type', 'application/json');
            const p = path.join(REPO, 'location', loc, `G_projection_${loc}_auto.json`);
            if (fs.existsSync(p)) res.end(fs.readFileSync(p, 'utf-8'));
            else { res.statusCode = 500; res.end(JSON.stringify({ error: '자동 캘리브레이션 실패', code, stderr: err.slice(-400) })); }
          });
        });
      });
      server.middlewares.use('/api/save-gproj', (req, res, next) => {
        if (req.method !== 'POST') return next();
        let body = '';
        req.on('data', (c) => (body += c));
        req.on('end', () => {
          try {
            const { loc, gproj, gcps } = JSON.parse(body);
            if (!/^[A-Za-z0-9_]+$/.test(loc || '')) throw new Error('bad loc');
            const dir = path.resolve(process.cwd(), '..', 'location', loc);
            fs.mkdirSync(dir, { recursive: true });
            fs.writeFileSync(path.join(dir, `G_projection_${loc}.json`), JSON.stringify(gproj, null, 4), 'utf-8');
            if (gcps) fs.writeFileSync(path.join(dir, '_gcps.json'), JSON.stringify(gcps, null, 2), 'utf-8');
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ ok: true, path: `location/${loc}/G_projection_${loc}.json` }));
          } catch (e) {
            res.statusCode = 400;
            res.end(JSON.stringify({ ok: false, error: String(e) }));
          }
        });
      });
    },
  };
}

// GT 라벨 저장/조회: eval/<loc>/labels.json  (평가용 정답 데이터)
function labelsPlugin() {
  const REPO = () => path.resolve(process.cwd(), '..');
  const labelPath = (loc) => {
    if (!/^[A-Za-z0-9_]+$/.test(loc || '')) throw new Error('bad loc');
    return path.join(REPO(), 'eval', loc, 'labels.json');
  };
  return {
    name: 'gt-labels',
    configureServer(server) {
      server.middlewares.use('/api/labels', (req, res, next) => {
        res.setHeader('Content-Type', 'application/json');
        try {
          if (req.method === 'GET') {
            const loc = new URL(req.url, 'http://x').searchParams.get('loc');
            const p = labelPath(loc);
            return res.end(fs.existsSync(p) ? fs.readFileSync(p, 'utf-8') : 'null');
          }
          if (req.method === 'POST') {
            let body = '';
            req.on('data', (c) => (body += c));
            return req.on('end', () => {
              try {
                const data = JSON.parse(body);
                const p = labelPath(data.loc);
                fs.mkdirSync(path.dirname(p), { recursive: true });
                fs.writeFileSync(p, JSON.stringify(data, null, 1), 'utf-8');
                res.end(JSON.stringify({ ok: true, path: `eval/${data.loc}/labels.json`,
                                         boxes: (data.boxes || []).length }));
              } catch (e) { res.statusCode = 400; res.end(JSON.stringify({ ok: false, error: String(e) })); }
            });
          }
          return next();
        } catch (e) { res.statusCode = 400; res.end(JSON.stringify({ ok: false, error: String(e) })); }
      });
    },
  };
}

export default defineConfig({
  plugins: [saveGprojPlugin(), llmAskPlugin(), labelsPlugin()],
  server: { host: true, port: 5174, strictPort: false },
});
