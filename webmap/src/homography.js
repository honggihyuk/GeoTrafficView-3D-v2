// 2D 호모그래피 DLT (의존성 없음). src(px) → dst(로컬 미터) 최소자승 추정, h33=1 고정.
// N≥4 대응점. 정규방정식 (AᵀA)h = Aᵀb 를 가우스 소거로 푼다.

export function computeHomography(src, dst) {
  if (src.length !== dst.length || src.length < 4) throw new Error('대응점 4쌍 이상 필요');
  const n = src.length;
  // 미지수 h = [h0..h7], H = [[h0,h1,h2],[h3,h4,h5],[h6,h7,1]]
  const ATA = Array.from({ length: 8 }, () => new Array(8).fill(0));
  const ATb = new Array(8).fill(0);
  const addRow = (row, rhs) => {
    for (let i = 0; i < 8; i++) {
      ATb[i] += row[i] * rhs;
      for (let j = 0; j < 8; j++) ATA[i][j] += row[i] * row[j];
    }
  };
  for (let k = 0; k < n; k++) {
    const [x, y] = src[k];
    const [X, Y] = dst[k];
    addRow([x, y, 1, 0, 0, 0, -x * X, -y * X], X);
    addRow([0, 0, 0, x, y, 1, -x * Y, -y * Y], Y);
  }
  const h = solve(ATA, ATb);
  return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}

// 가우스 소거(부분 피벗) — 8x8
function solve(Ain, bin) {
  const n = bin.length;
  const A = Ain.map((r, i) => [...r, bin[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    const d = A[c][c];
    if (Math.abs(d) < 1e-12) throw new Error('특이행렬 — 대응점이 일직선/중복일 수 있음');
    for (let j = c; j <= n; j++) A[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c];
      for (let j = c; j <= n; j++) A[r][j] -= f * A[c][j];
    }
  }
  return A.map((r) => r[n]);
}

// 검증용: H로 src를 투영
export function applyH(H, x, y) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w,
          (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
}

/**
 * Leave-One-Out 교차검증 RMS — **진짜 정확도 추정치**.
 * 4점 GCP는 자유도(8)와 동수라 적합 잔차가 항상 0이다(=검증 불가).
 * 한 점을 빼고 나머지로 H를 구해 그 점을 예측 → 예측오차 RMS(m). 5점 이상에서만 유효.
 */
export function looCvRms(src, dst) {
  if (src.length < 5) return null;
  let s = 0, n = 0;
  for (let k = 0; k < src.length; k++) {
    const s2 = src.filter((_, i) => i !== k);
    const d2 = dst.filter((_, i) => i !== k);
    let H;
    try { H = computeHomography(s2, d2); } catch { continue; }
    const q = applyH(H, src[k][0], src[k][1]);
    s += (q[0] - dst[k][0]) ** 2 + (q[1] - dst[k][1]) ** 2;
    n++;
  }
  return n ? Math.sqrt(s / n) : null;
}

/** GCP 분포 진단 — 세로(깊이) 커버리지가 좁으면 깊이 스케일이 틀어진다. */
export function gcpSpread(pxPoints, W, H) {
  if (!pxPoints.length) return { n: 0, vSpan: 0, hSpan: 0, redundancy: -4 };
  const ys = pxPoints.map((p) => p[1]), xs = pxPoints.map((p) => p[0]);
  return {
    n: pxPoints.length,
    vSpan: (Math.max(...ys) - Math.min(...ys)) / H,   // 세로 = 깊이 방향
    hSpan: (Math.max(...xs) - Math.min(...xs)) / W,
    redundancy: pxPoints.length - 4,                  // 0이면 잔차가 항상 0(검증 불가)
  };
}

// 3x3 행렬 곱
export function mul3(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}
export const translate3 = (dx, dy) => [[1, 0, dx], [0, 1, dy], [0, 0, 1]];

// N점 저차원 보정(하이브리드): 1점=평행이동, 2점↑=유사변환(회전+스케일+평행이동) 3x3
export function correction3(src, dst) {
  const n = src.length;
  if (n === 1) return translate3(dst[0][0] - src[0][0], dst[0][1] - src[0][1]);
  const ms = [0, 0], md = [0, 0];
  src.forEach((p, i) => { ms[0] += p[0]; ms[1] += p[1]; md[0] += dst[i][0]; md[1] += dst[i][1]; });
  ms[0] /= n; ms[1] /= n; md[0] /= n; md[1] /= n;
  let sxx = 0, sxy = 0, syx = 0, syy = 0, vars = 0;
  src.forEach((p, i) => {
    const ax = p[0] - ms[0], ay = p[1] - ms[1], bx = dst[i][0] - md[0], by = dst[i][1] - md[1];
    sxx += bx * ax; sxy += bx * ay; syx += by * ax; syy += by * ay; vars += ax * ax + ay * ay;
  });
  // 2x2 [[sxx,sxy],[syx,syy]] SVD 대신 최적 회전각 직접(유사변환 닫힌해)
  const a = sxx + syy, b = syx - sxy;      // s*cosθ, s*sinθ (부호주의)
  const theta = Math.atan2(b, a);
  const s = Math.hypot(a, b) / vars;
  const c = Math.cos(theta) * s, si = Math.sin(theta) * s;
  const R = [[c, -si], [si, c]];
  const t = [md[0] - (R[0][0] * ms[0] + R[0][1] * ms[1]), md[1] - (R[1][0] * ms[0] + R[1][1] * ms[1])];
  return [[R[0][0], R[0][1], t[0]], [R[1][0], R[1][1], t[1]], [0, 0, 1]];
}
