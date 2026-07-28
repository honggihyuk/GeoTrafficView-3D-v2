"""
HMM 맵매칭 (Newson & Krumm 2009) — 트랙 전체를 한 번에 풀어 차로 시퀀스를 확정한다.

**왜 프레임별 최근접이 아닌가**
  프레임마다 독립으로 가장 가까운 링크를 고르면 나란한 차로 사이를 왕복하며 깜빡인다
  (실측: 교차로에서 링크 위 점의 16%가 1m 이내 거리에서 방위 140° 이상 다른 링크에 붙었다).
  Viterbi는 **전이(도로를 따라 실제로 갈 수 있는가)** 를 함께 보므로 이 깜빡임이 사라지고,
  결과로 나온 링크 시퀀스가 곧 '이 차가 지나간 경로'가 된다.

모델:
  상태     : 관측 t에서의 후보 (link, s, d)
  방출확률 : 횡거리 d 의 가우시안         log p = -0.5 (d/σ)² - log(√(2π)σ)
  전이확률 : |직선거리 - 경로거리| 의 지수분포   log p = -|Δ|/β - log β
             + 연결성 게이트(도로를 따라 도달 불가면 -inf)

σ(sigma_lat) : 투영 오차의 횡방향 성분. CCTV 오블리크 투영은 종방향 σ가 훨씬 크므로
               횡방향만 방출로 보는 이 모델이 오히려 잘 맞는다.
β(beta)      : 경로거리와 직선거리의 허용 불일치. 커브·차로변경을 흡수한다.
"""
import math

import numpy as np

NEG_INF = -1e18


def viterbi_match(graph, obs, sigma_lat=2.0, beta=4.0, max_dist=25.0, k=6,
                  route_limit=300.0, dir_ref=None, dir_weight=2.0):
    """관측 시퀀스를 차로 시퀀스로 매칭.

    obs      : [(t_sec, np.array([x, y])), ...]  시간 오름차순
    dir_ref  : 관측별 진행방향 단위벡터 or None (있으면 방향 불일치를 방출확률에 반영)
    반환      : 관측과 같은 길이의 [(link_idx, s, d) or None]
    """
    n = len(obs)
    if n == 0:
        return []

    cands = [graph.project(p, max_dist=max_dist, k=k) for _, p in obs]
    if all(not c for c in cands):
        return [None] * n

    logsig = math.log(math.sqrt(2 * math.pi) * sigma_lat)
    logbeta = math.log(beta)

    def emit(i, c):
        li, s, d = c[0], c[1], c[2]
        e = -0.5 * (d / sigma_lat) ** 2 - logsig
        if dir_weight and dir_ref is not None and dir_ref[i] is not None:
            cos = float(dir_ref[i] @ graph.tangent(li, s))
            # 방향성 링크는 역방향까지 벌점, 양방향은 '축'만 맞으면 된다.
            pen = (1 - cos) / 2 if graph.links[li].directed else (1 - abs(cos)) / 2
            e -= dir_weight * pen
        return e

    # 첫 유효 관측 찾기
    start = next((i for i, c in enumerate(cands) if c), None)
    if start is None:
        return [None] * n

    V = [None] * n          # 관측별 [(logprob, prev_state_idx)]
    V[start] = [(emit(start, c), -1) for c in cands[start]]
    prev_i = start

    for i in range(start + 1, n):
        if not cands[i]:
            continue                                  # 후보 없는 관측은 건너뛰고 연결 유지
        dt_pos = float(np.linalg.norm(obs[i][1] - obs[prev_i][1]))
        cur = []
        for c in cands[i]:
            best, arg = NEG_INF, -1
            for j, (lp, _) in enumerate(V[prev_i]):
                if lp <= NEG_INF / 2:
                    continue
                pc = cands[prev_i][j]
                rd = graph.route_distance((pc[0], pc[1]), (c[0], c[1]), limit=route_limit)
                if not np.isfinite(rd):
                    continue
                tr = -abs(dt_pos - rd) / beta - logbeta
                v = lp + tr
                if v > best:
                    best, arg = v, j
            cur.append((best + emit(i, c) if arg >= 0 else NEG_INF, arg))
        if all(p <= NEG_INF / 2 for p, _ in cur):
            # 전이가 전부 끊겼다(가림 후 먼 재등장 등) → 여기서 트랙을 재시작한다.
            cur = [(emit(i, c), -1) for c in cands[i]]
        V[i] = cur
        prev_i = i

    # 역추적. 재시작(prev=-1)을 만나면 **거기서 멈추지 말고** 그 앞 구간을 독립 세그먼트로
    # 다시 역추적한다. 멈추면 가림 전 관측을 통째로 버리게 된다(실측: 관측의 20%가 사라졌다).
    out = [None] * n
    idxs = [i for i in range(n) if V[i] is not None]
    if not idxs:
        return out
    ptr = len(idxs) - 1
    j = int(np.argmax([p for p, _ in V[idxs[ptr]]]))
    while ptr >= 0:
        pos = idxs[ptr]
        if j < 0 or j >= len(cands[pos]):
            j = int(np.argmax([p for p, _ in V[pos]]))
        c = cands[pos][j]
        out[pos] = (c[0], c[1], c[2])
        prev_j = V[pos][j][1]
        ptr -= 1
        if ptr < 0:
            break
        j = prev_j if prev_j >= 0 else int(np.argmax([p for p, _ in V[idxs[ptr]]]))
    return out


def greedy_match(graph, obs, max_dist=25.0, k=6, dir_ref=None, dir_weight=2.0):
    """비교용 베이스라인 — 프레임별 독립 선택(현재 lane_snap 방식과 동일한 비용)."""
    out = []
    for i, (_, p) in enumerate(obs):
        cs = graph.project(p, max_dist=max_dist, k=k)
        if not cs:
            out.append(None)
            continue
        best, bc = None, None
        for c in cs:
            cost = abs(c[2])
            if dir_weight and dir_ref is not None and dir_ref[i] is not None:
                cos = float(dir_ref[i] @ graph.tangent(c[0], c[1]))
                pen = (1 - cos) / 2 if graph.links[c[0]].directed else (1 - abs(cos)) / 2
                cost += dir_weight * 4.0 * pen        # lane_snap 의 dir-weight 8m 와 같은 척도
            if bc is None or cost < bc:
                best, bc = c, cost
        out.append((best[0], best[1], best[2]))
    return out


def smooth_lateral(states, win=5):
    """횡오프셋 d 를 중앙값 필터로 다듬는다.

    d 는 검출 지터를 그대로 받는 성분이라 흔들린다. 차로변경은 유지하되(중앙값은
    계단 변화를 보존한다) 프레임 잡음만 죽인다.
    """
    idx = [i for i, s in enumerate(states) if s is not None]
    if len(idx) < 3:
        return states
    d = np.array([states[i][2] for i in idx], float)
    h = max(1, win // 2)
    sm = np.array([np.median(d[max(0, i - h):i + h + 1]) for i in range(len(d))])
    out = list(states)
    for k, i in enumerate(idx):
        out[i] = (states[i][0], states[i][1], float(sm[k]))
    return out


def isotonic(y):
    """PAVA — 단조 비감소 회귀. s 후진을 없애되 값을 한쪽으로 몰지 않는다.

    running max 로 자르면 s 가 잠깐 뒤로 갈 때 차가 그 자리에 '멈춰 섰다가' 따라잡는다.
    등장성 회귀는 위배 구간을 평균으로 눌러 자연스럽게 편다.
    """
    y = np.asarray(y, float).copy()
    n = len(y)
    if n < 2:
        return y
    val, cnt = [], []
    for v in y:
        val.append(v); cnt.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            v2, c2 = val.pop(), cnt.pop()
            v1, c1 = val.pop(), cnt.pop()
            val.append((v1 * c1 + v2 * c2) / (c1 + c2)); cnt.append(c1 + c2)
    out, k = np.empty(n), 0
    for v, c in zip(val, cnt):
        out[k:k + c] = v; k += c
    return out


def smooth_progress(S, win=9):
    """단조 진행량 S 를 **속도 평활 후 재적분**해 매끄럽게 만든다.

    등장성 회귀만 쓰면 위배 구간이 평평해져 차가 그 자리에 멈췄다가 튄다
    (실측 PANGYO_2: 프레임쌍의 33.9%가 Δs=0, 속도 변화 중앙 10.8 km/h/frame).
    속도를 이동평균한 뒤 다시 적분하면 단조성은 유지하면서 정지·급변이 사라진다.
    총 이동거리는 보존한다(끝점을 맞춰 스케일).
    """
    S = np.asarray(S, float)
    if len(S) < 3:
        return S
    v = np.diff(S)
    v = np.maximum(v, 0.0)
    k = max(3, int(win) | 1)
    pad = np.pad(v, k // 2, mode="edge")
    vs = np.convolve(pad, np.ones(k) / k, mode="valid")
    tot = v.sum()
    if vs.sum() > 1e-9 and tot > 1e-9:
        vs *= tot / vs.sum()                     # 총 이동거리 보존
    return np.concatenate([[S[0]], S[0] + np.cumsum(vs)])


def lane_lock(graph, states, route_limit=300.0, smooth_win=9):
    """트랙을 **하나의 연결된 차로 사슬**에 고정한다.

    맵매칭은 관측이 튀면 나란한 차로로 건너뛴다(실측 PANGYO_2: 링크 전환 29건이 전부
    도로를 따라 갈 수 없는 횡방향 점프였다). 화면에서는 차가 차선을 가로질러 순간이동하는
    것으로 보인다. 여기서는 Viterbi 결과 중 **도로를 따라 실제로 이어지는 가장 긴 구간**만
    남기고, 나머지 관측을 그 사슬 위로 다시 투영한다.

    반환: (states, chain) — chain 은 연결된 link_idx 리스트
    """
    idx = [i for i, s in enumerate(states) if s is not None]
    if len(idx) < 2:
        return states, []
    seq = []
    for i in idx:
        li = states[i][0]
        if not seq or seq[-1] != li:
            seq.append(li)
    # 연결된 최장 런 찾기
    runs, cur = [], [seq[0]]
    for a, b in zip(seq, seq[1:]):
        rd = graph.route_distance((a, graph.links[a].length), (b, 0.0), limit=route_limit)
        if np.isfinite(rd):
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    runs.append(cur)
    chain = max(runs, key=len)

    # 사슬 내 누적 오프셋(전역 S 좌표)
    off, acc = {}, 0.0
    for li in chain:
        off[li] = acc
        acc += graph.links[li].length

    out = list(states)
    S = []
    for i in idx:
        p = graph.to_xy(states[i][0], states[i][1], states[i][2])   # 원래 위치(월드)
        best = None
        for li in chain:
            s_, d_ = graph.project_onto(li, p)
            if best is None or abs(d_) < abs(best[2]):
                best = (li, s_, d_)
        out[i] = best
        S.append(off[best[0]] + best[1])

    # 사슬 위에서 s 를 단조화 → 속도 평활 재적분(정지·급변 제거)
    S = isotonic(S)
    if smooth_win > 1:
        S = smooth_progress(S, smooth_win)
    for k, i in enumerate(idx):
        s_glob = float(S[k])
        li = chain[0]
        for c in chain:                                   # 전역 S → (link, s)
            if s_glob <= off[c] + graph.links[c].length or c == chain[-1]:
                li = c
                break
        out[i] = (li, float(np.clip(s_glob - off[li], 0.0, graph.links[li].length)), out[i][2])
    return out, chain
