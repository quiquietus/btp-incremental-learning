// Session, API, WebSocket and the aggregation stepper used by both dashboards.
export const NODE_COLORS = { 1: '#2563eb', 2: '#0f766e', 3: '#9333ea', 4: '#c2410c', 5: '#ca8a04' };
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch {} },
};
export const session = {
  get token() { return store.get('btp_token'); },
  get user() { try { return JSON.parse(store.get('btp_user') || 'null'); } catch { return null; } },
  save(d) { store.set('btp_token', d.token); store.set('btp_user', JSON.stringify(d)); },
  clear() { store.set('btp_token', null); store.set('btp_user', null); },
};

export async function api(path, opts = {}, base = '') {
  const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
  if (session.token) headers.Authorization = 'Bearer ' + session.token;
  const r = await fetch(base + path, { ...opts, headers, body: opts.body && typeof opts.body !== 'string' ? JSON.stringify(opts.body) : opts.body });
  if (r.status === 401) { session.clear(); location.href = '/'; throw new Error('logged out'); }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

export const fmt = (x, d = 3) => (x == null || Number.isNaN(x)) ? '—' : Number(x).toFixed(d);
export const pct = (x, d = 1) => (x == null) ? '—' : (100 * x).toFixed(d) + '%';
export const sgn = (x, d = 4) => (x == null) ? '—' : (x >= 0 ? '+' : '') + Number(x).toFixed(d);
export const time = ts => new Date(ts * 1000).toLocaleTimeString();
export const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
export const nice = s => String(s).replace(/_/g, ' ');

export function connectWS(url, onEvent, onState) {
  let ws, closed = false, delay = 1000;
  const open = () => {
    ws = new WebSocket(url);
    ws.onopen = () => { delay = 1000; onState && onState(true); };
    ws.onmessage = m => { try { onEvent(JSON.parse(m.data)); } catch (e) { console.error(e); } };
    ws.onclose = () => { onState && onState(false); if (!closed) setTimeout(open, delay = Math.min(delay * 2, 15000)); };
  };
  open();
  return { close() { closed = true; ws && ws.close(); }, get ws() { return ws; } };
}
export function wsURL(httpBase, path) {
  const u = new URL(path, httpBase || location.href);
  u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
  return u.toString();
}

export function logLine(el, ts, html) {
  const d = document.createElement('div');
  d.innerHTML = `<span class="t">${time(ts)}</span>${html}`;
  el.prepend(d);
  while (el.children.length > 250) el.lastChild.remove();
}

// Human text for server events (used in both event logs)
export function describeServerEvent(e) {
  const n = k => `<b style="color:${NODE_COLORS[k] || 'inherit'}">client ${k}</b>`;
  switch (e.kind) {
    case 'round_opened': return `Round ${e.round} opened by ${esc(e.by)}; waiting for ${e.expected.map(n).join(', ')}`;
    case 'global_sent': return `Server → ${n(e.node)}: global model v${e.version} (${(e.bytes / 1e6).toFixed(2)} MB). out_b starts ${e.first_values.out_b.map(v => fmt(v, 4)).join(', ')}`;
    case 'update_received': return `${n(e.node)} → server: weights for round ${e.round}, n = ${e.n}, ‖d‖ = ${fmt(e.update_norm, 4)}, ${(e.bytes / 1e6).toFixed(2)} MB. out_b starts ${e.first_values.out_b.map(v => fmt(v, 4)).join(', ')}`;
    case 'agg_step': return `Aggregation step ${e.step}: ${esc(e.title)}`;
    case 'agg_progress': return `… ${esc(e.text)}`;
    case 'round_done': return `Round ${e.round} finished. Validation macro-F1 ${fmt(e.val_before)} → <b>${fmt(e.val_after)}</b>`;
    case 'round_timeout': return `Round ${e.round} timed out; aggregating the ${e.received.length} update(s) received`;
    case 'round_cancelled': return `Round ${e.round} cancelled: ${esc(e.reason)}`;
    case 'round_failed': return `<span class="pill-bad">Round failed: ${esc(e.error)}</span>`;
    case 'node_online': return `${n(e.node)} is online`;
    case 'settings_changed': return `Settings changed by ${esc(e.by)}: ${esc(e.settings.algorithm)}, η = ${e.settings.eta}, μ = ${e.settings.mu}`;
    case 'server_started': return `Server started (global model v${e.version}${e.placeholder ? ', placeholder model' : ''})`;
    default: return esc(e.kind);
  }
}

// live aggregation view
export class AggregationView {
  constructor(root, opts = {}) {
    this.root = root; this.me = opts.highlightNode || null;
    root.innerHTML = `
      <div class="row" style="justify-content:space-between"><div><b id="agg-title">No round running</b> <span class="quiet" id="agg-sub"></span></div>
        <span class="quiet" id="agg-prog"></span></div>
      <div class="steps" style="margin-top:8px">${[1, 2, 3, 4, 5].map(i => `<div class="step" data-s="${i}"><div class="n">STEP ${i}</div><div class="ttl">${['Updates received', 'Data weights', 'Weighted sum Δ', 'Server optimiser step', 'New global model'][i - 1]}</div><div class="body quiet">waiting</div></div>`).join('')}</div>
      <h3>One weight followed through the round (real numbers)</h3><div class="trace" id="agg-trace">Appears when a round reaches step 3.</div>
      <h3>Contribution of each client to the new global model</h3><div id="agg-contrib" class="quiet">Appears when a round finishes.</div>`;
    this.$ = s => root.querySelector(s);
  }
  stepEl(i) { return this.root.querySelector(`.step[data-s="${i}"]`); }
  reset(round) {
    for (let i = 1; i <= 5; i++) { const s = this.stepEl(i); s.className = 'step'; s.querySelector('.body').innerHTML = '<span class="quiet">waiting</span>'; }
    this.$('#agg-title').textContent = `Round ${round}`; this.$('#agg-prog').textContent = '';
  }
  set(i, html) {
    for (let j = 1; j <= 5; j++) { const s = this.stepEl(j); if (j < i) s.className = 'step done'; else if (j === i) s.className = 'step active'; }
    this.stepEl(i).querySelector('.body').innerHTML = html;
  }
  nodeName(k) { return `<b style="color:${NODE_COLORS[k]}">C${k}</b>${k === this.me ? ' (you)' : ''}`; }
  onEvent(e) {
    if (e.kind === 'round_opened') { this.reset(e.round); this.$('#agg-sub').textContent = `collecting updates from clients ${e.expected.join(', ')}`;
      this.set(1, `Waiting for ${e.expected.map(k => this.nodeName(k)).join(', ')}<div id="agg-recv"></div>`); }
    if (e.kind === 'update_received') {
      const box = this.root.querySelector('#agg-recv');
      if (box) box.insertAdjacentHTML('beforeend', `<div>${this.nodeName(e.node)} n=${e.n}, ‖d‖=${fmt(e.update_norm, 3)}, ${(e.bytes / 1e6).toFixed(2)} MB</div>`);
      this.$('#agg-sub').textContent = `received ${e.received.length} of ${e.expected.length}`;
    }
    if (e.kind === 'agg_progress') this.$('#agg-prog').textContent = e.text;
    if (e.kind === 'agg_step') this.step(e);
    if (e.kind === 'round_done') { this.$('#agg-sub').textContent = 'finished'; this.$('#agg-prog').textContent = ''; }
  }
  step(e) {
    const s = e.step;
    if (s === 1) { this.reset(e.round); this.set(1, e.clients.map(c => `<div>${this.nodeName(c.node)} n<sub>k</sub>=${c.n}<br><span class="mono">‖d<sub>k</sub>‖=${fmt(c.update_norm, 4)}</span></div>`).join('')); this.$('#agg-sub').textContent = 'aggregating'; }
    if (s === 2) this.set(2, `N = ${e.N}<br>` + e.clients.map(c => `<div>${this.nodeName(c.node)} p=${c.n}/${e.N}=<b>${fmt(c.p, 3)}</b><div class="bar"><i style="width:${100 * c.p}%;background:${NODE_COLORS[c.node]}"></i></div></div>`).join(''));
    if (s === 3) { this.set(3, `‖Δ‖ = <b>${fmt(e.delta_norm, 4)}</b><br>` + Object.entries(e.layer_delta_norms || {}).map(([k, v]) => `<div class="mono">${esc(k)}: ${fmt(v, 4)}</div>`).join('')); this.trace(e.trace, false); }
    if (s === 4) { const a = e.adam;
      this.set(4, (a ? `FedAdam η=${a.eta}, β₁=${a.beta1}, β₂=${a.beta2}, τ=${a.tau}<br>‖m‖=${fmt(a.m_norm, 4)}<br>` : 'FedAvg: w ← w + Δ<br>') + `‖step‖ = <b>${fmt(e.step_norm, 4)}</b><br>` +
        Object.entries(e.layer_step_mean_abs || {}).map(([k, v]) => `<div class="mono">${esc(k)}: mean |step| ${v.toExponential(2)}</div>`).join('') +
        (e.old_first_values && e.old_first_values.out_b ? `<div class="mono" style="margin-top:4px">out_b[0..3] old: ${e.old_first_values.out_b.map(v => fmt(v, 4)).join(', ')}<br>out_b[0..3] new: ${e.new_first_values.out_b.map(v => fmt(v, 4)).join(', ')}</div>` : ''));
      this.trace(e.trace, true); }
    if (s === 5) { const r = e.record;
      this.set(5, `Global v${r.round}<br>${r.val_before != null ? `validation macro-F1<br><b>${fmt(r.val_before)} → ${fmt(r.val_after)}</b>` : 'no server validation set'}`);
      this.stepEl(5).className = 'step done'; this.contrib(r); }
  }
  // Show a finished round (from /api/rounds/{n}) in the same stepper, without the live delays
  show(r) {
    this.reset(r.round); this.$('#agg-sub').textContent = r.simulated ? 'simulated on Kaggle (replay)' : 'finished (replay)';
    const T = r.trace || null;
    this.step({ step: 1, round: r.round, clients: r.clients });
    this.step({ step: 2, round: r.round, N: r.N || r.clients.reduce((a, c) => a + c.n, 0), clients: r.clients });
    this.step({ step: 3, delta_norm: r.delta_norm, layer_delta_norms: r.layer_delta_norms, trace: T });
    this.step({ step: 4, adam: r.adam, step_norm: r.step_norm, layer_step_mean_abs: r.layer_step_mean_abs,
      old_first_values: r.old_first_values, new_first_values: r.new_first_values, trace: T });
    this.step({ step: 5, record: r });
    if (!T) this.$('#agg-trace').textContent = 'This record has no single-weight trace (rounds simulated on Kaggle store only the summary numbers).';
  }
  trace(t, full) {
    if (!t) return;
    const L = [`<span class="h">Parameter ${esc(t.tensor)}[${t.index}]</span>   current global w_t = ${sgn(t.w_t, 6)}`];
    for (const c of t.clients) L.push(`<span style="color:${NODE_COLORS[c.node]}">client ${c.node}</span>: sent w_k = ${sgn(c.w_k, 6)}   d_k = w_k − w_t = ${sgn(c.d_k, 6)}   p_k = ${fmt(c.p_k, 3)}   c_k = p_k·d_k = ${sgn(c.c_k, 6)}`);
    L.push(`Δ = Σ c_k = ${t.clients.map(c => sgn(c.c_k, 6)).join(' ')} = <b>${sgn(t.delta, 6)}</b>`);
    if (full) {
      if (t.m_new != null) {
        L.push(`m = β₁·m_old + (1−β₁)·Δ = ${sgn(t.m_old, 6)} → <b>${sgn(t.m_new, 6)}</b>;   v = β₂·v_old + (1−β₂)·Δ² = ${t.v_old.toExponential(3)} → <b>${t.v_new.toExponential(3)}</b>`);
        L.push(`step = η·m / (√v + τ) = <b>${sgn(t.step, 6)}</b>`);
      } else L.push(`step = Δ = <b>${sgn(t.step, 6)}</b>`);
      L.push(`new global w_{t+1} = w_t + step = ${sgn(t.w_t, 6)} ${t.step >= 0 ? '+' : '−'} ${fmt(Math.abs(t.step), 6)} = <b>${sgn(t.w_new, 6)}</b>`);
    }
    this.$('#agg-trace').innerHTML = L.join('<br>');
  }
  contrib(r) {
    const cl = r.clients;
    const share = cl.map(c => `<span style="width:${Math.max(0, c.share) * 100}%;background:${NODE_COLORS[c.node]}">C${c.node} ${pct(c.share, 0)}</span>`).join('');
    const layers = Object.keys(cl[0].layer_share);
    const maxPhi = Math.max(1e-9, ...cl.map(c => Math.abs(c.shapley || 0)));
    this.$('#agg-contrib').innerHTML = `
      <div class="quiet" style="margin-bottom:6px">Share = ⟨c_k, Δ⟩ / ‖Δ‖² (sums to ${fmt(r.shares_sum, 4)}). Alignment = cos(d_k, Δ). Shapley = fair split of the validation macro-F1 gain (${r.shapley_sum != null ? 'sums to ' + sgn(r.shapley_sum, 4) : 'not computed'}).</div>
      <div class="stack">${share}</div>
      <table style="margin-top:8px"><tr><th>Client</th><th class="num">n_k</th><th class="num">p_k</th><th class="num">‖d_k‖</th><th class="num">Share</th><th class="num">Alignment</th><th class="num">Shapley</th><th class="num">Leave-one-out</th></tr>
      ${cl.map(c => `<tr class="${c.node === this.me ? 'hl' : ''}"><td>${this.nodeName(c.node)}</td><td class="num">${c.n}</td><td class="num">${fmt(c.p, 3)}</td><td class="num">${fmt(c.update_norm, 4)}</td><td class="num"><b>${pct(c.share)}</b></td><td class="num">${sgn(c.alignment, 3)}</td>
        <td class="num">${c.shapley != null ? `${sgn(c.shapley, 4)}<div class="bar"><i style="width:${50 * Math.abs(c.shapley) / maxPhi}%;margin-left:${c.shapley < 0 ? 50 - 50 * Math.abs(c.shapley) / maxPhi : 50}%;background:${c.shapley < 0 ? 'var(--red)' : NODE_COLORS[c.node]}"></i></div>` : '—'}</td>
        <td class="num">${c.loo != null ? sgn(c.loo, 4) : '—'}</td></tr>`).join('')}</table>
      <h3>Share per layer</h3>
      <table><tr><th>Layer</th>${cl.map(c => `<th class="num">C${c.node}</th>`).join('')}</tr>
      ${layers.map(L => `<tr><td>${esc(L)}</td>${cl.map(c => { const v = c.layer_share[L]; return `<td class="num" style="background:color-mix(in srgb, ${NODE_COLORS[c.node]} ${Math.round(Math.max(0, Math.min(1, v)) * 70)}%, transparent)">${pct(v, 0)}</td>`; }).join('')}</tr>`).join('')}</table>
      ${r.subset_scores ? `<h3>Validation macro-F1 of every subset (used for Shapley)</h3><div class="mono">${Object.entries(r.subset_scores).map(([k, v]) => `{${k}}: ${fmt(v)}`).join(' &nbsp; ')}</div>` : ''}`;
  }
}

// Stacked share history for several rounds
export function shareHistory(el, rounds, me) {
  if (!rounds.length) { el.innerHTML = '<span class="quiet">No rounds yet.</span>'; return; }
  el.innerHTML = `<table><tr><th>Round</th><th>Update share per client</th><th class="num">Val F1 after</th><th class="num">Shapley (C1, C2, C3)</th></tr>` +
    rounds.slice(-15).reverse().map(r => `<tr><td>${r.round}${r.simulated ? ' <span class="badge grey">sim</span>' : ''}</td><td style="min-width:180px"><div class="stack">${r.clients.map(c => `<span style="width:${Math.max(0, c.share) * 100}%;background:${NODE_COLORS[c.node]}">${pct(c.share, 0)}</span>`).join('')}</div></td>
      <td class="num">${fmt(r.val_after)}</td><td class="num">${r.clients.map(c => c.shapley != null ? sgn(c.shapley, 3) : '—').join(' ')}</td></tr>`).join('') + '</table>';
}

export async function doLogin(form, errEl, base = '', after) {
  errEl.textContent = '';
  const btn = form.querySelector('button'); btn.disabled = true; btn.textContent = 'Signing in… (a sleeping server can take a minute)';
  try {
    const d = await api('/api/auth/login', { method: 'POST', body: { username: form.username.value, password: form.password.value } }, base);
    session.save(d); after(d);
  } catch (e) { errEl.textContent = e.message; }
  btn.disabled = false; btn.textContent = 'Sign in';
}
