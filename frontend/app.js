/* ===================== J-Space Explorer — frontend ===================== */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (tag, cls, txt) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (txt != null) n.textContent = txt;
  return n;
};

const state = {
  config: null,
  analysis: null,
  sel: { layer: null, pos: null },
  pinned: null,       // pinned token string for the trajectory chart
};

/* ---------- API ---------- */
async function apiGET(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}
async function apiPOST(path, body) {
  const r = await fetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST ${path} → ${r.status}`);
  return r.json();
}

/* ---------- color scale for the heatmap ---------- */
const RAMP = [
  [0.00, [16, 32, 63]],
  [0.35, [47, 95, 176]],
  [0.65, [79, 140, 255]],
  [1.00, [240, 180, 41]],
];
function heatColor(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [a0, c0] = RAMP[i - 1], [a1, c1] = RAMP[i];
      const f = (t - a0) / (a1 - a0 || 1);
      const c = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return `rgb(240,180,41)`;
}
const REGION_OF = (region) => region;

/* ---------- init ---------- */
async function init() {
  try {
    state.config = await apiGET('/api/config');
  } catch (e) {
    $('#backend-name').textContent = 'offline';
    return;
  }
  const c = state.config;
  $('#backend-name').textContent = c.display_name;
  $('#foot-backend').textContent = `${c.display_name} · ${c.n_layers} layers`;

  // example prompt chips
  const ep = $('#example-prompts');
  c.example_prompts.forEach((p) => {
    const chip = el('button', 'ex-chip', p);
    chip.onclick = () => { $('#prompt').value = p; analyze(); };
    ep.appendChild(chip);
  });

  // feature chips (explore)
  const fc = $('#feature-chips');
  c.example_features.forEach((f) => {
    const chip = el('button', 'ex-chip', f);
    chip.onclick = () => { $('#search').value = f; runSearch(f); openFeature(f); };
    fc.appendChild(chip);
  });

  // steering concept select
  const sc = $('#steer-concept');
  c.steering_concepts.forEach((s) => {
    const o = el('option', null, s.label); o.value = s.id; sc.appendChild(o);
  });

  wireEvents();
  $('#prompt').value = c.example_prompts[0];
  analyze();
}

function wireEvents() {
  $('#analyze').onclick = analyze;
  $('#prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter') analyze(); });

  $$('.tab').forEach((t) => t.onclick = () => switchView(t.dataset.view));

  let stimer;
  $('#search').addEventListener('input', (e) => {
    clearTimeout(stimer);
    const q = e.target.value.trim();
    stimer = setTimeout(() => runSearch(q), 180);
  });

  $('#steer-strength').addEventListener('input', (e) => {
    $('#steer-val').textContent = Number(e.target.value).toFixed(2);
    runSteer();
  });
  $('#steer-concept').addEventListener('change', runSteer);
  $('#steer-prompt').addEventListener('input', () => { clearTimeout(stimer); stimer = setTimeout(runSteer, 200); });
}

function switchView(view) {
  $$('.tab').forEach((t) => t.classList.toggle('active', t.dataset.view === view));
  $$('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${view}`));
  if (view === 'steer' && !$('#steer-prompt').value) {
    $('#steer-prompt').value = $('#prompt').value || (state.config?.example_prompts[0] ?? '');
    runSteer();
  }
}

/* ===================== SLICE VIEWER ===================== */
async function analyze() {
  const prompt = $('#prompt').value.trim();
  if (!prompt) return;
  $('#analyze').textContent = '…';
  try {
    state.analysis = await apiPOST('/api/analyze', { prompt });
    state.pinned = null;
    // default selection: peak workspace cell
    state.sel = peakCell(state.analysis);
    renderHeatmap();
    renderSlices();
  } catch (e) {
    $('#heatmap-empty').textContent = 'Analysis failed: ' + e.message;
  } finally {
    $('#analyze').textContent = 'Analyze ▸';
  }
}

function peakCell(a) {
  let best = { layer: 0, pos: 0, v: -1 };
  a.activity.forEach((row, L) => row.forEach((v, P) => {
    if (v > best.v) best = { layer: L, pos: P, v };
  }));
  return { layer: best.layer, pos: best.pos };
}

function renderHeatmap() {
  const a = state.analysis;
  const wrap = $('#heatmap-wrap');
  wrap.innerHTML = '';
  const nCols = a.tokens.length;

  const grid = el('div', 'heatmap');
  // columns: 1 label col + one per token
  grid.style.gridTemplateColumns = `70px repeat(${nCols}, minmax(26px, 1fr))`;

  // header row (token labels) — layers descend, so put labels on top
  grid.appendChild(el('div', 'hm-corner'));
  a.tokens.forEach((t, P) => {
    const lbl = el('div', 'hm-col-label', t);
    lbl.title = `position ${P}: ${t}`;
    grid.appendChild(lbl);
  });

  // rows: deepest layer at top (motor) → layer 0 at bottom (sensory)
  for (let L = a.n_layers - 1; L >= 0; L--) {
    const region = a.layer_regions[L];
    const rl = el('div', 'hm-row-label');
    rl.appendChild(el('span', `band ${region}`));
    rl.appendChild(el('span', null, `L${L}`));
    rl.title = `layer ${L} · ${region}`;
    grid.appendChild(rl);

    for (let P = 0; P < nCols; P++) {
      const v = a.activity[L][P];
      const cell = el('div', 'hm-cell');
      cell.style.background = heatColor(v);
      cell.dataset.layer = L; cell.dataset.pos = P;
      cell.title = `L${L} · “${a.tokens[P]}” · activity ${v.toFixed(3)} · ${region}`;
      if (L === state.sel.layer && P === state.sel.pos) cell.classList.add('selected');
      cell.onclick = () => { state.sel = { layer: L, pos: P }; renderHeatmap(); renderSlices(); };
      grid.appendChild(cell);
    }
  }
  wrap.appendChild(grid);

  $('#heatmap-legend').innerHTML =
    '<span>low</span><div class="bar"></div><span>high workspace activity</span>';
}

function tokBar(tok, w, maxW, onPin) {
  const row = el('div', 'tok-bar' + (tok === state.pinned ? ' pinned' : ''));
  row.appendChild(el('div', 'tok-name', tok || '∅'));
  const track = el('div', 'tok-track');
  const fill = el('div', 'tok-fill');
  fill.style.width = `${Math.max(3, (w / (maxW || 1)) * 100)}%`;
  track.appendChild(fill);
  row.appendChild(track);
  row.appendChild(el('div', 'tok-w', w.toFixed(3)));
  row.onclick = () => onPin(tok);
  return row;
}

function readoutRow(label, region, activity, readout, highlight) {
  const row = el('div', 'readout-row' + (highlight ? ' hl' : ''));
  const head = el('div', 'rr-head');
  const lbl = el('div', 'lbl');
  lbl.appendChild(el('span', null, label));
  if (region) lbl.appendChild(el('span', `region-tag ${region}`, region));
  head.appendChild(lbl);
  if (activity != null) head.appendChild(el('span', 'rr-act', `act ${activity.toFixed(2)}`));
  row.appendChild(head);

  const list = el('div', 'tok-list');
  const maxW = Math.max(...readout.weights, 0.0001);
  readout.tokens.forEach((t, i) => {
    list.appendChild(tokBar(t, readout.weights[i], maxW, (tok) => {
      state.pinned = state.pinned === tok ? null : tok;
      renderSlices();
    }));
  });
  row.appendChild(list);
  return row;
}

function renderSlices() {
  const a = state.analysis;
  if (!a) return;
  const { layer, pos } = state.sel;

  // ----- middle: all layers at selected position -----
  $('#mid-sub').textContent = `@ position ${pos} · “${a.tokens[pos]}”`;
  const mid = $('#layer-readouts');
  mid.innerHTML = '';
  for (let L = a.n_layers - 1; L >= 0; L--) {
    mid.appendChild(readoutRow(
      `L${L}`, a.layer_regions[L], a.activity[L][pos],
      a.readouts[L][pos], L === layer));
  }

  // ----- right: all positions at selected layer -----
  $('#right-sub').textContent = `@ layer ${layer} · ${a.layer_regions[layer]}`;
  renderTrajectory();
  const right = $('#pos-readouts');
  right.innerHTML = '';
  a.tokens.forEach((t, P) => {
    right.appendChild(readoutRow(
      `“${t}”`, a.layer_regions[layer], a.activity[layer][P],
      a.readouts[layer][P], P === pos));
  });
}

/* ---------- trajectory chart: pinned token's weight across layers ---------- */
function renderTrajectory() {
  const wrap = $('#traj-wrap');
  wrap.innerHTML = '';
  const card = el('div', 'traj-card');
  const a = state.analysis;
  const { pos } = state.sel;

  if (!state.pinned) {
    card.appendChild(el('div', 'traj-title', 'Trajectory'));
    card.appendChild(el('div', 'traj-empty',
      'Click any token in a readout to trace its weight across all layers at this position.'));
    wrap.appendChild(card);
    return;
  }

  // weight of pinned token per layer at current position
  const series = [];
  for (let L = 0; L < a.n_layers; L++) {
    const r = a.readouts[L][pos];
    const idx = r.tokens.indexOf(state.pinned);
    series.push(idx >= 0 ? r.weights[idx] : 0);
  }
  const title = el('div', 'traj-title');
  title.innerHTML = `Token <b>${escapeHtml(state.pinned)}</b> across layers @ “${escapeHtml(a.tokens[pos])}”`;
  card.appendChild(title);
  card.appendChild(sparkline(series, a.layer_regions));
  wrap.appendChild(card);
}

function sparkline(series, regions) {
  const W = 320, H = 120, padL = 24, padB = 16, padT = 8, padR = 6;
  const iw = W - padL - padR, ih = H - padT - padB;
  const maxV = Math.max(...series, 0.05);
  const n = series.length;
  const x = (i) => padL + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v) => padT + ih - (v / maxV) * ih;

  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'traj');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'none');

  // region background bands
  const colors = { sensory: 'rgba(79,140,255,.10)', workspace: 'rgba(240,180,41,.12)', motor: 'rgba(192,132,252,.10)' };
  for (let i = 0; i < n; i++) {
    const r = document.createElementNS(NS, 'rect');
    r.setAttribute('x', x(i) - (iw / n) / 2); r.setAttribute('y', padT);
    r.setAttribute('width', iw / n); r.setAttribute('height', ih);
    r.setAttribute('fill', colors[regions[i]] || 'transparent');
    svg.appendChild(r);
  }
  // baseline
  const base = document.createElementNS(NS, 'line');
  base.setAttribute('x1', padL); base.setAttribute('x2', W - padR);
  base.setAttribute('y1', padT + ih); base.setAttribute('y2', padT + ih);
  base.setAttribute('stroke', '#263043');
  svg.appendChild(base);

  // area + line
  const dLine = series.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const dArea = `${dLine} L${x(n - 1).toFixed(1)},${padT + ih} L${x(0).toFixed(1)},${padT + ih} Z`;
  const area = document.createElementNS(NS, 'path');
  area.setAttribute('d', dArea); area.setAttribute('fill', 'rgba(240,180,41,.15)');
  svg.appendChild(area);
  const line = document.createElementNS(NS, 'path');
  line.setAttribute('d', dLine); line.setAttribute('fill', 'none');
  line.setAttribute('stroke', '#f0b429'); line.setAttribute('stroke-width', '2');
  svg.appendChild(line);
  // points
  series.forEach((v, i) => {
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', x(i)); c.setAttribute('cy', y(v)); c.setAttribute('r', 2.5);
    c.setAttribute('fill', '#f0b429');
    const tt = document.createElementNS(NS, 'title');
    tt.textContent = `L${i}: ${v.toFixed(3)}`;
    c.appendChild(tt);
    svg.appendChild(c);
  });
  return svg;
}

/* ===================== EXPLORE ===================== */
async function runSearch(q) {
  const box = $('#search-results');
  if (!q) { box.innerHTML = '<div class="empty-note">Try a concept above, or click a feature chip.</div>'; return; }
  let data;
  try { data = await apiGET('/api/search?q=' + encodeURIComponent(q)); }
  catch (e) { box.innerHTML = `<div class="empty-note">search failed: ${e.message}</div>`; return; }
  box.innerHTML = '';
  if (!data.results.length) { box.innerHTML = '<div class="empty-note">No matches. Try “paris”, “eight”, “moon”, “happy”…</div>'; return; }
  data.results.forEach((r) => box.appendChild(resultCard(r)));
}

function resultCard(r) {
  const card = el('div', 'result');
  card.appendChild(el('div', 'r-type', r.type));
  card.appendChild(el('div', 'r-title', r.title));
  if (r.type === 'concept') {
    const rel = el('div', 'r-rel');
    (r.related || []).forEach((t) => rel.appendChild(el('span', 'pill', t)));
    card.appendChild(rel);
    card.onclick = () => openFeature(r.id);
  } else if (r.type === 'activation') {
    const rel = el('div', 'r-rel');
    (r.tokens || []).forEach((t, i) => rel.appendChild(
      el('span', 'pill tok' + (i === r.position ? ' hot' : ''), t)));
    card.appendChild(rel);
    const note = el('div', 'r-type', `peak layer L${r.peak_layer} · workspace`);
    card.appendChild(note);
  }
  return card;
}

async function openFeature(concept) {
  switchView('explore');
  const dash = $('#feature-dash');
  dash.innerHTML = '<div class="empty-note">loading…</div>';
  let f;
  try { f = await apiGET('/api/feature?c=' + encodeURIComponent(concept)); }
  catch (e) { dash.innerHTML = `<div class="empty-note">${e.message}</div>`; return; }
  $('#feat-sub').textContent = `“${f.label}” · ${f.region}`;
  dash.innerHTML = '';

  // meta row
  const meta = el('div', 'fd-meta');
  meta.appendChild(metaBox('peak layer', 'L' + f.peak_layer));
  meta.appendChild(metaBox('region', f.region));
  meta.appendChild(metaBox('density', f.density));
  meta.appendChild(metaBox('examples', f.examples.length));
  dash.appendChild(meta);

  // related concepts
  if (f.related.length) {
    const sec = el('div', 'fd-section');
    sec.appendChild(el('h3', null, 'Verbalizable direction — top tokens'));
    const rel = el('div', 'r-rel');
    f.related.forEach((r) => {
      const p = el('span', 'pill', `${r.token} · ${r.weight}`);
      rel.appendChild(p);
    });
    sec.appendChild(rel);
    dash.appendChild(sec);
  }

  // activation histogram
  const hsec = el('div', 'fd-section');
  hsec.appendChild(el('h3', null, 'Activation distribution'));
  const hist = el('div', 'hist');
  const hm = Math.max(...f.histogram);
  f.histogram.forEach((v) => {
    const b = el('div', 'b'); b.style.height = `${(v / hm) * 100}%`;
    b.title = v; hist.appendChild(b);
  });
  hsec.appendChild(hist);
  dash.appendChild(hsec);

  // top activating examples
  const esec = el('div', 'fd-section');
  esec.appendChild(el('h3', null, 'Top activating examples'));
  f.examples.forEach((ex) => {
    const line = el('div', 'fd-example');
    ex.tokens.forEach((t, i) => {
      const a = ex.acts[i];
      const span = el('span', 't', t + ' ');
      if (a > 0) {
        span.style.background = `rgba(240,180,41,${0.15 + a * 0.55})`;
        span.style.color = a > 0.8 ? '#201700' : '#ffe9b0';
      }
      line.appendChild(span);
    });
    esec.appendChild(line);
  });
  dash.appendChild(esec);
}

function metaBox(label, val) {
  const d = el('div');
  d.appendChild(el('span', null, label));
  d.appendChild(el('b', null, String(val)));
  return d;
}

/* ===================== STEERING ===================== */
async function runSteer() {
  const prompt = $('#steer-prompt').value.trim();
  const concept = $('#steer-concept').value;
  const strength = Number($('#steer-strength').value);
  const box = $('#steer-results');
  if (!prompt) { box.innerHTML = '<div class="empty-note">Enter a prompt.</div>'; return; }
  let d;
  try { d = await apiPOST('/api/steer', { prompt, concept, strength }); }
  catch (e) { box.innerHTML = `<div class="empty-note">${e.message}</div>`; return; }

  box.innerHTML = '';
  box.appendChild(steerCol('Baseline continuation', d.baseline, false));
  box.appendChild(steerCol(`Steered → ${d.label || concept}`, d.steered, true));

  const gen = el('div', 'gen-box');
  gen.innerHTML = `<span class="muted">generation @ strength ${strength.toFixed(2)}:</span><br>` +
    escapeHtml(d.generation || (prompt + ' …')).replace(
      escapeHtml((d.steered || []).slice(0, 3).join(' ')),
      `<span class="steered-txt">${escapeHtml((d.steered || []).slice(0, 3).join(' '))}</span>`);
  box.appendChild(gen);
}

function steerCol(title, tokens, steered) {
  const col = el('div', 'steer-col' + (steered ? ' steered' : ''));
  col.appendChild(el('h3', null, title));
  const rel = el('div', 'r-rel');
  (tokens || []).forEach((t, i) => {
    const p = el('span', 'pill' + (steered ? '' : ' tok'), `${i + 1}. ${t}`);
    rel.appendChild(p);
  });
  col.appendChild(rel);
  return col;
}

/* ---------- utils ---------- */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

init();
