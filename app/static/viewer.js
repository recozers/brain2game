// Brain Twin viewer — plain ES module, no build step (three.js via the import map in index.html).
// See CONTRACTS.md "Viewer".  Query params: ?mock=1 (no network, synthetic data), ?modal=<base url>, ?sid=<id>.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadHiresAssets, createHiresBrain } from './brain.js';

const qs = new URLSearchParams(location.search);
const MOCK = ['1', 'true', 'yes'].includes((qs.get('mock') || '').toLowerCase());
const N = 20484;                            // fsaverage5 vertices (0..10241 left, 10242..20483 right)
let THRESH_LO = 0.25, THRESH_HI = 1.0;
const ADAPTIVE = true, THRESH_LO_FLOOR = 0.12, THRESH_MIN_SPAN = 0.25;  // per-frame percentile thresholds (p85 / p99.5)  // TRIBE v2 average-subject scale: active ~0.3-0.7, max ~1.5     // below lo -> base shading; lo..hi -> warm ramp
const FADE_S = 1.0;                         // crossfade between consecutive seconds
const POLL_MS = 1000;                       // /preds poll period
const GAP_SKIP_MS = 10000;                  // skip forward if a second never arrives
const MAX_AHEAD = 20, KEEP_AHEAD = 10, MIN_START_BUFFER = 4, HARD_JUMP = 60, GAP_FETCH_MS = 30000;      // catch-up if the buffer runs away (bursty backend)
const BAR_MIN = -0.2, BAR_MAX = 0.8;
const TRAIN_AFTER_S = 20;                    // 'Train my brain' appears after this many inferred seconds
const ASSET_BASE = new URL('assets/', import.meta.url);
const FALLBACK_SYSTEMS = ['early_visual', 'motion', 'faces', 'places', 'objects', 'attention', 'auditory',
  'somatomotor', 'frontal_control', 'language', 'default_mode'];

const $ = (id) => document.getElementById(id);
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (x, d = 1) => (x === null || x === undefined || Number.isNaN(Number(x))) ? '—' : Number(x).toFixed(d);
const randomSid = () => { const abc = 'abcdefghjkmnpqrstuvwxyz23456789'; let s = ''; for (let i = 0; i < 6; i++) s += abc[(Math.random() * abc.length) | 0]; return s; };
const withTimeout = (ms) => (typeof AbortSignal.timeout === 'function' ? AbortSignal.timeout(ms) : undefined);

// ------------------------------------------------------------------------------------------------
// state
// ------------------------------------------------------------------------------------------------
const state = {
  cfg: null, assets: null, atlas: null, brain: null,
  buffer: new Map(),          // second -> { vec: Float32Array(N), systems: {id: z} | null, top: [{name, z}] | null }
  nextSecond: 0,              // since= for the next poll (= last received + 1)
  playSecond: null,           // next second to display
  shownSecond: null, shownSystems: null, shownTop: null,
  waitingSince: null,
  secondsReceived: 0, lastInferS: null, secondsPredicted: null, busy: null,
  history: new Map(),          // second -> {system id: z} for every second we have seen (summary card)
  received: new Set(), maxReceived: -1, gapSince: null,   // fetch-cursor bookkeeping (see ingest)
  baseline: null,              // samples/baseline.json from the laptop server, if any
  pollErrors: 0, lastPollOk: null,
  phase: 'booting', phaseClass: '',
  finishing: false,
};

// ------------------------------------------------------------------------------------------------
// float16 -> float32 (no native Float16Array everywhere); lookup table built once (256 kB)
// ------------------------------------------------------------------------------------------------
const F16 = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    t[h] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return t;
})();
function decodeF16Base64(b64) {
  const bin = atob(b64);
  const out = new Float32Array(bin.length >> 1);
  for (let i = 0, j = 0; i < out.length; i++, j += 2) out[i] = F16[bin.charCodeAt(j) | (bin.charCodeAt(j + 1) << 8)]; // little-endian
  return out;
}

// ------------------------------------------------------------------------------------------------
// colours (three.js vertex colours are linear; the renderer converts to sRGB on output)
// ------------------------------------------------------------------------------------------------
const srgbToLinear = (c) => c.map((v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)));
const C_GYRUS = srgbToLinear([0.74, 0.74, 0.77]);
const C_SULCUS = srgbToLinear([0.33, 0.33, 0.36]);
const C_ORANGE = srgbToLinear([0.91, 0.31, 0.10]);   // deep orange
const C_YELLOW = srgbToLinear([1.00, 0.82, 0.25]);
const C_WHITE = [1, 1, 1];
const tmp3 = [0, 0, 0];

function rampInto(z, out) {  // z >= THRESH_LO
  const t = clamp((z - THRESH_LO) / (THRESH_HI - THRESH_LO), 0, 1);
  let a, b, u;
  if (t < 0.55) { a = C_ORANGE; b = C_YELLOW; u = t / 0.55; } else { a = C_YELLOW; b = C_WHITE; u = (t - 0.55) / 0.45; }
  out[0] = a[0] + (b[0] - a[0]) * u; out[1] = a[1] + (b[1] - a[1]) * u; out[2] = a[2] + (b[2] - a[2]) * u;
}
const _sortBuf = new Float32Array(N);
function adaptThresholds(vec) {  // colour the top ~15% of vertices this second, whatever the absolute scale
  _sortBuf.set(vec); _sortBuf.sort();
  const p85 = _sortBuf[Math.floor(0.85 * (N - 1))], p995 = _sortBuf[Math.floor(0.995 * (N - 1))];
  THRESH_LO = Math.max(THRESH_LO_FLOOR, p85);
  THRESH_HI = Math.max(THRESH_LO + THRESH_MIN_SPAN, p995);
  const lo = document.getElementById('tick-lo'), mid = document.getElementById('tick-mid'), hi = document.getElementById('tick-hi');
  if (lo) { lo.textContent = `<${THRESH_LO.toFixed(2)} base`; mid.textContent = ((THRESH_LO + THRESH_HI) / 2).toFixed(2); hi.textContent = THRESH_HI.toFixed(2); }
}
function baseColors(sulc) {  // dark grey in sulci (sulc > 0), light grey on gyri
  const out = new Float32Array(N * 3);
  for (let i = 0, o = 0; i < N; i++, o += 3) {
    const t = smoothstep(-0.35, 0.45, sulc[i]);
    out[o] = C_GYRUS[0] + (C_SULCUS[0] - C_GYRUS[0]) * t;
    out[o + 1] = C_GYRUS[1] + (C_SULCUS[1] - C_GYRUS[1]) * t;
    out[o + 2] = C_GYRUS[2] + (C_SULCUS[2] - C_GYRUS[2]) * t;
  }
  return out;
}
function activationColors(vec, base, out, glow) {
  for (let i = 0, o = 0; i < N; i++, o += 3) {
    const z = vec[i];
    if (!(z >= THRESH_LO)) { out[o] = base[o]; out[o + 1] = base[o + 1]; out[o + 2] = base[o + 2]; glow[i] = 0; continue; }
    rampInto(z, tmp3);
    const w = smoothstep(THRESH_LO, THRESH_LO + 0.3, z);   // soft edge just above threshold
    out[o] = base[o] + (tmp3[0] - base[o]) * w;
    out[o + 1] = base[o + 1] + (tmp3[1] - base[o + 1]) * w;
    out[o + 2] = base[o + 2] + (tmp3[2] - base[o + 2]) * w;
    glow[i] = w * (0.35 + 0.65 * clamp((z - THRESH_LO) / (THRESH_HI - THRESH_LO), 0, 1));
  }
}
const BAR_STOPS = [[255, 176, 112], [230, 57, 70], [122, 12, 46]];   // light theme: peach -> red -> deep rose
function barColor(z) {
  if (!(z >= THRESH_LO)) return '#cfd2d8';
  const t = clamp((z - THRESH_LO) / (THRESH_HI - THRESH_LO), 0, 1);
  const seg = t < 0.5 ? [BAR_STOPS[0], BAR_STOPS[1], t / 0.5] : [BAR_STOPS[1], BAR_STOPS[2], (t - 0.5) / 0.5];
  const c = seg[0].map((a, i) => Math.round(a + (seg[1][i] - a) * seg[2]));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ------------------------------------------------------------------------------------------------
// config + assets
// ------------------------------------------------------------------------------------------------
async function loadConfig() {
  const q = { modal: (qs.get('modal') || '').replace(/\/+$/, ''), sid: qs.get('sid') || '' };
  if (MOCK) return { modal_base_url: q.modal, sid: q.sid || randomSid(), source: 'mock' };
  try {
    const u = new URL('/config', location.origin);
    if (q.sid) u.searchParams.set('sid', q.sid);
    const r = await fetch(u, { signal: withTimeout(4000) });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const j = await r.json();
    return { modal_base_url: String(j.modal_base_url || q.modal || '').replace(/\/+$/, ''), sid: String(j.sid || q.sid || randomSid()), source: 'server' };
  } catch (e) {
    console.warn('[config] GET /config failed (' + e.message + '); using query params ?modal= & ?sid=');
    return { modal_base_url: q.modal, sid: q.sid || randomSid(), source: 'query' };
  }
}

async function fetchBin(name) {
  const r = await fetch(new URL(name, ASSET_BASE));
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.arrayBuffer();
}
async function loadAssets() {
  const r = await fetch(new URL('mesh.json', ASSET_BASE));
  if (!r.ok) throw new Error('mesh.json: HTTP ' + r.status);
  const mesh = await r.json();
  const [pos, fac, sul, atlas] = await Promise.all([
    fetchBin(mesh.files.positions), fetchBin(mesh.files.faces), fetchBin(mesh.files.sulc),
    fetch(new URL('atlas.json', ASSET_BASE)).then((x) => (x.ok ? x.json() : null)).catch(() => null),
  ]);
  const positions = new Float32Array(pos), faces = new Uint32Array(fac), sulc = new Float32Array(sul);
  if (positions.length !== N * 3) console.warn(`positions.f32 has ${positions.length / 3} vertices, expected ${N}`);
  if (!atlas) console.warn('[assets] atlas.json missing — bar labels fall back to system ids');
  return { mesh, positions, faces, sulc, atlas };
}

// ------------------------------------------------------------------------------------------------
// three.js brain
// ------------------------------------------------------------------------------------------------
function createBrain(assets) {
  const stage = $('stage');
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x0b0d12, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 1, 4000);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(assets.positions, 3));
  geo.setIndex(new THREE.BufferAttribute(assets.faces, 1));
  const colorAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3);
  colorAttr.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute('color', colorAttr);
  const glowAttr = new THREE.BufferAttribute(new Float32Array(N), 1);   // emissive weight per vertex
  glowAttr.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute('glow', glowAttr);
  geo.computeVertexNormals();
  geo.computeBoundingSphere();

  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, metalness: 0.0 });
  // active vertices also emit their own colour, so blobs stay bright on the shadowed side of the brain
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float glow;\nvarying float vGlow;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvGlow = glow;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vGlow;')
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += vColor.rgb * vGlow * 0.55;');
  };
  const mesh = new THREE.Mesh(geo, material);
  mesh.rotation.x = -Math.PI / 2;   // anatomical z (superior) -> three.js y (up); anterior -> -z
  scene.add(mesh);

  scene.add(new THREE.HemisphereLight(0xe9eeff, 0x2a2521, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(-0.6, 0.9, 1.0);
  camera.add(key); scene.add(camera);   // key light rides with the camera: whatever side faces the audience is lit
  const fill = new THREE.DirectionalLight(0xbfd0ff, 0.8); fill.position.set(1.0, -0.2, 1.0); scene.add(fill);
  const rim = new THREE.DirectionalLight(0xffe2c8, 0.6); rim.position.set(0.3, 0.5, 1.3); scene.add(rim);

  const radius = geo.boundingSphere.radius;
  const viewDir = new THREE.Vector3(-0.9, -0.95, 0.55).normalize();   // from the left, behind, above: visual cortex faces the audience
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.autoRotate = true; controls.autoRotateSpeed = 0.55;   // slow, until the user drags
  controls.minDistance = radius * 0.6; controls.maxDistance = radius * 6;
  let idleTimer = null;
  controls.addEventListener('start', () => { controls.autoRotate = false; clearTimeout(idleTimer); });
  controls.addEventListener('end', () => { clearTimeout(idleTimer); idleTimer = setTimeout(() => { controls.autoRotate = true; }, 45000); });

  function resize(initial) {
    const w = Math.max(1, stage.clientWidth), h = Math.max(1, stage.clientHeight);
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (initial) {
      let d = (radius / Math.sin((camera.fov * Math.PI) / 360)) * 0.8;
      if (camera.aspect < 1) d /= camera.aspect;
      camera.position.copy(viewDir).multiplyScalar(d);
      controls.target.set(0, 0, 0);
      controls.update();
    }
  }
  resize(true);
  new ResizeObserver(() => resize(false)).observe(stage);

  // per-vertex colours: base shading now, crossfaded activation later
  const base = baseColors(assets.sulc);
  colorAttr.array.set(base);
  colorAttr.needsUpdate = true;
  const fade = {
    from: new Float32Array(N * 3), to: new Float32Array(N * 3), gFrom: new Float32Array(N), gTo: new Float32Array(N), t: 1,
    setTarget(target, glow) { this.from.set(colorAttr.array); this.to.set(target); this.gFrom.set(glowAttr.array); this.gTo.set(glow); this.t = 0; },
    tick(dt) {
      if (this.t >= 1) return;
      this.t = Math.min(1, this.t + dt / FADE_S);
      const a = this.t, arr = colorAttr.array, f = this.from, to = this.to;
      for (let i = 0; i < arr.length; i++) arr[i] = f[i] + (to[i] - f[i]) * a;
      const g = glowAttr.array, gf = this.gFrom, gt = this.gTo;
      for (let i = 0; i < g.length; i++) g[i] = gf[i] + (gt[i] - gf[i]) * a;
      colorAttr.needsUpdate = true; glowAttr.needsUpdate = true;
    },
  };
  const target = new Float32Array(N * 3), targetGlow = new Float32Array(N);
  function showVector(vec) { if (ADAPTIVE) adaptThresholds(vec); activationColors(vec, base, target, targetGlow); fade.setTarget(target, targetGlow); }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    controls.update();
    fade.tick(dt);
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  return { renderer, scene, camera, controls, mesh, showVector, resize };
}

// ------------------------------------------------------------------------------------------------
// data: ingest /preds responses, buffer by second, play at 1 Hz
// ------------------------------------------------------------------------------------------------
function ingest(resp) {
  const secs = Array.isArray(resp.seconds) ? resp.seconds : [];
  if (secs.length) {
    const vals = resp._vectors || (resp.data_b64 ? decodeF16Base64(resp.data_b64) : null);
    const nv = resp.n_vertices || N;
    const sysBySec = new Map((resp.systems || []).map((s) => [s.second, s.z || null]));
    const topBySec = new Map((resp.regions || []).map((s) => [s.second, s.top || null]));
    secs.forEach((sec, k) => {
      const sz = sysBySec.get(sec); if (sz && !state.history.has(sec)) state.history.set(sec, sz);
      if (state.shownSecond !== null && sec <= state.shownSecond) return;   // already played (first write wins)
      let vec = vals ? vals.subarray(k * nv, (k + 1) * nv) : null;
      if (vec && vec.length !== N) { const v2 = new Float32Array(N); v2.set(vec.subarray(0, Math.min(N, vec.length))); vec = v2; }
      state.buffer.set(sec, { vec, systems: sysBySec.get(sec) || null, top: topBySec.get(sec) || null });
    });
    for (const sec of secs) state.received.add(sec);
    state.maxReceived = Math.max(state.maxReceived, ...secs);
  }
  // fetch cursor = first second we do not have yet: with parallel workers an earlier window can land after a
  // later one, so never jump the cursor past a hole; give a hole GAP_FETCH_MS, then abandon it
  while (state.received.has(state.nextSecond)) state.nextSecond++;
  if (state.maxReceived >= state.nextSecond) {
    if (state.gapSince === null) state.gapSince = performance.now();
    else if (performance.now() - state.gapSince > GAP_FETCH_MS) {
      console.info(`[fetch] second ${state.nextSecond} never arrived, moving on`);
      state.nextSecond++; state.gapSince = null;
      while (state.received.has(state.nextSecond)) state.nextSecond++;
    }
  } else state.gapSince = null;
  if (typeof resp.seconds_received === 'number') state.secondsReceived = resp.seconds_received;
  const st = resp.status || {};
  if (typeof st.seconds_received === 'number') state.secondsReceived = st.seconds_received;
  if (st.last_infer_s !== undefined) state.lastInferS = st.last_infer_s;
  if (st.seconds_predicted !== undefined) state.secondsPredicted = st.seconds_predicted;
  if (st.busy !== undefined) state.busy = st.busy;
}

function playbackPeriodMs() {
  // target buffer ~ one inference tick (+ margin); below it slow down (no stalls), above it catch up gently
  const tick = state.lastInferS ? clamp(state.lastInferS + 3, 6, 25) : 10;
  const buffered = state.buffer.size;
  if (state.playSecond === null) return 1000;
  // predictions arrive in bursts of ~one tick; we only need the buffer to stay above a few seconds at the
  // trough right before a burst, so slow down only when it is nearly empty and speed up when it is fat
  if (buffered < 3) return 1650;               // 0.6x realtime
  if (buffered < 6) return 1250;               // 0.8x
  if (buffered > tick + 6) return 800;         // 1.25x: too far behind, catch up
  return 1000;
}
function playbackTick() {
  const buf = state.buffer;
  if (buf.size === 0) return;
  const keys = [...buf.keys()].sort((a, b) => a - b);
  if (state.playSecond === null) {
    if (buf.size < MIN_START_BUFFER) return;          // build a small cushion before the first frame
    state.playSecond = keys[0];
  }
  const ahead = keys[keys.length - 1] - state.playSecond;
  if (ahead > HARD_JUMP) {                             // absurdly behind: jump
    state.playSecond = keys[keys.length - 1] - KEEP_AHEAD;
    for (const k of keys) if (k < state.playSecond) buf.delete(k);
    console.info('[play] catching up to second', state.playSecond);
  } else if (ahead > MAX_AHEAD && buf.has(state.playSecond) && buf.has(state.playSecond + 1)) {
    show(state.playSecond);                            // backend is faster than realtime: play 2 s this tick
  }
  if (buf.has(state.playSecond)) { show(state.playSecond); state.waitingSince = null; return; }
  // gap: wait up to GAP_SKIP_MS for the missing second, then skip forward to the next available one
  const now = performance.now();
  if (state.waitingSince === null) state.waitingSince = now;
  if (now - state.waitingSince > GAP_SKIP_MS) {
    const next = keys.find((k) => k > state.playSecond);
    if (next !== undefined) { console.info(`[play] second ${state.playSecond} never arrived, skipping to ${next}`); state.playSecond = next; state.waitingSince = null; show(next); }
  }
}
function show(sec) {
  const e = state.buffer.get(sec);
  state.buffer.delete(sec);
  for (const k of [...state.buffer.keys()]) if (k < sec) state.buffer.delete(k);
  if (e.vec) state.brain.showVector(e.vec);
  state.shownSecond = sec; state.shownSystems = e.systems; state.shownTop = e.top;
  state.playSecond = sec + 1;
  renderStatus(); renderBars();
}

async function pollLoop(cfg) {
  const base = cfg.modal_base_url;
  for (;;) {
    const t0 = performance.now();
    try {
      const u = `${base}/session/${encodeURIComponent(cfg.sid)}/preds?since=${state.nextSecond}`;
      const r = await fetch(u, { signal: withTimeout(6000), cache: 'no-store' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      ingest(await r.json());
      state.pollErrors = 0; state.lastPollOk = Date.now();
    } catch (e) {
      state.pollErrors++;
      if (state.pollErrors === 1 || state.pollErrors % 15 === 0) console.warn('[preds] poll failed:', e.message);
    }
    renderStatus();
    // back off to every 3 s once the backend has been unreachable for a while (less console noise, same recovery)
    const period = state.pollErrors >= 5 ? 3000 : POLL_MS;
    await sleep(Math.max(150, period - (performance.now() - t0)));
  }
}

// ------------------------------------------------------------------------------------------------
// mock: 2–3 smooth travelling blobs on the mesh + systems derived from the atlas (no network)
// ------------------------------------------------------------------------------------------------
function systemVertexIndex(atlas) {
  if (!atlas || !atlas.labels || !atlas.systems) return null;
  const nameIdx = new Map(atlas.names.map((n, i) => [n, i]));
  const labels = atlas.labels;
  const out = {};
  for (const [id, s] of Object.entries(atlas.systems)) {
    const set = new Set((s.regions || []).map((r) => nameIdx.get(r)).filter((x) => x !== undefined));
    const idx = [];
    for (let i = 0; i < labels.length; i++) if (set.has(labels[i])) idx.push(i);
    out[id] = Int32Array.from(idx);
  }
  return out;
}
function startMock(assets) {
  const pos = assets.positions, atlas = assets.atlas;
  const bb = assets.mesh.bbox || [-100, -110, -75, 100, 110, 75];
  const ext = [(bb[3] - bb[0]) / 2, (bb[4] - bb[1]) / 2, (bb[5] - bb[2]) / 2];
  const ctr = [(bb[3] + bb[0]) / 2, (bb[4] + bb[1]) / 2, (bb[5] + bb[2]) / 2];
  const blobs = [
    { f: [0.031, 0.047, 0.023], ph: [0.0, 1.3, 2.1], sigma: 23, amp: 1.12 },
    { f: [0.041, 0.029, 0.037], ph: [2.4, 0.4, 3.9], sigma: 29, amp: 1.0 },
    { f: [0.026, 0.053, 0.031], ph: [4.1, 2.9, 0.9], sigma: 19, amp: 1.2 },
  ];
  const sysIdx = systemVertexIndex(atlas);
  const sysIds = atlas && atlas.systems ? Object.keys(atlas.systems) : FALLBACK_SYSTEMS;
  const drift = Object.fromEntries(sysIds.map((id) => [id, 0]));
  const nLabels = atlas ? atlas.names.length : 0;
  const sums = new Float64Array(nLabels), cnts = new Int32Array(nLabels);
  const c = [0, 0, 0];

  function nearestVertex(p) {
    let best = 0, bd = Infinity;
    for (let i = 0, o = 0; i < N; i++, o += 3) {
      const dx = pos[o] - p[0], dy = pos[o + 1] - p[1], dz = pos[o + 2] - p[2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
  function synth(sec) {
    const vec = new Float32Array(N);
    for (let i = 0, o = 0; i < N; i++, o += 3) {   // slow low-frequency background wobble, ±0.35
      vec[i] = 0.35 * Math.sin(0.02 * pos[o + 1] + 0.4 * sec) * Math.cos(0.025 * pos[o + 2] + 0.23 * sec + 0.01 * pos[o]);
    }
    for (const b of blobs) {
      const p = [ctr[0] + 1.15 * ext[0] * Math.sin(b.f[0] * sec + b.ph[0]), ctr[1] + 1.15 * ext[1] * Math.sin(b.f[1] * sec + b.ph[1]), ctr[2] + 1.15 * ext[2] * Math.sin(b.f[2] * sec + b.ph[2])];
      const vi = nearestVertex(p);
      c[0] = pos[vi * 3]; c[1] = pos[vi * 3 + 1]; c[2] = pos[vi * 3 + 2];
      const amp = b.amp * (0.75 + 0.25 * Math.sin(0.35 * sec + b.ph[0]));
      const s2 = 2 * b.sigma * b.sigma, cut = 16 * b.sigma * b.sigma;
      for (let i = 0, o = 0; i < N; i++, o += 3) {
        const dx = pos[o] - c[0], dy = pos[o + 1] - c[1], dz = pos[o + 2] - c[2];
        const d = dx * dx + dy * dy + dz * dz;
        if (d < cut) vec[i] += amp * Math.exp(-d / s2);
      }
    }
    // systems: honest mean over each system's atlas vertices (+ a little random walk), top regions by mean
    const z = {}; let top = null;
    if (sysIdx) {
      for (const id of sysIds) {
        const idx = sysIdx[id]; let s = 0;
        for (let k = 0; k < idx.length; k++) s += vec[idx[k]];
        drift[id] = clamp(drift[id] + (Math.random() - 0.5) * 0.08, -0.25, 0.25);
        z[id] = idx.length ? s / idx.length + drift[id] : drift[id];
      }
      sums.fill(0); cnts.fill(0);
      for (let i = 0; i < N; i++) { const l = atlas.labels[i]; sums[l] += vec[i]; cnts[l]++; }
      top = [];
      for (let l = 1; l < nLabels; l++) if (cnts[l]) top.push({ name: atlas.names[l], z: sums[l] / cnts[l] });
      top.sort((a, b) => b.z - a.z); top = top.slice(0, 8).map((t) => ({ name: t.name, z: Math.round(t.z * 100) / 100 }));
    } else {
      sysIds.forEach((id, k) => { z[id] = 0.6 + 0.9 * Math.sin(0.1 * sec + k * 0.7); });
    }
    return {
      seconds: [sec], _vectors: vec, systems: [{ second: sec, z }], regions: top ? [{ second: sec, top }] : [],
      seconds_received: sec + 15.4 + 0.4 * Math.sin(sec * 0.5),
      status: { last_infer_s: 7.9 + 0.6 * Math.sin(sec * 0.7), busy: sec % 8 < 5, seconds_predicted: sec + 1 },
    };
  }
  let t = 0;
  ingest(synth(t++)); ingest(synth(t++));
  setInterval(() => { ingest(synth(t++)); state.secondsPredicted = Math.max(state.secondsPredicted || 0, t); renderStatus(); }, 1000);
}

// ------------------------------------------------------------------------------------------------
// panel: status, bars, QR
// ------------------------------------------------------------------------------------------------
function computePhase() {
  const cfg = state.cfg;
  if (state.finishing) return ['finishing', ''];
  if (MOCK) return ['mock data — no backend', ''];
  if (!cfg.modal_base_url) return ['no Modal URL: add ?modal=<base url> or ?mock=1', 'bad'];
  if (state.pollErrors >= 3) return [`can't reach Modal (${state.pollErrors} failed polls)`, 'bad'];
  if (state.lastPollOk === null) return ['connecting to Modal…', ''];
  if (!(state.secondsReceived > 0)) return ['waiting for the phone — scan the QR', ''];
  if (state.shownSecond === null) return ['predicting the first window…', ''];
  if (state.buffer.size === 0 && state.waitingSince !== null) return ['waiting for the next prediction…', ''];
  return ['live', 'ok'];
}
function renderStatus() {
  const [phase, cls] = computePhase();
  const ph = $('phase'); ph.textContent = phase; ph.className = cls;
  $('sid').textContent = state.cfg ? state.cfg.sid : '—';
  $('received').textContent = state.secondsReceived > 0 ? fmt(state.secondsReceived, 1) + ' s' : '—';
  const shown = state.shownSecond;
  $('brainTime').textContent = shown === null ? '—' : `${shown} s`;
  const stv = $('stageTimeVal');
  if (shown !== null) { const stalled = state.waitingSince !== null && state.buffer.size === 0; stv.textContent = `${shown} s${stalled ? ' …' : ''}`; stv.classList.toggle('pulse', stalled); }
  else if (state.secondsReceived > 0 || MOCK) { stv.textContent = 'listening…'; stv.classList.add('pulse'); }
  else { stv.textContent = '—'; stv.classList.remove('pulse'); }
  $('delay').textContent = shown === null || !(state.secondsReceived > 0) ? '—' : fmt(state.secondsReceived - shown, 1) + ' s';
  $('infer').textContent = state.lastInferS === null || state.lastInferS === undefined ? '—' : fmt(state.lastInferS, 1) + ' s' + (state.busy ? ' · busy' : '');
  $('buffered').textContent = `${state.buffer.size} s`;
  const btn = $('finishBtn');
  const ready = (state.secondsPredicted || 0) >= TRAIN_AFTER_S || state.finishing;
  if (ready && btn.hidden) { btn.hidden = false; btn.classList.add('pop'); }
  else if (!ready && !btn.hidden) btn.hidden = true;
  $('topRegions').textContent = state.shownTop && state.shownTop.length
    ? 'top: ' + state.shownTop.slice(0, 4).map((t) => `${t.name} ${fmt(t.z, 1)}`).join(' · ') : '';
}

let barRows = [];
function buildBars(atlas) {
  const systems = atlas && atlas.systems ? atlas.systems : null;
  const ids = systems ? Object.keys(systems) : FALLBACK_SYSTEMS;
  const host = $('bars'); host.innerHTML = '';
  barRows = ids.map((id) => {
    const label = systems && systems[id] && systems[id].label ? systems[id].label : id.replace(/_/g, ' ');
    const row = document.createElement('div'); row.className = 'row';
    row.title = systems && systems[id] ? `${systems[id].blurb || ''}\n${(systems[id].regions || []).length} regions, ${systems[id].vertex_count || 0} vertices` : id;
    row.innerHTML = `<span class="lbl"></span><div class="track"><div class="zero"></div><div class="thr"></div><div class="bar"></div></div><span class="val">—</span>`;
    row.querySelector('.lbl').textContent = label.replace(/\s*\(.*\)$/, '');
    host.appendChild(row);
    return { id, row, bar: row.querySelector('.bar'), val: row.querySelector('.val') };
  });
}
function renderBars() {
  const z = state.shownSystems || {};
  for (const r of barRows) {
    const v = z[r.id];
    if (v === undefined || v === null) { r.bar.style.width = '0%'; r.val.textContent = '—'; r.row.classList.remove('active'); continue; }
    r.bar.style.width = (clamp((v - BAR_MIN) / (BAR_MAX - BAR_MIN), 0, 1) * 100).toFixed(1) + '%';
    r.bar.style.background = barColor(v);
    r.val.textContent = fmt(v, 2);
    r.row.classList.toggle('active', v >= THRESH_LO);
  }
}

function renderQR(cfg, attempt = 0) {
  const base = cfg.modal_base_url || (MOCK ? 'https://brain-twin.example/mock' : '');
  const url = base ? `${base}/capture?sid=${encodeURIComponent(cfg.sid)}` : '';
  const box = $('qrBox'), none = $('qrNone');
  const showNone = (text) => { box.hidden = true; none.hidden = false; none.textContent = text; };
  if (!url) { showNone('No Modal URL configured. Set MODAL_BASE_URL in .env or open ?modal=<base url>'); $('qrUrl').textContent = ''; return; }
  $('qrUrl').textContent = url + (MOCK && !cfg.modal_base_url ? '  (mock)' : '');
  const lib = window.QRCode;
  if (!lib) {   // the CDN script (or its fallback) may still be loading: retry for ~15 s
    if (attempt < 30) { setTimeout(() => renderQR(cfg, attempt + 1), 500); return; }
    showNone('QR library did not load (offline?) — type the URL below into the phone instead');
    return;
  }
  box.hidden = false; none.hidden = true; box.innerHTML = '';
  try {
    if (typeof lib.toCanvas === 'function') {           // node-qrcode style API
      const canvas = document.createElement('canvas'); box.appendChild(canvas);
      lib.toCanvas(canvas, url, { width: 120, margin: 0, color: { dark: '#0b0d12', light: '#ffffff' } }, (err) => { if (err) console.warn('[qr]', err); });
    } else {                                              // qrcodejs (davidshimjs) API
      const opts = { text: url, width: 120, height: 120, colorDark: '#0b0d12', colorLight: '#ffffff' };
      if (lib.CorrectLevel) opts.correctLevel = lib.CorrectLevel.M;
      new lib(box, opts);
    }
  } catch (e) { console.warn('[qr] render failed', e); showNone('QR failed to render — type the URL below into the phone'); }
}

let toastTimer = null;
function toast(msg, kind = 'error', ms = 9000) {
  const el = $('toast'); el.textContent = msg; el.className = kind === 'info' ? 'info' : ''; el.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

// ------------------------------------------------------------------------------------------------
// finish flow: POST /finish -> poll GET /finish/{job} -> iframe game -> score message
// ------------------------------------------------------------------------------------------------
function appendLog(lines) {
  const pre = $('log');
  const shown = pre.childElementCount;
  for (let i = shown; i < lines.length; i++) {
    const d = document.createElement('div'); d.textContent = String(lines[i]); pre.appendChild(d);
  }
  [...pre.children].forEach((c, i) => c.classList.toggle('last', i === pre.childElementCount - 1));
  pre.scrollTop = pre.scrollHeight;
  const sl = $('sumLog'); if (sl) sl.textContent = lines.slice(-3).join('\n');
}
function setLogState(html) { $('logState').innerHTML = html; const el = $('sumState'); if (el) el.innerHTML = html; }


// ------------------------------------------------------------------------------------------------
// session summary card (shown while the agent writes the game)
// ------------------------------------------------------------------------------------------------
const HEAT_STOPS = [[240, 240, 237], [255, 190, 140], [230, 57, 70], [122, 12, 46]];
function heatColor(z, lo, hi) {
  const t = clamp((z - lo) / Math.max(hi - lo, 1e-6), 0, 1);
  const k = t < 0.33 ? [0, 1, t / 0.33] : t < 0.66 ? [1, 2, (t - 0.33) / 0.33] : [2, 3, (t - 0.66) / 0.34];
  const a = HEAT_STOPS[k[0]], b = HEAT_STOPS[k[1]];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * k[2])},${Math.round(a[1] + (b[1] - a[1]) * k[2])},${Math.round(a[2] + (b[2] - a[2]) * k[2])})`;
}
let summaryRows = {};
function showSummary() {
  const secs = [...state.history.keys()].sort((a, b) => a - b);
  const systems = state.atlas && state.atlas.systems ? state.atlas.systems : null;
  const ids = systems ? Object.keys(systems) : FALLBACK_SYSTEMS;
  const base = state.baseline && state.baseline.systems ? state.baseline.systems : null;
  const stats = ids.map((id) => {
    const zs = secs.map((sec) => (state.history.get(sec) || {})[id]).filter((v) => typeof v === 'number');
    const mean = zs.length ? zs.reduce((a, b) => a + b, 0) / zs.length : null;
    const b = base && base[id] ? base[id].mean : null;
    return { id, label: systems && systems[id] ? systems[id].label.replace(/\s*\(.*\)$/, '') : id.replace(/_/g, ' '), zs, mean, delta: mean !== null && b !== null ? mean - b : null };
  }).sort((a, b) => (b.mean ?? -9) - (a.mean ?? -9));
  const all = stats.flatMap((r) => r.zs).sort((a, b) => a - b);
  const hi = Math.max(0.35, all.length ? all[Math.floor(0.98 * (all.length - 1))] : 0.35), lo = 0.04;
  $('sumSub').textContent = `${secs.length} seconds inferred` + (base ? ` · compared with your usual (${state.baseline.n_clips || Object.keys(base).length ? (state.baseline.n_clips || '') + ' clips' : ''})`.replace('( clips)', '') : ' · no baseline yet, so systems are ranked against each other');
  $('sumT1').textContent = secs.length ? `${secs[secs.length - 1]} s` : '';
  const host = $('sumRows'); host.innerHTML = ''; summaryRows = {};
  const w = 600, hgt = 14;
  for (const r of stats) {
    const row = document.createElement('div'); row.className = 'srow';
    row.innerHTML = `<span class="lbl"></span><canvas width="${w}" height="${hgt}"></canvas><span class="mean"></span><span class="delta"></span>`;
    row.querySelector('.lbl').textContent = r.label;
    row.querySelector('.mean').textContent = r.mean === null ? '—' : fmt(r.mean, 2);
    const d = row.querySelector('.delta');
    if (r.delta === null) d.textContent = '';
    else { d.textContent = (r.delta >= 0 ? '+' : '−') + fmt(Math.abs(r.delta), 2); d.classList.toggle('low', r.delta < -0.04); d.classList.toggle('high', r.delta > 0.04); }
    const ctx = row.querySelector('canvas').getContext('2d');
    const n = Math.max(1, secs.length), cw = w / n;
    secs.forEach((sec, i) => { const z = (state.history.get(sec) || {})[r.id]; ctx.fillStyle = typeof z === 'number' ? heatColor(z, lo, hi) : '#f0f0ed'; ctx.fillRect(Math.floor(i * cw), 0, Math.ceil(cw) + 1, hgt); });
    host.appendChild(row); summaryRows[r.id] = row;
  }
  $('sumWhy').textContent = ''; $('sumLog').textContent = '';
  $('summary').hidden = false;
  $('logPanel').hidden = true;      // the card carries the log now
}
function annotateSummary(plan) {
  if (!plan || !plan.target_system) return;
  for (const [id, row] of Object.entries(summaryRows)) row.classList.toggle('target', id === plan.target_system);
  const title = plan.game && plan.game.title ? `Writing “${plan.game.title}” for you…` : 'Writing your game…';
  if (plan.why) $('sumWhy').textContent = plan.why;
  setLogState('<span class="spin"></span>' + title);
  const row = summaryRows[plan.target_system]; if (row) row.scrollIntoView({ block: 'nearest' });
}
function hideSummary() { $('summary').hidden = true; }

function openGame(url, plan) {
  hideSummary();
  const target = plan && plan.target_system;
  const label = target && state.atlas && state.atlas.systems && state.atlas.systems[target] ? state.atlas.systems[target].label : target || '';
  $('gameTitle').textContent = (plan && plan.game && plan.game.title) || 'Your game';
  $('gameWhy').textContent = [label && `target: ${label}`, plan && plan.why].filter(Boolean).join(' — ');
  $('score').textContent = '';
  const frame = $('gameFrame');
  if (url.startsWith('srcdoc:')) frame.srcdoc = url.slice(7); else frame.src = url;
  $('gameWrap').hidden = false;
  document.body.classList.add('game');
  state.brain.resize(false);
  frame.addEventListener('load', () => { try { frame.contentWindow.focus(); } catch (_) { /* cross-origin */ } }, { once: true });
}
window.addEventListener('message', (ev) => {
  const d = ev.data;
  if (!d || d.type !== 'brain-game-score') return;
  const label = d.target && state.atlas && state.atlas.systems && state.atlas.systems[d.target] ? state.atlas.systems[d.target].label : d.target;
  $('score').textContent = `Score ${d.score}` + (label ? ` · ${label}` : '');
  toast(`Round over — score ${d.score}${label ? ' for ' + label : ''}`, 'info', 12000);
  console.info('[game] score', d);
});
$('backBtn').addEventListener('click', () => {
  document.body.classList.remove('game');
  $('gameWrap').hidden = true;
  const frame = $('gameFrame'); frame.removeAttribute('srcdoc'); frame.src = 'about:blank';
  state.brain.resize(false);
  state.finishing = false;
  $('finishBtn').disabled = false;
  renderStatus();
});

async function pollFinish(jobId) {
  for (;;) {
    await sleep(1500);
    let j;
    try {
      const r = await fetch(new URL(`/finish/${encodeURIComponent(jobId)}`, location.origin), { cache: 'no-store', signal: withTimeout(8000) });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      j = await r.json();
    } catch (e) {
      appendLog([...$('log').children].map((c) => c.textContent).concat([`(poll failed: ${e.message}, retrying)`]));
      continue;
    }
    if (Array.isArray(j.log)) appendLog(j.log);
    if (j.state === 'done') {
      setLogState('done — loading the game');
      if (!j.game_url) { toast('Agent finished but returned no game_url.'); state.finishing = false; $('finishBtn').disabled = false; return; }
      openGame(j.game_url, j.plan);
      return;
    }
    if (j.state === 'error') {
      setLogState('<span style="color:var(--danger)">the agent hit an error</span>');
      toast('Game generation failed — see the log. The brain keeps running.'); hideSummary();
      state.finishing = false; $('finishBtn').disabled = false;
      return;
    }
    if (j.plan && j.plan.target_system && !state.summaryAnnotated) { state.summaryAnnotated = true; annotateSummary(j.plan); }
    else if (!state.summaryAnnotated) setLogState('<span class="spin"></span>picking the system that got the least exercise…');
  }
}

async function onFinish() {
  const btn = $('finishBtn');
  if (state.finishing) return;
  state.finishing = true; btn.disabled = true; state.summaryAnnotated = false;
  $('log').innerHTML = '';
  showSummary();
  setLogState('<span class="spin"></span>reading your session…');
  renderStatus();
  if (MOCK) return mockFinish();
  try {
    const r = await fetch(new URL('/finish', location.origin), {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ sid: state.cfg.sid }), signal: withTimeout(15000),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const j = await r.json();
    if (!j.job_id) throw new Error('no job_id in response');
    appendLog([`job ${j.job_id} started`]);
    pollFinish(j.job_id);
  } catch (e) {
    console.warn('[finish] POST /finish failed:', e.message);
    toast(`Couldn't reach the laptop server to finish (POST /finish: ${e.message}). The brain keeps running — start app/server.py and try again.`); hideSummary();
    setLogState('<span style="color:var(--danger)">laptop server not reachable</span>');
    state.finishing = false; btn.disabled = false; renderStatus();
  }
}

async function mockFinish() {
  const lines = ['Fetching session summary (mock)', 'Least-driven system: Visual motion (MT+/V5)', 'Planning a 60 s game: "Dot Drift"',
    'Generating HTML with the code model (mock)', 'Headless check: 0 console errors', 'Saving app/games/mock.html'];
  const acc = [];
  for (const l of lines) { acc.push(l); appendLog(acc); setLogState('<span class="spin"></span>' + l); await sleep(900); if (l.startsWith('Planning')) annotateSummary({ target_system: 'motion', why: 'Motion areas were the quietest of your session compared with your usual footage (mock data).', game: { title: 'Dot Drift (mock)' } }); }
  await sleep(2500);
  setLogState('done — loading the game');
  openGame('srcdoc:' + MOCK_GAME_HTML, { target_system: 'motion', why: 'motion areas were the quietest during your session (mock)', game: { title: 'Dot Drift (mock)' } });
}
const MOCK_GAME_HTML = `<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#0f1218;color:#e8eaf0;font:15px system-ui,sans-serif;overflow:hidden}
#hud{position:fixed;left:16px;top:12px;right:16px;display:flex;justify-content:space-between}
canvas{display:block;width:100vw;height:100vh;cursor:crosshair}</style></head><body>
<div id="hud"><span><b>Dot Drift</b> — click the drifting dots · target: Visual motion (MT+/V5)</span><span id="t">60 s</span><span id="s">score 0</span></div>
<canvas id="c"></canvas><script>
const c=document.getElementById('c'),x=c.getContext('2d');let W,H;function rs(){W=c.width=innerWidth;H=c.height=innerHeight}rs();addEventListener('resize',rs);
let dots=[],score=0,t0=performance.now(),over=false;function spawn(){const a=Math.random()*6.28;dots.push({x:Math.random()*W,y:Math.random()*H,vx:Math.cos(a)*(90+Math.random()*120),vy:Math.sin(a)*(90+Math.random()*120),r:14+Math.random()*10})}
for(let i=0;i<5;i++)spawn();c.addEventListener('pointerdown',e=>{if(over)return;for(let i=dots.length-1;i>=0;i--){const d=dots[i];if((d.x-e.clientX)**2+(d.y-e.clientY)**2<(d.r+6)**2){dots.splice(i,1);score++;spawn();if(score%4==0)spawn();break}}document.getElementById('s').textContent='score '+score});
let last=performance.now();function f(n){const dt=Math.min(.05,(n-last)/1000);last=n;const left=Math.max(0,60-(n-t0)/1000);document.getElementById('t').textContent=Math.ceil(left)+' s';
x.fillStyle='#0f1218';x.fillRect(0,0,W,H);for(const d of dots){d.x+=d.vx*dt;d.y+=d.vy*dt;if(d.x<d.r||d.x>W-d.r)d.vx*=-1;if(d.y<d.r||d.y>H-d.r)d.vy*=-1;x.beginPath();x.arc(d.x,d.y,d.r,0,6.28);x.fillStyle='#ffb347';x.fill()}
if(left<=0&&!over){over=true;x.fillStyle='#fff';x.font='40px system-ui';x.textAlign='center';x.fillText('Round over — '+score,W/2,H/2);parent.postMessage({type:'brain-game-score',score:score,target:'motion'},'*');return}requestAnimationFrame(f)}requestAnimationFrame(f);
window.startDemo=()=>{const id=setInterval(()=>{if(dots.length){const d=dots[0];c.dispatchEvent(new PointerEvent('pointerdown',{clientX:d.x,clientY:d.y}))}},400);setTimeout(()=>clearInterval(id),25000)};
<\/script></body></html>`;

// ------------------------------------------------------------------------------------------------
// boot
// ------------------------------------------------------------------------------------------------
async function main() {
  $('finishBtn').addEventListener('click', onFinish);
  const wantHires = !['0', 'false', 'no'].includes((qs.get('hires') || '').toLowerCase());
  const [cfg, assets, hires] = await Promise.all([
    loadConfig(), loadAssets(),
    wantHires ? loadHiresAssets(ASSET_BASE).catch((e) => { console.warn('[hires] unavailable, using fsaverage5 renderer:', e.message); return null; }) : null,
  ]);
  state.cfg = cfg; state.assets = assets; state.atlas = assets.atlas;
  fetch(new URL('/samples/baseline.json', location.origin)).then((r) => (r.ok ? r.json() : null)).then((b) => { state.baseline = b; }).catch(() => {});
  console.info('[brain-twin] config', cfg, 'mesh', assets.mesh.n_vertices, 'vertices', assets.mesh.n_faces, 'faces; atlas', assets.atlas && assets.atlas.atlas, '; hires', !!hires);
  state.brain = hires
    ? createHiresBrain(hires, { stage: $('stage'), fadeS: FADE_S, thresholdFn: (vec) => { if (ADAPTIVE) adaptThresholds(vec); return [THRESH_LO, THRESH_HI]; } })
    : createBrain(assets);
  const inflateBtn = $('inflateBtn');
  if (inflateBtn) {
    if (state.brain.setSurface) {
      inflateBtn.hidden = false;
      const label = () => { inflateBtn.textContent = state.brain.surface() === 'inflated' ? 'Show pial surface' : 'Inflate'; };
      inflateBtn.addEventListener('click', () => { state.brain.setSurface(state.brain.surface() === 'inflated' ? 'pial' : 'inflated'); label(); });
      window.addEventListener('keydown', (e) => { if (e.key === 'i' && !e.metaKey && !e.ctrlKey) inflateBtn.click(); });
      label();
    } else inflateBtn.hidden = true;
  }
  $('boot').hidden = true;
  buildBars(assets.atlas);
  renderQR(cfg);
  renderStatus(); renderBars();
  (function playbackLoop() { playbackTick(); setTimeout(playbackLoop, playbackPeriodMs()); })();
  if (MOCK) startMock(assets);
  else if (cfg.modal_base_url) pollLoop(cfg);
  else { renderStatus(); toast('No Modal base URL configured: the viewer is idle. Add ?modal=<base url> (or ?mock=1 for demo data).', 'error', 12000); }
}
main().catch((e) => {
  console.error('[brain-twin] boot failed', e);
  $('boot').textContent = 'failed to load: ' + e.message;
  $('boot').hidden = false;
});
