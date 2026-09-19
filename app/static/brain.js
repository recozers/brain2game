// brain.js — high-resolution brain renderer for Brain Twin.
// Full-res FreeSurfer fsaverage (163842 vertices per hemisphere): pial surface by default, GPU morph to
// white / inflated, activation interpolated from the 20484 TRIBE v2 vertices, custom shader ramp with
// emissive glow + bloom. Same interface as the fsaverage5 renderer in viewer.js (showVector / resize),
// plus setSurface(name) and surface().
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const FILES = ['pial.f32', 'white.f32', 'inflated.f32', 'faces.u32', 'curv.f32', 'sulc.f32', 'interp_idx.u32', 'interp_w.f32'];

async function bin(url, Ctor) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url.pathname} -> HTTP ${r.status}`);
  return new Ctor(await r.arrayBuffer());
}

/** Returns null (not an error) when the hires assets are not built. */
export async function loadHiresAssets(assetBase) {
  const base = new URL('hires/', assetBase);
  let meta;
  try {
    const r = await fetch(new URL('mesh.json', base));
    if (!r.ok) return null;
    meta = await r.json();
  } catch (e) { return null; }
  const t0 = performance.now();
  const [pial, white, inflated, faces, curv, sulc, idx, w] = await Promise.all([
    bin(new URL(FILES[0], base), Float32Array), bin(new URL(FILES[1], base), Float32Array), bin(new URL(FILES[2], base), Float32Array),
    bin(new URL(FILES[3], base), Uint32Array), bin(new URL(FILES[4], base), Float32Array), bin(new URL(FILES[5], base), Float32Array),
    bin(new URL(FILES[6], base), Uint32Array), bin(new URL(FILES[7], base), Float32Array),
  ]);
  const n = meta.n_vertices, k = meta.k || 4;
  if (pial.length !== n * 3 || inflated.length !== n * 3 || idx.length !== n * k || w.length !== n * k || curv.length !== n) {
    throw new Error('hires asset sizes do not match mesh.json');
  }
  console.info(`[hires] ${n} vertices, ${faces.length / 3} faces loaded in ${Math.round(performance.now() - t0)} ms`);
  return { meta, n, k, pial, white, inflated, faces, curv, sulc, idx, w };
}

const SURFACES = { pial: [0, 0], white: [1, 0], inflated: [0, 1] };
const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

export function createHiresBrain(assets, opts = {}) {
  const stage = opts.stage || document.getElementById('stage');
  const FADE_S = opts.fadeS ?? 1.0;
  const thresholdFn = opts.thresholdFn || (() => [0.25, 1.0]);
  const { n, k, pial, white, inflated, faces, curv, sulc, idx, w, meta } = assets;

  // ---- renderer / scene ------------------------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 1);   // tone mapping + bloom lift anything brighter; the CSS stage bg shows through the edges
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  stage.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(28, 1, 1, 6000);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;

  // ---- geometry: pial base + morph targets (white, inflated) ---------------------------------
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pial, 3));
  geo.setIndex(new THREE.BufferAttribute(faces, 1));
  geo.computeVertexNormals();
  const normalsFor = (pos) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(geo.index);
    g.computeVertexNormals();
    return g.getAttribute('normal').array;
  };
  geo.morphAttributes.position = [new THREE.BufferAttribute(white, 3), new THREE.BufferAttribute(inflated, 3)];
  geo.morphAttributes.normal = [new THREE.BufferAttribute(normalsFor(white), 3), new THREE.BufferAttribute(normalsFor(inflated), 3)];
  geo.setAttribute('aCurv', new THREE.BufferAttribute(curv, 1));
  geo.setAttribute('aSulc', new THREE.BufferAttribute(sulc, 1));
  const act = [new Float32Array(n), new Float32Array(n)];
  const actAttr = act.map((a) => { const b = new THREE.BufferAttribute(a, 1); b.setUsage(THREE.DynamicDrawUsage); return b; });
  geo.setAttribute('aAct0', actAttr[0]);
  geo.setAttribute('aAct1', actAttr[1]);
  geo.computeBoundingSphere();

  // ---- material: physical shading + activation ramp in the shader -----------------------------
  const uniforms = { uMix: { value: 0 }, uLo: { value: 0.25 }, uHi: { value: 1.0 }, uInflate: { value: 0 } };
  const material = new THREE.MeshPhysicalMaterial({
    color: 0xffffff, roughness: 0.62, metalness: 0.0, clearcoat: 0.2, clearcoatRoughness: 0.55, envMapIntensity: 0.4,
  });
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>
attribute float aCurv; attribute float aSulc; attribute float aAct0; attribute float aAct1;
uniform float uMix;
varying float vAct; varying float vCurv; varying float vSulc;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
vAct = mix(aAct0, aAct1, uMix); vCurv = aCurv; vSulc = aSulc;`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
uniform float uLo; uniform float uHi; uniform float uInflate;
varying float vAct; varying float vCurv; varying float vSulc;
vec3 hotRamp(float t) {
  vec3 c0 = vec3(0.55, 0.04, 0.05), c1 = vec3(1.0, 0.40, 0.04), c2 = vec3(1.0, 0.86, 0.28), c3 = vec3(1.0, 1.0, 1.0);
  if (t < 0.4) return mix(c0, c1, t / 0.4);
  if (t < 0.75) return mix(c1, c2, (t - 0.4) / 0.35);
  return mix(c2, c3, (t - 0.75) / 0.25);
}`)
      .replace('#include <color_fragment>', `#include <color_fragment>
float sulcal = smoothstep(-0.25, 0.5, vSulc);                       // 0 on gyral crowns, 1 deep in sulci
float curvy = smoothstep(-0.3, 0.3, vCurv);
float shade = mix(1.0, 0.40, mix(0.5 * curvy, sulcal, uInflate));   // pial: light curvature cue; inflated: full sulcal shading
vec3 tissue = vec3(0.70, 0.655, 0.635) * shade;
float actT = clamp((vAct - uLo) / max(uHi - uLo, 1e-3), 0.0, 1.0);
float actW = smoothstep(0.0, 0.3, actT);
vec3 hot = hotRamp(actT);
diffuseColor.rgb = mix(tissue, hot, actW);
vec3 hotEmissive = hot * actW * (0.25 + 0.75 * actT);`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
totalEmissiveRadiance += hotEmissive * 0.85;`);
  };
  const mesh = new THREE.Mesh(geo, material);
  mesh.rotation.x = -Math.PI / 2;   // anatomical z (superior) -> three.js y (up); anterior -> -z
  mesh.morphTargetInfluences = [0, 0];
  scene.add(mesh);

  // ---- lights ---------------------------------------------------------------------------------
  scene.add(new THREE.HemisphereLight(0xe9eeff, 0x2a2521, 0.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.15); key.position.set(-0.6, 0.9, 1.0);
  camera.add(key); scene.add(camera);   // key light rides with the camera
  const fill = new THREE.DirectionalLight(0xbfd0ff, 0.35); fill.position.set(1.0, -0.2, 1.0); scene.add(fill);
  const rim = new THREE.DirectionalLight(0xffe2c8, 0.3); rim.position.set(0.3, 0.5, 1.3); scene.add(rim);

  // ---- camera / controls -----------------------------------------------------------------------
  const bb = (meta.bbox && meta.bbox.inflated) || null;
  const inflRadius = bb ? 0.5 * Math.hypot(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]) : geo.boundingSphere.radius * 1.4;
  const radius = Math.max(geo.boundingSphere.radius, inflRadius);
  const viewDir = new THREE.Vector3(-0.9, -0.95, 0.55).normalize();   // left, behind, above: visual cortex faces the audience
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.autoRotate = true; controls.autoRotateSpeed = 0.5;
  controls.minDistance = radius * 0.5; controls.maxDistance = radius * 6;
  let idleTimer = null;
  controls.addEventListener('start', () => { controls.autoRotate = false; clearTimeout(idleTimer); });
  controls.addEventListener('end', () => { clearTimeout(idleTimer); idleTimer = setTimeout(() => { controls.autoRotate = true; }, 45000); });

  // ---- post: bloom on the hot vertices only ---------------------------------------------------
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.28, 0.15, 1.0);   // threshold 1.0: only emissive hot spots bloom; keep the halo tight
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  function resize(initial) {
    const w_ = Math.max(1, stage.clientWidth), h_ = Math.max(1, stage.clientHeight);
    renderer.setSize(w_, h_);
    composer.setSize(w_, h_);
    camera.aspect = w_ / h_;
    camera.updateProjectionMatrix();
    if (initial) {
      let d = (radius / Math.sin((camera.fov * Math.PI) / 360)) * 0.78;
      if (camera.aspect < 1) d /= camera.aspect;
      camera.position.copy(viewDir).multiplyScalar(d);
      controls.target.set(0, 0, 0);
      controls.update();
    }
  }
  resize(true);
  new ResizeObserver(() => resize(false)).observe(stage);

  // ---- activation: interpolate coarse -> fine, crossfade on the GPU ---------------------------
  let showing = 0;
  const mixAnim = { from: 0, to: 0, t: 1 };
  const thr = { fromLo: 0.25, fromHi: 1.0, toLo: 0.25, toHi: 1.0 };
  function showVector(vec) {
    const [lo, hi] = thresholdFn(vec);
    const other = 1 - showing, out = act[other];
    for (let i = 0, p = 0; i < n; i++, p += k) {
      let s = 0;
      for (let j = 0; j < k; j++) s += w[p + j] * vec[idx[p + j]];
      out[i] = s;
    }
    actAttr[other].needsUpdate = true;
    mixAnim.from = uniforms.uMix.value; mixAnim.to = other; mixAnim.t = 0;
    thr.fromLo = uniforms.uLo.value; thr.fromHi = uniforms.uHi.value; thr.toLo = lo; thr.toHi = hi;
    showing = other;
  }

  // ---- surface morph ----------------------------------------------------------------------------
  let current = 'pial';
  const morph = { from: [0, 0], to: [0, 0], t: 1, dur: 1.4 };
  function setSurface(name, animate = true) {
    const to = SURFACES[name] || SURFACES.pial;
    current = SURFACES[name] ? name : 'pial';
    morph.from = [...mesh.morphTargetInfluences]; morph.to = to; morph.t = animate ? 0 : 1;
    if (!animate) { mesh.morphTargetInfluences[0] = to[0]; mesh.morphTargetInfluences[1] = to[1]; uniforms.uInflate.value = to[1]; }
  }
  const surface = () => current;

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    controls.update();
    if (mixAnim.t < 1) {
      mixAnim.t = Math.min(1, mixAnim.t + dt / FADE_S);
      const a = mixAnim.t;
      uniforms.uMix.value = mixAnim.from + (mixAnim.to - mixAnim.from) * a;
      uniforms.uLo.value = thr.fromLo + (thr.toLo - thr.fromLo) * a;
      uniforms.uHi.value = thr.fromHi + (thr.toHi - thr.fromHi) * a;
    }
    if (morph.t < 1) {
      morph.t = Math.min(1, morph.t + dt / morph.dur);
      const e = easeInOut(morph.t);
      mesh.morphTargetInfluences[0] = morph.from[0] + (morph.to[0] - morph.from[0]) * e;
      mesh.morphTargetInfluences[1] = morph.from[1] + (morph.to[1] - morph.from[1]) * e;
      uniforms.uInflate.value = mesh.morphTargetInfluences[1];
    }
    composer.render();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  return { renderer, scene, camera, controls, mesh, showVector, resize, setSurface, surface, hires: true };
}
