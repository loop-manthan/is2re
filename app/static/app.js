/* Initial Structure to Relaxed Energy demo frontend.
   Three.js molecular graph viewer plus prediction comparison.
   Monochrome UI. The only color is the element-based atom coloring in the 3D view. */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SUBSCRIPTS = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"];

function formatFormula(s) {
  return String(s).replace(/[0-9]/g, (d) => SUBSCRIPTS[Number(d)]);
}

function formulaOf(adsorbate, catalyst) {
  return `${formatFormula(adsorbate)} on ${formatFormula(catalyst)}`;
}

/* Element colors (CPK convention). Used only for atoms in the 3D view. */
const ELEMENT_COLORS = {
  1: 0xf0f0f0, 2: 0xd9ffff, 3: 0xcc80ff, 4: 0xc2ff00, 5: 0xffb5b5, 6: 0x909090,
  7: 0x3050f8, 8: 0xff0d0d, 9: 0x90e050, 10: 0xb3e3f5, 11: 0xab5cf2, 12: 0x8aff00,
  13: 0xbfa6a6, 14: 0xf0c8a0, 15: 0xff8000, 16: 0xffff30, 17: 0x1ff01f, 18: 0x80d1e3,
  19: 0x8f40d4, 20: 0x3dff00, 21: 0xe6e6e6, 22: 0xbfc2c7, 23: 0xa6a6ab, 24: 0x8a99c7,
  25: 0x9c7ac7, 26: 0xe06633, 27: 0xf090a0, 28: 0x50d050, 29: 0xc88033, 30: 0x7d80b0,
  31: 0xc28f8f, 32: 0x668f8f, 33: 0xbd80e3, 34: 0xffa100, 35: 0xa62929, 36: 0x5cb8d1,
  40: 0x94e5ff, 41: 0x73c2c9, 42: 0x54b5b5, 44: 0x8fbed1, 45: 0x89bec9, 46: 0x55d5d5,
  47: 0x999999, 48: 0xffbd73, 50: 0x668080, 51: 0x9e63b5, 52: 0xd47a00, 73: 0x4d4d4d,
  74: 0x2194d6, 77: 0x175487, 78: 0xa1a1a1, 79: 0xffd123, 82: 0x575961,
};

const LABELS = {
  zero_shot: "Zero-shot",
  frozen_backbone: "Frozen backbone",
  full: "Full fine-tune",
};

const state = {
  structures: [],
  current: null,
};

let scene, camera, renderer, controls, structureGroup;

/* ---------- fetch helpers ---------- */

async function getJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

function showError() {
  document.getElementById("error-banner").classList.remove("hidden");
}

/* ---------- Three.js setup ---------- */

function initViewer() {
  const container = document.getElementById("viewer");
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xe6e6e6);

  camera = new THREE.PerspectiveCamera(
    50,
    container.clientWidth / Math.max(container.clientHeight, 1),
    0.1,
    2000,
  );
  camera.position.set(12, 9, 16);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const dir = new THREE.DirectionalLight(0xffffff, 0.85);
  dir.position.set(6, 10, 8);
  scene.add(dir);

  const observer = new ResizeObserver(resize);
  observer.observe(container);
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function resize() {
  const container = document.getElementById("viewer");
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (!w || !h) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

/* ---------- structure rendering ---------- */

function clearStructure() {
  if (structureGroup) {
    scene.remove(structureGroup);
    structureGroup.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    });
    structureGroup = null;
  }
}

function geometryForTag(tag) {
  if (tag === 0) return new THREE.SphereGeometry(0.5, 24, 16);
  if (tag === 1) return new THREE.OctahedronGeometry(0.58);
  return new THREE.TetrahedronGeometry(0.66);
}

function renderStructure(detail) {
  clearStructure();
  const group = new THREE.Group();
  const points = detail.positions.map((p) => new THREE.Vector3(p[0], p[1], p[2]));

  for (let i = 0; i < points.length; i += 1) {
    const z = detail.atomic_numbers[i];
    const color = ELEMENT_COLORS[z] !== undefined ? ELEMENT_COLORS[z] : 0x9a9a9a;
    const tag = detail.tags[i];
    const flat = tag !== 0;
    const mat = new THREE.MeshLambertMaterial({ color, flatShading: flat });
    const mesh = new THREE.Mesh(geometryForTag(tag), mat);
    mesh.position.copy(points[i]);
    group.add(mesh);
  }

  const cutoff = detail.cutoff;
  const edgePos = [];
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      if (points[i].distanceTo(points[j]) <= cutoff) {
        edgePos.push(points[i].x, points[i].y, points[i].z,
          points[j].x, points[j].y, points[j].z);
      }
    }
  }
  if (edgePos.length) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(edgePos, 3));
    const lines = new THREE.LineSegments(
      geo,
      new THREE.LineBasicMaterial({ color: 0x666666, transparent: true, opacity: 0.5 }),
    );
    group.add(lines);
  }

  addCell(group, detail.cell);

  structureGroup = group;
  scene.add(group);
  fitCamera(points);
}

function addCell(group, cell) {
  const o = new THREE.Vector3(0, 0, 0);
  const a = new THREE.Vector3(cell[0][0], cell[0][1], cell[0][2]);
  const b = new THREE.Vector3(cell[1][0], cell[1][1], cell[1][2]);
  const c = new THREE.Vector3(cell[2][0], cell[2][1], cell[2][2]);
  const corners = [
    o, a, b, c,
    a.clone().add(b), a.clone().add(c), b.clone().add(c),
    a.clone().add(b).add(c),
  ];
  const idx = [
    [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4],
    [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
  ];
  const pos = [];
  for (const [i, j] of idx) {
    pos.push(corners[i].x, corners[i].y, corners[i].z,
      corners[j].x, corners[j].y, corners[j].z);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  const lines = new THREE.LineSegments(
    geo,
    new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.7 }),
  );
  group.add(lines);
}

function fitCamera(points) {
  const center = new THREE.Vector3();
  for (const p of points) center.add(p);
  center.divideScalar(points.length);
  let radius = 0;
  for (const p of points) radius = Math.max(radius, p.distanceTo(center));
  const dist = radius * 3.2 + 4;
  const dir = new THREE.Vector3(1, 0.8, 1.3).normalize();
  camera.position.copy(center).add(dir.multiplyScalar(dist));
  controls.target.copy(center);
  controls.update();
}

/* ---------- panels ---------- */

function renderMeta(detail) {
  document.getElementById("viewer-meta").textContent = formulaOf(detail.adsorbate, detail.catalyst);
}

function renderResults(pred) {
  const gt = pred.ground_truth_energy;
  const maxErr = Math.max(...pred.predictions.map((p) => p.error), 1e-6);
  let html =
    `<div class="gt-row"><span class="gt-label">Ground truth</span>` +
    `<span class="gt-value">${gt.toFixed(3)} eV</span></div>`;
  for (const p of pred.predictions) {
    const width = (p.error / maxErr) * 100;
    html +=
      `<div class="pred-row">` +
      `<span class="name">${LABELS[p.variant] || p.variant}</span>` +
      `<span class="value">${p.energy.toFixed(3)} eV</span>` +
      `<span class="err">err ${p.error.toFixed(3)}</span>` +
      `<div class="bar-track"><div class="bar" style="width:${width.toFixed(1)}%"></div></div>` +
      `</div>`;
  }
  document.getElementById("results-body").innerHTML = html;
}

function renderList(structures) {
  const ul = document.getElementById("structure-list");
  ul.innerHTML = "";
  for (const s of structures) {
    const li = document.createElement("li");
    li.innerHTML = `<div class="formula">${formulaOf(s.adsorbate, s.catalyst)}</div>`;
    li.addEventListener("click", () => selectStructure(s.sid, li));
    ul.appendChild(li);
  }
  document.getElementById("list-status").textContent =
    `${structures.length} structures`;
}

/* ---------- data loading ---------- */

async function selectStructure(sid, li) {
  document.querySelectorAll("#structure-list li").forEach((x) => x.classList.remove("selected"));
  if (li) li.classList.add("selected");
  try {
    const [detail, pred] = await Promise.all([
      getJSON(`/structures/${sid}`),
      getJSON(`/structures/${sid}/predictions`),
    ]);
    state.current = detail;
    renderMeta(detail);
    renderStructure(detail);
    renderResults(pred);
  } catch (err) {
    showError();
  }
}

async function loadStructures() {
  const structures = await getJSON("/structures");
  state.structures = structures;
  renderList(structures);
  if (structures.length) {
    await selectStructure(structures[0].sid, document.querySelector("#structure-list li"));
  }
}

async function loadModelInfo() {
  const data = await getJSON("/model-info");
  const body = document.getElementById("model-info-body");
  body.innerHTML = data.variants
    .map((v) =>
      `<tr>` +
      `<td>${LABELS[v.variant] || v.variant}</td>` +
      `<td>${v.trainable_params.toLocaleString()}</td>` +
      `<td>${v.test_mae.toFixed(4)}</td>` +
      `<td>${(v.test_ewt * 100).toFixed(2)}%</td>` +
      `</tr>`,
    )
    .join("");
}

async function init() {
  initViewer();
  try {
    await Promise.all([loadModelInfo(), loadStructures()]);
  } catch (err) {
    showError();
  }
}

init();