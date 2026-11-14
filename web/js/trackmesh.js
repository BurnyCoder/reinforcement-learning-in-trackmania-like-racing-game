// trackmesh.js — three.js meshes for one track: the road ribbon, its underside, edge lines and gates.
//
// Global context: the drivable surface is exactly the track file's `vertices`/`triangles` (the same
// triangles cannon-es casts the wheel rays against, docs/formats.md), drawn flat-shaded. Everything
// else here is decoration: a slab underneath so the ribbon reads as a solid road floating in the sky,
// lighter edge lines, and gates at the start, each checkpoint and the finish.

import * as THREE from '../vendor/three/three.module.js';
import { roadFrame } from './roadframe.js';

const THICKNESS = 0.7; // m, visual depth of the road slab
const EDGE_WIDTH = 0.45; // m, width of the edge lines
const GATE_HEIGHT = 6.5; // m, posts above the road
export const GATE_COLORS = { start: 0x35d07f, checkpoint: 0x2cc6e0, finish: 0xff5a5f };

// Growable triangle soup with per-vertex colours (linear RGB, as three.js expects for vertex colours).
class Soup {
  constructor() { this.pos = []; this.col = []; }
  tri(a, b, c, color) {
    for (const v of [a, b, c]) { this.pos.push(v[0], v[1], v[2]); this.col.push(color.r, color.g, color.b); }
  }
  quad(a, b, c, d, color) { this.tri(a, b, c, color); this.tri(b, d, c, color); } // a-b on one side, c-d on the other
  geometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    g.computeVertexNormals(); // non-indexed: each triangle gets its own face normal (flat shading)
    return g;
  }
}

const add = (a, b, s = 1) => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
const down = (v) => [v[0], v[1] - THICKNESS, v[2]];
const lift = (v) => [v[0], v[1] + 0.03, v[2]]; // raise decals slightly above the road to avoid z-fighting

// Banner texture: text on a coloured strip (checkered ends for the finish).
function bannerTexture(text, color, checkered) {
  const c = document.createElement('canvas');
  c.width = 1024; c.height = 128;
  const g = c.getContext('2d');
  g.fillStyle = `#${new THREE.Color(color).getHexString()}`;
  g.fillRect(0, 0, c.width, c.height);
  if (checkered) {
    for (let x = 0; x < 6; x++) for (let y = 0; y < 4; y++) {
      g.fillStyle = (x + y) % 2 ? '#111' : '#fff';
      g.fillRect(x * 32, y * 32, 32, 32);
      g.fillRect(c.width - (x + 1) * 32, y * 32, 32, 32);
    }
  }
  g.fillStyle = '#0d1320';
  g.font = 'bold 84px system-ui, "Segoe UI", sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, c.width / 2, c.height / 2 + 4);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace; // the canvas holds sRGB colours
  tex.anisotropy = 4;
  return tex;
}

// One gate standing across the road at centreline sample i.
function buildGate(track, i, kind, text) {
  const { p, q } = roadFrame(track, i);
  const color = GATE_COLORS[kind];
  const span = track.half_width + 0.6; // posts just outside the road edges
  const group = new THREE.Group();
  group.position.copy(p);
  group.quaternion.copy(q); // local axes: x forward, y up, z right
  const postMat = new THREE.MeshStandardMaterial({ color: 0x263040, flatShading: true, roughness: 0.7 });
  const capMat = new THREE.MeshStandardMaterial({ color, flatShading: true, emissive: color, emissiveIntensity: 0.3 });
  for (const z of [-span, span]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.6, GATE_HEIGHT + 2, 0.6), postMat);
    post.position.set(0, GATE_HEIGHT / 2 - 1, z); // reaches 2 m below the road surface
    post.castShadow = true;
    group.add(post);
  }
  const tex = bannerTexture(text, color, kind === 'finish');
  const face = new THREE.MeshStandardMaterial({ map: tex, emissive: 0xffffff, emissiveMap: tex, emissiveIntensity: 0.25, roughness: 0.6 });
  const banner = new THREE.Mesh(new THREE.BoxGeometry(0.4, 1.5, 2 * span + 0.6), [face, face, capMat, capMat, capMat, capMat]); // text on the +x and -x faces
  banner.position.set(0, GATE_HEIGHT - 0.75, 0);
  group.add(banner);
  // A line across the road surface marks the exact crossing.
  const line = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 2 * track.half_width).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, depthWrite: false }));
  line.position.y = 0.04;
  group.add(line);
  return { group, face, kind };
}

// Build all track meshes. Returns { group, gates, bounds } where gates[k] belongs to checkpoint k.
export function buildTrack(track) {
  const V = track.vertices, L = track.left, M = track.center.length;
  const road = new Soup(), slab = new Soup(), edges = new Soup();
  const roadA = new THREE.Color(0x3b4252), roadB = new THREE.Color(0x343a49); // alternating 10 m bands show speed
  const under = new THREE.Color(0x5a5470), white = new THREE.Color(0xeef3f8), coral = new THREE.Color(0xff6b5e);

  track.triangles.forEach((t, k) => {
    const seg = track.triangle_segment[k];
    road.tri(V[t[0]], V[t[1]], V[t[2]], Math.floor(seg / 5) % 2 ? roadB : roadA);
    slab.tri(down(V[t[0]]), down(V[t[2]]), down(V[t[1]]), under); // underside, reversed winding
  });

  const isRoad = (i) => i >= 0 && i < M - 1 && !track.gap[i];
  for (let i = 0; i < M - 1; i++) {
    if (!isRoad(i)) continue;
    const l0 = V[2 * i], r0 = V[2 * i + 1], l1 = V[2 * i + 2], r1 = V[2 * i + 3]; // edge vertices of segment i
    slab.quad(l0, l1, down(l0), down(l1), under); // left wall
    slab.quad(r1, r0, down(r1), down(r0), under); // right wall
    if (!isRoad(i - 1)) slab.quad(r0, l0, down(r0), down(l0), under); // cap where a slab begins
    if (!isRoad(i + 1)) slab.quad(l1, r1, down(l1), down(r1), under); // cap where a slab ends
    const c = Math.floor(i / 3) % 2 ? coral : white; // alternating 6 m blocks
    edges.quad(lift(l0), lift(add(l0, L[i], -EDGE_WIDTH)), lift(l1), lift(add(l1, L[i + 1], -EDGE_WIDTH)), c);
    edges.quad(lift(add(r0, L[i], EDGE_WIDTH)), lift(r0), lift(add(r1, L[i + 1], EDGE_WIDTH)), lift(r1), c);
  }

  const group = new THREE.Group();
  const roadMesh = new THREE.Mesh(road.geometry(), new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.85 }));
  roadMesh.receiveShadow = true;
  const slabMesh = new THREE.Mesh(slab.geometry(), new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.9, side: THREE.DoubleSide }));
  const edgeMesh = new THREE.Mesh(edges.geometry(), new THREE.MeshStandardMaterial({
    vertexColors: true, flatShading: true, roughness: 0.6, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -1,
  }));
  edgeMesh.receiveShadow = true;
  group.add(roadMesh, slabMesh, edgeMesh);

  // Gates: start, intermediate checkpoints, finish (the last checkpoint).
  const cps = track.checkpoints;
  group.add(buildGate(track, track.start.index, 'start', 'START').group);
  const gates = cps.map((i, k) => {
    const last = k === cps.length - 1;
    const gate = buildGate(track, i, last ? 'finish' : 'checkpoint', last ? 'FINISH' : `CHECKPOINT ${k + 1}`);
    group.add(gate.group);
    return gate;
  });

  const bounds = new THREE.Box3().setFromBufferAttribute(roadMesh.geometry.getAttribute('position'));
  return { group, gates, bounds };
}

// Highlight the next checkpoint's banner, dim the passed ones.
export function setGateProgress(gates, next) {
  gates.forEach((g, k) => { g.face.emissiveIntensity = k === next ? 0.9 : k < next ? 0.05 : 0.25; });
}

// Free GPU memory of a track built by buildTrack.
export function disposeTrack(group) {
  group.traverse((o) => {
    if (!o.isMesh) return;
    o.geometry.dispose();
    for (const m of [].concat(o.material)) { m.map?.dispose(); m.dispose(); }
  });
  group.removeFromParent();
}
