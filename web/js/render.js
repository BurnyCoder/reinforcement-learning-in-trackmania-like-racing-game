// render.js — the three.js scene: sky, fog, lights, the haze far below, the track and the cameras.
//
// Global context: main.js owns one Renderer. It shows the selected track behind the menu with a slow
// orbit camera, and during a race a smoothed chase camera behind the focus car. The look is original
// and low-poly: a gradient sky, a road floating above a hazy cloud layer, flat-shaded meshes.
// three.js r187 (vendored, MIT): https://threejs.org/docs/

import * as THREE from '../vendor/three/three.module.js';
import { buildTrack, setGateProgress, disposeTrack } from './trackmesh.js';

const SKY_TOP = 0x2a64c9, HORIZON = 0xc9dcf0; // the fog colour equals the horizon so distance fades into the sky
const RACE_FOG = [80, 750]; // fog near/far (m) during a race

// Sky dome: colour by view direction, horizon colour at and below the horizon.
function createSky() {
  const material = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: { top: { value: new THREE.Color(SKY_TOP) }, horizon: { value: new THREE.Color(HORIZON) } },
    vertexShader: `varying vec3 vDir;
      void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform vec3 top; uniform vec3 horizon; varying vec3 vDir;
      void main() {
        float h = clamp(vDir.y, 0.0, 1.0);
        gl_FragColor = vec4(mix(horizon, top, pow(h, 0.55)), 1.0);
        #include <colorspace_fragment>
      }`, // colorspace_fragment converts the linear uniforms to the output colour space, like built-in materials
  });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(6000, 32, 16), material);
  sky.renderOrder = -1;
  sky.frustumCulled = false;
  return sky;
}

// Deterministic pseudo-random numbers (mulberry32) so a track's clouds are the same every visit.
function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The ground haze far below the road and a few low-poly clouds in between.
function buildEnvironment(bounds, name) {
  const group = new THREE.Group();
  const centre = bounds.getCenter(new THREE.Vector3());
  const haze = new THREE.Mesh(new THREE.PlaneGeometry(20000, 20000).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: 0x9fb6d6 }));
  haze.position.set(centre.x, bounds.min.y - 160, centre.z);
  group.add(haze);
  const rand = rng([...name].reduce((h, ch) => Math.imul(h ^ ch.charCodeAt(0), 16777619), 2166136261));
  const cloudMat = new THREE.MeshStandardMaterial({ color: 0xf4f8ff, flatShading: true, roughness: 1 });
  const size = bounds.getSize(new THREE.Vector3());
  for (let i = 0; i < 70; i++) {
    const cloud = new THREE.Group();
    const puffs = 3 + Math.floor(rand() * 4);
    for (let j = 0; j < puffs; j++) {
      const puff = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 0), cloudMat);
      puff.scale.set(10 + rand() * 14, 5 + rand() * 5, 9 + rand() * 12);
      puff.position.set(j * 11 - puffs * 5, rand() * 3, (rand() - 0.5) * 10);
      cloud.add(puff);
    }
    cloud.position.set(
      centre.x + (rand() - 0.5) * (size.x + 900),
      bounds.min.y - 40 - rand() * 90, // always well below the road
      centre.z + (rand() - 0.5) * (size.z + 900));
    cloud.rotation.y = rand() * Math.PI;
    group.add(cloud);
  }
  return group;
}

export class Renderer {
  constructor(container) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(HORIZON, ...RACE_FOG);
    this.camera = new THREE.PerspectiveCamera(68, 1, 0.3, 8000);
    this.sky = createSky();
    this.scene.add(this.sky);

    this.scene.add(new THREE.HemisphereLight(0xdcecff, 0x6b5f7c, 1.6)); // sky / ground bounce
    this.sun = new THREE.DirectionalLight(0xfff4e0, 2.4);
    this.sunOffset = new THREE.Vector3(40, 90, 25); // sun direction (from the target), fixed in the world
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(1024, 1024);
    Object.assign(this.sun.shadow.camera, { left: -14, right: 14, top: 14, bottom: -14, near: 1, far: 250 }); // a small box around the car
    this.sun.shadow.bias = -0.0005;
    this.scene.add(this.sun, this.sun.target);

    this.camDir = new THREE.Vector3(1, 0, 0); // smoothed horizontal heading of the chase camera
    this.camPos = new THREE.Vector3();
    this.track = null;
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const w = window.innerWidth, h = window.innerHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // Show a track (replaces the previous one).
  setTrack(track) {
    if (this.track?.name === track.name) return;
    if (this.trackObjects) { disposeTrack(this.trackObjects.group); disposeTrack(this.environment); }
    this.trackObjects = buildTrack(track);
    this.environment = buildEnvironment(this.trackObjects.bounds, track.name);
    this.scene.add(this.trackObjects.group, this.environment);
    this.track = track;
  }

  setNextCheckpoint(next) { if (this.trackObjects) setGateProgress(this.trackObjects.gates, next); }

  // Menu background: circle slowly around the whole track.
  orbit(seconds) {
    if (!this.trackObjects) return;
    const b = this.trackObjects.bounds, c = b.getCenter(new THREE.Vector3());
    const radius = 0.55 * b.getSize(new THREE.Vector3()).length() + 60;
    const a = seconds * 0.06;
    this.camera.position.set(c.x + radius * Math.cos(a), c.y + 0.32 * radius + 20, c.z + radius * Math.sin(a));
    this.camera.lookAt(c);
    this.scene.fog.near = radius * 0.6;
    this.scene.fog.far = radius * 3;
    this.setFov(55);
    this.placeSun(c);
  }

  // Chase camera behind and above the car, looking ahead. Heading and position are smoothed
  // exponentially (frame-rate independent); snap = jump straight there (start, respawn).
  follow(position, quaternion, speed, dt, snap = false) {
    [this.scene.fog.near, this.scene.fog.far] = RACE_FOG;
    const fwd = new THREE.Vector3(1, 0, 0).applyQuaternion(quaternion).setY(0); // chassis forward, flattened
    if (fwd.lengthSq() > 1e-4) {
      fwd.normalize();
      if (snap) this.camDir.copy(fwd); else this.camDir.lerp(fwd, 1 - Math.exp(-4 * dt)).normalize();
    }
    const dist = 9 + 0.05 * speed, height = 3.4 + 0.015 * speed;
    const target = position.clone().addScaledVector(this.camDir, -dist).add(new THREE.Vector3(0, height, 0));
    if (snap) this.camPos.copy(target); else this.camPos.lerp(target, 1 - Math.exp(-10 * dt));
    this.camera.position.copy(this.camPos);
    this.camera.lookAt(position.clone().addScaledVector(this.camDir, 6).add(new THREE.Vector3(0, 1.2, 0)));
    this.setFov(66 + 14 * Math.min(speed / 45, 1)); // wider at speed
    this.placeSun(position);
  }

  setFov(fov) {
    if (Math.abs(this.camera.fov - fov) < 0.05) return;
    this.camera.fov = fov;
    this.camera.updateProjectionMatrix();
  }

  // The shadow-casting sun follows the point of interest so its small shadow box covers the car.
  placeSun(target) {
    this.sun.target.position.copy(target);
    this.sun.position.copy(target).add(this.sunOffset);
  }

  render() {
    this.sky.position.copy(this.camera.position); // the dome is centred on the viewer
    this.renderer.render(this.scene, this.camera);
  }
}
