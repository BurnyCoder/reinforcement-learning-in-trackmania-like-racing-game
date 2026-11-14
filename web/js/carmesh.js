// carmesh.js — the low-poly car model: a body built from boxes and four cylinder wheels.
//
// Global context: the model is drawn in the chassis frame of the physics (+X forward, +Y up, +Z right,
// the same axes three.js uses), sized from vehicle.json: the body from chassis.half_extents, wheels
// at wheel_connections moved down the suspension by the current suspensionLength, radius wheel.radius,
// front wheels yawed by the steering angle. Used for the player, the AI car and translucent ghosts.

import * as THREE from '../vendor/three/three.module.js';

// opts: body (hex colour), accent (hex), ghost (bool: translucent, no shadows).
export function createCarMesh(cfg, { body = 0xff6b35, accent = 0x1b2230, ghost = false } = {}) {
  const [hx, hy, hz] = cfg.chassis.half_extents; // half length, half height, half width (m)
  const mat = (color, extra = {}) => new THREE.MeshStandardMaterial({
    color, flatShading: true, roughness: 0.6, metalness: 0.1,
    ...(ghost ? { transparent: true, opacity: 0.4, depthWrite: false, emissive: color, emissiveIntensity: 0.35 } : {}), ...extra,
  });
  const bodyMat = mat(body), accentMat = mat(accent), glassMat = mat(0x22303f, { roughness: 0.2 });
  const tyreMat = mat(0x16181c, { roughness: 0.9 }), hubMat = mat(0xc9d1db);
  const lightMat = mat(0xfff3c4, { emissive: 0xfff3c4, emissiveIntensity: ghost ? 0.3 : 1.2 });
  const tailMat = mat(0xff2a3c, { emissive: 0xff2a3c, emissiveIntensity: ghost ? 0.3 : 1.0 });

  const group = new THREE.Group();
  const box = (sx, sy, sz, m, x, y, z) => { // a box of size (sx, sy, sz) centred at (x, y, z) in the chassis frame
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), m);
    mesh.position.set(x, y, z);
    group.add(mesh);
    return mesh;
  };
  box(2 * hx, hy * 0.9, 2 * hz, bodyMat, 0, -hy * 0.35, 0); // main body slab
  box(2 * hx * 0.42, hy * 0.75, 2 * hz * 0.8, glassMat, -hx * 0.18, hy * 0.45, 0); // cabin
  box(2 * hx * 0.38, hy * 0.12, 2 * hz * 0.82, bodyMat, -hx * 0.2, hy * 0.86, 0); // roof
  box(hx * 0.5, hy * 0.25, 2 * hz * 0.9, bodyMat, hx * 0.55, hy * 0.22, 0); // bonnet bulge
  box(0.18, hy * 0.18, 2 * hz * 1.08, accentMat, hx + 0.02, -hy * 0.78, 0); // front splitter
  box(0.45, 0.07, 2 * hz * 1.05, accentMat, -hx + 0.25, hy * 1.05, 0); // rear wing
  for (const z of [-hz * 0.55, hz * 0.55]) box(0.12, hy * 0.5, 0.08, accentMat, -hx + 0.3, hy * 0.75, z); // wing struts
  for (const z of [-hz * 0.7, hz * 0.7]) {
    box(0.06, 0.14, 0.45, lightMat, hx + 0.01, -hy * 0.1, z); // headlights
    box(0.06, 0.12, 0.5, tailMat, -hx - 0.01, -hy * 0.1, z); // tail lights
  }

  // Wheels: pivot (suspension travel + steering yaw) -> spinner (roll about the axle, +Z).
  const r = cfg.wheel.radius, width = 0.34;
  const tyreGeo = new THREE.CylinderGeometry(r, r, width, 12).rotateX(Math.PI / 2); // axis along +Z
  const hubGeo = new THREE.CylinderGeometry(r * 0.55, r * 0.55, width + 0.02, 6).rotateX(Math.PI / 2);
  const wheels = cfg.wheel_connections.map((conn) => {
    const pivot = new THREE.Group(), spinner = new THREE.Group();
    spinner.add(new THREE.Mesh(tyreGeo, tyreMat), new THREE.Mesh(hubGeo, hubMat));
    pivot.add(spinner);
    pivot.position.set(conn[0], conn[1] - cfg.wheel.suspension_rest_length, conn[2]);
    group.add(pivot);
    return { pivot, spinner, conn };
  });
  group.traverse((o) => { if (o.isMesh) { o.castShadow = !ghost; o.renderOrder = ghost ? 1 : 0; } });

  return {
    group,
    // Chassis pose in world space.
    setPose(position, quaternion) { group.position.copy(position); group.quaternion.copy(quaternion); },
    // susp: suspension lengths (m) or null for rest length; steer: front-wheel angle (rad);
    // spin: wheel roll angles (rad) or null. Positive steer turns the wheels left, as in cannon-es.
    setWheels(susp, steer, spin) {
      wheels.forEach((w, i) => {
        w.pivot.position.y = w.conn[1] - (susp ? susp[i] : cfg.wheel.suspension_rest_length); // connection + direction * length
        w.pivot.rotation.y = cfg.front_wheels.includes(i) ? steer : 0;
        if (spin) w.spinner.rotation.z = -spin[i]; // cannon-es counts forward roll as positive about +Z; flip so the tread moves the right way
      });
    },
    dispose() {
      group.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
      group.removeFromParent();
    },
  };
}
