// physics.js — builds the cannon-es world, the track body and the car, and reads/writes the car state.
//
// Global context: this module is the single place where the browser game touches cannon-es
// (https://github.com/pmndrs/cannon-es, MIT, vendored unchanged in web/vendor/cannon-es).
// It has no DOM dependency, so the Node.js parity harness (tests/js/parity_harness.mjs) imports it
// too, and the PyTorch port in src/racingmania/sim/ is tested against exactly this setup.
// docs/physics.md lists what one cannon-es tick does; docs/formats.md defines vehicle.json and tracks.

import * as CANNON from '../../vendor/cannon-es/cannon-es.js';

// Re-export the engine so other core modules use the same instance.
export { CANNON };

// Create a world with fixed gravity and no sleeping.
// Local: World defaults are NaiveBroadphase and GSSolver; with the chassis filtered out of
// collisions they never produce equations, so they do not affect the car (docs/physics.md, step 2).
export function createWorld(cfg) {
  const world = new CANNON.World({ gravity: new CANNON.Vec3(...cfg.gravity) }); // gravity from vehicle.json
  world.allowSleep = false; // never put bodies to sleep: sleeping would freeze the car mid-race
  world.quatNormalizeSkip = 0; // normalise the chassis quaternion every tick (cannon-es default, set explicitly)
  world.quatNormalizeFast = false; // exact normalisation, which the port mirrors
  return world;
}

// Build a static body holding the track as one triangle mesh.
// Local: cannon-es stores Trimesh vertices as Float32Array and indices as Int16Array
// (Trimesh constructor), so a track must have fewer than 32,768 vertices.
// Global: the port casts its wheel rays against the same float32-rounded triangles.
export function createTrackBody(track) {
  const verts = track.vertices.flat(); // [[x,y,z],...] -> [x,y,z,x,y,z,...]
  const tris = track.triangles.flat(); // [[a,b,c],...] -> [a,b,c,...]
  if (track.vertices.length > 32767) throw new Error('track has too many vertices for a cannon-es Trimesh');
  const shape = new CANNON.Trimesh(verts, tris); // computes face normals (b-a)x(c-a), builds an octree
  const body = new CANNON.Body({ mass: 0, type: CANNON.Body.STATIC }); // static: never moves, infinite mass
  body.addShape(shape); // shape at the body origin; the body stays at (0,0,0) with identity rotation
  return body;
}

// Create the chassis body and the RaycastVehicle with four wheels from vehicle.json.
// Local: the box shape is added while the body is still at the origin with identity rotation, so
// cannon-es computes the inertia from the exact half extents (Body.updateMassProperties uses the
// world-space AABB of the shapes at the moment they are added).
export function createCar(world, cfg) {
  const c = cfg.chassis; // chassis block of vehicle.json
  const chassis = new CANNON.Body({ mass: c.mass }); // dynamic body, created at the origin
  chassis.addShape(new CANNON.Box(new CANNON.Vec3(...c.half_extents))); // box used only for mass and inertia
  chassis.linearDamping = c.linear_damping; // v *= (1 - d)^dt each tick (World.internalStep)
  chassis.angularDamping = c.angular_damping; // same for angular velocity
  chassis.allowSleep = false; // the car must never sleep
  chassis.collisionFilterGroup = 2; // its own group ...
  chassis.collisionFilterMask = 0; // ... colliding with nothing: only the wheel rays touch the track

  const vehicle = new CANNON.RaycastVehicle({
    chassisBody: chassis,
    indexRightAxis: 2, // +Z is right (cannon-es default, set explicitly)
    indexForwardAxis: 0, // +X is forward
    indexUpAxis: 1, // +Y is up
  });
  const w = cfg.wheel; // shared wheel parameters
  for (const conn of cfg.wheel_connections) {
    vehicle.addWheel({
      chassisConnectionPointLocal: new CANNON.Vec3(...conn), // where the suspension is attached, chassis frame
      directionLocal: new CANNON.Vec3(...w.direction), // suspension direction, (0,-1,0) = down
      axleLocal: new CANNON.Vec3(...w.axle), // wheel axle, (0,0,1) = along the right axis
      radius: w.radius,
      suspensionRestLength: w.suspension_rest_length,
      suspensionStiffness: w.suspension_stiffness,
      maxSuspensionTravel: w.max_suspension_travel,
      dampingCompression: w.damping_compression,
      dampingRelaxation: w.damping_relaxation,
      frictionSlip: w.friction_slip,
      rollInfluence: w.roll_influence,
      maxSuspensionForce: w.max_suspension_force,
      forwardAcceleration: w.forward_acceleration,
      sideAcceleration: w.side_acceleration,
      useCustomSlidingRotationalSpeed: false, // wheel spin is visual only; keep the cannon-es default
    });
  }
  vehicle.addToWorld(world); // adds the chassis and registers updateVehicle on the world's preStep event
  return { chassis, vehicle };
}

// Chassis state layout shared with the port: [px,py,pz, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz].
export const STATE_SIZE = 13;

// Copy a 13-number state into the chassis and reset the visual-only wheel spin.
// Local: previousPosition/previousQuaternion are also set so that interpolated rendering does not
// smear a teleport (respawn) across frames.
export function setChassisState(car, s) {
  const b = car.chassis;
  b.position.set(s[0], s[1], s[2]);
  b.quaternion.set(s[3], s[4], s[5], s[6]);
  b.velocity.set(s[7], s[8], s[9]);
  b.angularVelocity.set(s[10], s[11], s[12]);
  b.previousPosition.copy(b.position);
  b.previousQuaternion.copy(b.quaternion);
  b.force.set(0, 0, 0); // no leftover forces from before the teleport
  b.torque.set(0, 0, 0);
  b.updateInertiaWorld(true); // world inverse inertia must match the new orientation before the next tick
  for (const wheel of car.vehicle.wheelInfos) wheel.rotation = 0; // visual spin angle; see docs/physics.md
}

// Read the 13-number chassis state.
export function getChassisState(car) {
  const b = car.chassis;
  return [
    b.position.x, b.position.y, b.position.z,
    b.quaternion.x, b.quaternion.y, b.quaternion.z, b.quaternion.w,
    b.velocity.x, b.velocity.y, b.velocity.z,
    b.angularVelocity.x, b.angularVelocity.y, b.angularVelocity.z,
  ];
}

// Speed along the chassis forward axis (+X in the chassis frame), in m/s; negative when reversing.
export function forwardSpeed(car) {
  const fwd = car.chassis.quaternion.vmult(new CANNON.Vec3(1, 0, 0)); // chassis +X in world space
  return fwd.dot(car.chassis.velocity); // projection of the velocity on it
}

// Number of wheels whose suspension ray hit the track in the last tick.
export function wheelsInContact(car) {
  let n = 0;
  for (const wheel of car.vehicle.wheelInfos) if (wheel.isInContact) n += 1;
  return n;
}
