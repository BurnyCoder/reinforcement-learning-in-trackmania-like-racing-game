// rules.js — progress along the track, checkpoints, finish and "out" conditions, evaluated every tick.
//
// Global context: these rules decide what counts as finishing and failing, both for players in the
// browser and for the RL agent (the reward is the progress computed here). The PyTorch port
// (src/racingmania/sim/rules.py) implements the same arithmetic in the same order; the definitions
// are in docs/formats.md, section "Rules".

// Thresholds of the out conditions (docs/formats.md).
export const MAX_AIR_TICKS = 500; // 5 s with no wheel on the track
export const STALL_TICKS = 200; // 2 s without at least STALL_DISTANCE of new progress
export const STALL_DISTANCE = 0.5; // metres
export const SEARCH_BACK = 2; // nearest-sample search window [index - 2, index + 6]
export const SEARCH_AHEAD = 6;

// Precompute cumulative 3D arc length at each centreline sample: cum[0] = 0,
// cum[i+1] = cum[i] + |center[i+1] - center[i]|. Progress is reported in these metres.
export function cumulativeLength(center) {
  const cum = new Float64Array(center.length);
  for (let i = 1; i < center.length; i++) {
    const dx = center[i][0] - center[i - 1][0];
    const dy = center[i][1] - center[i - 1][1];
    const dz = center[i][2] - center[i - 1][2];
    cum[i] = cum[i - 1] + Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return cum;
}

// Nearest centreline sample to point p, searched only near the previous index.
// Local: squared 3D distance, ties resolved toward the lower index (strict <).
// Global: the window keeps the index following the car's own lap even where the track crosses
// itself (at least 8 m apart vertically by construction), and it is what the port vectorises.
export function nearestIndex(center, p, index) {
  const lo = Math.max(0, index - SEARCH_BACK);
  const hi = Math.min(center.length - 1, index + SEARCH_AHEAD);
  let best = lo, bestD = Infinity;
  for (let j = lo; j <= hi; j++) {
    const dx = p[0] - center[j][0], dy = p[1] - center[j][1], dz = p[2] - center[j][2];
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bestD) { bestD = d; best = j; }
  }
  return best;
}

// Project p onto segment k (center[k] -> center[k+1]); returns [t in 0..1, squared distance].
function projectOnSegment(center, p, k) {
  const a = center[k], b = center[k + 1];
  const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2]; // segment vector
  const wx = p[0] - a[0], wy = p[1] - a[1], wz = p[2] - a[2]; // from segment start to p
  const uu = ux * ux + uy * uy + uz * uz;
  let t = (wx * ux + wy * uy + wz * uz) / uu;
  t = Math.min(Math.max(t, 0), 1); // clamp to the segment
  const ex = wx - t * ux, ey = wy - t * uy, ez = wz - t * uz; // offset from the projected point
  return [t, ex * ex + ey * ey + ez * ez];
}

// Arc-length progress of p given its nearest sample j: project on the segments before and after
// j and keep the nearer projection (ties toward the earlier segment).
export function progressAt(center, cum, p, j) {
  let bestK = -1, bestT = 0, bestD = Infinity;
  for (const k of [j - 1, j]) {
    if (k < 0 || k > center.length - 2) continue; // segment must exist
    const [t, d] = projectOnSegment(center, p, k);
    if (d < bestD) { bestD = d; bestK = k; bestT = t; }
  }
  return cum[bestK] + bestT * (cum[bestK + 1] - cum[bestK]);
}

// Per-run rule state for one car on one track.
export class RaceRules {
  constructor(track) {
    this.track = track;
    this.cum = cumulativeLength(track.center); // metres at each sample
    this.cpS = track.checkpoints.map((i) => this.cum[i]); // progress needed for each checkpoint
  }

  // Start a run at chassis position p, with the nearest sample known to be `index`.
  reset(p, index) {
    this.index = nearestIndex(this.track.center, p, index);
    this.s = progressAt(this.track.center, this.cum, p, this.index); // current progress (m)
    this.anchorS = this.s; // progress at the last stall-timer reset
    this.stallTicks = 0;
    this.airTicks = 0;
    this.nextCheckpoint = 0; // index into track.checkpoints
    this.ticks = 0; // ticks since the start of the run
    this.finished = false;
    this.out = false;
    this.outReason = '';
  }

  // Update after one physics tick. p: chassis position; contacts: number of wheels on the track.
  // Returns the progress gained during this tick (metres, may be negative).
  update(p, contacts) {
    const center = this.track.center;
    this.ticks += 1;
    this.index = nearestIndex(center, p, this.index);
    const s = progressAt(center, this.cum, p, this.index);
    const gained = s - this.s;
    this.s = s;

    // Checkpoints in order; several can be passed in one tick. Finishing beats any out condition.
    while (this.nextCheckpoint < this.cpS.length && this.s >= this.cpS[this.nextCheckpoint]) this.nextCheckpoint += 1;
    if (this.nextCheckpoint === this.cpS.length) { this.finished = true; return gained; }

    // Out: below the track's floor at this sample (floor accounts for gaps; docs/formats.md).
    if (p[1] < this.track.floor[this.index]) { this.out = true; this.outReason = 'fell'; return gained; }
    // Out: airborne too long.
    this.airTicks = contacts === 0 ? this.airTicks + 1 : 0;
    if (this.airTicks >= MAX_AIR_TICKS) { this.out = true; this.outReason = 'airborne'; return gained; }
    // Out: stalled (no 0.5 m of new progress within 2 s).
    if (this.s >= this.anchorS + STALL_DISTANCE) { this.anchorS = this.s; this.stallTicks = 0; } else { this.stallTicks += 1; }
    if (this.stallTicks >= STALL_TICKS) { this.out = true; this.outReason = 'stalled'; }
    return gained;
  }
}
