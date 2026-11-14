// timing.js — time formatting and medal rules shared by the menu, the HUD and the finish screen.
//
// Global context: medal targets come from the track's certified reference run (track.reference.time,
// docs/formats.md): Gold = reference, Silver = 1.15x, Bronze = 1.35x, and Author = the best AI ghost
// time when an AI ghost exists. Without a reference there are no medals.

export const MEDALS = ['author', 'gold', 'silver', 'bronze']; // best first
export const MEDAL_NAMES = { author: 'Author', gold: 'Gold', silver: 'Silver', bronze: 'Bronze' };

const pad2 = (n) => String(n).padStart(2, '0');
const toCs = (sec) => Math.round(sec * 100); // centiseconds; race times are whole 0.01 s ticks

// mm:ss.cc, or a placeholder when there is no time.
export function formatTime(sec) {
  if (sec == null || !Number.isFinite(sec)) return '--:--.--';
  const cs = toCs(sec);
  return `${pad2(Math.floor(cs / 6000))}:${pad2(Math.floor(cs / 100) % 60)}.${pad2(cs % 100)}`;
}

// Signed difference, e.g. "-0.42" (faster) or "+1.07" (slower); minutes appear only when needed.
export function formatDelta(sec) {
  const cs = Math.abs(toCs(sec));
  const sign = toCs(sec) < 0 ? '-' : '+';
  const s = Math.floor(cs / 100), c = pad2(cs % 100);
  return s >= 60 ? `${sign}${Math.floor(s / 60)}:${pad2(s % 60)}.${c}` : `${sign}${s}.${c}`;
}

// Medal target times (s) rounded to centiseconds, so the shown target is exactly the one compared.
// Returns null when the track has no reference run.
export function medalTargets(track, authorTime = null) {
  const ref = track?.reference?.time;
  if (!(ref > 0)) return null;
  const round = (t) => toCs(t) / 100;
  const t = { gold: round(ref), silver: round(1.15 * ref), bronze: round(1.35 * ref) };
  if (authorTime > 0) t.author = round(authorTime);
  return t;
}

// Best medal earned by a time. Author also requires Gold, in case the AI is slower than the reference.
export function medalFor(time, targets) {
  if (!targets || time == null) return null;
  const cs = toCs(time), ok = (m) => targets[m] != null && cs <= toCs(targets[m]);
  if (ok('gold') && ok('author')) return 'author';
  return ['gold', 'silver', 'bronze'].find(ok) ?? null;
}
