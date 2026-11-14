// hud.js — the race overlay: timer, speed, checkpoints, split deltas, medal targets, messages, and the
// finish screen.
//
// Global context: main.js feeds the HUD from the RaceSession once per frame (values) and per event
// (checkpoint, respawn, finish). Elements are declared in index.html; this module only fills them.

import { formatTime, formatDelta, MEDALS, MEDAL_NAMES } from './timing.js';

const $ = (id) => document.getElementById(id);

export class Hud {
  constructor() {
    this.root = $('hud');
    this.el = {
      mode: $('hud-mode'), timer: $('hud-timer'), split: $('hud-split'), cp: $('hud-cp'),
      speed: $('hud-speed-value'), medals: $('hud-medals'), best: $('hud-best'), message: $('hud-message'),
      ai: $('hud-ai'),
    };
    this.finish = $('finish');
    this.splitUntil = 0; // performance.now() after which the split delta hides
    this.messageUntil = 0;
  }

  // Prepare for a race: mode ('race' | 'race-ai' | 'watch') and its label, medal targets (or none), personal best.
  start({ mode, modeLabel, targets, bestTime, cpTotal }) {
    this.root.hidden = false;
    this.root.dataset.mode = mode;
    this.finish.hidden = true;
    this.el.mode.textContent = modeLabel;
    this.cpTotal = cpTotal;
    this.el.best.textContent = bestTime != null ? `Best ${formatTime(bestTime)}` : 'No best yet';
    this.el.medals.replaceChildren(...(targets ? MEDALS.filter((m) => targets[m] != null).map((m) => {
      const li = document.createElement('li');
      li.innerHTML = `<span class="medal ${m}"></span><span>${MEDAL_NAMES[m]}</span><span class="t">${formatTime(targets[m])}</span>`;
      return li;
    }) : []));
    this.el.medals.hidden = !targets;
    this.el.split.textContent = '';
    this.el.message.textContent = '';
    this.el.ai.textContent = '';
  }

  hide() { this.root.hidden = true; this.finish.hidden = true; }

  // Per-frame values. cp: checkpoints passed (the finish counts as the last one).
  update({ time, speedKmh, cp, ai }) {
    const now = performance.now();
    this.el.timer.textContent = formatTime(time);
    this.el.speed.textContent = String(Math.round(speedKmh));
    const total = this.cpTotal - 1; // intermediate checkpoints; the last one is the finish
    this.el.cp.textContent = `Checkpoint ${Math.min(cp, total)} / ${total}`;
    if (now > this.splitUntil) this.el.split.textContent = '';
    if (now > this.messageUntil) this.el.message.textContent = '';
    if (ai != null) this.el.ai.textContent = ai;
  }

  // Show "+1.23" (red, slower) or "-0.45" (green, faster) against the best run at a checkpoint.
  showSplit(label, delta) {
    this.el.split.textContent = delta == null ? label : `${label}  ${formatDelta(delta)}`;
    this.el.split.className = delta == null ? '' : delta <= 0 ? 'faster' : 'slower';
    this.splitUntil = performance.now() + 2500;
  }

  message(text, ms = 2000) {
    this.el.message.textContent = text;
    this.messageUntil = performance.now() + ms;
  }

  // Finish screen. splits/bestSplits in seconds (last entry = finish); medal: key or null.
  showFinish({ title, time, splits, bestSplits, medal, note, extra }) {
    $('finish-title').textContent = title;
    $('finish-time').textContent = formatTime(time);
    const medalEl = $('finish-medal');
    medalEl.hidden = !medal;
    if (medal) medalEl.innerHTML = `<span class="medal ${medal}"></span>${MEDAL_NAMES[medal]} medal`;
    $('finish-note').textContent = note ?? '';
    $('finish-extra').textContent = extra ?? '';
    const rows = splits.map((t, k) => {
      const tr = document.createElement('tr');
      const d = bestSplits?.[k] != null ? t - bestSplits[k] : null;
      const name = k === splits.length - 1 ? 'Finish' : `Checkpoint ${k + 1}`;
      tr.innerHTML = `<td>${name}</td><td>${formatTime(t)}</td><td class="${d == null ? '' : d <= 0 ? 'faster' : 'slower'}">${d == null ? '' : formatDelta(d)}</td>`;
      return tr;
    });
    $('finish-splits').replaceChildren(...rows);
    this.finish.hidden = false;
  }
}
