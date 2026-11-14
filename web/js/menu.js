// menu.js — the start screen: track list with best times and medals, and the selected track's panel
// (medal targets, the AI drivers table with times, medals and ghost toggles, mode buttons).
//
// Global context: a passive view. main.js owns the data (index.json, track files, localStorage) and
// re-renders the menu through these functions; user actions come back through the callbacks.

import { formatTime, MEDALS, MEDAL_NAMES } from './timing.js';

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const medalBadge = (m) => (m ? `<span class="medal ${m}" title="${MEDAL_NAMES[m]} medal"></span>` : '<span class="medal none"></span>');

export class Menu {
  // callbacks: onSelect(name), onStart(mode), onGhostToggle(key, on), onModel(name)
  constructor(callbacks) {
    this.root = document.getElementById('menu');
    this.list = document.getElementById('track-list');
    this.detail = document.getElementById('track-detail');
    this.cb = callbacks;
    this.root.addEventListener('click', (e) => {
      const item = e.target.closest('[data-track]');
      if (item) this.cb.onSelect(item.dataset.track);
      const button = e.target.closest('button[data-mode]');
      if (button) this.cb.onStart(button.dataset.mode);
    });
    this.root.addEventListener('change', (e) => {
      if (e.target.dataset.ghost) this.cb.onGhostToggle(e.target.dataset.ghost, e.target.checked);
      if (e.target.name === 'driver') this.cb.onModel(e.target.value);
    });
  }

  show() { this.root.hidden = false; }
  hide() { this.root.hidden = true; }

  // rows: [{ name, title, best (s or null), medal (key or null) }]
  renderList(rows, selected) {
    this.list.innerHTML = rows.map((r, i) => `
      <li data-track="${esc(r.name)}" class="${r.name === selected ? 'selected' : ''}">
        <span class="num">${String(i + 1).padStart(2, '0')}</span>
        <span class="names"><span class="title">${esc(r.title)}</span><span class="name">${esc(r.name)}</span></span>
        <span class="best">${formatTime(r.best)}</span>${medalBadge(r.medal)}
      </li>`).join('');
  }

  // d: { title, name, info, best, medal, targets, bestGhost: { on, disabled }, canLive, canReplay,
  //      drivers: [{ name, description, tag, selected, live, finished, timeText, medal, ghostKey, ghostOn }] }
  // The AI drivers table is the panel's main part: one row per driver, best run first (main.js driversOf);
  // the chosen row drives "Race vs AI", "Watch AI" and "Watch replay". The buttons stay pinned at the bottom.
  renderDetail(d) {
    const sameTrack = d.name === this.detailName, top = this.detail.scrollTop; // keep the scroll when re-rendering a choice
    this.detailName = d.name;
    const targets = d.targets
      ? `<ul class="targets">${MEDALS.filter((m) => d.targets[m] != null).map((m) => `<li>${medalBadge(m)}<span>${MEDAL_NAMES[m]}</span><span class="t">${formatTime(d.targets[m])}</span></li>`).join('')}</ul>`
      : '<p class="muted small">No medal times: this track has no certified reference run yet.</p>';
    const rows = d.drivers.map((r) => `
      <li class="driver${r.selected ? ' selected' : ''}${r.live ? '' : ' recorded'}" data-driver="${esc(r.name)}">
        <label class="pick"><input type="radio" name="driver" value="${esc(r.name)}" ${r.selected ? 'checked' : ''}>
          <span class="who"><span class="dname">${esc(r.name)}</span>
            <span class="desc">${esc(r.description)}${r.tag ? ` · <em>${esc(r.tag)}</em>` : ''}</span></span></label>
        <span class="dtime${r.finished ? '' : ' out'}">${esc(r.timeText)}</span>${medalBadge(r.medal)}
        ${r.ghostKey ? `<label class="dghost" title="Show this run as a ghost"><input type="checkbox" data-ghost="${esc(r.ghostKey)}" ${r.ghostOn ? 'checked' : ''}></label>` : '<span class="dghost"></span>'}
      </li>`).join('');
    const drivers = d.drivers.length ? `
      <section class="drivers">
        <h3>AI drivers <span class="muted small">time on this track · medal · ghost</span></h3>
        <ol>${rows}</ol>
      </section>` : '';
    const live = d.canLive ? '' : 'disabled title="The chosen driver is a recorded run only"';
    this.detail.innerHTML = `
      <h2>${esc(d.title)}</h2>
      <p class="muted">${esc(d.name)} · ${esc(d.info)}</p>
      <p class="best-line">Your best <b>${formatTime(d.best)}</b> ${d.medal ? medalBadge(d.medal) : ''}
        <label class="ghost-best${d.bestGhost.disabled ? ' disabled' : ''}"><input type="checkbox" data-ghost="best" ${d.bestGhost.on ? 'checked' : ''} ${d.bestGhost.disabled ? 'disabled' : ''}> as ghost</label></p>
      ${targets}
      ${drivers}
      <div class="actions">
        <button data-mode="race" class="primary">Race <kbd>Enter</kbd></button>
        ${d.drivers.length ? `<button data-mode="race-ai" ${live}>Race vs AI</button><button data-mode="watch" ${live}>Watch AI</button>
        <button data-mode="replay" ${d.canReplay ? 'title="Play the chosen driver&#39;s recorded official run"' : 'disabled title="The chosen driver has no recorded run on this track"'}>Watch replay</button>` : ''}
      </div>`;
    this.detail.scrollTop = sameTrack ? top : 0;
  }
}
