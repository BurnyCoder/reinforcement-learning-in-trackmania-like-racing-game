// menu.js — the start screen: track list with best times and medals, and the selected track's panel
// (medal targets, ghost toggles, AI model choice, mode buttons).
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
      if (e.target.id === 'model-select') this.cb.onModel(e.target.value);
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

  // d: { title, name, info, best, medal, targets, ghosts: [{ key, label, on, disabled }], models, model, hasModels }
  renderDetail(d) {
    const targets = d.targets
      ? `<ul class="targets">${MEDALS.filter((m) => d.targets[m] != null).map((m) => `<li>${medalBadge(m)}<span>${MEDAL_NAMES[m]}</span><span class="t">${formatTime(d.targets[m])}</span></li>`).join('')}</ul>`
      : '<p class="muted small">No medal times: this track has no certified reference run yet.</p>';
    const ghosts = d.ghosts.map((g) => `
      <label class="${g.disabled ? 'disabled' : ''}"><input type="checkbox" data-ghost="${esc(g.key)}" ${g.on ? 'checked' : ''} ${g.disabled ? 'disabled' : ''}> ${esc(g.label)}</label>`).join('');
    const models = d.hasModels ? `
      <label class="model">AI driver <select id="model-select">${d.models.map((m) => `<option value="${esc(m.name)}" ${m.name === d.model ? 'selected' : ''} title="${esc(m.description ?? '')}">${esc(m.name)}</option>`).join('')}</select></label>` : '';
    this.detail.innerHTML = `
      <h2>${esc(d.title)}</h2>
      <p class="muted">${esc(d.name)} · ${esc(d.info)}</p>
      <p class="best-line">Your best <b>${formatTime(d.best)}</b> ${d.medal ? medalBadge(d.medal) : ''}</p>
      ${targets}
      <fieldset class="ghosts"><legend>Ghosts</legend>${ghosts}</fieldset>
      ${models}
      <div class="actions">
        <button data-mode="race" class="primary">Race <kbd>Enter</kbd></button>
        ${d.hasModels ? '<button data-mode="race-ai">Race vs AI</button><button data-mode="watch">Watch AI</button>' : ''}
        ${d.hasReplay ? '<button data-mode="replay" title="Replay a recorded AI run: the first ticked AI ghost, or the first one">Watch replay</button>' : ''}
      </div>`;
  }
}
