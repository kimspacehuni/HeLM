/* Navigation, task discovery and theme preferences. */
(function () {
  'use strict';
  const V = window.HelmViz, S = window.HelmScene;
  const traces = window.HELM_TRACES || [];
  const DESC = {
    A1: 'Count completed presses and stop at the requested number.',
    A2: 'Track which buttons are already pressed in a given sequence.',
    A3: 'Remember which window regions have been wiped.',
    A4: 'Keep track of searched drawers while looking for an object.',
    A5: 'Follow a multi-step pick, place and button-press procedure.',
    E1: 'Recall earlier presses to work out how many more are needed.',
    E2: 'Observe a person’s button order and reproduce it in the next episode.',
    E3: 'Remember the wiped regions and clean only the remaining area.',
    E4: 'Recall an object’s location and open the correct drawer directly.',
    E5: 'Remember the original plate and return the object to it.',
  };
  const grid = document.getElementById('task-grid');
  const cards = [];
  const filters = document.querySelectorAll('[data-filter]');
  function paint(card) {
    if (card.el.hidden) return;
    const size = V.fit(card.canvas, Math.min(210, Math.round(card.canvas.clientWidth * .62)));
    S.draw(card.canvas.getContext('2d'), size.w, size.h, card.task.id, card.state.work, card.state.epi, 0);
  }
  function filter(group) {
    filters.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.filter === group)));
    cards.forEach((c) => { c.el.hidden = group !== 'all' && c.task.group !== group; paint(c); });
  }
  function explore(task) {
    window.dispatchEvent(new CustomEvent('helm:select-task', { detail: { id: task.id } }));
    const explorer = document.getElementById('inspector');
    if (explorer) {
      document.getElementById('memory').scrollIntoView({ behavior: V.reduceMotion() ? 'auto' : 'smooth' });
      explorer.focus({ preventScroll: true });
    }
  }
  if (grid && V && S) {
    traces.forEach((task) => {
      const state = task.variants[0].episodes[0].steps[0];
      const article = document.createElement('article');
      article.className = 'task-card';
      const media = document.createElement('div'); media.className = 'task-media';
      const badge = document.createElement('span'); badge.className = 'task-id'; badge.textContent = task.id;
      const canvas = document.createElement('canvas'); canvas.className = 'task-canvas'; canvas.setAttribute('aria-hidden', 'true');
      media.append(badge, canvas);
      const body = document.createElement('div'); body.className = 'task-body';
      const title = document.createElement('h3'); title.className = 'task-name'; title.textContent = task.name;
      const description = document.createElement('p'); description.className = 'task-desc'; description.textContent = DESC[task.id];
      const memory = document.createElement('code'); memory.className = 'task-mem'; memory.textContent = state.work;
      const button = document.createElement('button'); button.type = 'button'; button.className = 'task-explore';
      button.textContent = 'Explore the memory ↗'; button.setAttribute('aria-label', 'Explore memory for ' + task.name);
      button.addEventListener('click', () => explore(task));
      body.append(title, description, memory, button); article.append(media, body); grid.append(article);
      cards.push({ task, state, canvas, el: article });
    });
    filter('intra');
    filters.forEach((button) => button.addEventListener('click', () => filter(button.dataset.filter)));
    window.addEventListener('resize', () => cards.forEach(paint));
    V.onTheme(() => cards.forEach(paint));
  }
  const root = document.documentElement;
  const toggle = document.getElementById('theme-toggle');
  const sun = '<svg viewBox="0 0 24 24" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M19 5l-1.5 1.5M6.5 17.5 5 19"/></svg>';
  const moon = '<svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round"><path d="M21 13.2A9 9 0 0 1 10.8 3a9 9 0 1 0 10.2 10.2z"/></svg>';
  function dark() { return root.dataset.theme ? root.dataset.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches; }
  function paintToggle() {
    if (!toggle) return;
    toggle.innerHTML = dark() ? sun : moon;
    toggle.setAttribute('aria-label', 'Switch to ' + (dark() ? 'light' : 'dark') + ' theme');
  }
  if (toggle) toggle.addEventListener('click', () => {
    root.dataset.theme = dark() ? 'light' : 'dark';
    try { localStorage.setItem('helm-theme', root.dataset.theme); } catch (_) {}
    paintToggle();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', paintToggle);
  paintToggle();
  const navLinks = Array.from(document.querySelectorAll('.nav-links a'));
  const sections = navLinks.map((a) => document.querySelector(a.getAttribute('href')));
  function spy() {
    const threshold = window.scrollY + window.innerHeight * .3;
    let active = 0;
    sections.forEach((section, i) => { if (section && section.offsetTop <= threshold) active = i; });
    navLinks.forEach((a, i) => {
      a.classList.toggle('is-on', i === active);
      if (i === active) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current');
    });
  }
  window.addEventListener('scroll', spy, { passive: true });
  spy();
  function revealHash() {
    if (!location.hash) return;
    const target = document.getElementById(location.hash.slice(1));
    if (target && target.tagName === 'DETAILS') target.open = true;
  }
  window.addEventListener('hashchange', revealHash);
  revealHash();
  function pruneVisitorMap() {
    const box = document.querySelector('.visitor-map');
    if (!box) return;
    const rendered = () => Array.from(box.children).some((el) => el.tagName !== 'SCRIPT');
    if (!rendered()) setTimeout(() => { if (!rendered()) box.hidden = true; }, 3000);
  }
  window.addEventListener('load', pruneVisitorMap);
})();
