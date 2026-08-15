/* HeLM project page — chrome, task grid, and the small shared behaviours.
 *
 * The task cards used to be placeholders waiting on rollout footage. They now render
 * the same thing the inspector does, at a glance: the scene drawn from the task's real
 * memory string, stepping through the episode. One rAF loop drives all ten so a grid
 * of canvases stays cheap, and it only runs while the section is on screen.
 */
(function () {
  'use strict';

  const V = window.HelmViz, S = window.HelmScene, TRACES = window.HELM_TRACES || [];

  /* ------------------------------------------------------------------ copy */

  const DESC = {
    A1: 'Press the blue button exactly N times, then stop. Frames are visually near-identical, so the count can only come from memory.',
    A2: 'Press three coloured buttons in a given order. Requires tracking which sub-goals are already done.',
    A3: 'Wipe the bottom, middle and top regions in order. A wiped region looks much like an unwiped one.',
    A4: 'Open drawers clockwise until the object turns up, then stop — which means recalling the drawers already searched.',
    A5: 'Move the banana between plates with a button press in between: a multi-step procedure with an interleaved sub-goal.',
    E1: 'Press N times; a new instruction then asks for M presses in total. The agent must recall N to infer M − N remain.',
    E2: 'Watch the order a person presses, then reproduce that order in the next episode.',
    E3: 'Some window regions are wiped in the first episode; the next instruction only says “wipe the remaining part”.',
    E4: 'Observe which drawer holds the object, then open that drawer directly — without A4’s sequential search.',
    E5: 'After two transfers, return the banana to the plate it began on — which is no longer observable anywhere.',
  };

  /* ------------------------------------------------------------- task cards */

  const grid = document.getElementById('task-grid');
  const cards = [];

  function buildCards() {
    if (!grid || !TRACES.length) return;

    TRACES.forEach((task) => {
      /* Show the longest variant. The first one is often the degenerate case — A1's is
         a single press — which makes for a card that shows almost nothing happening. */
      const variant = task.variants.reduce((best, v) => {
        const n = v.episodes.reduce((s, e) => s + e.steps.length, 0);
        return n > best.n ? { v: v, n: n } : best;
      }, { v: task.variants[0], n: 0 }).v;

      const steps = [];
      variant.episodes.forEach((ep, ei) =>
        ep.steps.forEach((st) => steps.push({ st: st, ep: ei })));

      const card = document.createElement('article');
      card.className = 'task-card';
      card.dataset.group = task.group;
      card.innerHTML =
        '<div class="task-media"><span class="task-id">' + task.id + '</span>' +
        '<canvas class="task-canvas"></canvas></div>' +
        '<div class="task-body">' +
          '<div class="task-name">' + task.name + '</div>' +
          '<div class="task-desc">' + DESC[task.id] + '</div>' +
          '<code class="task-mem"></code>' +
        '</div>';
      grid.appendChild(card);

      cards.push({
        task: task, steps: steps, i: 0,
        cv: card.querySelector('.task-canvas'),
        mem: card.querySelector('.task-mem'),
        el: card,
      });
    });

    cards.forEach(paint);
  }

  function paint(c) {
    const m = c.steps[c.i];
    const size = V.fit(c.cv, Math.round(c.cv.clientWidth * 0.72));
    S.draw(c.cv.getContext('2d'), size.w, size.h, c.task.id, m.st.work, m.st.epi, 0);
    /* Working memory is the line that changes; episodic only matters at a boundary. */
    c.mem.textContent = m.st.epi && m.st.epi !== 'None' && m.st.work.indexOf('done') === 0
      ? m.st.epi : m.st.work;
  }

  function stepCards() {
    cards.forEach((c) => { c.i = (c.i + 1) % c.steps.length; paint(c); });
  }

  /* ------------------------------------------------------------------ tabs */

  const tabs = document.querySelectorAll('.tab');
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.setAttribute('aria-selected', String(t === tab)));
      const f = tab.dataset.filter;
      cards.forEach((c) => {
        c.el.style.display = (f === 'all' || c.task.group === f) ? '' : 'none';
      });
    });
  });

  /* ------------------------------------------------------- nav + scroll spy */

  const nav = document.getElementById('nav');
  const navLinks = nav ? Array.from(nav.querySelectorAll('a[href^="#"]')) : [];
  const sections = navLinks
    .map((a) => document.getElementById(a.getAttribute('href').slice(1)))
    .filter(Boolean);

  function spy() {
    if (!sections.length) return;
    const y = window.scrollY + window.innerHeight * 0.3;
    let active = 0;
    sections.forEach((s, i) => { if (s.offsetTop <= y) active = i; });
    navLinks.forEach((a, i) => a.classList.toggle('is-on', i === active));
    if (nav) nav.classList.toggle('is-stuck', window.scrollY > 120);
  }

  /* ---------------------------------------------------------------- reveals */

  /* Sections fade up once. The stylesheet already neutralises this under
     prefers-reduced-motion, so nothing here needs to branch on it. */
  function reveals() {
    const targets = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) {
      targets.forEach((t) => t.classList.add('is-in'));
      return;
    }
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add('is-in'); io.unobserve(e.target); }
      });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });
    targets.forEach((t) => io.observe(t));
  }

  /* ------------------------------------------------------------------ theme */

  const SUN =
    '<svg viewBox="0 0 24 24" stroke-linecap="round"><circle cx="12" cy="12" r="4.5"/>' +
    '<path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4"/></svg>';
  const MOON =
    '<svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M21 13.2A9 9 0 0 1 10.8 3a9 9 0 1 0 10.2 10.2z"/></svg>';

  const root = document.documentElement;
  const toggle = document.getElementById('theme-toggle');

  function currentTheme() {
    return root.dataset.theme ||
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  }
  function paintToggle() {
    if (toggle) toggle.innerHTML = currentTheme() === 'dark' ? SUN : MOON;
  }

  const saved = localStorage.getItem('helm-theme');
  if (saved) root.dataset.theme = saved;
  paintToggle();

  if (toggle) {
    toggle.addEventListener('click', () => {
      const next = currentTheme() === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      localStorage.setItem('helm-theme', next);
      paintToggle();
    });
  }

  /* ------------------------------------------------------- visitor map guard */

  /* The footer card is styled for the mapmyvisitors widget, so it renders as an empty
   * white bar whenever the widget does not inject anything. That is not a rare edge
   * case: content blockers block this tracker by default, so a good share of visitors
   * would see the empty bar. Hide the card unless the widget actually rendered.
   *
   * map.js writes synchronously, but it fetches its data first, so re-check once
   * before giving up. */
  function pruneVisitorMap() {
    const box = document.querySelector('.visitor-map');
    if (!box) return;
    const rendered = () =>
      Array.from(box.children).some((el) => el.tagName !== 'SCRIPT');
    if (rendered()) return;
    setTimeout(() => { if (!rendered()) box.style.display = 'none'; }, 3000);
  }

  /* ------------------------------------------------------------------- boot */

  buildCards();
  reveals();
  spy();

  window.addEventListener('scroll', spy, { passive: true });
  window.addEventListener('load', pruneVisitorMap);
  window.addEventListener('resize', () => cards.forEach(paint));
  if (V) V.onTheme(() => cards.forEach(paint));

  /* One timer for the whole grid, paused while the section is off screen. */
  if (grid && cards.length && V && !V.reduceMotion()) {
    let acc = 0;
    V.animate(grid, (dt) => {
      acc += dt;
      if (acc >= 1.6) { acc = 0; stepCards(); }
    });
  }
})();
