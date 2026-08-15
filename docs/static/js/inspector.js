/* Memory inspector — replays HeLM's real memory states step by step.
 *
 * Everything shown here is read straight out of static/js/traces.js, which is built
 * from helm_datasets/taskspecs/ by docs/tools/build_traces.py. All 51 task variants
 * from the paper are present; none of it is illustrative.
 *
 * Playback mirrors the paper's two-stage inference rather than just sliding a value:
 * each step dwells in DETECT while the HLP watches for its event, then flips briefly
 * to UPDATE when that event fires and the memory is rewritten.
 */
(function () {
  'use strict';

  const V = window.HelmViz, S = window.HelmScene, DATA = window.HELM_TRACES;
  const root = document.getElementById('inspector');
  if (!root || !DATA || !V || !S) return;

  const el = (id) => root.querySelector('#' + id);
  const tabsBox = el('insp-tabs');
  const variantSel = el('insp-variant');
  const instrBox = el('insp-instruction');
  const canvas = el('insp-canvas');
  const memBox = el('insp-mem');
  const lineBox = el('insp-timeline');
  const playBtn = el('insp-play');
  const ctx = canvas.getContext('2d');

  const DWELL = 0.78;          /* fraction of a step spent in DETECT */
  const STEP_MS = 1750;

  /* Open on A2 by default: it is the sequence from the teaser figure, and unlike A1's
     shortest variant it has enough steps that the timeline reads as a timeline.
     ?task=E2 overrides it, so a specific task can be linked to directly. */
  const wanted = new URLSearchParams(location.search).get('task');
  let task = (wanted && DATA.find((t) => t.id === wanted.toUpperCase())) ||
             DATA.find((t) => t.id === 'A2') || DATA[0];
  let variant = task.variants[0], moments = [];
  let idx = 0, phase = 0, playing = true, dragging = false;

  /* ------------------------------------------------------------- flattening */

  /* One flat list of moments so the scrubber can cross episode boundaries. Each
     moment keeps its episode index, which is what makes the reset/promote visible. */
  function flatten(v) {
    const out = [];
    v.episodes.forEach((ep, ei) => {
      ep.steps.forEach((st, si) => {
        out.push({
          ep: ei, si: si, step: st,
          epStart: si === 0 && ei > 0,
          instruction: ep.instruction,
          lastGoal: lastGoalOf(ep),
        });
      });
    });
    return out;
  }

  /* The terminal state reads "task done (None)", which drops the goal. Carry the
     previous step's goal forward so the scene can still draw a finished episode. */
  function lastGoalOf(ep) {
    for (let i = ep.steps.length - 1; i >= 0; i--) {
      const w = S.parseWork(ep.steps[i].work);
      if (!w.done) return w.kind === 'count' ? w.goal : w.goal.length;
    }
    return 0;
  }

  /* ------------------------------------------------------------------ chrome */

  function buildTabs() {
    tabsBox.innerHTML = '';
    ['intra', 'inter'].forEach((grp) => {
      const wrap = document.createElement('div');
      wrap.className = 'insp-tabgroup';
      const cap = document.createElement('span');
      cap.className = 'insp-tabcap';
      cap.textContent = grp === 'intra' ? 'Intra-episode' : 'Inter-episode';
      wrap.appendChild(cap);
      DATA.filter((t) => t.group === grp).forEach((t) => {
        const b = document.createElement('button');
        b.className = 'insp-tab';
        b.type = 'button';
        b.textContent = t.id;
        b.title = t.name;
        b.setAttribute('aria-pressed', String(t === task));
        b.addEventListener('click', () => selectTask(t));
        wrap.appendChild(b);
      });
      tabsBox.appendChild(wrap);
    });
  }

  function syncTabs() {
    let i = 0;
    const order = DATA.filter((t) => t.group === 'intra').concat(DATA.filter((t) => t.group === 'inter'));
    tabsBox.querySelectorAll('.insp-tab').forEach((b) => {
      b.setAttribute('aria-pressed', String(order[i++] === task));
    });
  }

  function buildVariants() {
    variantSel.innerHTML = '';
    task.variants.forEach((v, i) => {
      const o = document.createElement('option');
      o.value = String(i);
      o.textContent = v.label;
      variantSel.appendChild(o);
    });
    variantSel.value = String(task.variants.indexOf(variant));
  }

  function buildTimeline() {
    lineBox.innerHTML = '';
    moments.forEach((m, i) => {
      if (m.epStart) {
        const div = document.createElement('span');
        div.className = 'insp-epdiv';
        div.innerHTML = '<i></i><b>new instruction</b>';
        lineBox.appendChild(div);
      }
      const d = document.createElement('button');
      d.className = 'insp-dot';
      d.type = 'button';
      d.dataset.i = String(i);
      d.setAttribute('aria-label', 'step ' + (m.si + 1) + ' of episode ' + (m.ep + 1));
      d.addEventListener('click', () => { goTo(i); pause(); });
      lineBox.appendChild(d);
    });
  }

  /* ------------------------------------------------------------------ render */

  function memRow(name, value, changed, tone) {
    return '<div class="mem-row' + (changed ? ' is-changed' : '') + (tone ? ' t-' + tone : '') + '">' +
      '<span class="mem-key">' + name + '</span>' +
      '<span class="mem-val">' + escape(value) + '</span></div>';
  }

  function escape(s) {
    return String(s == null ? '—' : s).replace(/[&<>"]/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  function render() {
    const m = moments[idx], prev = idx > 0 ? moments[idx - 1] : null;
    const st = m.step;
    const detecting = phase < DWELL;
    const atEnd = st.event == null;

    instrBox.textContent = m.instruction;
    instrBox.dataset.ep = 'Episode ' + (m.ep + 1);

    const workChanged = prev && prev.step.work !== st.work;
    const epiChanged = prev && prev.step.epi !== st.epi;
    const reset = m.epStart;

    memBox.innerHTML =
      '<div class="mem-head">' +
        '<span class="mem-mode ' + (atEnd ? 'is-done' : detecting ? 'is-detect' : 'is-update') + '">' +
          (atEnd ? 'DONE' : detecting ? 'DETECT' : 'UPDATE') + '</span>' +
        '<span class="mem-step">t = ' + m.si + '</span>' +
      '</div>' +
      memRow('Event', atEnd ? '—' : (detecting ? 'watching for “' + st.event + '”' : st.event), !detecting && !atEnd, 'event') +
      memRow('Action_Command', st.cmd, prev && prev.step.cmd !== st.cmd) +
      '<div class="mem-sep"><span>Language memory ' + (reset ? '· working resets, episodic is promoted' : '') + '</span></div>' +
      memRow('Working_Memory', st.work, workChanged, 'work') +
      memRow('Episodic_Context', st.epi, epiChanged, 'epi');

    lineBox.querySelectorAll('.insp-dot').forEach((d) => {
      const i = +d.dataset.i;
      d.classList.toggle('is-on', i === idx);
      d.classList.toggle('is-past', i < idx);
    });

    drawScene();
  }

  function drawScene() {
    const m = moments[idx];
    const size = V.fit(canvas, Math.max(205, Math.min(265, canvas.clientWidth * 0.55)));
    S.draw(ctx, size.w, size.h, task.id, m.step.work, m.step.epi, m.lastGoal);
  }

  /* ---------------------------------------------------------------- playback */

  function goTo(i) {
    idx = V.clamp(i, 0, moments.length - 1);
    phase = 0;
    render();
  }

  function pause() { playing = false; syncPlay(); }
  function syncPlay() {
    playBtn.setAttribute('aria-pressed', String(playing));
    playBtn.textContent = playing ? '❚❚' : '▶';
    playBtn.title = playing ? 'Pause' : 'Play';
  }

  function selectTask(t) {
    task = t; variant = t.variants[0];
    moments = flatten(variant);
    buildVariants(); buildTimeline(); syncTabs(); goTo(0);
  }

  /* Drag anywhere along the timeline to scrub, matching the feel of dragging a clip. */
  function scrubFrom(e) {
    const dots = Array.from(lineBox.querySelectorAll('.insp-dot'));
    if (!dots.length) return;
    let best = 0, bestD = Infinity;
    dots.forEach((d, i) => {
      const r = d.getBoundingClientRect();
      const dist = Math.abs(e.clientX - (r.left + r.width / 2));
      if (dist < bestD) { bestD = dist; best = i; }
    });
    goTo(best);
  }

  lineBox.addEventListener('pointerdown', (e) => {
    dragging = true; pause(); scrubFrom(e);
    try { lineBox.setPointerCapture(e.pointerId); } catch (_) {}
  });
  lineBox.addEventListener('pointermove', (e) => { if (dragging) scrubFrom(e); });
  ['pointerup', 'pointercancel'].forEach((ev) =>
    lineBox.addEventListener(ev, () => { dragging = false; }));

  playBtn.addEventListener('click', () => { playing = !playing; syncPlay(); });
  variantSel.addEventListener('change', () => {
    variant = task.variants[+variantSel.value];
    moments = flatten(variant);
    buildTimeline(); goTo(0);
  });

  root.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') { goTo(idx + 1); pause(); e.preventDefault(); }
    if (e.key === 'ArrowLeft') { goTo(idx - 1); pause(); e.preventDefault(); }
  });

  /* ------------------------------------------------------------------- boot */

  buildTabs();
  moments = flatten(variant);
  buildVariants(); buildTimeline(); render(); syncPlay();

  V.onTheme(drawScene);
  window.addEventListener('resize', drawScene);

  if (V.reduceMotion()) { playing = false; syncPlay(); }

  V.animate(root, (dt) => {
    if (playing && !dragging && dt > 0) {
      const wasDetect = phase < DWELL;
      phase += dt * 1000 / STEP_MS;
      if (phase >= 1) { phase = 0; idx = (idx + 1) % moments.length; render(); }
      else if (wasDetect !== (phase < DWELL)) render();     /* only on a mode flip */
    }
  });
})();
