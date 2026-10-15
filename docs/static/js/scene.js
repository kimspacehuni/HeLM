/* Real setup photos from paper Fig. 4, paired with symbolic TaskSpec progress.
 * Photos are fixed illustrations of the setup, never live or replayed observations.
 */
window.HelmScene = (function () {
  'use strict';
  const list = (s) => !s || s === 'none' ? [] : s.split(',').map((x) => x.trim()).filter(Boolean);
  function parseWork(s) {
    if (!s || /^task done/.test(s)) return { done: true, got: [], goal: [] };
    let m = s.match(/^Count:\s*(\d+)\s*\(Goal:\s*(\d+)\)/);
    if (m) return { kind: 'count', n: +m[1], goal: +m[2], got: [], done: false };
    m = s.match(/^Progress:\s*(.*?)\s*\(Goal:\s*(.*?)\)\s*$/);
    if (m) return { kind: 'progress', got: list(m[1]), goal: list(m[2]), done: false };
    return { kind: 'unknown', got: [], goal: [], done: false };
  }
  function parseEpi(s) {
    if (!s || s === 'None') return null;
    let m = s.match(/^Previous_Count:\s*(\d+)/);
    if (m) return { kind: 'count', n: +m[1] };
    m = s.match(/^Previous_Progress:\s*(.*)$/);
    return m ? { kind: 'progress', items: list(m[1]) } : null;
  }
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pretty = (s) => {
    const text = s.replace(/^origin_/, 'original: ').replace('_to_', ' → ').replace('press_orange', 'orange button').replace(/_/g, ' ');
    return text.charAt(0).toUpperCase() + text.slice(1);
  };
  function photo(id, name) {
    if (!/^[AE][1-5]$/.test(id)) return '';
    return '<img class="task-photo" src="static/images/tasks/' + id.toLowerCase() + '.webp" width="334" height="280" alt="' + esc(name + ' — real-world experiment setup from paper Figure 4') + '" decoding="async" loading="lazy">';
  }
  function step(label, state, detail, color) {
    return '<div class="state-step is-' + state + '"><span class="state-symbol' + (color ? ' color-' + color : '') + '" aria-hidden="true">' +
      (state === 'done' ? '✓' : state === 'next' ? '→' : state === 'earlier' ? '↳' : '·') + '</span><span class="state-step-label">' + esc(label) + '</span>' +
      ((detail || state === 'next') ? '<span class="state-step-detail">' + esc(detail || 'Next') + '</span>' : '') + '</div>';
  }
  function render(target, task, episode, index, compact) {
    const selected = episode.steps[index];
    const work = parseWork(selected.work), epi = parseEpi(selected.epi);
    const beforeEnd = parseWork(episode.steps[Math.max(0, episode.steps.length - 2)].work);
    const goal = work.done ? beforeEnd.goal : work.goal;
    const observed = !Array.isArray(goal) ? false : goal.length === 1 && goal[0] === 'observe';
    const recorded = epi && epi.kind === 'progress' ? epi.items : [];
    const got = work.done ? (observed ? recorded : Array.isArray(goal) ? goal.filter((item) => recorded.includes(item)) : []) : work.got;
    let content = '', title = '', summary = '', context = '';
    if (work.kind === 'count' || (!Array.isArray(goal) && Number.isFinite(goal))) {
      const n = work.done ? goal : work.n;
      const earlier = task.id === 'E1' && !work.done && epi && epi.kind === 'count' ? epi.n : 0;
      title = 'Press count'; summary = n + ' / ' + goal;
      for (let i = 0; i < goal; i++) content += step('Press ' + (i + 1), i < earlier ? 'earlier' : i < n ? 'done' : i === n ? 'next' : 'pending', '', 'blue');
      if (earlier) context = earlier + ' press' + (earlier === 1 ? '' : 'es') + ' remembered from the previous episode';
    } else if (observed) {
      title = task.id === 'E4' ? 'Observe the object location' : 'Observe the human’s order';
      summary = work.done ? 'Recorded' : 'Observing';
      // Reveal only observations already in memory, never the future event sequence.
      got.forEach((item) => { content += step(pretty(item), 'done', '', /^(red|green|blue)$/.test(item) ? item : null); });
      if (!work.done) content += step(task.id === 'E4' ? 'Watch for the object' : 'Watch the next press', 'next');
    } else if (task.id === 'A3' || task.id === 'E3') {
      title = 'Window regions'; summary = got.length + ' / ' + goal.length + ' in this episode';
      ['top','middle','bottom'].forEach((region) => {
        const earlier = recorded.includes(region) && !got.includes(region);
        const state = got.includes(region) ? 'done' : earlier ? 'earlier' : goal[got.length] === region ? 'next' : 'pending';
        content += step(pretty(region), state, earlier ? 'Earlier episode' : state === 'done' ? 'Wiped' : state === 'next' ? 'Wipe next' : 'Pending');
      });
      if (task.id === 'E3' && recorded.length && !work.done) context = 'Previously wiped: ' + recorded.map(pretty).join(', ');
    } else if (task.id === 'A4' || task.id === 'E4') {
      title = task.id === 'A4' ? 'Search the drawers' : 'Retrieve from memory';
      summary = work.done ? 'Complete' : task.id === 'E4' ? 'Location recalled' : got.length + ' checked';
      ['top_left','top_right','bottom_left','bottom_right'].forEach((drawer) => {
        const remembered = task.id === 'E4' && recorded.includes(drawer);
        const state = got.includes(drawer) ? 'done' : !work.done && goal[got.length] === drawer ? 'next' : remembered ? 'earlier' : 'pending';
        content += step(pretty(drawer), state, state === 'next' ? 'Open next' : state === 'done' ? (task.id === 'E4' ? 'Opened' : work.done && drawer === got[got.length - 1] ? 'Object found' : 'Checked') : remembered ? 'Remembered' : '');
      });
      if (task.id === 'E4' && recorded.length) context = 'Object location: ' + recorded.map(pretty).join(', ');
    } else {
      title = task.id === 'A2' || task.id === 'E2' ? 'Button order' : 'Action sequence';
      summary = got.length + ' / ' + goal.length;
      goal.forEach((item, i) => { content += step(pretty(item), i < got.length ? 'done' : i === got.length ? 'next' : 'pending', '', /^(red|green|blue)$/.test(item) ? item : null); });
      if (task.id === 'E2' && recorded.length && !work.done) context = 'Remembered order: ' + recorded.map(pretty).join(' → ');
      const origin = recorded.find((item) => item.startsWith('origin_'));
      if (task.id === 'E5' && origin) context = 'Original plate: ' + pretty(origin.slice(7));
    }
    target.className = 'task-scene' + (compact ? ' is-compact' : '');
    if (target.dataset.task !== task.id || !target.querySelector('.scene-state')) {
      target.innerHTML = '<div class="scene-photo">' + photo(task.id, task.name) + '<span class="scene-photo-label">EXPERIMENT SETUP · FIG. 4</span></div><div class="scene-state"></div>';
      target.dataset.task = task.id;
    }
    target.querySelector('.scene-state').innerHTML = '<div class="scene-state-heading"><strong>' + esc(title) + '</strong><span>' + esc(summary) + '</span></div>' +
      '<div class="state-steps' + ((task.id === 'A4' || task.id === 'E4') && !observed ? ' is-drawers' : '') + '">' + content + '</div>' +
      (context && !compact ? '<div class="scene-context"><span>EPISODIC CONTEXT</span>' + esc(context) + '</div>' : '');

  }
  return {parseWork, parseEpi, photo, render};
})();
