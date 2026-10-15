/* A manual, task-specification-backed introduction to language memory. */
(function () {
  'use strict';
  const V = window.HelmViz, S = window.HelmScene;
  const task = (window.HELM_TRACES || []).find((t) => t.id === 'A2');
  const variant = task && task.variants.find((v) => v.id === 'press_RGB');
  const canvas = document.getElementById('hero-canvas');
  if (!V || !S || !variant || !canvas) return;
  const states = variant.episodes[0].steps;
  const buttons = document.querySelectorAll('[data-hero-step]');
  const progress = document.getElementById('hero-progress');
  const action = document.getElementById('hero-action');
  let index = 1;
  function draw() {
    const state = states[index];
    const size = V.fit(canvas, Math.min(230, Math.max(170, canvas.clientWidth * .48)));
    S.draw(canvas.getContext('2d'), size.w, size.h, 'A2', state.work, state.epi, 3);
  }
  function select(i) {
    index = i;
    const state = states[index];
    progress.textContent = state.cmd === 'done'
      ? state.epi.replace('Previous_Progress:', 'Progress:')
      : state.work.replace(/ \(Goal:.*\)$/, '');
    action.textContent = state.cmd === 'done' ? 'Task complete ✓' : state.cmd[0].toUpperCase() + state.cmd.slice(1) + ' →';
    buttons.forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.heroStep === index)));
    canvas.setAttribute('aria-label', 'Button sequence schematic. ' + state.work + '. Next action: ' + state.cmd + '.');
    draw();
  }
  buttons.forEach((b) => b.addEventListener('click', () => select(+b.dataset.heroStep)));
  window.addEventListener('resize', draw);
  V.onTheme(draw);
  select(index);
})();
