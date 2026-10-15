/* A manual, task-specification-backed introduction to language memory. */
(function () {
  'use strict';
  const V = window.HelmViz, S = window.HelmScene;
  const task = (window.HELM_TRACES || []).find((t) => t.id === 'A2');
  const variant = task && task.variants.find((v) => v.id === 'press_RGB');
  const scene = document.getElementById('hero-scene');
  if (!V || !S || !variant || !scene) return;
  const states = variant.episodes[0].steps;
  const buttons = document.querySelectorAll('[data-hero-step]');
  const progress = document.getElementById('hero-progress');
  const action = document.getElementById('hero-action');
  let index = 1;
  function draw() {
    S.render(scene, task, variant.episodes[0], index, true);
  }
  function select(i) {
    index = i;
    const state = states[index];
    progress.textContent = state.cmd === 'done'
      ? state.epi.replace('Previous_Progress:', 'Progress:')
      : state.work.replace(/ \(Goal:.*\)$/, '');
    action.textContent = state.cmd === 'done' ? 'Task complete ✓' : state.cmd[0].toUpperCase() + state.cmd.slice(1) + ' →';
    buttons.forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.heroStep === index)));
    draw();
  }
  buttons.forEach((b) => b.addEventListener('click', () => select(+b.dataset.heroStep)));
  select(index);
})();
