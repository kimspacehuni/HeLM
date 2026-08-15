/* Shared canvas helpers for the page's diagrams.
 *
 * Every diagram here is drawn rather than exported as an image, for one reason that
 * matters on this page specifically: it has a light/dark toggle, and a rasterised PDF
 * crop can only ever sit on a forced white plate. Canvas colours are read out of the
 * stylesheet at draw time, so a theme flip repaints instead of looking broken.
 */
window.HelmViz = (function () {
  'use strict';

  /* Read a CSS custom property off :root. Diagrams call this every frame, so the
     stylesheet stays the single source of truth for colour in both themes. */
  function css(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function palette() {
    return {
      ink:    css('--fg', '#16181d'),
      text:   css('--fg-muted', '#5b6472'),
      faint:  css('--fg-faint', '#8a93a2'),
      line:   css('--border', '#e3e6ec'),
      track:  css('--viz-track', '#eef1f5'),
      panel:  css('--bg-card', '#ffffff'),
      accent: css('--accent', '#2f5fe0'),
      ours:   css('--c-ours', '#4a6fd4'),
      key:    css('--c-keyframe', '#e0932a'),
      pcmb:   css('--c-pcmb', '#4f9d69'),
      naive:  css('--c-naive', '#c2452f'),
      detect: css('--c-detect', '#8b5cf6'),
      update: css('--c-update', '#2f6df0'),
      blue:   css('--c-blue', '#2f6df0'),
      green:  css('--c-green', '#3f9e56'),
      red:    css('--c-red', '#d94a3d'),
      white:  css('--c-plate-white', '#e9edf3'),
      orange: css('--c-orange', '#e0932a'),
      banana: css('--c-banana', '#e8c33c'),
    };
  }

  /* Size a canvas for the device pixel ratio. Returns CSS-pixel dimensions, so all
     drawing code can work in layout units and ignore the ratio entirely. */
  function fit(cv, cssHeight) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || cv.parentElement.clientWidth || 600;
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(cssHeight * dpr);
    cv.style.height = cssHeight + 'px';
    cv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: w, h: cssHeight };
  }

  function rr(ctx, x, y, w, h, r) {
    r = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  const ease = (t) => t * t * (3 - 2 * t);              /* smoothstep */
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;

  function mix(c1, c2, t) {
    const p = (c) => {
      const s = c.replace('#', '');
      const n = s.length === 3 ? s.split('').map((x) => x + x).join('') : s;
      return [0, 2, 4].map((i) => parseInt(n.slice(i, i + 2), 16));
    };
    const [a, b] = [p(c1), p(c2)];
    return 'rgb(' + a.map((v, i) => Math.round(lerp(v, b[i], t))).join(',') + ')';
  }

  /* Repaint on a theme flip. Covers both paths: the explicit toggle stamps
     data-theme on <html>, and the untouched default follows the OS setting. */
  function onTheme(fn) {
    new MutationObserver(fn).observe(document.documentElement, {
      attributes: true, attributeFilter: ['data-theme'],
    });
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener ? mq.addEventListener('change', fn) : mq.addListener(fn);
  }

  /* Run an rAF loop only while the element is on screen — several of these diagrams
     animate continuously and there is no reason to burn frames off-screen. */
  function animate(el, step) {
    let visible = true, running = false, last = null;
    function frame(ts) {
      if (last === null) last = ts;
      const dt = Math.min(0.05, (ts - last) / 1000);
      last = ts;
      step(dt);
      if (visible) requestAnimationFrame(frame);
      else running = false;
    }
    function start() {
      if (running) return;
      running = true; last = null;
      requestAnimationFrame(frame);
    }
    /* Paint one frame up front. Otherwise a diagram below the fold stays blank until
       it is scrolled to, and briefly shows an empty canvas on the way in. */
    step(0);

    if ('IntersectionObserver' in window) {
      new IntersectionObserver((es) => {
        visible = es[0].isIntersecting;
        if (visible) start();
      }, { threshold: 0 }).observe(el);
    } else start();
    return { redraw: () => step(0) };
  }

  const reduceMotion = () =>
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  return { css, palette, fit, rr, ease, clamp, lerp, mix, onTheme, animate, reduceMotion };
})();
