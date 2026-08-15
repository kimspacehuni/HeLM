/* The two result diagrams.
 *
 * 1. Success rates, with a toggle between the subtask-level score and the strict one.
 *    The strict metric is the interesting half of the story and the paper leaves it in
 *    an appendix table, so animating between the two makes the collapse legible:
 *    HeLM barely moves, every baseline falls off a cliff.
 *
 * 2. Inference cost against task length, driven by a slider.
 */
(function () {
  'use strict';
  const V = window.HelmViz;
  if (!V) return;

  /* ------------------------------------------------------- success rate chart */

  /* Subtask-level from Fig. 5; strict from Tab. 7. */
  const GROUPS = ['Average', 'A1', 'A2', 'A3', 'A4', 'A5'];
  const METHODS = [
    { key: 'ours', name: 'HeLM (Ours)',
      subtask: [85.7, 100, 92.2, 96.6, 66.6, 73.3],
      strict:  [81.3, 100, 87.8, 96.6, 66.7, 55.6] },
    { key: 'key', name: 'Keyframe-HLP',
      subtask: [72.9, 90, 91.1, 60, 79.2, 44.4],
      strict:  [29.3, 0, 86.7, 20, 0, 40] },
    { key: 'pcmb', name: 'PCMB-π₀',
      subtask: [39.5, 100, 24.4, 33.3, 32.9, 6.7],
      strict:  [0.4, 0, 2.2, 0, 0, 0] },
    { key: 'naive', name: 'Naïve-π₀',
      subtask: [16.7, 31.1, 6.6, 33.3, 8.3, 4.4],
      strict:  [0.4, 0, 2.2, 0, 0, 0] },
  ];

  function successChart() {
    const cv = document.getElementById('chart-success');
    if (!cv) return;
    const ctx = cv.getContext('2d');
    const readout = document.getElementById('chart-success-note');
    const btns = document.querySelectorAll('[data-metric]');

    let target = 0, t = 0;                 /* 0 = subtask, 1 = strict */

    function draw() {
      const pal = V.palette();
      const { w: W, h: H } = V.fit(cv, cv.clientWidth < 560 ? 340 : 300);
      ctx.clearRect(0, 0, W, H);

      const padL = 34, padR = 10, padT = 24, padB = 46;
      const plotW = W - padL - padR, plotH = H - padT - padB;
      const colors = { ours: pal.ours, key: pal.key, pcmb: pal.pcmb, naive: pal.naive };

      ctx.strokeStyle = pal.line;
      ctx.lineWidth = 1;
      ctx.textBaseline = 'middle';
      for (let g = 0; g <= 100; g += 25) {                    /* gridlines */
        const y = padT + plotH - (g / 100) * plotH;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
        ctx.fillStyle = pal.faint;
        ctx.font = '500 10px ui-monospace, monospace';
        ctx.textAlign = 'right';
        ctx.fillText(String(g), padL - 6, y);
      }

      const gw = plotW / GROUPS.length, bw = Math.min(22, (gw - 14) / METHODS.length);
      GROUPS.forEach((name, gi) => {
        const gx = padL + gi * gw;
        if (gi === 0) {                                       /* average band */
          ctx.fillStyle = V.mix(pal.line, pal.panel, 0.45);
          ctx.fillRect(gx + 2, padT - 6, gw - 4, plotH + 10);
        }
        const span = METHODS.length * bw + (METHODS.length - 1) * 3;
        let x = gx + gw / 2 - span / 2;
        METHODS.forEach((m) => {
          const val = V.lerp(m.subtask[gi], m.strict[gi], t);
          const bh = Math.max(1, (val / 100) * plotH);
          V.rr(ctx, x, padT + plotH - bh, bw, bh, 2);
          ctx.fillStyle = colors[m.key];
          ctx.globalAlpha = m.key === 'ours' ? 1 : 0.88;
          ctx.fill();
          ctx.globalAlpha = 1;
          if (bw > 10) {
            ctx.fillStyle = pal.text;
            ctx.font = '600 9px ui-monospace, monospace';
            ctx.textAlign = 'center';
            ctx.fillText(val.toFixed(val < 10 ? 1 : 0), x + bw / 2, padT + plotH - bh - 8);
          }
          x += bw + 3;
        });
        ctx.fillStyle = gi === 0 ? pal.ink : pal.text;
        ctx.font = (gi === 0 ? '700 ' : '500 ') + '11px Inter, system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(name, gx + gw / 2, H - padB + 16);
      });

      ctx.fillStyle = pal.faint;                              /* axis title */
      ctx.save();
      ctx.translate(11, padT + plotH / 2); ctx.rotate(-Math.PI / 2);
      ctx.font = '500 10px Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Success rate (%)', 0, 0);
      ctx.restore();
    }

    function setMetric(v) {
      target = v;
      btns.forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.metric === v)));
      if (readout) {
        readout.textContent = v === 0
          ? 'Subtask-level: the share of subtasks completed. Partial progress counts.'
          : 'Strict: a trial counts only if every subtask completes, in order, and the agent then stops. HeLM gives up 4.4 points; Keyframe-HLP loses 43.6 and both π₀ baselines go to the floor.';
      }
    }

    btns.forEach((b) => b.addEventListener('click', () => setMetric(+b.dataset.metric)));
    setMetric(0);
    V.onTheme(draw);
    window.addEventListener('resize', draw);
    V.animate(cv, (dt) => {
      const d = target - t;
      if (Math.abs(d) > 0.001) { t += d * Math.min(1, dt * 5); draw(); }
      else if (t !== target) { t = target; draw(); }
    });
    draw();
  }

  /* ----------------------------------------------------------- efficiency */

  /* Digitised from Fig. 7 — the paper plots these rather than tabulating them. The
     endpoints match the values stated in the text: Keyframe-HLP reaches ~28 GB on an
     8-event episode while HeLM stays under 6, and HeLM ends ~2.5x faster. */
  const LEN = [1, 2, 3, 4, 5, 6, 7, 8];
  const LAT = { ours: [2.05, 2.05, 2.04, 2.04, 2.05, 2.05, 2.06, 2.06],
                key:  [1.90, 1.96, 2.25, 2.52, 3.00, 3.49, 4.22, 4.81] };
  const VRAM = { ours: [6.2, 6.15, 6.12, 6.12, 6.13, 6.15, 6.15, 6.16],
                 key:  [6.3, 7.5, 9.0, 11.7, 14.5, 18.2, 23.0, 28.0] };

  function efficiencyChart() {
    const cv = document.getElementById('chart-eff');
    if (!cv) return;
    const ctx = cv.getContext('2d');
    const slider = document.getElementById('eff-len');
    const readout = document.getElementById('chart-eff-note');
    let n = 8;

    function panel(x0, w, pad, series, unit, maxY, titleText, pal, H) {
      const padT = 26, padB = 34;
      const plotH = H - padT - padB, plotW = w - pad * 2;
      const X = (i) => x0 + pad + (i / (LEN.length - 1)) * plotW;
      const Y = (v) => padT + plotH - (v / maxY) * plotH;

      ctx.strokeStyle = pal.line; ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const y = padT + plotH - (g / 4) * plotH;
        ctx.beginPath(); ctx.moveTo(x0 + pad, y); ctx.lineTo(x0 + w - pad, y); ctx.stroke();
      }

      [['key', pal.key], ['ours', pal.ours]].forEach(([k, col]) => {
        ctx.strokeStyle = col; ctx.lineWidth = 2.4;
        ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        ctx.beginPath();
        series[k].slice(0, n).forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
        ctx.stroke();
        const i = n - 1, v = series[k][i];
        ctx.beginPath(); ctx.arc(X(i), Y(v), 4, 0, Math.PI * 2);
        ctx.fillStyle = col; ctx.fill();
        ctx.strokeStyle = pal.panel; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = col;
        ctx.font = '700 12px ui-monospace, monospace';
        ctx.textAlign = X(i) > x0 + w - 60 ? 'right' : 'left';
        ctx.fillText(v.toFixed(unit === 'GB' ? 1 : 2) + ' ' + unit,
          X(i) + (X(i) > x0 + w - 60 ? -8 : 8), Y(v) - 12);
      });

      ctx.fillStyle = pal.text;
      ctx.font = '600 11px Inter, system-ui, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(titleText, x0 + pad, 12);
      ctx.fillStyle = pal.faint;
      ctx.font = '500 10px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('task length ' + n, x0 + w / 2, H - 10);
    }

    function draw() {
      const pal = V.palette();
      const stacked = cv.clientWidth < 560;
      const { w: W, h: H } = V.fit(cv, stacked ? 380 : 210);
      ctx.clearRect(0, 0, W, H);
      if (stacked) {
        panel(0, W, 16, LAT, 's', 5.2, 'Latency', pal, H / 2);
        ctx.save(); ctx.translate(0, H / 2);
        panel(0, W, 16, VRAM, 'GB', 30, 'VRAM', pal, H / 2);
        ctx.restore();
      } else {
        panel(0, W / 2, 16, LAT, 's', 5.2, 'Latency', pal, H);
        panel(W / 2, W / 2, 16, VRAM, 'GB', 30, 'VRAM', pal, H);
      }
    }

    /* Absolute values rather than a latency ratio. These points are digitised off
       Fig. 7, so a computed speed-up lands near but not exactly on the 2.5x the paper
       states, and printing a slightly different number right under that sentence reads
       as an error. The VRAM ratio does match, so it is worth stating. */
    function setN(v) {
      n = +v;
      if (readout) {
        const i = n - 1;
        const lean = VRAM.key[i] / VRAM.ours[i];
        readout.textContent = n <= 2
          ? 'At short horizons HeLM is slightly slower — it generates its memory as text, token by token.'
          : 'At task length ' + n + ', HeLM answers in ' + LAT.ours[i].toFixed(2) + ' s on '
            + VRAM.ours[i].toFixed(1) + ' GB, while Keyframe-HLP needs ' + LAT.key[i].toFixed(2)
            + ' s and ' + VRAM.key[i].toFixed(1) + ' GB — ' + lean.toFixed(1)
            + '× the memory, because it pays for every image it keeps.';
      }
      draw();
    }

    if (slider) { slider.addEventListener('input', (e) => setN(e.target.value)); setN(slider.value); }
    else setN(8);
    V.onTheme(draw);
    window.addEventListener('resize', draw);
  }

  successChart();
  efficiencyChart();
})();
