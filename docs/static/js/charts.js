/* Intra-episode results: Figure 5 and Table 8 of the latest manuscript. */
(function () {
  'use strict';
  const V = window.HelmViz;
  if (!V) return;

  /* ------------------------------------------------------- success rate chart */

  /* Subtask-level from Fig. 5; strict from Table 8 of the latest manuscript. */
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
      strict:  [0, 0, 0, 0, 0, 0] },
    { key: 'naive', name: 'Naïve-π₀',
      subtask: [16.7, 31.1, 6.6, 33.3, 8.3, 4.4],
      strict:  [0, 0, 0, 0, 0, 0] },
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
      if (V.reduceMotion()) { t = target; draw(); }
      btns.forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.metric === v)));
      if (readout) {
        readout.textContent = v === 0
          ? 'Subtask-level: the share of subtasks completed. Partial progress counts.'
          : 'Strict task success requires every subtask to complete in order with proper termination: HeLM 81.3%, Keyframe-HLP 29.3%, PCMB-π₀ 0%, Naïve-π₀ 0% (Table 8).';
      }
    }

    btns.forEach((b) => b.addEventListener('click', () => setMetric(+b.dataset.metric)));
    setMetric(0);
    window.addEventListener('resize', draw);
    V.animate(cv, (dt) => {
      const d = target - t;
      if (Math.abs(d) > 0.001) { t += d * Math.min(1, dt * 5); draw(); }
      else if (t !== target) { t = target; draw(); }
    });
    draw();
  }

  successChart();
})();
