/* Animated architecture diagram.
 *
 * The static figure in the paper can show the two modes but not the thing that makes
 * them worth having: most passes through DETECT find nothing and cost nothing, and the
 * memory is only rewritten when an event actually fires. So the loop runs here, idling
 * through two quiet cycles for every one that triggers an update.
 */
(function () {
  'use strict';
  const V = window.HelmViz;
  const cv = document.getElementById('chart-arch');
  if (!cv || !V) return;
  const ctx = cv.getContext('2d');
  const note = document.getElementById('chart-arch-note');

  const QUIET = 2;              /* quiet detect cycles between each triggered update */
  const T_IN = 0.9, T_DET = 0.9, T_UPD = 1.2, T_OUT = 1.0;

  let t = 0, cycle = 0, lastLabel = '';

  function box(x, y, w, h, fill, stroke, lw) {
    V.rr(ctx, x, y, w, h, 8);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1.5; ctx.stroke(); }
  }

  function text(s, x, y, col, size, weight, align) {
    ctx.fillStyle = col;
    ctx.font = (weight || 600) + ' ' + size + 'px Inter, system-ui, sans-serif';
    ctx.textAlign = align || 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(s, x, y);
  }

  function mono(s, x, y, col, size, align) {
    ctx.fillStyle = col;
    ctx.font = '600 ' + size + 'px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = align || 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(s, x, y);
  }

  function arrow(x1, y1, x2, y2, col, lw, dash) {
    ctx.strokeStyle = col;
    ctx.lineWidth = lw || 1.6;
    ctx.setLineDash(dash || []);
    ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.setLineDash([]);
    const a = Math.atan2(y2 - y1, x2 - x1), s = 5;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - s * Math.cos(a - 0.5), y2 - s * Math.sin(a - 0.5));
    ctx.lineTo(x2 - s * Math.cos(a + 0.5), y2 - s * Math.sin(a + 0.5));
    ctx.closePath();
    ctx.fillStyle = col; ctx.fill();
  }

  function pulse(x, y, col, r) {
    ctx.beginPath(); ctx.arc(x, y, r || 5, 0, Math.PI * 2);
    ctx.fillStyle = col; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, (r || 5) + 3.5, 0, Math.PI * 2);
    ctx.strokeStyle = col; ctx.globalAlpha = 0.35; ctx.lineWidth = 2; ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function draw() {
    const pal = V.palette();
    const narrow = cv.clientWidth < 620;
    const { w: W, h: H } = V.fit(cv, narrow ? 470 : 400);
    ctx.clearRect(0, 0, W, H);

    const fires = cycle % (QUIET + 1) === QUIET;
    const total = T_IN + T_DET + (fires ? T_UPD : 0) + T_OUT;
    const p = t;

    const cx = W / 2;
    const colW = Math.min(430, W - 40);
    const left = cx - colW / 2;

    /* --- rows ------------------------------------------------------------- */
    const yIn = 26;
    const yHLP = 66, hHLP = narrow ? 190 : 168;
    const yMem = yHLP + hHLP + 16, hMem = 56;
    const yLLP = yMem + hMem + 26, hLLP = 46;
    const yOut = yLLP + hLLP + 22;

    /* inputs */
    mono('o' + 'ₜ', left + 46, yIn, pal.text, 12);
    mono('I', left + colW - 46, yIn, pal.text, 12);
    arrow(left + 46, yIn + 12, left + 46, yHLP - 4, pal.line, 1.4);
    arrow(left + colW - 46, yIn + 12, left + colW - 46, yHLP - 4, pal.line, 1.4);

    /* HLP shell */
    box(left, yHLP, colW, hHLP, V.mix(pal.accent, pal.panel, 0.94), pal.line, 1.5);
    text('High-Level Policy', left + 14, yHLP + 18, pal.ink, 12.5, 700, 'left');
    mono('fine-tuned VLM', left + colW - 14, yHLP + 18, pal.faint, 10, 'right');

    /* the two modes */
    /* The gap between the two modes has to carry the arrow and its event/none label,
       so it is wider than a purely visual gutter would be. */
    const mw = (colW - 84) / 2, mh = 62;
    const mx1 = left + 20, mx2 = left + 64 + mw, my = yHLP + 42;
    const inDetect = p >= T_IN && p < T_IN + T_DET;
    const inUpdate = fires && p >= T_IN + T_DET && p < T_IN + T_DET + T_UPD;

    box(mx1, my, mw, mh, inDetect ? V.mix(pal.detect, pal.panel, 0.8) : pal.panel,
        inDetect ? pal.detect : pal.line, inDetect ? 2 : 1.2);
    text('DETECT', mx1 + mw / 2, my + 22, inDetect ? pal.detect : pal.text, 11.5, 700);
    mono('eₜ ~ π_det', mx1 + mw / 2, my + 42, pal.faint, 10);

    box(mx2, my, mw, mh, inUpdate ? V.mix(pal.update, pal.panel, 0.8) : pal.panel,
        inUpdate ? pal.update : pal.line, inUpdate ? 2 : 1.2);
    text('UPDATE', mx2 + mw / 2, my + 22, inUpdate ? pal.update : pal.text, 11.5, 700);
    mono('Mₜ₊₁ ~ π_upd', mx2 + mw / 2, my + 42, pal.faint, 10);

    /* event edge between them, dashed while nothing has fired */
    arrow(mx1 + mw + 3, my + mh / 2, mx2 - 5, my + mh / 2,
          fires ? pal.detect : pal.faint, fires ? 2 : 1.3, fires ? [] : [4, 4]);
    mono(fires ? 'event' : 'none', (mx1 + mw + mx2) / 2, my + mh / 2 - 14,
         fires ? pal.detect : pal.faint, 9.5);

    /* the quiet path: detect loops straight back on itself */
    if (!fires) {
      const ly = my + mh + 16;
      ctx.strokeStyle = pal.faint; ctx.lineWidth = 1.3; ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(mx1 + mw / 2, my + mh);
      ctx.lineTo(mx1 + mw / 2, ly);
      ctx.lineTo(mx1 - 8, ly);
      ctx.lineTo(mx1 - 8, my + mh / 2);
      ctx.stroke();
      ctx.setLineDash([]);
      arrow(mx1 - 8, my + mh / 2 + 6, mx1 - 8, my + mh / 2 - 2, pal.faint, 1.3);
      mono('no update', mx1 + mw / 2, ly + 11, pal.faint, 9.5);
    }

    /* memory */
    box(left, yMem, colW, hMem, pal.panel, inUpdate ? pal.update : pal.line, inUpdate ? 2 : 1.2);
    mono('Language memory  Mₜ', left + 12, yMem + 15, pal.faint, 9.5, 'left');
    const halfW = (colW - 36) / 2;
    box(left + 12, yMem + 24, halfW, 22, V.mix(pal.accent, pal.panel, 0.9), null);
    mono('working', left + 12 + halfW / 2, yMem + 35, pal.accent, 9.5);
    box(left + 24 + halfW, yMem + 24, halfW, 22, V.mix(pal.faint, pal.panel, 0.88), null);
    mono('episodic', left + 24 + halfW + halfW / 2, yMem + 35, pal.text, 9.5);

    /* command down to the LLP */
    arrow(cx, yMem + hMem + 2, cx, yLLP - 4, pal.line, 1.6);
    mono('zₜ', cx + 14, (yMem + hMem + yLLP) / 2, pal.faint, 10, 'left');

    box(left, yLLP, colW, hLLP, V.mix(pal.green, pal.panel, 0.92), pal.line, 1.5);
    text('Low-Level Policy', left + 14, yLLP + hLLP / 2, pal.ink, 12.5, 700, 'left');
    mono('π₀ VLA', left + colW - 14, yLLP + hLLP / 2, pal.faint, 10, 'right');

    arrow(cx, yLLP + hLLP + 2, cx, yOut + 4, pal.line, 1.6);
    mono('aₜ', cx + 14, yOut - 4, pal.faint, 10, 'left');

    /* --- travelling pulse -------------------------------------------------- */
    let px = null, py = null, pc = pal.accent;
    if (p < T_IN) {
      const k = V.ease(p / T_IN);
      px = left + 46; py = V.lerp(yIn + 12, my + mh / 2, k);
      if (k > 0.55) px = V.lerp(left + 46, mx1 + mw / 2, V.ease((k - 0.55) / 0.45));
    } else if (inDetect) {
      px = mx1 + mw / 2; py = my + mh / 2; pc = pal.detect;
    } else if (inUpdate) {
      const k = V.ease((p - T_IN - T_DET) / T_UPD);
      pc = pal.update;
      if (k < 0.4) { px = V.lerp(mx1 + mw / 2, mx2 + mw / 2, k / 0.4); py = my + mh / 2; }
      else { px = mx2 + mw / 2; py = V.lerp(my + mh / 2, yMem + 35, (k - 0.4) / 0.6); }
    } else {
      const k = V.ease((p - T_IN - T_DET - (fires ? T_UPD : 0)) / T_OUT);
      px = V.lerp(fires ? mx2 + mw / 2 : mx1 + mw / 2, cx, Math.min(1, k * 2));
      py = V.lerp(fires ? yMem + 35 : my + mh / 2, yOut, k);
      pc = fires ? pal.update : pal.faint;
    }
    if (px != null) pulse(px, py, pc);

    const lbl = fires ? 'update' : 'quiet';
    if (note && lbl !== lastLabel) {
      lastLabel = lbl;
      note.textContent = fires
        ? 'An event fired. UPDATE rewrites working memory and emits the next action command.'
        : 'DETECT found nothing. Memory is untouched and the previous command stands — most steps look like this.';
    }
    return total;
  }

  let total = 4;
  V.onTheme(draw);
  window.addEventListener('resize', draw);
  V.animate(cv, (dt) => {
    if (!V.reduceMotion()) {
      t += dt;
      if (t >= total) { t = 0; cycle++; }
    }
    total = draw();
  });
})();
