/* Scene renderer for the memory inspector.
 *
 * The point of this file is the point of the paper. Every scene below is drawn purely
 * by parsing HeLM's memory strings — no scene graph, no simulator state, nothing but
 * the text the policy actually carries. That is only possible because the memory is
 * explicit language; a latent memory vector could not drive a picture like this.
 *
 * The strings come in exactly three shapes (see helm_datasets/taskspecs/):
 *   Count: 2 (Goal: 3)
 *   Progress: blue, green (Goal: blue, green, red)
 *   task done (None)
 * and episodic context in two:
 *   Previous_Count: 3
 *   Previous_Progress: blue, green, red
 */
window.HelmScene = (function () {
  'use strict';

  const V = window.HelmViz;

  /* ---------------------------------------------------------------- parsing */

  const list = (s) =>
    !s || s === 'none' ? [] : s.split(',').map((x) => x.trim()).filter(Boolean);

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
    if (m) return { kind: 'progress', items: list(m[1]) };
    return null;
  }

  const KIND = {
    A1: 'pips', E1: 'pips',
    A2: 'buttons', E2: 'buttons',
    A3: 'window', E3: 'window',
    A4: 'drawers', E4: 'drawers',
    A5: 'plates', E5: 'plates',
  };

  /* ---------------------------------------------------------------- drawing */

  function label(ctx, text, x, y, col, size, weight) {
    ctx.fillStyle = col;
    ctx.font = (weight || 600) + ' ' + (size || 11) + 'px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x, y);
  }

  function disc(ctx, x, y, r, fill, stroke, lw) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 2; ctx.stroke(); }
  }

  /* A1 / E1 — one button, and a pip per required press. Pips already banked by a
     previous episode are drawn hollow, which is the whole trick of the E1 task. */
  function pips(ctx, W, H, st, pal) {
    const w = st.work, prev = st.epi && st.epi.kind === 'count' ? st.epi.n : 0;
    const goal = w.done ? (st.lastGoal || prev) : w.goal;
    const n = w.done ? goal : w.n;

    const cy = H * 0.42;
    disc(ctx, W / 2, cy, 30, pal.blue, null);
    disc(ctx, W / 2, cy, 30, null, V.mix(pal.blue, '#000000', 0.25), 2);
    disc(ctx, W / 2, cy - 4, 22, V.mix(pal.blue, '#ffffff', 0.25), null);

    const total = Math.max(1, goal), gap = 10, r = 7;
    const span = total * (r * 2) + (total - 1) * gap;
    let x = W / 2 - span / 2 + r;
    const py = H * 0.78;
    for (let i = 0; i < total; i++, x += r * 2 + gap) {
      if (i < prev) disc(ctx, x, py, r, null, pal.faint, 2);          /* banked earlier */
      else if (i < n) disc(ctx, x, py, r, pal.blue, null);            /* this episode */
      else disc(ctx, x, py, r, pal.track, null);
    }
    label(ctx, prev ? n + ' / ' + goal + '  (' + prev + ' before this episode)' : n + ' / ' + goal,
      W / 2, H - 12, pal.text, 11);
  }

  /* A2 / E2 — three buttons in their physical order, badged with the order the task
     asks for. During E2's first episode nothing is pressed, only watched. */
  function buttons(ctx, W, H, st, pal) {
    const w = st.work;
    const observing = w.goal.length === 1 && w.goal[0] === 'observe';
    const order = observing ? [] : w.goal;
    const cols = { blue: pal.blue, green: pal.green, red: pal.red };
    const names = ['blue', 'green', 'red'];

    const gap = 26, r = 26;
    const span = names.length * r * 2 + (names.length - 1) * gap;
    let x = W / 2 - span / 2 + r;
    const cy = H * 0.46;

    names.forEach((nm) => {
      const hit = w.got.indexOf(nm) >= 0 || w.done;
      const idx = order.indexOf(nm);
      disc(ctx, x, cy, r, hit ? cols[nm] : V.mix(cols[nm], pal.panel, 0.72), null);
      disc(ctx, x, cy, r, null, hit ? V.mix(cols[nm], '#000000', 0.25) : pal.line, 2);
      if (hit) {
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 3; ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(x - 9, cy); ctx.lineTo(x - 2, cy + 7); ctx.lineTo(x + 10, cy - 7);
        ctx.stroke();
      }
      if (idx >= 0) label(ctx, String(idx + 1), x, cy + r + 15, pal.faint, 11);
      x += r * 2 + gap;
    });

    label(ctx, observing ? 'watching — ' + (w.got.length ? w.got.join(' → ') : 'nothing yet')
                         : (w.got.length ? w.got.join(' → ') : 'nothing pressed yet'),
      W / 2, H - 14, pal.text, 11);
  }

  /* A3 / E3 — the window, bottom band at the bottom. Regions cleaned in an earlier
     episode stay clean, which is exactly what "wipe the remaining part" relies on. */
  function windowScene(ctx, W, H, st, pal) {
    const w = st.work;
    const prev = st.epi && st.epi.kind === 'progress' ? st.epi.items : [];
    const rows = ['top', 'middle', 'bottom'];
    const bw = Math.min(300, W - 60), bx = (W - bw) / 2;
    const bh = 34, gap = 8, top = H * 0.5 - (rows.length * bh + (rows.length - 1) * gap) / 2;

    rows.forEach((nm, i) => {
      const y = top + i * (bh + gap);
      const done = w.done || w.got.indexOf(nm) >= 0;
      const earlier = prev.indexOf(nm) >= 0;
      const target = w.goal.indexOf(nm) >= 0;

      V.rr(ctx, bx, y, bw, bh, 5);
      ctx.fillStyle = done || earlier ? V.mix(pal.accent, pal.panel, 0.82) : pal.track;
      ctx.fill();
      ctx.strokeStyle = target && !done ? pal.accent : pal.line;
      ctx.lineWidth = target && !done ? 2 : 1;
      ctx.stroke();

      if (!(done || earlier)) {                                  /* grime hatching */
        ctx.save();
        ctx.beginPath(); ctx.rect(bx, y, bw, bh); ctx.clip();
        ctx.strokeStyle = pal.line; ctx.lineWidth = 1;
        for (let d = -bh; d < bw; d += 7) {
          ctx.beginPath(); ctx.moveTo(bx + d, y + bh); ctx.lineTo(bx + d + bh, y); ctx.stroke();
        }
        ctx.restore();
      }
      label(ctx, nm, bx + bw / 2, y + bh / 2, done || earlier ? pal.accent : pal.faint, 11);
      if (earlier && !(w.got.indexOf(nm) >= 0)) label(ctx, 'earlier', bx + bw - 30, y + bh / 2, pal.faint, 9);
    });
  }

  /* A4 / E4 — the four drawers. A4 sweeps them in a fixed order; E4 walks straight to
     the one episodic context names, so the target is drawn as a called shot. */
  function drawers(ctx, W, H, st, pal) {
    const w = st.work;
    const prev = st.epi && st.epi.kind === 'progress' ? st.epi.items : [];
    const cells = [['top_left', 0, 0], ['top_right', 1, 0], ['bottom_left', 0, 1], ['bottom_right', 1, 1]];
    const target = w.goal.length === 1 && w.goal[0] !== 'observe' ? w.goal[0] : null;
    const known = prev.length === 1 ? prev[0] : null;

    const cw = 104, ch = 46, gap = 10;
    const ox = W / 2 - (cw * 2 + gap) / 2, oy = H * 0.5 - (ch * 2 + gap) / 2 - 6;

    cells.forEach(([nm, cx, cy]) => {
      const x = ox + cx * (cw + gap), y = oy + cy * (ch + gap);
      const opened = w.got.indexOf(nm) >= 0;
      const isTarget = nm === target || nm === known;

      V.rr(ctx, x, y, cw, ch, 5);
      ctx.fillStyle = opened ? V.mix(pal.faint, pal.panel, 0.85) : pal.track;
      ctx.fill();
      ctx.strokeStyle = isTarget ? pal.accent : pal.line;
      ctx.lineWidth = isTarget ? 2 : 1;
      ctx.stroke();

      if (opened) {                                        /* drawer pulled out */
        V.rr(ctx, x + 8, y + ch - 13, cw - 16, 10, 3);
        ctx.fillStyle = pal.panel; ctx.fill();
        ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.stroke();
      } else {
        V.rr(ctx, x + cw / 2 - 12, y + ch / 2 - 2, 24, 4, 2);
        ctx.fillStyle = pal.line; ctx.fill();
      }
      if (isTarget) disc(ctx, x + cw - 15, y + 14, 6, pal.orange, null);
      label(ctx, nm.replace('_', '-'), x + cw / 2, y + ch / 2 + (opened ? -6 : 14), pal.faint, 9);
    });

    label(ctx, known ? 'episodic context says: ' + known.replace('_', '-')
                     : (w.got.length ? 'searched ' + w.got.length + ' of 4' : 'nothing searched yet'),
      W / 2, H - 12, known ? pal.accent : pal.text, 11);
  }

  /* A5 / E5 — three plates and the button. The banana sits wherever the last completed
     transfer put it; the ring marks where it started, which is all E5 has to go on. */
  function plates(ctx, W, H, st, pal) {
    const w = st.work;
    const epi = st.epi && st.epi.kind === 'progress' ? st.epi.items : [];
    const names = ['blue', 'red', 'white'];
    const cols = { blue: pal.blue, red: pal.red, white: pal.white };

    const moves = w.got.filter((g) => g.indexOf('_to_') > 0);
    const allGoal = w.goal.filter((g) => g.indexOf('_to_') > 0);
    const originTag = epi.find((g) => g.indexOf('origin_') === 0);

    let origin = originTag ? originTag.split('_')[1]
      : (allGoal.length ? allGoal[0].split('_to_')[0] : 'blue');
    let at = moves.length ? moves[moves.length - 1].split('_to_')[1] : origin;
    if (w.done && allGoal.length) at = allGoal[allGoal.length - 1].split('_to_')[1];

    const pressed = w.done || w.got.indexOf('press_orange') >= 0;
    const needsPress = w.goal.indexOf('press_orange') >= 0;

    const r = 34, gap = 34;
    const span = names.length * r * 2 + (names.length - 1) * gap;
    let x = W / 2 - span / 2 + r;
    const cy = H * 0.44;

    names.forEach((nm) => {
      disc(ctx, x, cy, r, V.mix(cols[nm], pal.panel, nm === 'white' ? 0.25 : 0.68), null);
      disc(ctx, x, cy, r, null, nm === origin ? pal.faint : pal.line, nm === origin ? 2.5 : 1.5);
      disc(ctx, x, cy, r - 9, null, pal.line, 1);
      if (nm === origin) label(ctx, 'origin', x, cy + r + 14, pal.faint, 9);
      if (nm === at) {                                            /* the banana */
        ctx.save();
        ctx.translate(x, cy - 3); ctx.rotate(-0.35);
        ctx.beginPath();
        ctx.ellipse(0, 0, 19, 8, 0, 0, Math.PI * 2);
        ctx.fillStyle = pal.banana; ctx.fill();
        ctx.strokeStyle = V.mix(pal.banana, '#000000', 0.3); ctx.lineWidth = 1.5; ctx.stroke();
        ctx.restore();
      }
      x += r * 2 + gap;
    });

    if (needsPress || pressed) {
      const bx = W / 2, by = cy + r + 34;
      disc(ctx, bx, by, 12, pressed ? pal.orange : V.mix(pal.orange, pal.panel, 0.7), null);
      disc(ctx, bx, by, 12, null, pressed ? V.mix(pal.orange, '#000000', 0.25) : pal.line, 2);
      label(ctx, pressed ? 'pressed' : 'press', bx + 44, by, pal.faint, 9);
    }
    label(ctx, 'banana on ' + at + (origin !== at ? '  ·  started on ' + origin : ''),
      W / 2, H - 12, pal.text, 11);
  }

  const RENDER = { pips, buttons, window: windowScene, drawers, plates };

  /* Scenes are square-ish and centred; the caller owns the canvas element. */
  function draw(ctx, W, H, taskId, workStr, epiStr, lastGoal) {
    const pal = V.palette();
    ctx.clearRect(0, 0, W, H);
    const st = { work: parseWork(workStr), epi: parseEpi(epiStr), lastGoal: lastGoal };
    const fn = RENDER[KIND[taskId]];
    if (fn) fn(ctx, W, H, st, pal);
  }

  return { parseWork, parseEpi, draw, kindFor: (id) => KIND[id] };
})();
