/* HeLM project page.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *  TO ADD A ROLLOUT VIDEO: drop the file into static/videos/ and set `video`
 *  below to its filename. Leave it as null to keep showing the placeholder.
 *
 *    { id: 'A1', ..., video: 'A1_press_n_times.mp4' }
 *
 *  Recommended encode (keeps the page light — aim for < 3 MB per clip):
 *    ffmpeg -i raw.mp4 -vf "scale=-2:480" -c:v libx264 -crf 30 \
 *           -preset slow -pix_fmt yuv420p -movflags +faststart -an out.mp4
 * ─────────────────────────────────────────────────────────────────────────────
 */

const TASKS = [
  // ---------------------------------------------------------------- intra
  {
    id: 'A1', group: 'intra',
    name: 'Press N Times',
    desc: 'Press the blue button exactly N times, then stop. Frames are visually near-identical, so the count must come from memory.',
    memory: 'Count: 2 (Goal: 3)',
    video: null,
  },
  {
    id: 'A2', group: 'intra',
    name: 'Press in Order',
    desc: 'Press three colored buttons in a specified order. Requires tracking which sub-goals are already done.',
    memory: 'Progress: blue (Goal: B, G, R)',
    video: null,
  },
  {
    id: 'A3', group: 'intra',
    name: 'Wipe the Window',
    desc: 'Wipe the bottom, middle, and top regions in order. A wiped region looks much like an unwiped one.',
    memory: 'Progress: bottom (Goal: B, M, T)',
    video: null,
  },
  {
    id: 'A4', group: 'intra',
    name: 'Find the Object',
    desc: 'Open drawers clockwise until the toy hamburger is found, then stop — requires recalling which drawers were already searched.',
    memory: 'Searched: top_left (Goal: find)',
    video: null,
  },
  {
    id: 'A5', group: 'intra',
    name: 'Pick, Place and Press',
    desc: 'Move the banana between plates with a button press in between. A multi-step procedure with an interleaved sub-goal.',
    memory: 'Banana: red; Target: press',
    video: null,
  },
  // ---------------------------------------------------------------- inter
  {
    id: 'E1', group: 'inter',
    name: 'Cumulative Press',
    desc: 'Press N times, then a new instruction asks for M presses in total. The agent must recall N to infer M − N remaining.',
    memory: 'Previous_Count: 3',
    video: null,
  },
  {
    id: 'E2', group: 'inter',
    name: 'Human-Order Press',
    desc: 'Observe the sequence a person presses, then reproduce that same order in the next episode.',
    memory: 'Previous_Progress: blue, green, red',
    video: null,
  },
  {
    id: 'E3', group: 'inter',
    name: 'Remaining Wipe',
    desc: 'Some window regions are wiped in the first episode; the next instruction only says "wipe the remaining part".',
    memory: 'Previous_Progress: bottom, middle',
    video: null,
  },
  {
    id: 'E4', group: 'inter',
    name: 'Direct Retrieval',
    desc: 'Observe which drawer holds the object, then open that drawer directly — without the sequential search of A4.',
    memory: 'Previous_Location: top_right',
    video: null,
  },
  {
    id: 'E5', group: 'inter',
    name: 'Original Restore',
    desc: 'After two transfers, return the banana to the plate it originally started on — which is no longer observable.',
    memory: 'Origin: blue plate',
    video: null,
  },
];

/* ------------------------------------------------------------------ render */

const PLACEHOLDER_ICON =
  '<svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">' +
  '<rect x="2" y="4" width="14" height="16" rx="2"/><path d="m16 10 6-3v10l-6-3"/></svg>';

function taskCard(t) {
  const media = t.video
    ? `<video src="static/videos/${t.video}" muted loop playsinline preload="metadata"
              aria-label="${t.id}: ${t.name} rollout"></video>`
    : `<div class="placeholder">${PLACEHOLDER_ICON}
         <div>Rollout video pending</div>
         <code>static/videos/${t.id}_*.mp4</code>
       </div>`;

  const card = document.createElement('article');
  card.className = 'task-card';
  card.dataset.group = t.group;
  card.innerHTML = `
    <div class="task-media">
      <span class="task-id">${t.id}</span>
      ${media}
    </div>
    <div class="task-body">
      <div class="task-name">${t.name}</div>
      <div class="task-desc">${t.desc}</div>
      <span class="task-mem">${t.memory}</span>
    </div>`;
  return card;
}

const grid = document.getElementById('task-grid');
TASKS.forEach((t) => grid.appendChild(taskCard(t)));

/* Play rollouts on hover / on screen, pause otherwise — avoids 10 concurrent decodes. */
grid.querySelectorAll('.task-card').forEach((card) => {
  const v = card.querySelector('video');
  if (!v) return;
  card.addEventListener('mouseenter', () => v.play().catch(() => {}));
  card.addEventListener('mouseleave', () => { v.pause(); v.currentTime = 0; });
});

if ('IntersectionObserver' in window) {
  const io = new IntersectionObserver(
    (entries) => entries.forEach((e) => {
      const v = e.target.querySelector('video');
      if (v && !e.isIntersecting) v.pause();
    }),
    { threshold: 0 }
  );
  grid.querySelectorAll('.task-card').forEach((c) => io.observe(c));
}

/* ------------------------------------------------------------------ filter */

const tabs = document.querySelectorAll('.tab');
tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    tabs.forEach((t) => t.setAttribute('aria-selected', String(t === tab)));
    const f = tab.dataset.filter;
    grid.querySelectorAll('.task-card').forEach((card) => {
      card.style.display = (f === 'all' || card.dataset.group === f) ? '' : 'none';
    });
  });
});

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

window.addEventListener('load', pruneVisitorMap);

/* ------------------------------------------------------------------- theme */

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

function paint() {
  toggle.innerHTML = currentTheme() === 'dark' ? SUN : MOON;
}

const saved = localStorage.getItem('helm-theme');
if (saved) root.dataset.theme = saved;
paint();

toggle.addEventListener('click', () => {
  const next = currentTheme() === 'dark' ? 'light' : 'dark';
  root.dataset.theme = next;
  localStorage.setItem('helm-theme', next);
  paint();
});
