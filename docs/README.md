# HeLM project page

Static site served by GitHub Pages from `main` → `/docs`.
No build step, no dependencies, no framework — plain HTML/CSS/JS.

Live URL: <https://kimspacehuni.github.io/HeLM/>

## Enabling GitHub Pages

Repo **Settings → Pages → Build and deployment**:
- Source: `Deploy from a branch`
- Branch: `main`, folder: `/docs`

`.nojekyll` is present so Jekyll does not reprocess the files.

## Local preview

```bash
cd docs && python3 -m http.server 8899
# open http://localhost:8899/
```

## Layout

```
docs/
├── index.html                 # all page copy lives here
├── static/
│   ├── css/style.css          # single light-mode stylesheet
│   ├── js/
│   │   ├── viz.js             # shared canvas helpers + palette
│   │   ├── scene.js           # setup photos and symbolic task progress
│   │   ├── traces.js          # GENERATED — real memory states, all 51 variants
│   │   ├── hero.js            # manual introduction using the A2 task spec
│   │   ├── inspector.js       # the memory inspector
│   │   ├── charts.js          # success-rate chart + efficiency chart
│   │   └── main.js            # task discovery and nav
│   ├── figures/               # figures extracted from the paper PDF
│   ├── images/                # video poster + social card
│   └── videos/                # overview video
└── .nojekyll
```

## Figures and presentation

The page uses a fixed light palette, independent of the OS preference and any
previously saved theme setting. Diagram colours come from the CSS custom
properties in `style.css` through `viz.js`.

The main method illustration is paper Figure 2 (`static/figures/fig2_arch.webp`).
It spans the content width. On smaller screens it scrolls horizontally to keep
its labels readable; a full-size link is available below the figure.

The paper mascots appear as small brand marks near the title and in the footer.
`mascot-mark.webp` preserves transparency and `mascot-signature.webp` preserves
the second image's white background. Both are 192 px wide; a separate 48 px PNG
serves as the favicon. Original large PNGs are kept local and ignored by Git.

## Reading order and interaction

The publication header leads with the full paper title, a short secondary subtitle,
and Code / arXiv (coming soon) / Video links. An abstract precedes the problem,
method, memory explorer, task suite and results. The optional button-sequence
example and detailed tables are expandable.

The introductory example uses the `press_RGB` states from `traces.js`. It advances only when a
visitor selects a step. The memory explorer also starts paused. Task cards open
the corresponding task in that explorer; keyboard controls and reduced-motion
preferences are supported. Main content stays visible if JavaScript fails.

## The memory inspector

`static/js/traces.js` is generated from `helm_datasets/taskspecs/` — the same task
specifications used to synthesise the training data. It includes 51 task variants.
Regenerate from the repository root after editing a taskspec:

```bash
python tools/build_page_traces.py
```

The explorer pairs a fixed reference photograph of the real experiment setup
with symbolic progress derived from the selected TaskSpec state (`scene.js`).
The photos in `static/images/tasks/` are resized from the supplied 4000 × 3000
originals to 1200 × 900 WebP, with orientation applied and no cropping. Native
`<img>` elements use `width: 100%; height: auto` so each photograph remains fully
visible in cards, the explorer and the introductory example. EXIF data is not
included in the web assets. No generated robot imagery is used.

Completed steps, the next step and remembered context are displayed separately.
Observation tasks reveal only events already in memory, not the future sequence.

Examples of memory strings:

```
Count: 2 (Goal: 3)
Progress: blue, green (Goal: blue, green, red)
task done (None)
```

`?task=E2#memory` deep-links to a task in the explorer. Setup photos stay fixed while the symbolic progress and memory follow the selected
task-specification state. Playback is not recorded robot inference.

## Numbers on this page

All success rates come from the paper (Fig. 5, Tables 1, 3, 7, 10), not from the
top-level `README.md`.

**Efficiency values are approximate.** `LAT` and `VRAM` in `charts.js` are
digitised from paper Fig. 7, not raw benchmark measurements. At 8 events the
estimates are 2.06 s / 6.16 GB for HeLM and 4.81 s / 28 GB for Keyframe-HLP:
approximately 2.3× faster inference and 4.5× lower VRAM usage. The README,
page summary, chart and social description use these estimates consistently.

## Social preview card

`og:image` points at `static/images/og_card.jpg` via an **absolute** URL — scrapers
do not resolve relative paths. It is a JPEG, not the WebP source, because some
scrapers (LinkedIn especially) still do not render WebP previews.

It is generated from the teaser figure. Run from `docs/`:

```bash
python -c "
from PIL import Image
im = Image.open('static/figures/fig1_teaser.webp').convert('RGB')
im = im.resize((1200, round(1200*im.size[1]/im.size[0])), Image.LANCZOS)
im.save('static/images/og_card.jpg', 'JPEG', quality=86, optimize=True, progressive=True)
"
```

If the page URL ever changes, the absolute URLs in the `og:` block must change with it.

## Visitor map

The footer embeds a [mapmyvisitors.com](https://mapmyvisitors.com/) widget. Stats are
public at <https://mapmyvisitors.com/web/1c78b>.

Two things to know before editing it:

- `map.js` injects the widget where the tag sits, so it must stay in the body and must
  **not** get `async` / `defer`.
- The widget renders on a white background (`cl=ffffff`), so it sits in a light
  `.visitor-map` card. **Do not** duplicate the tag —
  a `display:none` copy still fires the request and double-counts every visitor.
- `pruneVisitorMap()` in `main.js` hides the card when the widget injects nothing, so
  it does not show as an empty white bar. Content blockers block this tracker by
  default, so that path is common, not an edge case.

If the map never appears, check the endpoint directly — it should return an image:

```bash
curl -so /dev/null -w '%{http_code}\n' \
  'https://mapmyvisitors.com/map.png?d=<HASH>&cl=ffffff'
```

A `500` there means the tracker is not being served for that hash, which is a
mapmyvisitors-side issue and not something the page can fix.

## Regenerating figures from the paper

Figures were cropped from the full-version PDF at 600 dpi. To redo one after a
paper revision:

```bash
pdftoppm -r 600 -f 6 -l 6 -png -singlefile paper.pdf page6
magick page6.png -crop 3400x1520+900+555 +repage -fuzz 2% -trim +repage fig.png
magick fig.png -resize 1800x\> -strip -quality 88 docs/static/figures/fig.webp
```

Adjust the `-crop WxH+X+Y` box per figure; `-trim` removes the surrounding
whitespace so the box only needs to be close, not exact. Photo-heavy figures go
to WebP; flat-color charts are often smaller as PNG — check both.
