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
├── tools/build_traces.py      # taskspecs -> static/js/traces.js
├── static/
│   ├── css/style.css          # single stylesheet, light + dark
│   ├── js/
│   │   ├── viz.js             # shared canvas helpers + theme-aware palette
│   │   ├── scene.js           # draws a task scene by parsing a memory string
│   │   ├── traces.js          # GENERATED — real memory states, all 51 variants
│   │   ├── inspector.js       # the memory inspector
│   │   ├── charts.js          # success-rate chart + efficiency chart
│   │   ├── arch.js            # animated architecture diagram
│   │   └── main.js            # task grid, nav, theme, reveals
│   ├── figures/               # figures extracted from the paper PDF
│   ├── images/                # video poster + social card
│   └── videos/                # overview video
└── .nojekyll
```

## Diagrams are drawn, not exported

Every diagram except the photo-based figures is a `<canvas>` drawn at runtime. The
reason is the theme toggle: a rasterised PDF crop can only ever sit on a forced white
plate, which is what made the page look like a template.

Two rules follow from that, and breaking either one is the likely cause of a diagram
that looks wrong in one theme:

1. **Colour lives in `style.css` only.** `viz.js` reads the CSS custom properties at
   draw time. A colour hard-coded in a `.js` file will not follow a theme flip.
2. **Register for repaints.** A diagram must call `HelmViz.onTheme(draw)` and
   `window.addEventListener('resize', draw)`, or it will go stale.

`HelmViz.animate(el, step)` runs an rAF loop only while `el` is on screen, and paints
one frame immediately so nothing below the fold shows an empty canvas.

### Scroll reveals are gated on `.js`

An inline script in `<head>` sets `class="js"` on `<html>`, and the reveal styles are
scoped to `.js .reveal`. Without that guard a script error would leave every section
at `opacity: 0` and the page would read as blank rather than merely unanimated. Keep
the guard if you touch the reveal CSS.

## The memory inspector

`static/js/traces.js` is generated from `helm_datasets/taskspecs/` — the same task
specifications used to synthesise the training data. All 51 variants from the paper
are present, so nothing in the inspector is illustrative. Regenerate after editing a
taskspec:

```bash
python docs/tools/build_traces.py
```

The scene beside the memory panel is drawn purely by parsing the memory string
(`scene.js`) — no simulator, no scene graph. That is the paper's claim made literal:
explicit language memory is enough to reconstruct the world state, and a latent memory
vector could not drive the same picture. If you add a task, add a renderer there.

Memory strings come in exactly three shapes, which is what makes the parsing tractable:

```
Count: 2 (Goal: 3)
Progress: blue, green (Goal: blue, green, red)
task done (None)
```

`?task=E2` deep-links the inspector to a task, which is also how the inter-episode
behaviour gets checked without clicking.

## Numbers on this page

All success rates come from the paper (Fig. 5, Tables 1, 3, 7, 10), not from the
top-level `README.md`.

**The efficiency chart is the exception.** The paper plots Fig. 7 without tabulating
it, so `LAT` and `VRAM` in `charts.js` are digitised off that figure. The endpoints
match the values stated in the text (~28 GB vs under 6 GB, 4.5× the memory), but the
latency ratio lands at 2.3× rather than the 2.5× the abstract claims. The readout
therefore prints absolute values instead of a latency ratio, so the page does not
contradict its own prose. **If the exact measurements exist, put them in `charts.js`
and the readout can state the ratio directly.**

## Social preview card

`og:image` points at `static/images/og_card.jpg` via an **absolute** URL — scrapers
do not resolve relative paths. It is a JPEG, not the WebP source, because some
scrapers (LinkedIn especially) still do not render WebP previews.

It is generated from the teaser figure, which is already almost exactly the 1.91:1
card ratio, so no cropping is needed:

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
  `.visitor-map` card in both themes. **Do not** duplicate the tag to theme-switch it —
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

## Still open

Search `index.html` for `TODO(`:

1. `TODO(authors)` — the `.authors` / `.affil` block is **commented out**. Fill in the
   real author list and un-comment it.
2. `TODO(paper)` / `TODO(arxiv)` — both buttons render as dashed, non-clickable "soon"
   chips. Swap each back to an `<a>` with a real `href`.
3. `TODO(bibtex)` — no author list, and the entry is `@misc` rather than a venue entry.
   The venue is deliberately unclaimed anywhere on the page until the paper is
   accepted; `static/images/demo_poster.jpg` still says "CoRL 2026 Demo Video" because
   it is a frame of the video and would need re-rendering.

Also: the personal site still lives at `kimhuni.github.io` (the expiring school
account) and needs its own migration before linking to it is durable.

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
