# HeLM project page

Static site served by GitHub Pages from `main` → `/docs`.
No build step, no dependencies — plain HTML/CSS/JS.

Live URL (after enabling Pages): <https://kimhuni.github.io/HeLM/>

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
│   ├── css/style.css          # single stylesheet, light + dark
│   ├── js/main.js             # task-card data + grid + theme toggle
│   ├── figures/               # figures extracted from the paper PDF
│   ├── images/demo_poster.jpg # poster frame for the overview video
│   └── videos/                # overview video + per-task rollouts
└── .nojekyll
```

## Before publishing

Search `index.html` for `TODO(` — three items are intentionally left unfilled so
that nothing fake renders in the meantime:

1. `TODO(authors)` — the `.authors` / `.affil` block is **commented out**. Fill in
   the real author list and un-comment it.
2. `TODO(paper)` / `TODO(arxiv)` — both buttons render as dashed, non-clickable
   "soon" chips (`<span class="btn disabled">`). Swap each back to an `<a>` with a
   real `href` when the PDF / arXiv entry exists.
3. `TODO(bibtex)` — the `author` field is omitted from the BibTeX entry.

Also link the page from the personal site (`kimhuni.github.io`) once it is live.

## Adding per-task rollout videos

Task cards render from the `TASKS` array at the top of `static/js/main.js`.
Each entry starts with `video: null`, which renders a "Rollout video pending"
placeholder. To fill one in, drop the encoded clip into `static/videos/` and set
the filename:

```js
{ id: 'A1', ..., video: 'A1_press_n_times.mp4' }
```

Cards play on hover and pause when scrolled off screen.

### Encoding rollouts

Keep clips small — they all load on one page. Target under 3 MB each:

```bash
ffmpeg -i raw.mp4 -vf "scale=-2:480" -c:v libx264 -crf 30 \
       -preset slow -pix_fmt yuv420p -movflags +faststart -an out.mp4
```

`-an` drops audio (rollouts do not need it) and `+faststart` moves the index to
the front so playback starts before the file finishes downloading.

GitHub blocks files over 100 MB and Pages sites should stay well under 1 GB
total. Current asset footprint is ~2.9 MB.

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

## Numbers on this page

All success rates come from the paper (Fig. 5, Tables 1, 3, 7, 10), not from the
top-level `README.md`.
