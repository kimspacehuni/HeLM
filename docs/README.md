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
│   │   ├── charts.js          # subtask / strict task success chart
│   │   └── main.js            # task discovery and nav
│   ├── figures/               # figures extracted from the paper PDF
│   ├── images/                # video poster + social card
│   └── videos/                # overview and six task recordings
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

The publication header leads with the full paper title, the authors and affiliation,
and Code / arXiv / Video links. The original framework demo video follows the header. An abstract precedes the problem,
method, memory explorer, task suite and results. The optional button-sequence
example and detailed tables are expandable.

The introductory example uses the `press_RGB` states from `traces.js`. It advances only when a
visitor selects a step. The memory explorer also starts paused. Task cards open
the corresponding task in the expandable explorer; keyboard controls and reduced-motion
preferences are supported. Main content stays visible if JavaScript fails.

## The memory inspector

`static/js/traces.js` is generated from `helm_datasets/taskspecs/` — the same task
specifications used to synthesise the training data. It includes 51 task variants.
Regenerate from the repository root after editing a taskspec:

```bash
python tools/build_page_traces.py
```

The explorer selects reference photographs for the current action or its outcome
when a verified example is available. Human demonstrations follow the observed
colour, and robot button-press photos follow the command. Window photos match
bottom / middle / top in the robot’s frame (confirmed against the source images). Repeated presses can
reuse the same reference. The photos are examples, not synchronized rollout frames.
When an exact drawer-position match is unavailable, the caption marks
the image as a setup reference.

The source photos in `static/images/tasks/` are cropped by 600 px (15%) on the
left of the 4000 × 3000 originals, then reduced to 600 × 529 WebP. The top, bottom
and right edges are retained. Native `<img>` elements keep `height: auto`; cards
limit display width to 260 px and the explorer to 320 px. EXIF data is excluded.

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

Content follows the manuscript supplied on 2026-11-01, titled
“HeLM: Hierarchical Explicit Language Memory for Non-Markovian Manipulation”.
Author order: Gyeonghun Kim, Donghoon Kim, Minsoo Choi, Suyun Lee;
affiliation: Seoul National University. Gyeonghun Kim links to the supplied homepage.

- Figure 5: intra-episode subtask success; Table 8: strict task success.
  Strict PCMB-π₀ and Naïve-π₀ results are zero for all tasks.
- Table 1: inter-episode subtask success.
- Tables 2 and 3: memory / HLP fine-tuning and two-stage inference ablations.
- Table 4: compositional generalization; Table 11: per-task LIBERO-Mem results.
- Figure 6: original paper plot for task-length robustness and memory scaling.
  The old digitized efficiency slider was removed. The text reports the paper's
  up-to-4.5× memory and 2.5× inference claims; these are not new measurements.

The paper evaluates **56 variants**. The repository explorer currently contains
**51 task-specification variants**, explicitly labelled on the page. The missing
five paper variants are not fabricated or counted as implemented examples.
Task categories use “Intra-episode tasks” and “Inter-episode tasks”; individual
names follow Table 6. Figure 2 was re-exported from the latest PDF with its entire
right-hand label retained.

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

## Recorded task videos

The original `helm_demo.mp4` is the main demo below the paper title.
The Real-world tasks section shows category buttons and a compact task-name list.
Selecting a task reveals only that task's panel; changing category defaults to A1
(intra-episode) or E2 (inter-episode). Switching tasks pauses the hidden recording.
The memory example opens from the selected task's “Inspect memory” button.

Intra-episode recordings map to A1, A3, A4 and A5; A2 uses a reference photograph.
Inter-episode tasks E1–E5 remain in the text list; E2 Human-Order Press is the only
inter-episode recording. Other inter-episode panels contain text and the memory
link, without photographs or videos. The E3 media files remain preserved but are
not referenced by the task UI.

Recordings are full speed and do not contain inferred memory logs or timestamps.
Web copies use H.264, 1280 × 720, CRF 24, AAC 96 kbps and faststart.
Posters are actual frames exported as 960 × 540 WebP. All players use
`preload="none"`, controls and inline playback. Only one player can play at a
time; changing the task filter pauses videos that become hidden.

The arXiv buttons link to https://arxiv.org/abs/2611.02660; the Paper PDF link
uses https://arxiv.org/pdf/2611.02660. The BibTeX entry includes this identifier.
