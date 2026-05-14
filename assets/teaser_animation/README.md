# Teaser-animation source

Backup of the source that produces the 46&nbsp;s explainer video shown on the project page (`docs/index.html` &rarr; "Video Explanation") and in the repo README.

The rendered MP4 lives at [`docs/assets/atv_teaser.mp4`](../../docs/assets/atv_teaser.mp4); the source here regenerates it.

## Files

- **`index.html`** &mdash; self-contained SVG + CSS + JS animation. Open it in a browser to preview the looping version, or load with `?record=1` to expose `window.setT(t)` for deterministic frame capture.
- **`record.mjs`** &mdash; Node script using `puppeteer-core` against the system-installed Google Chrome. Walks the timeline frame-by-frame and writes 1280&times;720 PNGs into `frames/`.
- **`package.json`** &mdash; pins `puppeteer-core` and exposes the `record`, `encode:mp4`, `encode:gif`, `build`, and `clean` scripts.
- **`package-lock.json`** &mdash; pinned dependency tree.

## Rebuild

```bash
cd assets/teaser_animation
npm install            # one time
npm run build          # record + encode mp4 + encode gif
```

Outputs land next to the scripts:
- `atv_teaser.mp4` (H.264, 1280&times;720, 30&nbsp;fps, ~1.1&nbsp;MB)
- `atv_teaser.gif` (960&times;540, 20&nbsp;fps, ~2.4&nbsp;MB)

After rebuilding, copy the MP4 over the one served by the project page:

```bash
cp atv_teaser.mp4 ../../docs/assets/atv_teaser.mp4
```

## Requirements

- Node 20 or newer.
- `ffmpeg` on `PATH` (the `encode:*` scripts use `libx264` and `palettegen`/`paletteuse`).
- Google Chrome at `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` (the path is hard-coded in `record.mjs` for macOS &mdash; adjust the `CHROME` constant if you're on Linux/Windows).

## Notes on the timeline

The animation is 46&nbsp;s total at 30&nbsp;fps (1380 frames), structured as:

| Range | Act |
|---|---|
| 0&ndash;16&nbsp;s | Scene phase &mdash; setup + Token Replacement Test demo cycling through 5 variants |
| 16&ndash;30&nbsp;s | Depth-task results (2&times;2 bar grid) with right-side row callouts and &sect;A.2 caveat |
| 30&ndash;42&nbsp;s | TRT beyond depth (Mirage / Mull-Tokens / CoVT table) + closing line |
| 42&ndash;46&nbsp;s | Final card &mdash; TRT as the contribution, finding as the sub-line |

`TOTAL_MS` in `index.html` and `TOTAL` in `record.mjs` must stay in sync.
