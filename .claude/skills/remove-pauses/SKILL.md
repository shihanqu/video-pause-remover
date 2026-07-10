---
name: remove-pauses
description: Cut still/pause sections out of a video at source quality, keeping only motion. Use when asked to remove pauses, dead time, or freezes from a screen recording or video, condense a video to constant movement, or find out where a video is still. Input mp4/mov; macOS Apple Silicon.
---

# Remove pauses from a video

This repo ships a GPU-accelerated pause remover with a headless CLI. Always use
the CLI (`cli.py`), not the web UI, unless the user asks to tune interactively.

## Setup (once per machine)

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Requires `ffmpeg` on PATH (`brew install ffmpeg`) and Apple Silicon (MLX).

## Workflow

**1. Dry-run first.** Analysis is cached, so this costs seconds and the real
export reuses it:

```sh
.venv/bin/python cli.py "input.mp4" --dry-run --json --quiet
```

Read `removed_pct` and `pauses_cut` (list of `[start, end]` seconds). Sanity-check
against the user's intent before writing anything:

- `removed_pct` near 0 → threshold too strict for this footage, raise `--threshold`.
- `removed_pct` implausibly high (cutting real content) → lower `--threshold`, or a
  persistent overlay (clock, webcam bubble, blinking cursor) is masking stillness →
  add `--ignore` over it.

**2. Export.**

```sh
.venv/bin/python cli.py "input.mp4" --json --quiet [-o "out.mp4"]
```

Default output is `<input>.nostills.<ext>` next to the source. Never overwrites
the input.

**3. Verify from the report** (all fields are in the `export` object):
`decode_errors` must be null, `av_desync_ms` small (< ~50), `copied_pct` is the
share of output that is bit-identical stream copy. Report these to the user.

## Flags

| flag | default | meaning |
|---|---|---|
| `--threshold PCT` | 10 | % of frame area that must change to count as motion. 0 = any change at all is motion (only frozen frames are cut). Typical: 0–2 for camera/noisy footage, 5–20 for screen recordings. |
| `--min-pause SEC` | 0.4 | only cut pauses at least this long |
| `--pad SEC` | 0.1 | motion padding kept around every cut |
| `--ignore X0,Y0,X1,Y1` | — | normalized (0–1) rect whose changes don't count; repeatable |
| `--focus X0,Y0,X1,Y1` | — | only changes inside this rect count; repeatable |
| `--mode reencode` | smart | full re-encode fallback if a player glitches at smart-cut boundaries |

## Cautions

- Audio during pauses is cut with the video: if the footage has narration over
  still frames (e.g. slides), warn the user before exporting.
- `"out_path": null` with exit code 0 means nothing crossed the threshold —
  not an error.
- Analysis cache lives in `cache/` keyed on file identity; safe to delete.
