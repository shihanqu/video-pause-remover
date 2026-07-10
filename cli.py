#!/usr/bin/env python
"""pause-remover CLI: cut still sections out of a video, no UI required.

Examples:
  python cli.py recording.mp4                        # export with defaults
  python cli.py recording.mp4 --dry-run --json      # inspect what would be cut
  python cli.py recording.mp4 --threshold 2 --min-pause 1.0 -o out.mp4
  python cli.py recording.mp4 --ignore 0.9,0.0,1.0,0.1   # mask a corner clock

Exit codes: 0 = success (including "nothing to cut"), 1 = error.
With --json, a single JSON object is printed to stdout; progress goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import analysis as ana
import export as exp
from segmentation import compute_keep_segments


def _rect(spec: str, mode: str) -> dict:
    try:
        x0, y0, x1, y1 = (float(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit(f"bad --{mode} rect '{spec}': expected X0,Y0,X1,Y1 in 0..1")
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "mode": mode}


def main() -> None:
    p = argparse.ArgumentParser(
        prog="pause-remover", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="video file (mp4/mov, h264/hevc)")
    p.add_argument("-o", "--output", help="output path (default: <input>.nostills.<ext>)")
    p.add_argument("--threshold", type=float, default=10.0, metavar="PCT",
                   help="percent of frame area that must change for a frame to count "
                        "as motion; 0 = any change at all is motion (default: 10)")
    p.add_argument("--min-pause", type=float, default=0.4, metavar="SEC",
                   help="only cut pauses at least this long (default: 0.4)")
    p.add_argument("--pad", type=float, default=0.1, metavar="SEC",
                   help="motion kept around each cut so movement never starts abruptly (default: 0.1)")
    p.add_argument("--mode", choices=["smart", "reencode"], default="smart",
                   help="smart = stream-copy source bit-identical, re-encode only "
                        "sub-GOP cut boundaries (default); reencode = re-encode everything")
    p.add_argument("--ignore", action="append", default=[], metavar="X0,Y0,X1,Y1",
                   help="normalized rect whose changes are ignored (repeatable)")
    p.add_argument("--focus", action="append", default=[], metavar="X0,Y0,X1,Y1",
                   help="normalized rect; only changes inside it count (repeatable)")
    p.add_argument("--backend", choices=["mlx", "cupy", "numpy"],
                   help="compute backend (default: auto - MLX on Apple, CuPy on NVIDIA, NumPy CPU)")
    p.add_argument("--dry-run", action="store_true", help="analyze and report cuts, do not export")
    p.add_argument("--json", action="store_true", help="machine-readable report on stdout")
    p.add_argument("--quiet", action="store_true", help="no progress output")
    args = p.parse_args()

    if args.backend:
        os.environ["PAUSE_REMOVER_BACKEND"] = args.backend
    src = Path(args.input)
    if not src.exists():
        sys.exit(f"error: {src} not found")
    out_path = Path(args.output) if args.output else src.with_name(f"{src.stem}.nostills{src.suffix}")

    progress = None
    if not args.quiet:
        progress = lambda pr: print(f"\ranalyzing… {pr*100:3.0f}%", end="", file=sys.stderr, flush=True)
    try:
        res = ana.analyze(str(src), progress_cb=progress)
    except Exception as e:
        sys.exit(f"error: analysis failed: {e}")
    if not args.quiet:
        print("", file=sys.stderr)

    meta = res["meta"]
    rects = [_rect(r, "ignore") for r in args.ignore] + [_rect(r, "focus") for r in args.focus]
    frac = ana.aggregate_curves(res, rects)["frac"]
    seg = compute_keep_segments(frac, meta["analysis_fps"], meta["duration"],
                                tau=args.threshold / 100.0,
                                min_still_s=args.min_pause, pad_s=args.pad)

    report = {
        "input": str(src),
        "duration": meta["duration"],
        "threshold_pct": args.threshold,
        "kept_seconds": seg["kept_seconds"],
        "removed_seconds": seg["removed_seconds"],
        "removed_pct": seg["removed_pct"],
        "pauses_cut": seg["cut"],
        "segments_kept": seg["keep"],
        "out_path": None,
    }

    if not args.dry_run and seg["cut"] and seg["keep"]:
        if not args.quiet:
            print(f"exporting {len(seg['keep'])} segments…", file=sys.stderr)
        try:
            report["export"] = exp.smart_cut(str(src), seg["keep"], meta, str(out_path), mode=args.mode)
            report["out_path"] = str(out_path)
        except Exception as e:
            sys.exit(f"error: export failed: {e}")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{src.name}: {meta['duration']:.1f}s, {len(seg['cut'])} pauses -> "
              f"remove {seg['removed_seconds']:.1f}s ({seg['removed_pct']}%)")
        for a, b in seg["cut"]:
            print(f"  cut {a:8.2f} – {b:8.2f}  ({b - a:.2f}s)")
        if report["out_path"]:
            e = report["export"]
            print(f"wrote {report['out_path']}: {e['out_duration']}s, "
                  f"{e['copied_pct']}% stream-copied, A/V delta {e['av_desync_ms']}ms, "
                  f"decode {'CLEAN' if not e['decode_errors'] else 'WARNINGS: ' + e['decode_errors']}")
        elif not args.dry_run:
            print("nothing to cut (or nothing to keep) — no file written")


if __name__ == "__main__":
    main()
