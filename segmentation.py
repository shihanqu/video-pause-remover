"""Turn a per-frame change curve into keep/cut segments.

Mirrors the algorithm in static/index.html (computeSegments) so the CLI and the
UI make identical decisions: hysteresis threshold -> absorb short stills ->
pad motion outward -> collapse to segments.
"""

from __future__ import annotations

import numpy as np


def compute_keep_segments(frac: np.ndarray, fps: int, duration: float,
                          tau: float, min_still_s: float = 0.4,
                          pad_s: float = 0.1, exit_ratio: float = 0.5) -> dict:
    """frac: fraction-of-area-changed per frame pair. tau: motion threshold (0..1);
    tau == 0 means any nonzero change counts as motion."""
    n = len(frac)
    exit_t = tau * exit_ratio
    min_still = round(min_still_s * fps)
    pad = round(pad_s * fps)

    motion = np.zeros(n, dtype=bool)
    m = frac[0] > tau
    for i in range(n):
        v = frac[i]
        m = (v > exit_t) if m else (v > tau)
        motion[i] = m

    # still runs shorter than the minimum pause stay as motion
    i = 0
    while i < n:
        if motion[i]:
            i += 1
            continue
        j = i
        while j < n and not motion[j]:
            j += 1
        if j - i < min_still:
            motion[i:j] = True
        i = j

    # pad motion outward
    if pad > 0:
        src = np.flatnonzero(motion)
        for i in src:
            motion[max(0, i - pad):i + pad + 1] = True

    keep, cut = [], []
    i = 0
    while i < n:
        j = i
        while j < n and motion[j] == motion[i]:
            j += 1
        a = i / fps
        b = duration if j >= n else j / fps
        (keep if motion[i] else cut).append([round(a, 4), round(b, 4)])
        i = j

    kept = sum(b - a for a, b in keep)
    return {
        "keep": keep,
        "cut": cut,
        "kept_seconds": round(kept, 3),
        "removed_seconds": round(duration - kept, 3),
        "removed_pct": round(100 * (1 - kept / duration), 1) if duration else 0.0,
    }
