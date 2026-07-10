"""Video motion analysis: VideoToolbox decode -> MLX GPU tile metrics -> cached curves.

One pass per video. Caches a per-frame *spatial grid* of change metrics (8px tiles
at analysis resolution), so region masks and thresholds re-aggregate instantly
without touching the video again.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent / "cache"
ANALYSIS_FPS = 30          # temporal grid for stillness decisions (source is VFR-safe: fps filter fills)
MAX_DIM = 288              # longest side of analysis frames
TILE = 8                   # tile size in analysis pixels -> region-mask granularity
CHUNK = 512                # frames per MLX batch
GATE_FLOOR = 1.5 / 255.0   # minimum per-pixel noise gate


def ffprobe_meta(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
        capture_output=True, check=True,
    ).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    duration = float(info["format"]["duration"])
    return {
        "path": path,
        "duration": duration,
        "width": v["width"],
        "height": v["height"],
        "codec": v["codec_name"],
        "profile": v.get("profile", ""),
        "pix_fmt": v.get("pix_fmt", "yuv420p"),
        "avg_fps": float(Fraction(v["avg_frame_rate"])) if v.get("avg_frame_rate", "0/0") != "0/0" else None,
        "nb_frames": int(v.get("nb_frames", 0)) or None,
        "has_audio": a is not None,
        "audio_rate": int(a["sample_rate"]) if a else None,
        "audio_channels": int(a["channels"]) if a else None,
        "size_bytes": int(info["format"]["size"]),
    }


def keyframe_times(path: str) -> list[float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,flags", "-of", "csv", path],
        capture_output=True, check=True,
    ).stdout.decode()
    kfs = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) >= 3 and "K" in parts[2] and parts[1] != "N/A":
            kfs.append(float(parts[1]))
    return sorted(kfs)


def _cache_key(path: str) -> str:
    st = os.stat(path)
    raw = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|v3"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _analysis_dims(w: int, h: int) -> tuple[int, int]:
    scale = MAX_DIM / max(w, h)
    aw = max(TILE, round(w * scale / TILE) * TILE)
    ah = max(TILE, round(h * scale / TILE) * TILE)
    return aw, ah


def analyze(path: str, progress_cb=None) -> dict:
    """Return cached analysis, computing it if needed.

    Result arrays:
      tile_mean: (N-1, Ty, Tx) float16 - mean |luma delta| per tile (0..1)
      tile_frac: (N-1, Ty, Tx) float16 - fraction of tile pixels changed beyond noise gate
    """
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(path)
    cache_file = CACHE_DIR / f"{key}.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        return {"meta": meta, "tile_mean": z["tile_mean"], "tile_frac": z["tile_frac"]}

    import mlx.core as mx

    meta = ffprobe_meta(path)
    aw, ah = _analysis_dims(meta["width"], meta["height"])
    tx, ty = aw // TILE, ah // TILE
    frame_bytes = aw * ah

    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-hwaccel", "videotoolbox",
        "-i", path,
        "-vf", f"fps={ANALYSIS_FPS},scale={aw}:{ah}:flags=area,format=gray",
        "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 64)

    t0 = time.time()
    means, fracs = [], []
    gate = None
    prev_tail: np.ndarray | None = None  # last frame of previous chunk (overlap)
    n_frames = 0
    expected = int(meta["duration"] * ANALYSIS_FPS) + 2

    def read_chunk(n: int) -> np.ndarray | None:
        want = n * frame_bytes
        buf = proc.stdout.read(want)
        if not buf:
            return None
        n_got = len(buf) // frame_bytes
        if n_got == 0:
            return None
        return np.frombuffer(buf[: n_got * frame_bytes], dtype=np.uint8).reshape(n_got, ah, aw)

    while True:
        chunk = read_chunk(CHUNK)
        if chunk is None:
            break
        n_frames += chunk.shape[0]
        if prev_tail is not None:
            chunk = np.concatenate([prev_tail[None], chunk], axis=0)
        prev_tail = chunk[-1].copy()

        x = mx.array(chunk).astype(mx.float32) / 255.0
        # kill global exposure / flicker shifts before differencing
        x = x - x.mean(axis=(1, 2), keepdims=True)
        d = mx.abs(x[1:] - x[:-1])
        if gate is None:
            # estimate per-pixel noise floor from this first batch (MAD-based)
            sample = np.array(d).ravel()[::13]
            sigma = 1.4826 * float(np.median(np.abs(sample)))
            gate = max(GATE_FLOOR, 4.0 * sigma)
        if d.shape[0] == 0:
            continue
        dt = d.reshape(d.shape[0], ty, TILE, tx, TILE)
        t_mean = dt.mean(axis=(2, 4)).astype(mx.float16)
        t_frac = (dt > gate).astype(mx.float32).mean(axis=(2, 4)).astype(mx.float16)
        mx.eval(t_mean, t_frac)
        means.append(np.array(t_mean))
        fracs.append(np.array(t_frac))
        if progress_cb:
            progress_cb(min(0.99, n_frames / expected))

    proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg decode failed (rc={rc})")
    if not means:
        raise RuntimeError("no frames decoded")

    tile_mean = np.concatenate(means, axis=0)
    tile_frac = np.concatenate(fracs, axis=0)
    elapsed = time.time() - t0

    meta.update({
        "analysis_fps": ANALYSIS_FPS,
        "analysis_dims": [aw, ah],
        "grid": [tx, ty],
        "tile": TILE,
        "gate": gate,
        "n_analysis_frames": n_frames,
        "n_pairs": int(tile_mean.shape[0]),
        "analysis_seconds": round(elapsed, 2),
        "analysis_fps_rate": round(n_frames / elapsed, 1),
        "keyframes": keyframe_times(path),
    })
    np.savez_compressed(
        cache_file,
        tile_mean=tile_mean, tile_frac=tile_frac,
        meta_json=np.array(json.dumps(meta)),
    )
    if progress_cb:
        progress_cb(1.0)
    return {"meta": meta, "tile_mean": tile_mean, "tile_frac": tile_frac}


def aggregate_curves(analysis: dict, rects: list[dict] | None = None) -> dict:
    """Collapse cached tile metrics into per-frame-pair curves under a region mask.

    rects: [{x0,y0,x1,y1, mode}] normalized 0..1 coords; mode 'ignore' or 'focus'.
    Instant - pure numpy over cached tiles, no video access.
    """
    tx, ty = analysis["meta"]["grid"]
    w = np.ones((ty, tx), dtype=bool)
    if rects:
        focus = [r for r in rects if r.get("mode") == "focus"]
        if focus:
            w[:] = False
            for r in focus:
                w[_tile_slice(r, tx, ty)] = True
        for r in rects:
            if r.get("mode") == "ignore":
                w[_tile_slice(r, tx, ty)] = False
    if not w.any():
        w[:] = True  # degenerate mask -> full frame
    wn = w.astype(np.float32) / w.sum()
    frac = (analysis["tile_frac"].astype(np.float32) * wn).sum(axis=(1, 2))
    mean = (analysis["tile_mean"].astype(np.float32) * wn).sum(axis=(1, 2))
    return {"frac": frac, "mean": mean, "mask_tiles": int(w.sum()), "total_tiles": int(tx * ty)}


def _tile_slice(r: dict, tx: int, ty: int) -> tuple[slice, slice]:
    x0 = int(np.clip(np.floor(min(r["x0"], r["x1"]) * tx), 0, tx))
    x1 = int(np.clip(np.ceil(max(r["x0"], r["x1"]) * tx), 0, tx))
    y0 = int(np.clip(np.floor(min(r["y0"], r["y1"]) * ty), 0, ty))
    y1 = int(np.clip(np.ceil(max(r["y0"], r["y1"]) * ty), 0, ty))
    return slice(y0, max(y1, y0 + 1)), slice(x0, max(x1, x0 + 1))


if __name__ == "__main__":
    import sys

    res = analyze(sys.argv[1], progress_cb=lambda p: print(f"\r{p*100:5.1f}%", end=""))
    print()
    m = res["meta"]
    print(json.dumps({k: v for k, v in m.items() if k != "keyframes"}, indent=2))
    curves = aggregate_curves(res)
    f = curves["frac"]
    print(f"frac curve: n={len(f)} zero={np.mean(f == 0)*100:.1f}% "
          f"p50={np.percentile(f, 50):.5f} p90={np.percentile(f, 90):.5f} max={f.max():.4f}")
