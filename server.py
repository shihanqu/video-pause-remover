"""Local web app for the pause remover: analysis state, curves, video serving, export."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

import analysis as ana
import export as exp

ROOT = Path(__file__).parent
PORT = 8765

app = FastAPI()
STATE: dict = {"status": "idle", "progress": 0.0, "path": None, "analysis": None, "error": None}
LOCK = threading.Lock()


def list_videos() -> list[str]:
    vids = [p.name for p in ROOT.iterdir()
            if p.suffix.lower() in (".mp4", ".mov", ".m4v")
            and ".nostills" not in p.name and not p.name.startswith(".")]
    return sorted(vids, key=lambda n: (ROOT / n).stat().st_mtime, reverse=True)


def start_analysis(path: Path) -> None:
    with LOCK:
        STATE.update(status="analyzing", progress=0.0, path=str(path), analysis=None, error=None)

    def work():
        try:
            res = ana.analyze(str(path), progress_cb=lambda p: STATE.update(progress=p))
            with LOCK:
                STATE.update(status="ready", analysis=res, progress=1.0)
        except Exception as e:  # surface to UI
            with LOCK:
                STATE.update(status="error", error=str(e))

    threading.Thread(target=work, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/state")
def state():
    out = {"status": STATE["status"], "progress": STATE["progress"],
           "path": STATE["path"], "error": STATE["error"], "files": list_videos()}
    if STATE["status"] == "ready":
        meta = dict(STATE["analysis"]["meta"])
        meta["n_keyframes"] = len(meta.pop("keyframes"))
        out["meta"] = meta
    return out


@app.post("/api/load")
async def load(req: Request):
    body = await req.json()
    path = ROOT / body["name"]
    if not path.exists():
        raise HTTPException(404, "file not found")
    start_analysis(path)
    return {"ok": True}


@app.post("/api/curves")
async def curves(req: Request):
    if STATE["status"] != "ready":
        raise HTTPException(409, "analysis not ready")
    body = await req.json()
    c = ana.aggregate_curves(STATE["analysis"], body.get("rects") or [])
    return JSONResponse({
        "frac": [round(float(x), 6) for x in c["frac"]],
        "mean": [round(float(x), 6) for x in c["mean"]],
        "fps": STATE["analysis"]["meta"]["analysis_fps"],
        "mask_tiles": c["mask_tiles"], "total_tiles": c["total_tiles"],
    })


@app.post("/api/export")
async def do_export(req: Request):
    if STATE["status"] != "ready":
        raise HTTPException(409, "analysis not ready")
    body = await req.json()
    src = Path(STATE["path"])
    out_path = src.with_name(f"{src.stem}.nostills{src.suffix}")
    try:
        report = exp.smart_cut(str(src), body["segments"], STATE["analysis"]["meta"],
                               str(out_path), mode=body.get("mode", "smart"))
    except Exception as e:
        raise HTTPException(500, str(e))
    return report


@app.post("/api/reveal")
async def reveal(req: Request):
    body = await req.json()
    p = Path(body["path"])
    if p.exists():
        subprocess.Popen(["open", "-R", str(p)])
    return {"ok": True}


@app.get("/video")
def video(request: Request):
    if not STATE["path"]:
        raise HTTPException(404)
    path = Path(STATE["path"])
    size = path.stat().st_size
    rng = request.headers.get("range")
    if not rng:
        return FileResponse(path, media_type="video/mp4")
    try:
        unit, _, spec = rng.partition("=")
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = min(int(end_s) if end_s else size - 1, size - 1)
    except ValueError:
        raise HTTPException(416)
    if start >= size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def stream(s=start, e=end):
        with open(path, "rb") as f:
            f.seek(s)
            left = e - s + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk

    return StreamingResponse(stream(), status_code=206, media_type="video/mp4", headers={
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
    })


def main():
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        vids = list_videos()
        if not vids:
            print("no video files found in", ROOT)
            sys.exit(1)
        target = ROOT / vids[0]
    start_analysis(target)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
