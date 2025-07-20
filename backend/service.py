# backend/service.py
from __future__ import annotations

import io, os, sys, time, base64, gc, logging, atexit
from functools import wraps
from contextlib import nullcontext
from typing import Tuple

import numpy as np
import psutil
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── TripoSR local path ────────────────────────────────────────────────────────
TRIPOSR_PATH = os.path.join(os.path.dirname(__file__), "TripoSR-main")
sys.path.insert(0, TRIPOSR_PATH)

# Disable the optional CUDA-only torchmcubes extension → fallback to PyMCubes
os.environ["TSR_DISABLE_TORCHMCUBES"] = "1"

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("service.log")],
)
log = logging.getLogger(__name__)
atexit.register(lambda: (sys.stdout.flush(), sys.stderr.flush()))

# ── helpers ───────────────────────────────────────────────────────────────────
def timing(fn):
    @wraps(fn)
    def wrap(*a, **k):
        t0, cpu0 = time.time(), psutil.cpu_percent(None)
        out = fn(*a, **k)
        t1, cpu1 = time.time(), psutil.cpu_percent(None)
        log.info(f"{fn.__name__}: {t1 - t0:.2f}s | CPU {cpu0:.1f}%→{cpu1:.1f}%")
        return out
    return wrap


class LRU:
    """Tiny LRU cache (stores last 10 binary responses)."""
    def __init__(self, n: int = 10):
        self.n, self.d, self.o = n, {}, []

    def get(self, k):
        if k in self.d:
            self.o.remove(k)
            self.o.append(k)
            return self.d[k]

    def put(self, k, v):
        if len(self.d) >= self.n:
            del self.d[self.o.pop(0)]
        self.d[k] = v
        self.o.append(k)


def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# ── model loading ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"▶ loading models on {DEVICE}")

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
)
from controlnet_aux import CannyDetector
from tsr.system import TSR
from tsr.utils import resize_foreground  # remove_background unused here

edge_det = CannyDetector()

cnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
).to(DEVICE)

sd = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=cnet, torch_dtype=torch.float16
).to(DEVICE)
sd.scheduler = EulerAncestralDiscreteScheduler.from_config(sd.scheduler.config)
try:
    sd.enable_xformers_memory_efficient_attention()
except Exception:
    log.warning("xformers not available")
sd.enable_model_cpu_offload()
sd.enable_attention_slicing()

tsr = (
    TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    .to(DEVICE)
    .eval()
)
if hasattr(tsr, "renderer"):
    tsr.renderer.set_chunk_size(8192)

log.info("✔ models ready")

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
cache = LRU()


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        gpu_mb=(
            torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        ),
        cpu=psutil.cpu_percent(None),
        mem=psutil.virtual_memory().percent,
    )


def sharpest(img_list):
    """Return PIL image with the highest Laplacian variance (sharpest)."""
    import cv2

    scores = [
        cv2.Laplacian(cv2.cvtColor(np.array(i), cv2.COLOR_RGBA2GRAY), cv2.CV_64F).var()
        for i in img_list
    ]
    return img_list[int(np.argmax(scores))]


# ───────────────────────────── /generate ─────────────────────────────────────
@app.post("/generate")
@timing
def generate():
    if not request.is_json:
        return jsonify(error="JSON body required"), 400
    data = request.json
    if "sketch" not in data:
        return jsonify(error="missing 'sketch'"), 400

    key = data["sketch"][:120] + data.get("prompt", "")
    if (buf := cache.get(key)) is not None:
        return send_file(
            buf, mimetype="model/gltf-binary", download_name="model.glb", as_attachment=True
        )

    # 1 – Decode PNG → PIL
    try:
        png_bytes = base64.b64decode(data["sketch"].split(",", 1)[1])
        sketch = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception as e:
        return jsonify(error=f"bad image data: {e}"), 400

    prompt = data.get("prompt", "a clean 3-D asset")

    # 2 – Edge map
    edge = edge_det(sketch)
    del sketch

    # 3 – Stable Diffusion + ControlNet
    with torch.no_grad():
        concept = sd(
            prompt,
            image=edge,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]
    clear_gpu()

    # 4 – Foreground resize
    concept = resize_foreground(concept, 1.0)

    # 5 – TripoSR latent codes
    with torch.no_grad(), (
        torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext()
    ):
        codes = tsr([concept], device=DEVICE)
    clear_gpu()

    # 6 – Render & choose sharpest
    with torch.no_grad(), (
        torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext()
    ):
        views = tsr.render(codes, n_views=4, height=512, width=512, return_type="pil")[0]
    concept_view = sharpest(views)  # not strictly needed, kept for UI

    # 7 – Mesh extraction (64³ for full quality)
    with torch.no_grad(), (
        torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext()
    ):
        mesh = tsr.extract_mesh(codes, resolution=64, threshold=20.0)[0]
    del codes
    clear_gpu()

    # 8 – Parametrize + texture
    import xatlas, trimesh

    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    uv_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[vmapping],
        faces=indices,
        vertex_normals=mesh.vertex_normals[vmapping],
        visual=trimesh.visual.TextureVisuals(uv=uvs),
    )

    # create texture
    texture = concept.resize((1024, 1024))
    material = trimesh.visual.material.SimpleMaterial(image=texture)
    uv_mesh.visual.material = material

    # 9 – Export single binary glTF (.glb)
    glb_bytes = trimesh.exchange.gltf.export_glb(uv_mesh)
    buf = io.BytesIO(glb_bytes)

    cache.put(key, buf)
    clear_gpu()
    return send_file(
        buf, mimetype="model/gltf-binary", download_name="model.glb", as_attachment=True
    )


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
