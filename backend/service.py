# backend/service.py
from __future__ import annotations   # Python 3.8‒3.11 friendly postponed-evaluation

# ─────────────────────────────────────────────────────────────────────────────
#  Standard libs
# ─────────────────────────────────────────────────────────────────────────────
import io, os, sys, time, base64, gc, logging, atexit
from functools import wraps
from contextlib import nullcontext
from typing import Tuple, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
#  Third-party libs
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import psutil
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────────────────────
#  TripoSR local path & torchmcubes fallback
# ─────────────────────────────────────────────────────────────────────────────
TRIPOSR_PATH = os.path.join(os.path.dirname(__file__), "TripoSR-main")
sys.path.insert(0, TRIPOSR_PATH)

# Disable CUDA-only torchmcubes so TripoSR will attempt import but we stub it
os.environ["TSR_DISABLE_TORCHMCUBES"] = "1"

# ---- stub begins -----------------------------------------------------------
# If torchmcubes isn’t installed, create a tiny wrapper that forwards calls to
# pure-Python PyMCubes (already in requirements)
try:
    import torchmcubes  # noqa: F401 (real module present → nothing to do)
except ModuleNotFoundError:
    import types, PyMCubes, numpy as _np, torch as _torch

    def _marching_cubes(
        density: _torch.Tensor, thresh: float
    ) -> Tuple[_torch.Tensor, _torch.Tensor]:
        """Fallback marching-cubes using PyMCubes and returning torch tensors."""
        vol = density.detach().cpu().numpy().astype(_np.float32)
        verts, faces = PyMCubes.marching_cubes(vol, thresh)
        return _torch.from_numpy(verts), _torch.from_numpy(faces)

    stub = types.ModuleType("torchmcubes")
    stub.marching_cubes = _marching_cubes
    sys.modules["torchmcubes"] = stub
# ---- stub ends -------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("service.log")],
)
log = logging.getLogger(__name__)
atexit.register(lambda: (sys.stdout.flush(), sys.stderr.flush()))

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def timing(fn):
    """Decorator to log wall-time & CPU usage of endpoint calls."""
    @wraps(fn)
    def wrapper(*a, **k):
        t0, cpu0 = time.time(), psutil.cpu_percent(None)
        out = fn(*a, **k)
        t1, cpu1 = time.time(), psutil.cpu_percent(None)
        log.info(f"{fn.__name__}: {t1-t0:.2f}s | CPU {cpu0:.1f}%→{cpu1:.1f}%")
        return out
    return wrapper


class LRU:
    """Tiny LRU cache (holds the last *n* binary results in memory)."""
    def __init__(self, n: int = 10):
        self.n, self.d, self.o = n, {}, []

    def get(self, key: str):
        if key in self.d:
            self.o.remove(key)
            self.o.append(key)
            return self.d[key]

    def put(self, key: str, value: io.BytesIO):
        if len(self.d) >= self.n:
            del self.d[self.o.pop(0)]
        self.d[key] = value
        self.o.append(key)


def clear_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
#  Load models
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"▶ loading models on {DEVICE}")

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
)
from controlnet_aux import CannyDetector
from tsr.system import TSR
from tsr.utils import resize_foreground  # remove_background not needed here

edge_det = CannyDetector()

cnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
).to(DEVICE)

sd = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=cnet,
    torch_dtype=torch.float16,
).to(DEVICE)
sd.scheduler = EulerAncestralDiscreteScheduler.from_config(sd.scheduler.config)
try:
    sd.enable_xformers_memory_efficient_attention()
except Exception:
    log.warning("xformers not available – falling back to standard attention")
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

# ─────────────────────────────────────────────────────────────────────────────
#  Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
cache = LRU()

# Health-check ---------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify(
        status="ok",
        gpu_mb=(torch.cuda.memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0,
        cpu=psutil.cpu_percent(None),
        mem=psutil.virtual_memory().percent,
    )

# Utility: pick sharpest of several renders ----------------------------------
def sharpest(img_list):
    import cv2
    scores = [
        cv2.Laplacian(cv2.cvtColor(np.array(i), cv2.COLOR_RGBA2GRAY), cv2.CV_64F).var()
        for i in img_list
    ]
    return img_list[int(np.argmax(scores))]

# Main /generate endpoint ----------------------------------------------------
@app.post("/generate")
@timing
def generate():
    if not request.is_json:
        return jsonify(error="JSON body required"), 400
    data: Dict[str, Any] = request.json
    if "sketch" not in data:
        return jsonify(error="missing 'sketch'"), 400

    cache_key = data["sketch"][:120] + data.get("prompt", "")
    if (buf := cache.get(cache_key)) is not None:
        # Cache-hit: return previous GLB
        return send_file(
            buf,
            mimetype="model/gltf-binary",
            download_name="model.glb",
            as_attachment=True,
        )

    # 1. decode base64 PNG → PIL
    try:
        png_bytes = base64.b64decode(data["sketch"].split(",", 1)[1])
        sketch = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception as e:
        return jsonify(error=f"bad image data: {e}"), 400

    prompt = data.get("prompt", "a clean 3-D asset")

    # 2. Edge map
    edge = edge_det(sketch)
    del sketch

    # 3. Stable Diffusion + ControlNet
    with torch.no_grad():
        concept = sd(
            prompt,
            image=edge,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]
    clear_gpu()

    # 4. resize foreground
    concept = resize_foreground(concept, 1.0)

    # 5. TripoSR latent codes
    with torch.no_grad(), (
        torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext()
    ):
        codes = tsr([concept], device=DEVICE)
    clear_gpu()

    # 6. Render 4 views & choose sharpest (for UI preview / texture fallback)
    with torch.no_grad(), (
        torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext()
    ):
        views = tsr.render(codes, n_views=4, height=512, width=512, return_type="pil")[0]
    concept_view = sharpest(views)

    # 7. Extract high-res mesh (64³, threshold 20)
    with torch.no_grad(), (
        torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext()
    ):
        mesh = tsr.extract_mesh(codes, resolution=64, threshold=20.0)[0]
    del codes
    clear_gpu()

    # 8. UV-unwrap with xatlas & bake texture
    import xatlas, trimesh

    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    uv_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[vmapping],
        faces=indices,
        vertex_normals=mesh.vertex_normals[vmapping],
        visual=trimesh.visual.TextureVisuals(uv=uvs),
    )

    texture = concept.resize((1024, 1024))
    material = trimesh.visual.material.SimpleMaterial(image=texture)
    uv_mesh.visual.material = material

    # 9. Export as single binary glTF (.glb)
    glb_bytes = trimesh.exchange.gltf.export_glb(uv_mesh)
    buf = io.BytesIO(glb_bytes)

    cache.put(cache_key, buf)
    clear_gpu()
    return send_file(
        buf,
        mimetype="model/gltf-binary",
        download_name="model.glb",
        as_attachment=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
