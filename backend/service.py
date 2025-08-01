#!/usr/bin/env python3
# service.py  — Mono3D backend
# ----------------------------------------------------------------------
# Generates a 2-D concept (SD-ControlNet) → removes backdrop →
# TripoSR → renders                © 2025
# ----------------------------------------------------------------------

import io, os, sys, time, base64, gc, logging, atexit, types
from functools import wraps
from contextlib import nullcontext

# ── third-party -------------------------------------------------------
import numpy as np
import psutil
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── paths -------------------------------------------------------------
TRIPOSR_PATH = os.path.join(os.path.dirname(__file__), "TripoSR-main")
sys.path.insert(0, TRIPOSR_PATH)

# ---------------------------------------------------------------------
#  Torchmcubes shim — **must run BEFORE the first TripoSR import**
# ---------------------------------------------------------------------
def setup_torchmcubes_fallback():
    """
    TripoSR imports `torchmcubes`. If the CUDA wheel isn’t present
    (e.g. on Colab), we emulate it with pure-Python PyMCubes.
    """
    try:
        import torchmcubes  # noqa: F401
        logging.info("✅ Using native torchmcubes extension")
        return
    except ImportError:
        logging.info("🔧 torchmcubes not found – falling back to PyMCubes")

    try:
        import PyMCubes, torch

        # create a fake module
        mcube_mod = types.ModuleType("torchmcubes")

        def marching_cubes(vol: torch.Tensor, thresh: float = 0.0):
            vol_np = vol.detach().cpu().numpy()
            verts, faces = PyMCubes.marching_cubes(vol_np, thresh)
            verts = torch.from_numpy(verts).to(vol.device)
            faces = torch.from_numpy(faces.astype(np.int32)).to(vol.device)
            return verts, faces

        mcube_mod.marching_cubes = marching_cubes  # type: ignore
        sys.modules["torchmcubes"] = mcube_mod
        logging.info("✅ PyMCubes shim registered as torchmcubes")

    except ImportError as e:
        raise ImportError(
            "Neither torchmcubes nor PyMCubes is available. "
            "Add `pymcubes` to requirements or install torchmcubes."
        ) from e


setup_torchmcubes_fallback()

# ---------------------------------------------------------------------
# Disable CUDA-only torchmcubes kernels (just in case)
os.environ["TSR_DISABLE_TORCHMCUBES"] = "1"

# ── logging -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("service.log")],
)
log = logging.getLogger(__name__)
atexit.register(lambda: (sys.stdout.flush(), sys.stderr.flush()))

# ── timing decorator --------------------------------------------------
def timing(fn):
    @wraps(fn)
    def wrap(*a, **k):
        start, cpu0 = time.time(), psutil.cpu_percent(None)
        result = fn(*a, **k)
        end, cpu1 = time.time(), psutil.cpu_percent(None)
        log.info(f"{fn.__name__}: {end - start:.2f}s | CPU {cpu0:.1f}%→{cpu1:.1f}%")
        return result

    return wrap


# ── tiny LRU cache (last 10 responses) --------------------------------
class LRU:
    def __init__(self, n=10):
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


cache = LRU()

# ── helpers -----------------------------------------------------------
def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# ── model loading -----------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"▶ loading models on {DEVICE}")

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
)
from controlnet_aux import CannyDetector

from tsr.system import TSR
from tsr.utils import resize_foreground, remove_background

edge_det = CannyDetector(low_threshold=64, high_threshold=128)

cnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
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
    log.warning("xformers not available")
sd.enable_model_cpu_offload()
sd.enable_attention_slicing()

tsr = (
    TSR.from_pretrained(
        "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
    )
    .to(DEVICE)
    .eval()
)
if hasattr(tsr, "renderer"):
    tsr.renderer.set_chunk_size(8192)

log.info("✔ models ready")

# ── Flask app ---------------------------------------------------------
app = Flask(__name__)
CORS(app)


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


# ── helper: sharpest of N views --------------------------------------
def sharpest(img_list):
    import cv2

    scores = [
        cv2.Laplacian(cv2.cvtColor(np.array(i), cv2.COLOR_RGBA2GRAY), cv2.CV_64F).var()
        for i in img_list
    ]
    return img_list[int(np.argmax(scores))]


# ── /generate ---------------------------------------------------------
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
            buf, mimetype="image/png", download_name="3d_image.png", as_attachment=True
        )

    try:
        png_bytes = base64.b64decode(data["sketch"].split(",", 1)[1])
        sketch = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception as e:
        return jsonify(error=f"bad image data: {e}"), 400

    prompt = data.get("prompt", "a clean 3-D asset")

    # 1) Canny edges from opaque RGB
    edge = edge_det(sketch.convert("RGB"))

    # 2) SD-ControlNet
    with torch.no_grad():
        concept = sd(
            prompt,
            image=edge,
            num_inference_steps=63,
            guidance_scale=9.96,
            height=768,
            width=768,
        ).images[0]
    clear_gpu()

    # 3) clean + centre
    concept = remove_background(concept)
    concept = resize_foreground(concept, 0.8)

    # 4) TripoSR
    with torch.no_grad(), (
        torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext()
    ):
        codes = tsr([concept], device=DEVICE)
    clear_gpu()

    # 5) render & pick sharpest
    with torch.no_grad(), (
        torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext()
    ):
        views = tsr.render(codes, n_views=4, height=512, width=512, return_type="pil")[0]
    img = sharpest(views)

    # 6) buffer → cache → send
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)

    cache.put(key, buf)
    clear_gpu()

    return send_file(
        buf, mimetype="image/png", download_name="3d_image.png", as_attachment=True
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
