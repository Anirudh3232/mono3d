#!/usr/bin/env python3
# service.py  –  Mono3D backend (optimised)

# ───────────────────────────────────────────────────────────────
# Standard & third-party imports
# ───────────────────────────────────────────────────────────────
from typing import Callable, Optional, Tuple
import sys, os, io, time, types, importlib, logging, atexit, gc, base64
from functools import wraps
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import psutil

# ───────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("service.log")],
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────
# Add TripoSR to path
# ───────────────────────────────────────────────────────────────
TRIPOSR_PATH = os.path.join(os.path.dirname(__file__), "TripoSR-main")
sys.path.insert(0, TRIPOSR_PATH)

# ───────────────────────────────────────────────────────────────
# torchmcubes fallback  →  pymcubes
# ───────────────────────────────────────────────────────────────
def _setup_torchmcubes_fallback() -> None:
    try:
        import torchmcubes                         # noqa: F401
        logger.info("✅ native torchmcubes found")
        return
    except ModuleNotFoundError:
        logger.info("🔧 torchmcubes missing – patching with pymcubes")

    try:
        import pymcubes as mc, torch, numpy as _np

        mod = types.ModuleType("torchmcubes")

        def marching_cubes(vol: torch.Tensor, thresh: float = 0.0):
            vol_np = vol.detach().cpu().numpy()
            v, f = mc.marching_cubes(vol_np, thresh)
            return (
                torch.from_numpy(v).to(vol.device),
                torch.from_numpy(f.astype(_np.int32)).to(vol.device),
            )

        mod.marching_cubes = marching_cubes          # type: ignore
        sys.modules["torchmcubes"] = mod
        logger.info("✅ pymcubes shim registered as torchmcubes")
    except ModuleNotFoundError as e:
        raise ImportError(
            "Neither torchmcubes nor pymcubes is available. "
            "Run `pip install pymcubes` or build torchmcubes."
        ) from e


_setup_torchmcubes_fallback()

# ───────────────────────────────────────────────────────────────
# Mock cache classes (bypass HF/Accelerate memory caches)
# ───────────────────────────────────────────────────────────────
class MockCache:
    def __init__(self, *_, **__):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, *args, **kwargs):
        return None

    def dim(self): return 0
    def size(self, dim=None): return (0,) if dim is None else 0
    def to(self, device): self.device = device; return self
    def update(self, *_, **__): pass
    def get_decoder_cache(self, *_, **__): return self
    def get_encoder_cache(self, *_, **__): return self
    def __getattr__(self, _): return self

class MockEncoderDecoderCache(MockCache):
    @property
    def encoder(self): return self
    @property
    def decoder(self): return self

# patch transformers / accelerate
import transformers, diffusers.models.attention_processor
for _n in ("Cache", "DynamicCache", "EncoderDecoderCache"):
    for _modname in (
        "transformers",
        "transformers.cache_utils",
        "transformers.models.encoder_decoder",
    ):
        try:
            _mod = importlib.import_module(_modname)
            if not hasattr(_mod, _n):
                setattr(_mod, _n, MockEncoderDecoderCache)
        except ImportError:
            pass
diffusers.models.attention_processor.AttnProcessor2_0 = MockCache
import transformers.models.llama.modeling_llama
transformers.models.llama.modeling_llama.AttnProcessor2_0 = MockCache
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "cached_download"):
    _hf_hub.cached_download = _hf_hub.hf_hub_download
_acc_mem = importlib.import_module("accelerate.utils.memory")
if not hasattr(_acc_mem, "clear_device_cache"):
    _acc_mem.clear_device_cache = lambda *a, **k: None

# ───────────────────────────────────────────────────────────────
# Utility decorators / helpers
# ───────────────────────────────────────────────────────────────
def _flush():
    sys.stdout.flush()
    sys.stderr.flush()
atexit.register(_flush)

def timing(fn):
    @wraps(fn)
    def wrap(*a, **k):
        t0, cpu0 = time.time(), psutil.cpu_percent(None)
        out = fn(*a, **k)
        t1, cpu1 = time.time(), psutil.cpu_percent(None)
        logger.info(f"{fn.__name__}: {t1-t0:.2f}s | CPU {cpu0:.1f}%→{cpu1:.1f}%")
        return out
    return wrap

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(clear_gpu, "_cnt"):
            clear_gpu._cnt += 1
        else:
            clear_gpu._cnt = 0
        if clear_gpu._cnt % 3 == 0:
            gc.collect()

def gpu_mem_mb() -> float:
    return torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0

# ───────────────────────────────────────────────────────────────
# Optimisation parameter helpers
# ───────────────────────────────────────────────────────────────
class OptimizedParameters:
    DEFAULT_INFERENCE_STEPS = 30   # was 63
    DEFAULT_GUIDANCE_SCALE = 7.5   # was 9.96
    DEFAULT_RENDER_RES   = 512

    @classmethod
    def get(cls, data):
        return dict(
            num_inference_steps=int(data.get("num_inference_steps", cls.DEFAULT_INFERENCE_STEPS)),
            guidance_scale=float(data.get("guidance_scale", cls.DEFAULT_GUIDANCE_SCALE)),
            render_resolution=int(data.get("render_resolution", cls.DEFAULT_RENDER_RES)),
        )

class LRU:
    def __init__(self, n=10):
        self.n, self.d, self.o = n, {}, []
    def get(self, k):
        if k in self.d:
            self.o.remove(k); self.o.append(k)
            return self.d[k]
    def put(self, k, v):
        if len(self.d) >= self.n:
            del self.d[self.o.pop(0)]
        self.d[k] = v; self.o.append(k)

cache = LRU()

# ───────────────────────────────────────────────────────────────
# Start heavy imports (after all monkey-patches)
# ───────────────────────────────────────────────────────────────
logger.info("Starting model initialisation …"); _flush()

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
from controlnet_aux import CannyDetector
import rembg

logger.info("Loading TripoSR from %s", TRIPOSR_PATH)
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

# ───────── device
DEV = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", DEV)
if DEV == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

# ───────── edge detector
edge_det = CannyDetector()

# ───────── ControlNet
cnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
).to(DEV)

# ───────── Stable Diffusion
sd = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=cnet,
    torch_dtype=torch.float16,
).to(DEV)
sd.scheduler = EulerAncestralDiscreteScheduler.from_config(sd.scheduler.config)
try:
    sd.enable_xformers_memory_efficient_attention()
    logger.info("xformers attention enabled")
except Exception:
    logger.warning("xformers unavailable, using default attention")
sd.enable_model_cpu_offload()
sd.enable_attention_slicing()
sd.enable_vae_slicing()

# ───────── TripoSR
triposr = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
if hasattr(triposr, "renderer"):
    triposr.renderer.set_chunk_size(8192)
triposr.to(DEV).eval()

logger.info("✔ all models ready"); _flush()

# ───────────────────────────────────────────────────────────────
# Flask app
# ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
rembg_session = rembg.new_session()
last_concept_image = None  # optional debugging endpoint

@app.get("/health")
def health():
    return jsonify(
        status="ok",
        gpu_mb=gpu_mem_mb(),
        cpu=psutil.cpu_percent(None),
        mem=psutil.virtual_memory().percent,
    )

@app.get("/latest_concept_image")
def latest():
    global last_concept_image
    if last_concept_image is None:
        return jsonify(error="no concept generated yet"), 404
    buf = io.BytesIO()
    last_concept_image.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

# ───────────────────────────────────────────────────────────────
# /generate endpoint
# ───────────────────────────────────────────────────────────────
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
        return send_file(buf, mimetype="image/png", download_name="3d.png", as_attachment=True)

    # decode PNG
    try:
        png_bytes = base64.b64decode(data["sketch"].split(",", 1)[1])
        sketch = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception as e:
        return jsonify(error=f"bad image data: {e}"), 400

    prm = data.get("prompt", "a clean 3-D asset")
    params = OptimizedParameters.get(data)

    # edge map
    edge = edge_det(sketch); del sketch

    # Stable Diffusion
    with torch.no_grad():
        concept = sd(
            prm, image=edge,
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
        ).images[0]
    clear_gpu()
    global last_concept_image
    last_concept_image = concept.copy()

    # background removal & resize
    try:
        proc = remove_background(concept, rembg_session)
        proc = resize_foreground(proc, 0.85)
        arr  = np.array(proc).astype(np.float32) / 255.0
        arr  = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        proc = Image.fromarray((arr * 255).astype(np.uint8))
    except Exception as e:
        logger.warning("rembg failed: %s – using original", e)
        proc = concept

    # TripoSR codes
    with torch.no_grad(), (autocast() if DEV == "cuda" else nullcontext()):
        codes = triposr([proc], device=DEV)
    clear_gpu()

    # render
    with torch.no_grad(), (autocast() if DEV == "cuda" else nullcontext()):
        imgs = triposr.render(
            codes, n_views=1,
            height=params["render_resolution"],
            width=params["render_resolution"],
            return_type="pil",
        )[0]

    img = imgs[0]
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    cache.put(key, buf)
    clear_gpu()

    return send_file(buf, mimetype="image/png", download_name="3d.png", as_attachment=True)

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
