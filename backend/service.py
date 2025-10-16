#!/usr/bin/env python3
# service.py  –  Mono3D backend (optimised)

# ───────────────────────────────────────────────────────────────
# Standard & third-party imports
# ───────────────────────────────────────────────────────────────
import sys
import os
import io
import time
import types
import importlib
import logging
import atexit
import gc
import base64
from functools import wraps
from contextlib import nullcontext

import numpy as np
import torch
from torch.cuda.amp import autocast

from PIL import Image, ImageFilter
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import psutil

# ───────────────────────────────────────────────────────────────
# Logging (define early so we can use logger in any fallback code)
# ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("service.log")],
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────
# Optional dependency shims (huggingface_hub / accelerate)
# ───────────────────────────────────────────────────────────────
try:
    import huggingface_hub as _hf_hub  # type: ignore

    if not hasattr(_hf_hub, "cached_download"):
        _hf_hub.cached_download = _hf_hub.hf_hub_download  # pragma: no cover
except Exception:  # pragma: no cover
    logger.warning("huggingface_hub not available; relying on diffusers internal download")
    _hf_hub = None

try:
    _acc_mem = importlib.import_module("accelerate.utils.memory")
    if not hasattr(_acc_mem, "clear_device_cache"):
        _acc_mem.clear_device_cache = lambda *a, **k: None
except Exception:  # pragma: no cover
    logger.warning("accelerate.utils.memory not available")

# peft shim: Some environments have incompatible peft↔huggingface_hub versions
# Diffusers optionally imports peft.PeftModel; if that import fails due to
# environment mismatch, we register a minimal stub so pipelines can load.
def _setup_peft_shim() -> None:
    try:
        import peft  # noqa: F401
        return
    except Exception as e:  # pragma: no cover
        logger.info(f"peft unavailable ({e.__class__.__name__}: {e}) – registering minimal shim")

    try:
        import torch.nn as nn
    except Exception:  # very unlikely; if torch missing nothing else works anyway
        nn = None  # type: ignore

    mod = types.ModuleType("peft")
    try:
        import importlib.machinery as _machinery
        mod.__spec__ = _machinery.ModuleSpec("peft", loader=None)  # type: ignore[attr-defined]
        mod.__path__ = []  # type: ignore[attr-defined]
    except Exception:
        pass

    if nn is not None:
        class PeftModel(nn.Module):  # type: ignore[name-defined]
            def __init__(self, base_model: nn.Module | None = None, *_a, **_k):
                super().__init__()
                self.base_model = base_model

            def forward(self, *a, **k):  # noqa: D401
                if self.base_model is None:
                    raise RuntimeError("peft shim: no base model provided")
                return self.base_model(*a, **k)

        mod.PeftModel = PeftModel  # type: ignore[attr-defined]
    else:
        def _noop(*_a, **_k):  # pragma: no cover
            raise RuntimeError("peft shim: torch.nn not available")

        mod.PeftModel = _noop  # type: ignore[attr-defined]

    # Provide a stable __file__ for inspection tools
    try:
        mod.__file__ = __file__  # type: ignore[attr-defined]
    except Exception:
        pass

    # Provide minimal submodule `peft.tuners` expected by some import paths
    tuners_mod = types.ModuleType("peft.tuners")
    try:
        import importlib.machinery as _machinery2
        tuners_mod.__spec__ = _machinery2.ModuleSpec("peft.tuners", loader=None)  # type: ignore[attr-defined]
        tuners_mod.__path__ = []  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        tuners_mod.__file__ = __file__  # type: ignore[attr-defined]
    except Exception:
        pass

    # Provide nested submodule `peft.tuners.tuners_utils` and make it permissive
    tuners_utils_mod = types.ModuleType("peft.tuners.tuners_utils")
    try:
        import importlib.machinery as _machinery3
        tuners_utils_mod.__spec__ = _machinery3.ModuleSpec("peft.tuners.tuners_utils", loader=None)  # type: ignore[attr-defined]
        tuners_utils_mod.__path__ = []  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        tuners_utils_mod.__file__ = __file__  # type: ignore[attr-defined]
    except Exception:
        pass

    # Minimal symbols used by some integrations
    class BaseTunerLayer:  # type: ignore[too-many-instance-attributes]
        def __init__(self, *_a, **_k):
            pass

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def train(self, _mode: bool = True):
            return self

        def state_dict(self, *_a, **_k):
            return {}

        def load_state_dict(self, *_a, **_k):
            return None

    tuners_utils_mod.BaseTunerLayer = BaseTunerLayer  # type: ignore[attr-defined]

    tuners_mod.tuners_utils = tuners_utils_mod  # type: ignore[attr-defined]

    mod.tuners = tuners_mod  # type: ignore[attr-defined]

    sys.modules["peft"] = mod
    sys.modules["peft.tuners"] = tuners_mod
    sys.modules["peft.tuners.tuners_utils"] = tuners_utils_mod

_setup_peft_shim()

# ───────────────────────────────────────────────────────────────
# Add TripoSR to path
# ───────────────────────────────────────────────────────────────
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
    if "backend" in current_dir:
        current_dir = os.path.join(current_dir, "backend")
    else:
        current_dir = os.path.join(current_dir, "backend")

TRIPOSR_PATH = os.path.join(current_dir, "TripoSR-main")
if TRIPOSR_PATH not in sys.path:
    sys.path.insert(0, TRIPOSR_PATH)

# ───────────────────────────────────────────────────────────────
# torchmcubes fallback  →  pymcubes (safe shim)
# ───────────────────────────────────────────────────────────────
def _setup_torchmcubes_fallback() -> None:
    try:
        import torchmcubes  # noqa: F401
        logger.info("✅ native torchmcubes found")
        return
    except Exception as e:
        # Catch both missing module and binary import errors (e.g., Colab/py3.12 ABI issues)
        logger.info(f"🔧 torchmcubes unavailable ({e.__class__.__name__}: {e}) – patching with pymcubes")

    try:
        import pymcubes as mc  # type: ignore

        mod = types.ModuleType("torchmcubes")

        def marching_cubes(vol: torch.Tensor, thresh: float = 0.0):
            vol_np = vol.detach().cpu().numpy()
            v, f = mc.marching_cubes(vol_np, thresh)
            return (
                torch.from_numpy(v).to(vol.device),
                torch.from_numpy(f.astype(np.int32)).to(vol.device),
            )

        mod.marching_cubes = marching_cubes  # type: ignore[attr-defined]
        sys.modules["torchmcubes"] = mod
        logger.info("✅ pymcubes shim registered as torchmcubes")
    except ModuleNotFoundError as e:
        logger.warning(f"Neither torchmcubes nor pymcubes is available: {e}")
        # Provide a dummy module to avoid import-time crashes in optional paths
        mod = types.ModuleType("torchmcubes")

        def _dummy(*_a, **_k):  # pragma: no cover
            raise RuntimeError("torchmcubes not available")

        mod.marching_cubes = _dummy  # type: ignore[attr-defined]
        sys.modules["torchmcubes"] = mod


_setup_torchmcubes_fallback()

# ───────────────────────────────────────────────────────────────
# Mock cache classes (bypass/neutralize some HF cache types if missing)
# ───────────────────────────────────────────────────────────────
class MockCache:
    def __init__(self, *_: object, **__: object) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, *_a: object, **_k: object) -> None:
        return None

    def dim(self) -> int:
        return 0

    def size(self, dim: int | None = None):
        return (0,) if dim is None else 0

    def to(self, device):
        self.device = device
        return self

    def update(self, *_a: object, **_k: object) -> None:  # pragma: no cover
        return None

    def get_decoder_cache(self, *_a: object, **_k: object):  # pragma: no cover
        return self

    def get_encoder_cache(self, *_a: object, **_k: object):  # pragma: no cover
        return self

    def __getattr__(self, _name: str):  # pragma: no cover
        return self


class MockEncoderDecoderCache(MockCache):
    @property
    def encoder(self):
        return self

    @property
    def decoder(self):
        return self


try:
    import transformers  # noqa: F401
    import diffusers.models.attention_processor  # noqa: F401

    for _cache_name in ("Cache", "DynamicCache", "EncoderDecoderCache"):
        for _modname in (
            "transformers",
            "transformers.cache_utils",
            "transformers.models.encoder_decoder",
        ):
            try:
                _mod = importlib.import_module(_modname)
                if not hasattr(_mod, _cache_name):
                    setattr(_mod, _cache_name, MockEncoderDecoderCache)
            except Exception:
                pass
except Exception:
    # Patching is best-effort; continue regardless
    pass

# ───────────────────────────────────────────────────────────────
# Utility decorators / helpers
# ───────────────────────────────────────────────────────────────
def _flush() -> None:
    sys.stdout.flush()
    sys.stderr.flush()


atexit.register(_flush)


def timing(fn):
    @wraps(fn)
    def wrap(*a, **k):
        t0, cpu0 = time.time(), psutil.cpu_percent(None)
        out = fn(*a, **k)
        t1, cpu1 = time.time(), psutil.cpu_percent(None)
        logger.info(f"{fn.__name__}: {t1 - t0:.2f}s | CPU {cpu0:.1f}%→{cpu1:.1f}%")
        return out

    return wrap


def clear_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(clear_gpu, "_cnt"):
            clear_gpu._cnt += 1  # type: ignore[attr-defined]
        else:
            clear_gpu._cnt = 0  # type: ignore[attr-defined]
        if clear_gpu._cnt % 3 == 0:  # type: ignore[attr-defined]
            gc.collect()
        logger.info(f"GPU memory cleared. Allocated: {gpu_mem_mb():.1f}MB")


def gpu_mem_mb() -> float:
    return torch.cuda.memory_allocated() / 1024.0**2 if torch.cuda.is_available() else 0.0


def gpu_mem_total_mb() -> float:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_properties(0).total_memory / 1024.0**2
        except Exception:
            return 0.0
    return 0.0


def log_memory_usage() -> None:
    if torch.cuda.is_available():
        allocated = gpu_mem_mb()
        total = max(gpu_mem_total_mb(), 1e-9)
        cpu_mem = psutil.virtual_memory().percent
        logger.info(
            f"GPU: {allocated:.1f}MB / {total:.1f}MB ({allocated / total * 100:.1f}%) | CPU RAM: {cpu_mem:.1f}%"
        )
    else:
        cpu_mem = psutil.virtual_memory().percent
        logger.info(f"CPU RAM: {cpu_mem:.1f}%")


# ───────────────────────────────────────────────────────────────
# Optimisation parameter helpers
# ───────────────────────────────────────────────────────────────
class GenerationParameters:
    DEFAULT_INFERENCE_STEPS = 60
    DEFAULT_GUIDANCE_SCALE = 10.0
    DEFAULT_RENDER_RES = 1024
    DEFAULT_UPSCALE_FACTOR = 2

    @classmethod
    def get(cls, data: dict):
        return dict(
            num_inference_steps=int(data.get("num_inference_steps", cls.DEFAULT_INFERENCE_STEPS)),
            guidance_scale=float(data.get("guidance_scale", cls.DEFAULT_GUIDANCE_SCALE)),
            render_resolution=int(data.get("render_resolution", cls.DEFAULT_RENDER_RES)),
            upscale_factor=int(data.get("upscale_factor", cls.DEFAULT_UPSCALE_FACTOR)),
        )


class LRU:
    def __init__(self, n: int = 10) -> None:
        self.n = n
        self.d: dict[str, io.BytesIO] = {}
        self.o: list[str] = []

    def get(self, k: str) -> io.BytesIO | None:
        if k in self.d:
            self.o.remove(k)
            self.o.append(k)
            return self.d[k]
        return None

    def put(self, k: str, v: io.BytesIO) -> None:
        if len(self.d) >= self.n:
            del self.d[self.o.pop(0)]
        self.d[k] = v
        self.o.append(k)


cache = LRU()

# ───────────────────────────────────────────────────────────────
# Start heavy imports (after all monkey-patches)
# ───────────────────────────────────────────────────────────────
logger.info("Starting model initialisation …")
_flush()

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
)
from controlnet_aux import CannyDetector
import rembg

logger.info("Loading TripoSR from %s", TRIPOSR_PATH)
try:
    from tsr.system import TSR  # type: ignore
    from tsr.utils import remove_background, resize_foreground  # type: ignore
    TRIPOSR_AVAILABLE = True
except Exception as e:  # pragma: no cover
    logger.warning(f"TripoSR not available: {e}")
    logger.info("Continuing with 2D image generation only")
    TRIPOSR_AVAILABLE = False

    def remove_background(img, session):  # type: ignore
        return img

    def resize_foreground(img, factor: float):  # type: ignore
        return img

# ───────── device
DEV = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", DEV)
if DEV == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Mixed precision policy: prefer fp16 on CUDA, fp32 otherwise, override with env
USE_FP32_ENV = os.environ.get("MONO3D_FP32", "false").lower() == "true"
if DEV == "cuda" and not USE_FP32_ENV:
    DTYPE = torch.float16
else:
    DTYPE = torch.float32

# ───────── edge detector
edge_det = CannyDetector()

# ───────── ControlNet
cnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=DTYPE,
)
if DEV == "cuda":
    cnet.to(DEV)

# ───────── Stable Diffusion
sd = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=cnet,
    torch_dtype=DTYPE,
)
if DEV == "cuda":
    sd.to(DEV)
sd.scheduler = EulerAncestralDiscreteScheduler.from_config(sd.scheduler.config)
try:
    sd.enable_xformers_memory_efficient_attention()  # optional
    logger.info("xformers attention enabled")
except Exception:
    logger.warning("xformers unavailable, using default attention")

if DEV == "cuda":
    # Prefer VRAM-friendly settings on GPU
    sd.enable_attention_slicing()
    sd.enable_vae_slicing()
    try:
        torch.cuda.set_per_process_memory_fraction(0.9)
    except Exception:
        pass
    logger.info("✅ GPU memory optimization enabled")
else:
    # CPU path kept minimal & stable
    try:
        sd.enable_model_cpu_offload()  # no-op if unsupported
        logger.info("⚠️ Using CPU offload")
    except Exception:
        pass

# ───────── TripoSR (optional, not required for 2D generation)
if TRIPOSR_AVAILABLE:
    try:
        triposr = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        if hasattr(triposr, "renderer"):
            try:
                triposr.renderer.set_chunk_size(8192)
            except Exception:
                pass
        triposr.to(DEV).eval()
        logger.info("✅ TripoSR loaded successfully")
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to load TripoSR: {e}")
        triposr = None
else:
    triposr = None
    logger.info("✅ Continuing without TripoSR (2D generation only)")

logger.info("✔ all models ready")
_flush()

# ───────────────────────────────────────────────────────────────
# Flask app
# ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
rembg_session = rembg.new_session()
last_concept_image: Image.Image | None = None


@app.get("/health")
def health():
    if torch.cuda.is_available():
        gpu_allocated = gpu_mem_mb()
        gpu_total = gpu_mem_total_mb()
        gpu_percent = (gpu_allocated / gpu_total * 100.0) if gpu_total > 0 else 0.0
    else:
        gpu_allocated = 0.0
        gpu_total = 0.0
        gpu_percent = 0.0

    return jsonify(
        status="ok",
        device=DEV,
        gpu_mb=round(gpu_allocated, 1),
        gpu_total_mb=round(gpu_total, 1),
        gpu_percent=round(gpu_percent, 1),
        cpu=psutil.cpu_percent(None),
        cpu_mem=psutil.virtual_memory().percent,
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
# Color and prompt utilities
# ───────────────────────────────────────────────────────────────
def enhance_image_quality(image: Image.Image) -> Image.Image:
    """Apply subtle enhancements to improve perceptual quality."""
    from PIL import ImageEnhance

    # Sharpness
    image = ImageEnhance.Sharpness(image).enhance(1.4)
    # Contrast
    image = ImageEnhance.Contrast(image).enhance(1.15)
    # Color
    image = ImageEnhance.Color(image).enhance(1.10)
    # Brightness
    image = ImageEnhance.Brightness(image).enhance(1.02)
    # Unsharp mask and a very light denoise
    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=2))
    image = image.filter(ImageFilter.MedianFilter(size=1))
    return image


def normalize_color(color_str: str | None) -> str:
    if not color_str:
        return ""

    color_map = {
        "red": "red",
        "blue": "blue",
        "green": "green",
        "yellow": "yellow",
        "purple": "purple",
        "orange": "orange",
        "pink": "pink",
        "brown": "brown",
        "black": "black",
        "white": "white",
        "gray": "gray",
        "grey": "gray",
        "gold": "golden",
        "silver": "silver",
        "cyan": "cyan",
        "magenta": "magenta",
        "lime": "lime green",
        "navy": "navy blue",
        "maroon": "maroon",
        "teal": "teal",
    }

    color_lower = color_str.lower().strip()
    return color_map.get(color_lower, color_lower)


# ───────────────────────────────────────────────────────────────
# Concept optimization function
# ───────────────────────────────────────────────────────────────
@timing
def optimize_concept(edge_image: Image.Image, prompt: str, params: dict) -> tuple[Image.Image, dict]:
    """Generate optimized concept image using Stable Diffusion with ControlNet."""
    enhanced_prompt = f"{prompt}, high quality, detailed, sharp, professional, masterpiece"

    try:
        with torch.no_grad(), (autocast(dtype=DTYPE) if DEV == "cuda" else nullcontext()):
            result = sd(
                prompt=enhanced_prompt,
                image=edge_image,
                num_inference_steps=int(params.get("num_inference_steps", GenerationParameters.DEFAULT_INFERENCE_STEPS)),
                guidance_scale=float(params.get("guidance_scale", GenerationParameters.DEFAULT_GUIDANCE_SCALE)),
                width=edge_image.size[0],
                height=edge_image.size[1],
            )

        concept_image = result.images[0]

        # Mild sharpness + contrast pass
        from PIL import ImageEnhance

        concept_image = ImageEnhance.Sharpness(concept_image).enhance(1.3)
        concept_image = ImageEnhance.Contrast(concept_image).enhance(1.1)
        return concept_image, params
    except Exception as e:
        logger.error(f"Error in optimize_concept: {str(e)}", exc_info=True)
        raise


# ───────────────────────────────────────────────────────────────
# /generate endpoint
# ───────────────────────────────────────────────────────────────
@app.post("/generate")
@timing
def generate():  # type: ignore[override]
    log_memory_usage()

    if not request.is_json:
        return jsonify(error="JSON body required"), 400
    data = request.json or {}
    if "sketch" not in data:
        return jsonify(error="missing 'sketch'"), 400

    key = (data.get("sketch", "")[:120] or "") + (data.get("prompt", "") or "")
    if (buf := cache.get(key)) is not None:
        buf.seek(0)
        return send_file(buf, mimetype="image/png", download_name="3d_image.png", as_attachment=True)

    try:
        # Decode input sketch data URI
        sketch_b64 = data["sketch"]
        if "," not in sketch_b64:
            raise ValueError("Invalid data URI format - missing comma separator")
        png_bytes = base64.b64decode(sketch_b64.split(",", 1)[1])
        sketch = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

        # Build prompt (optionally colorized)
        base_prompt = data.get("prompt", "a clean 3-D asset, beautiful, high quality")
        color = normalize_color(data.get("color"))
        prm = f"{base_prompt}, {color} color, vibrant {color} tones, {color} highlights" if color else base_prompt

        params = GenerationParameters.get(data)

        # Extract edges and run concept generation
        edge = edge_det(sketch)
        del sketch

        concept, best_params = optimize_concept(edge, prm, params)
        clear_gpu()

        global last_concept_image
        last_concept_image = concept.copy()

        # Background removal & composite on mid-gray, resize to keep foreground prominence
        try:
            proc = remove_background(concept, rembg_session)
            proc = resize_foreground(proc, 0.85)
            arr = np.array(proc).astype(np.float32) / 255.0
            arr = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
            proc = Image.fromarray((arr * 255.0).astype(np.uint8))
        except Exception as e:
            logger.warning(f"rembg failed: {e} – using original concept")
            proc = concept

        # Hi-res upscale and enhancement pipeline
        upscale_factor = max(1, int(params.get("upscale_factor", GenerationParameters.DEFAULT_UPSCALE_FACTOR)))
        upscale_size = (proc.size[0] * upscale_factor, proc.size[1] * upscale_factor)
        proc = proc.resize(upscale_size, Image.LANCZOS)

        target_res = int(params.get("render_resolution", GenerationParameters.DEFAULT_RENDER_RES))
        target_size = (target_res, target_res)
        proc = proc.resize(target_size, Image.LANCZOS)

        img = enhance_image_quality(proc)

        # Save result to buffer
        buf = io.BytesIO()
        img.save(buf, "PNG", compress_level=4)
        buf.seek(0)

        cache.put(key, buf)
        clear_gpu()
        return send_file(buf, mimetype="image/png", download_name="3d_image.png", as_attachment=True)
    except Exception as e:
        logger.error(f"Error in generate: {str(e)}", exc_info=True)
        return jsonify(error=f"Server error: {str(e)}"), 500


if __name__ == "__main__":
    # Standard entrypoint (avoids background-thread Jupyter pattern)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False, use_reloader=False)


