import typing as t
import sys, os, io, time, types, importlib, logging, atexit, gc, base64
from functools import wraps
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from PIL import Image, ImageFilter  # Re-added for unsharp mask
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import psutil


import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "cached_download"):
    _hf_hub.cached_download = _hf_hub.hf_hub_download
_acc_mem = importlib.import_module("accelerate.utils.memory")
if not hasattr(_acc_mem, "clear_device_cache"):
    _acc_mem.clear_device_cache = lambda *a, **k: None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler('service.log')],
)
logger = logging.getLogger(__name__)

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

def _setup_torchmcubes_fallback() -> None:
    try:
        import torchmcubes                         # noqa: F401
        logger.info(" native torchmcubes found")
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
        logger.info(" pymcubes shim registered as torchmcubes")
    except ModuleNotFoundError as e:
        raise ImportError(
            "Neither torchmcubes nor pymcubes is available. "
            "Run `pip install pymcubes` or build torchmcubes."
        ) from e


_setup_torchmcubes_fallback()

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
import transformers.models.llama.modeling_llama
transformers.models.llama.modeling_llama.AttnProcessor2_0 = MockCache

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
        logger.info(f"GPU memory cleared. Allocated: {gpu_mem_mb():.1f}MB")

def gpu_mem_mb() -> float:
    return torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0

def gpu_mem_total_mb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 2 if torch.cuda.is_available() else 0

def log_memory_usage():
    """Log current memory usage"""
    if torch.cuda.is_available():
        allocated = gpu_mem_mb()
        total = gpu_mem_total_mb()
        cpu_mem = psutil.virtual_memory().percent
        logger.info(f"GPU: {allocated:.1f}MB / {total:.1f}MB ({allocated/total*100:.1f}%) | CPU RAM: {cpu_mem:.1f}%")
    else:
        cpu_mem = psutil.virtual_memory().percent
        logger.info(f"CPU RAM: {cpu_mem:.1f}%")

class GenerationParameters:
    DEFAULT_INFERENCE_STEPS = 60   # Higher for more detail and quality
    DEFAULT_GUIDANCE_SCALE = 10.0  # Higher for better prompt adherence
    DEFAULT_RENDER_RES = 1024      # Higher resolution for better quality
    DEFAULT_UPSCALE_FACTOR = 2     # Moderate upscale to maintain quality

    @classmethod
    def get(cls, data):
        return dict(
            num_inference_steps=int(data.get("num_inference_steps", cls.DEFAULT_INFERENCE_STEPS)),
            guidance_scale=float(data.get("guidance_scale", cls.DEFAULT_GUIDANCE_SCALE)),
            render_resolution=int(data.get("render_resolution", cls.DEFAULT_RENDER_RES)),
            upscale_factor=int(data.get("upscale_factor", cls.DEFAULT_UPSCALE_FACTOR)),
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

logger.info("Starting model initialisation …"); _flush()

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
from controlnet_aux import CannyDetector
import rembg

logger.info("Loading TripoSR from %s", TRIPOSR_PATH)
try:
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
    TRIPOSR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TripoSR not available: {e}")
    logger.info("Continuing with 2D image generation only")
    TRIPOSR_AVAILABLE = False
    # Create dummy functions for compatibility
    def remove_background(img, session):
        return img
    def resize_foreground(img, factor):
        return img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", DEV)
if DEV == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

USE_FP32 = os.environ.get("MONO3D_FP32", "false").lower() == "true"
DTYPE = torch.float32 if USE_FP32 else torch.float16

edge_det = CannyDetector()

cnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=DTYPE,
).to(DEV)

sd = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=cnet,
    torch_dtype=DTYPE,
).to(DEV)
sd.scheduler = EulerAncestralDiscreteScheduler.from_config(sd.scheduler.config)
try:
    sd.enable_xformers_memory_efficient_attention()
    logger.info("xformers attention enabled")
except Exception:
    logger.warning("xformers unavailable, using default attention")

if DEV == "cuda":
    sd.enable_attention_slicing()
    sd.enable_vae_slicing()
    torch.cuda.set_per_process_memory_fraction(0.9)
    logger.info("✅ GPU memory optimization enabled")
else:
    # Fallback for CPU
    sd.enable_model_cpu_offload()
    logger.info("⚠️ Using CPU offload (not recommended for performance)")

if TRIPOSR_AVAILABLE:
    try:
        triposr = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        if hasattr(triposr, "renderer"):
            triposr.renderer.set_chunk_size(8192)
        triposr.to(DEV).eval()
        logger.info("✅ TripoSR loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load TripoSR: {e}")
        triposr = None
else:
    triposr = None
    logger.info("✅ Continuing without TripoSR (2D generation only)")

logger.info("✔ all models ready"); _flush()

app = Flask(__name__)
CORS(app)
rembg_session = rembg.new_session()
last_concept_image = None  # optional debugging endpoint

@app.get("/health")
def health():
    if torch.cuda.is_available():
        gpu_allocated = gpu_mem_mb()
        gpu_total = gpu_mem_total_mb()
        gpu_percent = (gpu_allocated / gpu_total) * 100
    else:
        gpu_allocated = 0
        gpu_total = 0
        gpu_percent = 0
    
    return jsonify(
        status="ok",
        device=DEV,
        gpu_mb=gpu_allocated,
        gpu_total_mb=gpu_total,
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

def enhance_image_quality(image):
    """Apply advanced image enhancement to match reference quality"""
    from PIL import ImageEnhance, ImageFilter
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.4)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.15)
    
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.1)
    
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.02)
    
    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=2))
    
    image = image.filter(ImageFilter.MedianFilter(size=1))
    
    return image
def normalize_color(color_str):
    """Normalize and validate color input"""
    if not color_str:
        return ""
    
    # Common color mappings
    color_map = {
        "red": "red", "blue": "blue",  "green": "green", "yellow": "yellow", "purple": "purple", "orange": "orange","pink": "pink", "brown": "brown",
        "black": "black", "white": "white", "gray": "gray","grey": "gray","gold": "golden","silver": "silver",
        "cyan": "cyan", "magenta": "magenta","lime": "lime green", "navy": "navy blue", "maroon": "maroon","teal": "teal"
    }
    
    color_lower = color_str.lower().strip()
    return color_map.get(color_lower, color_lower)

@timing
def optimize_concept(edge_image, prompt):
    """Generate optimized concept image using Stable Diffusion with ControlNet"""
    try:
        enhanced_prompt = f"{prompt}, high quality, detailed, sharp, professional, masterpiece"
        
        with torch.no_grad(), (autocast(dtype=DTYPE) if DEV == "cuda" else nullcontext()):
            result = sd(
                prompt=enhanced_prompt,
                image=edge_image,
                num_inference_steps=GenerationParameters.DEFAULT_INFERENCE_STEPS,
                guidance_scale=GenerationParameters.DEFAULT_GUIDANCE_SCALE,
                width=edge_image.size[0],
                height=edge_image.size[1],
            )
        
        concept_image = result.images[0]
        
        from PIL import ImageEnhance
        
        enhancer = ImageEnhance.Sharpness(concept_image)
        concept_image = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Contrast(concept_image)
        concept_image = enhancer.enhance(1.1)
        
        return concept_image, {
            "num_inference_steps": GenerationParameters.DEFAULT_INFERENCE_STEPS,
            "guidance_scale": GenerationParameters.DEFAULT_GUIDANCE_SCALE
        }
    except Exception as e:
        logger.error(f"Error in optimize_concept: {str(e)}", exc_info=True)
        raise

@app.post("/generate")
@timing
def generate():
    log_memory_usage()
    
    if not request.is_json:
        return jsonify(error="JSON body required"), 400
    data = request.json
    if "sketch" not in data:
        return jsonify(error="missing 'sketch'"), 400

    key = data["sketch"][:120] + data.get("prompt", "")
    if (buf := cache.get(key)) is not None:
        buf.seek(0)
        return send_file(buf, mimetype="image/png", download_name="3d_image.png", as_attachment=True)

    try:
        if "," not in data["sketch"]:
            raise ValueError("Invalid data URI format - missing comma separator")
        png_bytes = base64.b64decode(data["sketch"].split(",", 1)[1])
        sketch = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

        base_prompt = data.get("prompt", "a clean 3-D asset, beautiful, high quality")
        color = normalize_color(data.get("color", ""))
        
        if color:
            prm = f"{base_prompt}, {color} color, vibrant {color} tones, {color} highlights"
        else:
            prm = base_prompt
            
        params = GenerationParameters.get(data)
        edge = edge_det(sketch); del sketch

        concept, best_params = optimize_concept(edge, prm)
        clear_gpu()
        global last_concept_image
        last_concept_image = concept.copy()

        # Background removal & resize
        try:
            proc = remove_background(concept, rembg_session)
            proc = resize_foreground(proc, 0.85)
            arr  = np.array(proc).astype(np.float32) / 255.0
            arr  = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
            proc = Image.fromarray((arr * 255).astype(np.uint8))
        except Exception as e:
            logger.warning("rembg failed: %s – using original", e)
            proc = concept

        upscale_size = (proc.size[0] * params["upscale_factor"], proc.size[1] * params["upscale_factor"])
        proc = proc.resize(upscale_size, Image.LANCZOS)


        target_size = (params["render_resolution"], params["render_resolution"])
        concept = concept.resize(target_size, Image.LANCZOS)
        
        img = enhance_image_quality(concept)

        buf = io.BytesIO()
        img.save(buf, "PNG", compress_level=4)
        buf.seek(0)

        cache.put(key, buf)
        clear_gpu()
        return send_file(buf, mimetype="image/png", download_name="3d_image.png", as_attachment=True)
    except Exception as e:
        logger.error(f"Error in generate: {str(e)}", exc_info=True)
        return jsonify(error=f"Server error: {str(e)}"), 500

import threading

def run_app():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

thread = threading.Thread(target=run_app)
thread.daemon = True
thread.start()

print(" Service running in background!")
print("Use other cells to test it!")
