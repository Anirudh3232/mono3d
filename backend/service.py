from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import sys, os, base64, io, gc, time, types, importlib, logging, atexit
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from functools import wraps
from torch.cuda.amp import autocast
from contextlib import nullcontext
import psutil
import subprocess

# ENHANCED COMPATIBILITY PATCHES - Must come first
import transformers

# Mock cache classes for compatibility
class MockCache:
    def __init__(self, *args, **kwargs):
        pass
    def update(self, *args, **kwargs):
        pass
    def get_decoder_cache(self, *args, **kwargs):
        return self
    def get_encoder_cache(self, *args, **kwargs):
        return self
    def __call__(self, *args, **kwargs):
        return args if args else None

class MockEncoderDecoderCache(MockCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    @property
    def encoder(self):
        return self
    @property
    def decoder(self):
        return self

class MockHybridCache(MockCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# Apply patches to transformers module
for cache_name, cache_class in [
    ("Cache", MockCache),
    ("DynamicCache", MockCache),
    ("EncoderDecoderCache", MockEncoderDecoderCache),
    ("HybridCache", MockHybridCache)
]:
    if not hasattr(transformers, cache_name):
        setattr(transformers, cache_name, cache_class)

# Ensure patches are available in __init__ for PEFT
transformers.__dict__.update({
    'Cache': MockCache,
    'DynamicCache': MockCache,
    'EncoderDecoderCache': MockEncoderDecoderCache,
    'HybridCache': MockHybridCache
})

# Additional patches for submodules
try:
    import transformers.cache_utils as _tcu
    for name, cls in [("Cache", MockCache), ("DynamicCache", MockCache), ("EncoderDecoderCache", MockEncoderDecoderCache)]:
        if not hasattr(_tcu, name):
            setattr(_tcu, name, cls)
except ImportError:
    pass

try:
    import transformers.models.encoder_decoder as _ted
    if not hasattr(_ted, "EncoderDecoderCache"):
        setattr(_ted, "EncoderDecoderCache", MockEncoderDecoderCache)
except ImportError:
    pass

# Continue with your existing compatibility patches
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "cached_download"):
    _hf_hub.cached_download = _hf_hub.hf_hub_download

_acc_mem = importlib.import_module("accelerate.utils.memory")
if not hasattr(_acc_mem, "clear_device_cache"):
    _acc_mem.clear_device_cache = lambda *a, **k: None

import diffusers.models.attention_processor
diffusers.models.attention_processor.AttnProcessor2_0 = MockCache

import transformers.models.llama.modeling_llama
transformers.models.llama.modeling_llama.AttnProcessor2_0 = MockCache

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        cpu_start = psutil.cpu_percent(interval=None)
        result = f(*args, **kwargs)
        end = time.time()
        cpu_end = psutil.cpu_percent(interval=None)
        logger.info(f"{f.__name__} took {end-start:.2f}s, CPU: {cpu_start:.1f}% -> {cpu_end:.1f}%")
        return result
    return wrapper

def _flush():
    """Force flush stdout/stderr to ensure logs are visible"""
    sys.stdout.flush()
    sys.stderr.flush()

atexit.register(_flush)

def clear_gpu_memory():
    """Aggressive GPU memory clearing for OOM prevention"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def gpu_mem_mb():
    return (torch.cuda.memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0

class OptimizedParameters:
    """Memory-optimized parameters to prevent CUDA OOM"""
    
    # Reduced for memory efficiency
    DEFAULT_INFERENCE_STEPS = 20  # Reduced from 30
    DEFAULT_GUIDANCE_SCALE = 7.0  # Reduced from 7.5
    DEFAULT_N_VIEWS = 2           # Reduced from 4 to save memory
    DEFAULT_HEIGHT = 256          # Reduced from 512 to save memory
    DEFAULT_WIDTH = 256           # Reduced from 512 to save memory
    
    @classmethod
    def get_optimized_params(cls, data):
        """Get memory-optimized parameters"""
        return {
            'num_inference_steps': int(data.get("num_inference_steps", cls.DEFAULT_INFERENCE_STEPS)),
            'guidance_scale': float(data.get("guidance_scale", cls.DEFAULT_GUIDANCE_SCALE)),
            'n_views': int(data.get("n_views", cls.DEFAULT_N_VIEWS)),
            'height': int(data.get("height", cls.DEFAULT_HEIGHT)),
            'width': int(data.get("width", cls.DEFAULT_WIDTH))
        }

class ResultCache:
    """Simple caching system to avoid redundant computations"""
    
    def __init__(self, max_size=5):  # Reduced cache size
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key):
        if key in self.cache:
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        if key not in self.access_order:
            self.access_order.append(key)

result_cache = ResultCache()

# Sharpest view selection function
def sharpest(img_list):
    """Return PIL image with highest Laplacian variance."""
    import cv2
    scores = [
        cv2.Laplacian(cv2.cvtColor(np.array(i), cv2.COLOR_RGBA2GRAY), cv2.CV_64F).var()
        for i in img_list
    ]
    return img_list[int(np.argmax(scores))]

print("Starting memory-optimized service initialization...")
try:
    logger.info("Importing diffusers...")
    _flush()
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
    from controlnet_aux import CannyDetector

    # TripoSR setup
    logger.info("Setting up TripoSR...")
    triposr_path = os.path.join(os.path.dirname(__file__), "TripoSR-main")
    if not os.path.exists(triposr_path):
        logger.info("Cloning TripoSR from GitHub...")
        subprocess.run(["git", "clone", "https://github.com/VAST-AI-Research/TripoSR.git", triposr_path], check=True)
        logger.info("✅ TripoSR cloned")

    if triposr_path not in sys.path:
        sys.path.insert(0, triposr_path)

    # Import TripoSR
    try:
        from tsr.system import TSR
        from tsr.utils import resize_foreground, remove_background
        logger.info("✅ TripoSR imported successfully")
    except ImportError as e:
        logger.error(f"❌ TripoSR import failed: {e}")
        raise

    # Load models with memory optimization
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading models on {DEVICE} with memory optimization...")
    
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.8)  # Reduced from 0.9

    # Load TripoSR with memory optimization
    triposr_model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml", 
        weight_name="model.ckpt"
    )
    
    # Aggressive memory optimization
    triposr_model.to(DEVICE)
    if DEVICE == "cuda":
        triposr_model = triposr_model.half()  # Use half precision
    triposr_model.eval()
    
    clear_gpu_memory()
    logger.info("✅ TripoSR loaded with memory optimization")

    app = Flask(__name__)
    CORS(app)

    # Health endpoint
    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok", 
            "gpu_mb": gpu_mem_mb(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "triposr_available": True
        })

    # Test endpoint
    @app.route("/test", methods=["GET", "POST"])
    def test():
        return jsonify({"message": "Memory-optimized server is working!", "method": request.method})

    # Load other models with memory management
    logger.info("Loading edge detector...")
    _flush()
    app.edge_det = CannyDetector()
    
    logger.info("Loading ControlNet...")
    _flush()
    app.cnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(DEVICE)
    
    logger.info("Loading Stable Diffusion...")
    _flush()
    app.sd = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=app.cnet,
        torch_dtype=torch.float16).to(DEVICE)
    
    app.sd.scheduler = EulerAncestralDiscreteScheduler.from_config(app.sd.scheduler.config)
    
    try:
        app.sd.enable_xformers_memory_efficient_attention()
        logger.info("xformers enabled")
    except Exception:
        logger.warning("xformers not available")
    
    # Memory optimization
    app.sd.enable_model_cpu_offload()
    app.sd.enable_attention_slicing()
    app.sd.enable_vae_slicing()
    
    if DEVICE == "cuda":
        app.sd = app.sd.half()

    def ensure_module_on_device(module, target_device):
        """Helper function to ensure all tensors are on the right device"""
        if module is None:
            return None
        module.to(target_device)
        return module

    # Setup TripoSR
    app.triposr = triposr_model
    app.triposr = ensure_module_on_device(app.triposr, DEVICE)
    
    clear_gpu_memory()
    logger.info("✅ All models loaded with memory optimization")
    _flush()

    # Memory-optimized generate endpoint
    @app.post("/generate")
    @timing
    def generate():
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            data = request.json
            if "sketch" not in data:
                return jsonify({"error": "Missing sketch"}), 400

            # Check cache first
            cache_key = f"{data['sketch'][:50]}_{data.get('prompt', '')}"  # Shorter key
            cached_result = result_cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                cached_result.seek(0)
                return send_file(cached_result, mimetype="image/png", download_name="3d_render.png", as_attachment=True)

            # Decode input
            try:
                png = base64.b64decode(data["sketch"].split(",", 1)[1])
                pil = Image.open(io.BytesIO(png)).convert("RGBA")
                # Resize input to reduce memory usage
                if pil.size[0] > 512 or pil.size[1] > 512:
                    pil = pil.resize((512, 512), Image.Resampling.LANCZOS)
            except Exception as e:
                return jsonify({"error": f"Bad image data: {str(e)}"}), 400

            prompt = data.get("prompt", "a clean 3-D asset")
            params = OptimizedParameters.get_optimized_params(data)
            logger.info(f"Using memory-optimized parameters: {params}")

            # Clear memory before each step
            clear_gpu_memory()

            # Edge detection
            edge = app.edge_det(pil)
            del pil
            clear_gpu_memory()

            # Stable Diffusion with memory management
            with torch.no_grad():
                concept = app.sd(
                    prompt, image=edge,
                    num_inference_steps=params['num_inference_steps'], 
                    guidance_scale=params['guidance_scale']
                ).images[0]
            del edge
            clear_gpu_memory()

            # Resize concept if needed
            concept = resize_foreground(concept, 1.0)
            clear_gpu_memory()

            # TripoSR with memory management
            with torch.no_grad():
                with torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext():
                    codes = app.triposr([concept], device=DEVICE)
            clear_gpu_memory()

            # Render with reduced parameters
            with torch.no_grad():
                with torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext():
                    views = app.triposr.render(
                        codes, 
                        n_views=params['n_views'], 
                        height=params['height'], 
                        width=params['width'], 
                        return_type="pil"
                    )[0]
            del codes
            clear_gpu_memory()

            # Select sharpest view
            final_image = sharpest(views)
            del views

            # Return PNG
            buf = io.BytesIO()
            final_image.save(buf, "PNG")
            buf.seek(0)

            # Cache result
            result_cache.put(cache_key, io.BytesIO(buf.getvalue()))

            clear_gpu_memory()
            _flush()
            return send_file(buf, mimetype="image/png", download_name="3d_render.png", as_attachment=True)

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM: {e}")
            clear_gpu_memory()
            return jsonify({"error": "GPU memory insufficient. Try reducing image size or parameters."}), 500
        except Exception as e:
            logger.error("Error in /generate", exc_info=True)
            clear_gpu_memory()
            return jsonify({"error": str(e)}), 500

    if __name__ == "__main__":
        logger.info("🚀 Starting memory-optimized TripoSR service")
        app.run(host="0.0.0.0", port=5000)

except Exception:
    logger.error("❌ Error during initialization", exc_info=True)
    _flush()
    raise
