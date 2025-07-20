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

# Mock classes for compatibility
class MockCache:
    def __init__(self, *args, **kwargs):
        pass
    def update(self, *args, **kwargs):
        pass
    def get_decoder_cache(self, *args, **kwargs):
        return self
    def get_encoder_cache(self, *args, **kwargs):
        return self

class MockEncoderDecoderCache(MockCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    @property
    def encoder(self):
        return self
    @property
    def decoder(self):
        return self

# Compatibility patches
import transformers
for _n in ("Cache", "DynamicCache", "EncoderDecoderCache"):
    if not hasattr(transformers, _n):
        setattr(transformers, _n, MockEncoderDecoderCache)

try:
    import transformers.cache_utils as _tcu
    for _n in ("Cache", "DynamicCache", "EncoderDecoderCache"):
        if not hasattr(_tcu, _n):
            setattr(_tcu, _n, MockEncoderDecoderCache)
except ImportError:
    pass

# Hugging Face compatibility
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "cached_download"):
    _hf_hub.cached_download = _hf_hub.hf_hub_download

# Accelerate compatibility
_acc_mem = importlib.import_module("accelerate.utils.memory")
if not hasattr(_acc_mem, "clear_device_cache"):
    _acc_mem.clear_device_cache = lambda *a, **k: None

# Attention processor patches
import diffusers.models.attention_processor
diffusers.models.attention_processor.AttnProcessor2_0 = MockCache

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('service.log')
    ]
)
logger = logging.getLogger(__name__)

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
    """Optimized GPU memory clearing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def gpu_mem_mb():
    return (torch.cuda.memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0

class ResultCache:
    """Simple caching system"""
    def __init__(self, max_size=10):
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

print("Starting service initialization...")
try:
    logger.info("Importing diffusers...")
    _flush()
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
    from controlnet_aux import CannyDetector
    
    # TripoSR setup
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"        
    logger.info(f"Loading TripoSR model on {DEVICE}...")
    
    triposr_path = os.path.join(os.getcwd(), "backend", "TripoSR-main")
    if triposr_path not in sys.path:
        sys.path.insert(0, triposr_path)
    
    try:
        from tsr.system import TSR
        logger.info("TSR module imported successfully")
        triposr_model = TSR()
        triposr_model.to(DEVICE)
        triposr_model.eval()
        logger.info("TripoSR model loaded")
    except ImportError as e:
        logger.error(f"Failed to import TSR: {e}")
        triposr_model = None
    
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
            "triposr_available": triposr_model is not None
        })

    # Test endpoint
    @app.route("/test", methods=["GET", "POST"])
    def test():
        return jsonify({"message": "Server is working!", "method": request.method})

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device}")
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info("Loading edge detector...")
    _flush()
    app.edge_det = CannyDetector()
    
    logger.info("Loading ControlNet...")
    _flush()
    app.cnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
    
    logger.info("Loading Stable Diffusion...")
    _flush()
    app.sd = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=app.cnet,
        torch_dtype=torch.float16).to(device)
    
    app.sd.scheduler = EulerAncestralDiscreteScheduler.from_config(app.sd.scheduler.config)
    
    try:
        app.sd.enable_xformers_memory_efficient_attention()
        logger.info("xformers enabled")
    except Exception:
        logger.warning("xformers not available")
    
    app.sd.enable_model_cpu_offload()
    app.sd.enable_attention_slicing()
    app.sd.enable_vae_slicing()

    logger.info("Setting up TripoSR...")
    _flush()
    
    app.triposr = triposr_model
    if app.triposr and device == "cuda":
        app.triposr = app.triposr.half()
    
    logger.info("✅ Models ready")
    _flush()

    # Main generate endpoint - returns PNG instead of ZIP
    @app.post("/generate")
    @timing
    def generate():
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            data = request.json
            if "sketch" not in data:
                return jsonify({"error": "Missing sketch"}), 400

            # Check cache
            cache_key = f"{data['sketch'][:100]}_{data.get('prompt', '')}"
            cached_result = result_cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                cached_result.seek(0)
                return send_file(cached_result, mimetype="image/png", download_name="3d_render.png", as_attachment=True)

            # Decode input
            try:
                png = base64.b64decode(data["sketch"].split(",", 1)[1])
                pil = Image.open(io.BytesIO(png)).convert("RGBA")
            except Exception as e:
                return jsonify({"error": f"Bad image data: {str(e)}"}), 400

            prompt = data.get("prompt", "a clean 3D asset")
            
            # Parameters
            num_inference_steps = int(data.get("num_inference_steps", 30))
            guidance_scale = float(data.get("guidance_scale", 7.5))

            logger.info(f"Processing prompt: '{prompt}'")

            # Edge detection
            edge = app.edge_det(pil)
            del pil

            # Stable Diffusion
            with torch.no_grad():
                concept = app.sd(
                    prompt, image=edge,
                    num_inference_steps=num_inference_steps, 
                    guidance_scale=guidance_scale
                ).images[0]
            del edge
            clear_gpu_memory()

            # Scene generation
            with torch.cuda.amp.autocast() if device == "cuda" else nullcontext():
                codes = app.triposr([concept], device=device)
            clear_gpu_memory()

            # Render 4 views and select sharpest (like your original working code)
            with torch.no_grad():
                with torch.cuda.amp.autocast() if device == "cuda" else nullcontext():
                    views = app.triposr.render(codes, n_views=4, height=512, width=512, return_type="pil")[0]
            
            # Select the sharpest view
            final_image = sharpest(views)
            
            clear_gpu_memory()

            # Return PNG image
            buf = io.BytesIO()
            final_image.save(buf, "PNG")
            buf.seek(0)

            # Cache the result
            result_cache.put(cache_key, io.BytesIO(buf.getvalue()))

            _flush()
            return send_file(buf, mimetype="image/png", download_name="3d_render.png", as_attachment=True)

        except Exception as e:
            logger.error("Error in /generate", exc_info=True)
            _flush()
            clear_gpu_memory()
            return jsonify({"error": str(e)}), 500

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000)

except Exception:
    logger.error("❌ Error during initialization", exc_info=True)
    _flush()
    raise
