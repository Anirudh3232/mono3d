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


try:
    import transformers.models.encoder_decoder as _ted

    if not hasattr(_ted, "EncoderDecoderCache"):
        setattr(_ted, "EncoderDecoderCache", MockEncoderDecoderCache)
except ImportError:
    pass


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
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

atexit.register(_flush)



def clear_gpu_memory():
    """Optimized GPU memory clearing with reduced CPU overhead"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def gpu_mem_mb():
    return (torch.cuda.memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0


class OptimizedParameters:
    """Optimized default parameters to reduce CPU usage"""
    
    # Reduced inference steps for faster generation
    DEFAULT_INFERENCE_STEPS = 30  # Reduced from 63
    DEFAULT_GUIDANCE_SCALE = 7.5  # Reduced from 9.96 for faster convergence
    
    # Optimized mesh parameters
    DEFAULT_MESH_THRESHOLD = 20.0  # Reduced from 25.0 for faster extraction
    DEFAULT_SMOOTHING_ITERATIONS = 0  # Disabled by default to reduce CPU
    
    # Preview mode optimizations
    PREVIEW_RESOLUTION = 24  # Reduced from 32
    FULL_RESOLUTION = 64    # Reduced from 128
    
    @classmethod
    def get_optimized_params(cls, data):
        """Get optimized parameters based on request data"""
        return {
            'num_inference_steps': int(data.get("num_inference_steps", cls.DEFAULT_INFERENCE_STEPS)),
            'guidance_scale': float(data.get("guidance_scale", cls.DEFAULT_GUIDANCE_SCALE)),
            'smoothing_iterations': int(data.get("smoothing_iterations", cls.DEFAULT_SMOOTHING_ITERATIONS)),
            'mesh_threshold': float(data.get("mesh_threshold", cls.DEFAULT_MESH_THRESHOLD)),
            'preview_resolution': cls.PREVIEW_RESOLUTION,
            'full_resolution': cls.FULL_RESOLUTION
        }


class ResultCache:
    """Simple caching system to avoid redundant computations"""
    
    def __init__(self, max_size=10):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key):
        if key in self.cache:
            # Move to end of access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        if key not in self.access_order:
            self.access_order.append(key)


result_cache = ResultCache()



print("Starting optimized service initialization …")
try:
    logger.info("Importing diffusers …");
    _flush()
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
    # TripoSR imports
    TRIPOSR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TripoSR-main")
    sys.path.append(TRIPOSR_PATH)
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
    import rembg

    # Load TripoSR model at startup
    CHECKPOINT_DIR = os.path.join(TRIPOSR_PATH, "checkpoints")
    CONFIG_NAME = "config.yaml"
    WEIGHT_NAME = "model.ckpt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading TripoSR model on {DEVICE} ...")
    triposr_model = TSR.from_pretrained(
        CHECKPOINT_DIR,
        config_name=CONFIG_NAME,
        weight_name=WEIGHT_NAME,
    )
    triposr_model.to(DEVICE)
    triposr_model.eval()
    logger.info("TripoSR model loaded.")
    rembg_session = rembg.new_session()

    
    app = Flask(__name__);
    CORS(app)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/generate")
    def generate():
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            data = request.json
            if "sketch" not in data:
                return jsonify({"error": "Missing sketch"}), 400

            # Decode input image
            try:
                png = base64.b64decode(data["sketch"].split(",", 1)[1])
                pil = Image.open(io.BytesIO(png)).convert("RGBA")
            except Exception as e:
                return jsonify({"error": f"Bad image data: {str(e)}"}), 400

            # Preprocess: remove background and resize
            pil = remove_background(pil, rembg_session)
            pil = resize_foreground(pil, ratio=0.85)
            pil = pil.convert("RGB")

            # Run TripoSR model
            with torch.no_grad():
                scene_codes = triposr_model([pil], device=DEVICE)
                render_images = triposr_model.render(scene_codes, n_views=1, return_type="pil")
                rendered_img = render_images[0][0]  # First view of first image

            # Return rendered PNG
            output_buffer = io.BytesIO()
            rendered_img.save(output_buffer, format="PNG")
            output_buffer.seek(0)

            clear_gpu_memory()
            _flush()
            return send_file(
                output_buffer,
                mimetype="image/png",
                as_attachment=False
            )

        except Exception as e:
            logger.error("Error in /generate", exc_info=True)
            _flush()
            clear_gpu_memory()
            return jsonify({"error": str(e)}), 500

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000)

except Exception:
    logger.error("❌ Error during initialization", exc_info=True);
    _flush()
    raise
