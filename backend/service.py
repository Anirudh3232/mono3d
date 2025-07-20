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

# Mock classes for compatibility (unchanged from your code)
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

# Compatibility patches (unchanged)
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

# Logging setup (unchanged)
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
    """Optimized GPU memory clearing with reduced CPU overhead"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(clear_gpu_memory, '_gc_counter'):
            clear_gpu_memory._gc_counter += 1
        else:
            clear_gpu_memory._gc_counter = 0
        
        if clear_gpu_memory._gc_counter % 3 == 0:
            gc.collect()

def gpu_mem_mb():
    return (torch.cuda.memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0

class OptimizedParameters:
    """Optimized default parameters to reduce CPU usage"""
    
    DEFAULT_INFERENCE_STEPS = 30
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_N_VIEWS = 4
    DEFAULT_HEIGHT = 512
    DEFAULT_WIDTH = 512
    
    @classmethod
    def get_optimized_params(cls, data):
        """Get optimized parameters based on request data"""
        return {
            'num_inference_steps': int(data.get("num_inference_steps", cls.DEFAULT_INFERENCE_STEPS)),
            'guidance_scale': float(data.get("guidance_scale", cls.DEFAULT_GUIDANCE_SCALE)),
            'n_views': int(data.get("n_views", cls.DEFAULT_N_VIEWS)),
            'height': int(data.get("height", cls.DEFAULT_HEIGHT)),
            'width': int(data.get("width", cls.DEFAULT_WIDTH))
        }

class ResultCache:
    """Simple caching system to avoid redundant computations"""
    
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

# ROBUST MARCHING CUBES IMPLEMENTATION WITH MULTIPLE FALLBACKS
def setup_marching_cubes():
    """Setup marching cubes with robust fallback chain"""
    
    # Disable CUDA-only torchmcubes to force fallback
    os.environ["TSR_DISABLE_TORCHMCUBES"] = "1"
    
    try:
        # First try: torchmcubes (if available and compiled with matching CUDA)
        import torchmcubes
        logger.info("✅ Using torchmcubes")
        return True
    except (ImportError, AttributeError) as e:
        logger.warning(f"torchmcubes not available: {e}")
    
    try:
        # Second try: PyMCubes
        import PyMCubes
        import types
        import numpy as _np
        import torch as _torch
        
        def _marching_cubes(vol: _torch.Tensor, thresh: float = 0.0):
            """PyMCubes fallback for marching cubes"""
            v, f = PyMCubes.marching_cubes(vol.detach().cpu().numpy(), thresh)
            return (_torch.from_numpy(v).to(vol.device, dtype=vol.dtype),
                    _torch.from_numpy(f.astype(_np.int64)).to(vol.device))
        
        # Create stub module
        stub = types.ModuleType("torchmcubes")
        stub.marching_cubes = _marching_cubes
        sys.modules["torchmcubes"] = stub
        logger.info("✅ Using PyMCubes fallback")
        return True
        
    except ImportError:
        logger.warning("PyMCubes not available")
    
    try:
        # Third try: scikit-image (most reliable)
        from skimage import measure
        import types
        import numpy as _np
        import torch as _torch
        
        def _marching_cubes(vol: _torch.Tensor, thresh: float = 0.0):
            """Scikit-image fallback for marching cubes"""
            verts, faces, _, _ = measure.marching_cubes(
                vol.detach().cpu().numpy(), 
                level=thresh
            )
            return (_torch.from_numpy(verts).to(vol.device, dtype=vol.dtype),
                    _torch.from_numpy(faces.astype(_np.int64)).to(vol.device))
        
        # Create stub module
        stub = types.ModuleType("torchmcubes")
        stub.marching_cubes = _marching_cubes
        sys.modules["torchmcubes"] = stub
        logger.info("✅ Using scikit-image marching cubes fallback")
        return True
        
    except ImportError:
        logger.error("❌ No marching cubes implementation available")
        return False

# Setup TripoSR path and imports
def setup_triposr():
    """Setup TripoSR with proper path handling"""
    
    # Add TripoSR path
    triposr_path = "/content/mono3d/backend/TripoSR-main"
    if triposr_path not in sys.path:
        sys.path.insert(0, triposr_path)
        logger.info(f"Added TripoSR path: {triposr_path}")
    
    # Alternative path for local development
    alt_path = os.path.join(os.path.dirname(__file__), "TripoSR-main")
    if alt_path not in sys.path and os.path.exists(alt_path):
        sys.path.insert(0, alt_path)
        logger.info(f"Added alternative TripoSR path: {alt_path}")
    
    try:
        from tsr.system import TSR
        from tsr.utils import resize_foreground, remove_background
        logger.info("✅ TripoSR imported successfully")
        return TSR, resize_foreground, remove_background
    except ImportError as e:
        logger.error(f"❌ Failed to import TripoSR: {e}")
        
        # Try to clone if in Colab
        if 'google.colab' in sys.modules:
            logger.info("Attempting to clone TripoSR from GitHub...")
            try:
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/VAST-AI-Research/TripoSR.git", 
                    triposr_path
                ], check=True)
                logger.info("TripoSR cloned successfully")
                
                # Retry import
                from tsr.system import TSR
                from tsr.utils import resize_foreground, remove_background
                logger.info("✅ TripoSR imported after cloning")
                return TSR, resize_foreground, remove_background
            except Exception as clone_error:
                logger.error(f"Failed to clone TripoSR: {clone_error}")
        
        raise

print("Starting optimized service initialization …")
try:
    logger.info("Importing diffusers …")
    _flush()
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
    from controlnet_aux import CannyDetector

    # Setup marching cubes with fallbacks
    if not setup_marching_cubes():
        raise RuntimeError("No marching cubes implementation available")

    # Setup TripoSR
    TSR, resize_foreground, remove_background = setup_triposr()

    # Load TripoSR model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"        
    logger.info(f"Loading TripoSR model on {DEVICE}...")
    
    triposr_model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml", 
        weight_name="model.ckpt"
    )
    triposr_model.to(DEVICE)
    triposr_model.eval()
    logger.info("✅ TripoSR model loaded")

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
            "triposr_available": True,
            "device": DEVICE
        })

    # Test endpoint
    @app.route("/test", methods=["GET", "POST"])
    def test():
        return jsonify({"message": "TripoSR server is working!", "method": request.method})

    # Load models
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.9)

    logger.info("Loading edge detector …")
    _flush()
    app.edge_det = CannyDetector()
    
    logger.info("Loading ControlNet …")
    _flush()
    app.cnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(DEVICE)
    
    logger.info("Loading Stable Diffusion …")
    _flush()
    app.sd = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=app.cnet,
        torch_dtype=torch.float16).to(DEVICE)
    
    app.sd.scheduler = EulerAncestralDiscreteScheduler.from_config(app.sd.scheduler.config)
    
    try:
        app.sd.enable_xformers_memory_efficient_attention()
        logger.info("xformers enabled")
    except Exception:
        logger.warning("xformers not available — using plain attention")
    
    app.sd.enable_model_cpu_offload()
    app.sd.enable_attention_slicing()
    app.sd.enable_vae_slicing()

    logger.info("Setting up TripoSR …")
    _flush()

    def ensure_module_on_device(module, target_device):
        """Helper function to ensure all tensors in a module are on the right device"""
        if module is None:
            return None
        module.to(target_device)
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, torch.Tensor):
                    setattr(module, attr_name, attr.to(target_device))
                elif hasattr(attr, 'to'):
                    attr.to(target_device)
            except Exception:
                continue
        return module

    # Setup TripoSR model
    app.triposr = triposr_model
    app.triposr = ensure_module_on_device(app.triposr, DEVICE)
    
    if hasattr(app.triposr, 'renderer'):
        app.triposr.renderer = ensure_module_on_device(app.triposr.renderer, DEVICE)
        if hasattr(app.triposr.renderer, 'triplane'):
            app.triposr.renderer.triplane = app.triposr.renderer.triplane.to(DEVICE)

    if DEVICE == "cuda":
        app.triposr = app.triposr.half()
        if hasattr(app.triposr, 'renderer'):
            app.triposr.renderer = app.triposr.renderer.half()

    app.triposr.eval()
    logger.info(f"TripoSR loaded on {DEVICE}")
    _flush()

    logger.info("✅ All models ready")
    _flush()

    # ───────────── Optimized /generate endpoint ─────────────
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
            cache_key = f"{data['sketch'][:100]}_{data.get('prompt', '')}"
            cached_result = result_cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                cached_result.seek(0)
                return send_file(
                    cached_result,
                    as_attachment=True,
                    download_name='3d_render.png',
                    mimetype='image/png'
                )

            # Decode input image
            try:
                png = base64.b64decode(data["sketch"].split(",", 1)[1])
                pil = Image.open(io.BytesIO(png)).convert("RGBA")
                logger.info(f"Decoded sketch: {pil.size}")
            except Exception as e:
                return jsonify({"error": f"Bad image data: {str(e)}"}), 400

            prompt = data.get("prompt", "a clean 3-D asset")
            params = OptimizedParameters.get_optimized_params(data)
            logger.info(f"Processing prompt: '{prompt}' with params: {params}")

            # Edge detection
            logger.info("Generating Canny edges...")
            edge = app.edge_det(pil)
            del pil

            # Stable Diffusion
            logger.info("Running Stable Diffusion...")
            with torch.no_grad():
                concept = app.sd(
                    prompt, image=edge,
                    num_inference_steps=params['num_inference_steps'], 
                    guidance_scale=params['guidance_scale']
                ).images[0]
            del edge
            clear_gpu_memory()

            # Resize foreground for TripoSR
            logger.info("Resizing foreground...")
            concept = resize_foreground(concept, 1.0)

            # TripoSR scene generation
            logger.info("Generating 3D scene codes...")
            with torch.no_grad():
                with torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext():
                    codes = app.triposr([concept], device=DEVICE)
            clear_gpu_memory()

            # Render multiple views
            logger.info("Rendering views...")
            with torch.no_grad():
                with torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext():
                    views = app.triposr.render(
                        codes, 
                        n_views=params['n_views'], 
                        height=params['height'], 
                        width=params['width'], 
                        return_type="pil"
                    )[0]
            clear_gpu_memory()

            # Select sharpest view
            logger.info(f"Selecting sharpest from {len(views)} views...")
            final_image = sharpest(views)

            # Return PNG image
            buf = io.BytesIO()
            final_image.save(buf, "PNG")
            buf.seek(0)

            # Cache the result
            result_cache.put(cache_key, io.BytesIO(buf.getvalue()))

            clear_gpu_memory()
            _flush()
            logger.info("✅ Generation completed successfully")
            
            return send_file(
                buf,
                as_attachment=True,
                download_name='3d_render.png',
                mimetype='image/png'
            )

        except Exception as e:
            logger.error("Error in /generate", exc_info=True)
            _flush()
            clear_gpu_memory()
            return jsonify({"error": str(e)}), 500

    if __name__ == "__main__":
        logger.info("🚀 Starting TripoSR service on port 5000")
        app.run(host="0.0.0.0", port=5000)

except Exception:
    logger.error("❌ Error during initialization", exc_info=True)
    _flush()
    raise
