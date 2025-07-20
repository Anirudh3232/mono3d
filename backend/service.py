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

# Continue with existing compatibility patches
try:
    import huggingface_hub as _hf_hub
    if not hasattr(_hf_hub, "cached_download"):
        _hf_hub.cached_download = _hf_hub.hf_hub_download
except ImportError:
    pass

try:
    _acc_mem = importlib.import_module("accelerate.utils.memory")
    if not hasattr(_acc_mem, "clear_device_cache"):
        _acc_mem.clear_device_cache = lambda *a, **k: None
except ImportError:
    pass

try:
    import diffusers.models.attention_processor
    diffusers.models.attention_processor.AttnProcessor2_0 = MockCache
except ImportError:
    pass

try:
    import transformers.models.llama.modeling_llama
    transformers.models.llama.modeling_llama.AttnProcessor2_0 = MockCache
except ImportError:
    pass

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
    DEFAULT_INFERENCE_STEPS = 20
    DEFAULT_GUIDANCE_SCALE = 7.0
    DEFAULT_N_VIEWS = 2
    DEFAULT_HEIGHT = 256
    DEFAULT_WIDTH = 256
    
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
    
    def __init__(self, max_size=5):
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
    try:
        import cv2
        scores = [
            cv2.Laplacian(cv2.cvtColor(np.array(i), cv2.COLOR_RGBA2GRAY), cv2.CV_64F).var()
            for i in img_list
        ]
        return img_list[int(np.argmax(scores))]
    except ImportError:
        logger.warning("OpenCV not available, returning first image")
        return img_list[0] if img_list else None

# Robust marching cubes fallback
try:
    import torchmcubes
    logger.info("✅ Using torchmcubes")
except ImportError:
    try:
        import PyMCubes
        import types
        import numpy as _np
        import torch as _torch
        
        def _marching_cubes(vol: _torch.Tensor, thresh: float = 0.0):
            v, f = PyMCubes.marching_cubes(vol.detach().cpu().numpy(), thresh)
            return (_torch.from_numpy(v).to(vol.device, dtype=vol.dtype),
                    _torch.from_numpy(f.astype(_np.int64)).to(vol.device))
        
        stub = types.ModuleType("torchmcubes")
        stub.marching_cubes = _marching_cubes
        sys.modules["torchmcubes"] = stub
        logger.info("✅ Using PyMCubes fallback")
    except ImportError:
        try:
            from skimage import measure
            import types
            import numpy as _np
            import torch as _torch
            
            def _marching_cubes(vol: _torch.Tensor, thresh: float = 0.0):
                verts, faces, _, _ = measure.marching_cubes(
                    vol.detach().cpu().numpy(), level=thresh
                )
                return (_torch.from_numpy(verts).to(vol.device, dtype=vol.dtype),
                        _torch.from_numpy(faces.astype(_np.int64)).to(vol.device))
            
            stub = types.ModuleType("torchmcubes")
            stub.marching_cubes = _marching_cubes
            sys.modules["torchmcubes"] = stub
            logger.info("✅ Using scikit-image fallback")
        except ImportError:
            logger.warning("No marching cubes implementation available")

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
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/VAST-AI-Research/TripoSR.git", 
                triposr_path
            ], check=True)
            logger.info("✅ TripoSR cloned successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone TripoSR: {e}")
            raise

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
        torch.cuda.set_per_process_memory_fraction(0.8)

    # Load TripoSR with memory optimization
    triposr_model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml", 
        weight_name="model.ckpt"
    )
    
    # Memory optimization for TripoSR
    triposr_model.to(DEVICE)
    if DEVICE == "cuda":
        triposr_model = triposr_model.half()
    triposr_model.eval()
    
    clear_gpu_memory()
    logger.info("✅ TripoSR loaded with memory optimization")

    app = Flask(__name__)
    CORS(app)

    # Health endpoint
    @app.route("/health", methods=["GET"])
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
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=torch.float16
    ).to(DEVICE)
    
    logger.info("Loading Stable Diffusion...")
    _flush()
    # FIXED: Proper initialization with torch_dtype instead of calling .half() later
    app.sd = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=app.cnet,
        torch_dtype=torch.float16
    ).to(DEVICE)
    
    app.sd.scheduler = EulerAncestralDiscreteScheduler.from_config(app.sd.scheduler.config)
    
    # Enable memory optimizations
    try:
        app.sd.enable_xformers_memory_efficient_attention()
        logger.info("✅ xformers enabled")
    except Exception as e:
        logger.warning(f"xformers not available: {e}")
    
    # Memory optimization techniques
    app.sd.enable_model_cpu_offload()
    app.sd.enable_attention_slicing()
    app.sd.enable_vae_slicing()

    def ensure_module_on_device(module, target_device):
        """Helper function to ensure all tensors are on the right device"""
        if module is None:
            return None
        try:
            module.to(target_device)
            return module
        except Exception as e:
            logger.warning(f"Could not move module to {target_device}: {e}")
            return module

    # Setup TripoSR
    app.triposr = triposr_model
    app.triposr = ensure_module_on_device(app.triposr, DEVICE)
    
    clear_gpu_memory()
    logger.info("✅ All models loaded with memory optimization")
    _flush()

    # Memory-optimized generate endpoint
    @app.route("/generate", methods=["POST"])
    @timing
    def generate():
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            
            data = request.json
            if not data or "sketch" not in data:
                return jsonify({"error": "Missing sketch in request"}), 400

            # Check cache first
            cache_key = f"{data['sketch'][:50]}_{data.get('prompt', '')}"
            cached_result = result_cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                cached_result.seek(0)
                return send_file(
                    cached_result, 
                    mimetype="image/png", 
                    download_name="3d_render.png", 
                    as_attachment=True
                )

            # Decode input image
            try:
                sketch_data = data["sketch"]
                if "," in sketch_data:
                    png = base64.b64decode(sketch_data.split(",", 1)[1])
                else:
                    png = base64.b64decode(sketch_data)
                
                pil = Image.open(io.BytesIO(png)).convert("RGBA")
                
                # Resize input to reduce memory usage
                if pil.size[0] > 512 or pil.size[1] > 512:
                    pil = pil.resize((512, 512), Image.Resampling.LANCZOS)
                    
            except Exception as e:
                logger.error(f"Image decode error: {e}")
                return jsonify({"error": f"Bad image data: {str(e)}"}), 400

            prompt = data.get("prompt", "a clean 3-D asset")
            params = OptimizedParameters.get_optimized_params(data)
            logger.info(f"Using parameters: {params}")

            # Clear memory before processing
            clear_gpu_memory()

            # Edge detection
            try:
                edge = app.edge_det(pil)
                del pil
                clear_gpu_memory()
            except Exception as e:
                logger.error(f"Edge detection failed: {e}")
                return jsonify({"error": f"Edge detection failed: {str(e)}"}), 500

            # Stable Diffusion with memory management
            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext():
                        result = app.sd(
                            prompt, 
                            image=edge,
                            num_inference_steps=params['num_inference_steps'], 
                            guidance_scale=params['guidance_scale']
                        )
                        concept = result.images[0]
                del edge
                clear_gpu_memory()
            except Exception as e:
                logger.error(f"Stable Diffusion failed: {e}")
                return jsonify({"error": f"Stable Diffusion failed: {str(e)}"}), 500

            # Resize concept if needed
            try:
                concept = resize_foreground(concept, 1.0)
                clear_gpu_memory()
            except Exception as e:
                logger.error(f"Resize failed: {e}")
                return jsonify({"error": f"Resize failed: {str(e)}"}), 500

            # TripoSR processing
            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext():
                        codes = app.triposr([concept], device=DEVICE)
                clear_gpu_memory()

                # Render views
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
            except Exception as e:
                logger.error(f"TripoSR processing failed: {e}")
                return jsonify({"error": f"TripoSR processing failed: {str(e)}"}), 500

            # Select sharpest view
            try:
                final_image = sharpest(views)
                del views
                
                if final_image is None:
                    return jsonify({"error": "Failed to generate final image"}), 500

            except Exception as e:
                logger.error(f"View selection failed: {e}")
                return jsonify({"error": f"View selection failed: {str(e)}"}), 500

            # Return result as PNG
            try:
                buf = io.BytesIO()
                final_image.save(buf, "PNG")
                buf.seek(0)

                # Cache result
                result_cache.put(cache_key, io.BytesIO(buf.getvalue()))

                clear_gpu_memory()
                _flush()
                
                return send_file(
                    buf, 
                    mimetype="image/png", 
                    download_name="3d_render.png", 
                    as_attachment=True
                )
            except Exception as e:
                logger.error(f"Image saving failed: {e}")
                return jsonify({"error": f"Image saving failed: {str(e)}"}), 500

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM: {e}")
            clear_gpu_memory()
            return jsonify({
                "error": "GPU memory insufficient. Try reducing image size or parameters."
            }), 500
        except Exception as e:
            logger.error("Unexpected error in /generate", exc_info=True)
            clear_gpu_memory()
            return jsonify({"error": f"Server error: {str(e)}"}), 500

    if __name__ == "__main__":
        logger.info("🚀 Starting memory-optimized TripoSR service")
        app.run(host="0.0.0.0", port=5000, debug=False)

except Exception as e:
    logger.error("❌ Error during initialization", exc_info=True)
    _flush()
    raise
