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
        if args:
            first_arg = args[0]
            if hasattr(first_arg, 'dim'):
                return first_arg
            return args[0] if len(args) == 1 else args
        return None
    
    def forward(self, *args, **kwargs):
        if args:
            return args[0]
        return None

class MockAttentionProcessor:
    def __init__(self):
        pass
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        return hidden_states

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
    diffusers.models.attention_processor.AttnProcessor2_0 = MockAttentionProcessor
except ImportError:
    pass

try:
    import transformers.models.llama.modeling_llama
    transformers.models.llama.modeling_llama.AttnProcessor2_0 = MockAttentionProcessor
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

def ensure_tensor_from_output(data):
    """Convert diffusers BaseOutput or tuple outputs to tensors safely"""
    if hasattr(data, 'images'):
        return data.images[0]
    elif hasattr(data, 'to_tuple'):
        tuple_data = data.to_tuple()
        return tuple_data[0] if tuple_data else None
    elif isinstance(data, tuple):
        first_element = data[0]
        if hasattr(first_element, 'images'):
            return first_element.images[0]
        return first_element
    elif hasattr(data, 'dim'):
        return data
    elif torch.is_tensor(data):
        return data
    else:
        return torch.tensor(data) if not isinstance(data, torch.Tensor) else data

def safe_resize_foreground(image, ratio=1.0):
    """Safely resize image with comprehensive error handling"""
    try:
        if image is None:
            raise ValueError("Input image is None")
        
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")
        
        if ratio == 1.0:
            logger.info("Resize ratio is 1.0, returning original image")
            return image
        
        width, height = image.size
        logger.info(f"Original image size: {width}x{height}")
        
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        new_width = max(new_width, 64)
        new_height = max(new_height, 64)
        
        logger.info(f"Resizing to: {new_width}x{new_height}")
        
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized
        
    except Exception as e:
        logger.error(f"Safe resize failed: {str(e)}")
        logger.info("Returning original image as fallback")
        return image

# FIXED: Proper image preparation for TripoSR
def prepare_image_for_triposr(pil_image):
    """Prepare PIL image for TripoSR processing"""
    try:
        # Ensure we have an RGBA image
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        
        # TripoSR typically expects square images
        width, height = pil_image.size
        if width != height:
            # Make it square by padding with white background
            max_size = max(width, height)
            new_image = Image.new('RGBA', (max_size, max_size), (255, 255, 255, 255))
            paste_x = (max_size - width) // 2
            paste_y = (max_size - height) // 2
            new_image.paste(pil_image, (paste_x, paste_y))
            pil_image = new_image
        
        # Resize to a size that TripoSR can handle efficiently
        target_size = 256  # Smaller size for memory efficiency
        if pil_image.size[0] != target_size:
            pil_image = pil_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {target_size}x{target_size} for TripoSR")
        
        logger.info(f"✅ Image prepared for TripoSR: {pil_image.size}, mode: {pil_image.mode}")
        return pil_image
        
    except Exception as e:
        logger.error(f"Image preparation failed: {e}")
        return pil_image  # Return original on error

class OptimizedParameters:
    """Memory-optimized parameters to prevent CUDA OOM"""
    
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

            # Enhanced Stable Diffusion with proper BaseOutput handling
            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext():
                        result = app.sd(
                            prompt, 
                            image=edge,
                            num_inference_steps=params['num_inference_steps'], 
                            guidance_scale=params['guidance_scale'],
                            return_dict=True
                        )
                        
                        concept = ensure_tensor_from_output(result)
                        
                        if concept is None:
                            raise ValueError("Stable Diffusion returned None concept")
                        
                        logger.info(f"Concept type: {type(concept)}, size: {concept.size if hasattr(concept, 'size') else 'unknown'}")
                        
                del edge, result
                clear_gpu_memory()
            except Exception as e:
                logger.error(f"Stable Diffusion failed: {e}")
                return jsonify({"error": f"Stable Diffusion failed: {str(e)}"}), 500

            # Enhanced resize with better error handling and fallback
            try:
                logger.info(f"About to resize concept: {type(concept)}")
                
                try:
                    concept = resize_foreground(concept, 1.0)
                    logger.info("✅ TripoSR resize_foreground successful")
                except Exception as resize_error:
                    logger.warning(f"TripoSR resize_foreground failed: {resize_error}")
                    logger.info("🔄 Falling back to safe_resize_foreground")
                    concept = safe_resize_foreground(concept, 1.0)
                
                if concept is None:
                    raise ValueError("Concept is None after resize")
                    
                clear_gpu_memory()
                logger.info("✅ Resize completed successfully")
                
            except Exception as e:
                logger.error(f"All resize methods failed: {e}")
                return jsonify({"error": f"Resize failed: {str(e)}"}), 500

            # FIXED: TripoSR processing with proper PIL Image handling
            try:
                # Prepare the image properly for TripoSR
                concept_prepared = prepare_image_for_triposr(concept)
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext():
                        # FIXED: Pass PIL Image directly in a list (TripoSR's expected format)
                        logger.info("Calling TripoSR with prepared PIL image")
                        raw_codes = app.triposr([concept_prepared], device=DEVICE)
                        codes = ensure_tensor_from_output(raw_codes)
                        del raw_codes, concept_prepared
                clear_gpu_memory()

                # Render views
                with torch.no_grad():
                    with torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext():
                        raw_views = app.triposr.render(
                            codes, 
                            n_views=params['n_views'], 
                            height=params['height'], 
                            width=params['width'], 
                            return_type="pil"
                        )
                        views = ensure_tensor_from_output(raw_views)
                        del raw_views
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
