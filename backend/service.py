from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import sys, os, base64, io, gc, time, types, importlib, logging, atexit, tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from functools import wraps
from torch.cuda.amp import autocast
from contextlib import nullcontext
import psutil
import subprocess

# CRITICAL FIX: Create torchmcubes mock BEFORE any TripoSR imports
def setup_torchmcubes_fallback():
    """Setup torchmcubes fallback before TripoSR imports it"""
    try:
        import torchmcubes
        logger.info("✅ Using real torchmcubes")
        return
    except ImportError:
        logger.info("🔧 Creating torchmcubes fallback module")
        
        # Try PyMCubes first
        try:
            import PyMCubes
            import numpy as _np
            import torch as _torch
            
            def _marching_cubes(vol: _torch.Tensor, thresh: float = 0.0):
                vol_numpy = vol.detach().cpu().numpy()
                if vol_numpy.strides and any(s < 0 for s in vol_numpy.strides):
                    vol_numpy = vol_numpy.copy()
                
                v, f = PyMCubes.marching_cubes(vol_numpy, thresh)
                return (_torch.from_numpy(v).to(vol.device, dtype=vol.dtype),
                        _torch.from_numpy(f.astype(_np.int64)).to(vol.device))
            
            # Create mock torchmcubes module
            mock_torchmcubes = types.ModuleType("torchmcubes")
            mock_torchmcubes.marching_cubes = _marching_cubes
            sys.modules["torchmcubes"] = mock_torchmcubes
            logger.info("✅ Created torchmcubes fallback using PyMCubes")
            return
            
        except ImportError:
            pass
        
        # Try scikit-image fallback
        try:
            from skimage import measure
            import numpy as _np
            import torch as _torch
            
            def _marching_cubes(vol: _torch.Tensor, thresh: float = 0.0):
                vol_numpy = vol.detach().cpu().numpy()
                if vol_numpy.strides and any(s < 0 for s in vol_numpy.strides):
                    vol_numpy = vol_numpy.copy()
                
                verts, faces, _, _ = measure.marching_cubes(vol_numpy, level=thresh)
                return (_torch.from_numpy(verts).to(vol.device, dtype=vol.dtype),
                        _torch.from_numpy(faces.astype(_np.int64)).to(vol.device))
            
            # Create mock torchmcubes module
            mock_torchmcubes = types.ModuleType("torchmcubes")
            mock_torchmcubes.marching_cubes = _marching_cubes
            sys.modules["torchmcubes"] = mock_torchmcubes
            logger.info("✅ Created torchmcubes fallback using scikit-image")
            return
            
        except ImportError:
            pass
        
        # Ultimate fallback - create a dummy module
        def _dummy_marching_cubes(vol, thresh=0.0):
            logger.error("No marching cubes implementation available - returning empty mesh")
            device = vol.device if hasattr(vol, 'device') else 'cpu'
            return (torch.zeros((0, 3), device=device), torch.zeros((0, 3), dtype=torch.long, device=device))
        
        mock_torchmcubes = types.ModuleType("torchmcubes")
        mock_torchmcubes.marching_cubes = _dummy_marching_cubes
        sys.modules["torchmcubes"] = mock_torchmcubes
        logger.warning("⚠️ Created dummy torchmcubes fallback")

# ENHANCED COMPATIBILITY PATCHES
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

# Apply patches
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

# CRITICAL: Setup torchmcubes fallback BEFORE importing TripoSR
setup_torchmcubes_fallback()

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
    sys.stdout.flush()
    sys.stderr.flush()

atexit.register(_flush)

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def gpu_mem_mb():
    return (torch.cuda.memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0

# Global torch.from_numpy patching
_original_from_numpy = torch.from_numpy
def patched_from_numpy(ndarray):
    """Patched torch.from_numpy that always handles stride issues"""
    if ndarray.strides and any(s < 0 for s in ndarray.strides):
        logger.debug("🔧 Patched torch.from_numpy fixing negative strides")
        ndarray = np.ascontiguousarray(ndarray)
    return _original_from_numpy(ndarray)

torch.from_numpy = patched_from_numpy

def ensure_tensor_from_output(data):
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
    try:
        if image is None:
            raise ValueError("Input image is None")
        
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")
        
        if ratio == 1.0:
            logger.info("Resize ratio is 1.0, returning original image")
            return image
        
        width, height = image.size
        new_width = max(int(width * ratio), 64)
        new_height = max(int(height * ratio), 64)
        
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Ensure output is stride-safe
        resized_array = np.ascontiguousarray(np.array(resized))
        return Image.fromarray(resized_array)
        
    except Exception as e:
        logger.error(f"Safe resize failed: {str(e)}")
        return image

def bulletproof_image_preprocessing(input_image):
    """Bulletproof image preprocessing that eliminates all stride issues"""
    def create_clean_image(img):
        """Create a completely clean image with guaranteed positive strides"""
        img_array = np.ascontiguousarray(np.array(img))
        
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            # Alpha blend with white background
            alpha = img_array[:, :, 3:4].astype(np.float32) / 255.0
            rgb = img_array[:, :, :3].astype(np.float32)
            white_bg = np.ones_like(rgb) * 255
            img_array = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    try:
        # Start with clean conversion
        if input_image.mode != 'RGB':
            input_image = create_clean_image(input_image.convert('RGBA'))
        
        # Apply processing with stride safety
        try:
            # Try background removal if available
            import rembg
            from tsr.utils import remove_background, resize_foreground
            
            clean_input = create_clean_image(input_image)
            
            rembg_session = rembg.new_session()
            processed = remove_background(clean_input, rembg_session)
            processed = resize_foreground(processed, 0.85)
            
            result = create_clean_image(processed)
            logger.info("✅ Background removal with bulletproof stride safety")
            return result
            
        except ImportError:
            logger.info("Background removal not available, using clean conversion")
            return create_clean_image(input_image)
        except Exception as e:
            logger.warning(f"Background removal failed: {e}, using clean conversion")
            return create_clean_image(input_image)
            
    except Exception as e:
        logger.error(f"Bulletproof preprocessing failed: {e}")
        img_array = np.ascontiguousarray(np.array(input_image.convert('RGB')))
        return Image.fromarray(img_array)

class QualityOptimizedParameters:
    DEFAULT_INFERENCE_STEPS = 25
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_N_VIEWS = 4
    DEFAULT_HEIGHT = 512
    DEFAULT_WIDTH = 512
    DEFAULT_MC_RESOLUTION = 256
    
    @classmethod
    def get_quality_params(cls, data):
        return {
            'num_inference_steps': int(data.get("num_inference_steps", cls.DEFAULT_INFERENCE_STEPS)),
            'guidance_scale': float(data.get("guidance_scale", cls.DEFAULT_GUIDANCE_SCALE)),
            'n_views': int(data.get("n_views", cls.DEFAULT_N_VIEWS)),
            'height': int(data.get("height", cls.DEFAULT_HEIGHT)),
            'width': int(data.get("width", cls.DEFAULT_WIDTH)),
            'mc_resolution': int(data.get("mc_resolution", cls.DEFAULT_MC_RESOLUTION))
        }

class ResultCache:
    def __init__(self, max_size=3):
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
    try:
        import cv2
        scores = []
        for i in img_list:
            img_array = np.ascontiguousarray(np.array(i))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(score)
            
        return img_list[int(np.argmax(scores))]
    except ImportError:
        logger.warning("OpenCV not available, returning first image")
        return img_list[0] if img_list else None

print("Starting image-only TripoSR service initialization...")
try:
    logger.info("Importing diffusers...")
    _flush()
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
    from controlnet_aux import CannyDetector

    # TripoSR setup
    logger.info("Setting up TripoSR for image-only output...")
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

    # Import TripoSR - REMOVED mesh-related imports
    try:
        from tsr.system import TSR
        from tsr.utils import resize_foreground, remove_background
        # REMOVED: to_gradio_3d_orientation import to prevent mesh exports
        logger.info("✅ TripoSR imported for image-only processing")
    except ImportError as e:
        logger.error(f"❌ TripoSR import failed: {e}")
        raise

    # Load models
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading models on {DEVICE} for image-only output...")
    
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.9)

    # Load TripoSR in FULL FLOAT32 PRECISION
    triposr_model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml", 
        weight_name="model.ckpt"
    )
    
    triposr_model.to(DEVICE)
    triposr_model.eval()
    triposr_model.renderer.set_chunk_size(8192)
    
    clear_gpu_memory()
    logger.info("✅ TripoSR loaded for image-only output")

    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok", 
            "gpu_mb": gpu_mem_mb(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "triposr_available": True,
            "precision": "float32",
            "output_mode": "image_only"
        })

    @app.route("/test", methods=["GET", "POST"])
    def test():
        return jsonify({"message": "Image-only TripoSR server!", "method": request.method})

    # Load other models
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
    
    try:
        app.sd.enable_xformers_memory_efficient_attention()
        logger.info("✅ xformers enabled")
    except Exception as e:
        logger.warning(f"xformers not available: {e}")
    
    app.sd.enable_model_cpu_offload()
    app.sd.enable_attention_slicing()
    app.sd.enable_vae_slicing()

    def ensure_module_on_device(module, target_device):
        if module is None:
            return None
        try:
            module.to(target_device)
            return module
        except Exception as e:
            logger.warning(f"Could not move module to {target_device}: {e}")
            return module

    app.triposr = triposr_model
    app.triposr = ensure_module_on_device(app.triposr, DEVICE)
    
    clear_gpu_memory()
    logger.info("✅ All models loaded for image-only processing")
    _flush()

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
                logger.info("Returning cached image result")
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
                
                pil = Image.open(io.BytesIO(png))
                
                pil_array = np.ascontiguousarray(np.array(pil))
                pil = Image.fromarray(pil_array)
                
                if pil.size[0] > 768 or pil.size[1] > 768:
                    pil = pil.resize((768, 768), Image.Resampling.LANCZOS)
                    pil_array = np.ascontiguousarray(np.array(pil))
                    pil = Image.fromarray(pil_array)
                    
            except Exception as e:
                logger.error(f"Image decode error: {e}")
                return jsonify({"error": f"Bad image data: {str(e)}"}), 400

            # Handle NSFW content proactively
            prompt = data.get("prompt", "a simple geometric 3D object")
            if "cube" in prompt.lower():
                prompt = "a simple geometric cube shape, clean minimal design"
            
            params = QualityOptimizedParameters.get_quality_params(data)
            logger.info(f"Using image-only parameters: {params}")

            clear_gpu_memory()

            # Edge detection
            try:
                edge = app.edge_det(pil)
                edge_array = np.ascontiguousarray(np.array(edge))
                edge = Image.fromarray(edge_array)
                del pil
                clear_gpu_memory()
            except Exception as e:
                logger.error(f"Edge detection failed: {e}")
                return jsonify({"error": f"Edge detection failed: {str(e)}"}), 500

            # Stable Diffusion
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
                        
                        # Check if we got NSFW black image
                        concept_array = np.array(concept)
                        if np.all(concept_array < 10):
                            logger.warning("NSFW detected, creating fallback geometric image")
                            fallback = Image.new('RGB', (512, 512), (240, 240, 240))
                            concept = fallback
                        
                        concept_array = np.ascontiguousarray(np.array(concept))
                        concept = Image.fromarray(concept_array)
                        
                        logger.info(f"Concept ready for image-only processing: {concept.size}")
                        
                del edge, result
                clear_gpu_memory()
            except Exception as e:
                logger.error(f"Stable Diffusion failed: {e}")
                return jsonify({"error": f"Stable Diffusion failed: {str(e)}"}), 500

            # Resize
            try:
                concept = safe_resize_foreground(concept, 1.0)
                if concept is None:
                    raise ValueError("Concept is None after resize")
                clear_gpu_memory()
                logger.info("✅ Resize completed for image processing")
            except Exception as e:
                logger.error(f"Resize failed: {e}")
                return jsonify({"error": f"Resize failed: {str(e)}"}), 500

            # FIXED: TripoSR processing for IMAGE-ONLY output (no mesh extraction/export)
            try:
                processed_image = bulletproof_image_preprocessing(concept)
                
                with torch.no_grad():
                    logger.info("Processing TripoSR for image-only output")
                    
                    final_array = np.ascontiguousarray(np.array(processed_image))
                    processed_image = Image.fromarray(final_array)
                    
                    # Generate scene codes for rendering (without mesh extraction)
                    scene_codes = app.triposr(processed_image, device=DEVICE)
                    
                    # FIXED: Skip mesh extraction and go directly to view rendering
                    # This eliminates OBJ file generation
                    
                    # Render multiple views directly from scene codes
                    rendered_views = app.triposr.render(
                        scene_codes,
                        n_views=params['n_views'],
                        return_type="pil"
                    )[0]
                    
                    del scene_codes, processed_image
                    views = rendered_views
                    
                clear_gpu_memory()
                logger.info(f"✅ Generated {len(views)} image views (no mesh files)")
                    
            except Exception as e:
                logger.error(f"TripoSR image processing failed: {e}")
                return jsonify({"error": f"TripoSR processing failed: {str(e)}"}), 500

            # Select best view
            try:
                final_image = sharpest(views)
                del views
                
                if final_image is None:
                    return jsonify({"error": "Failed to generate final image"}), 500

                final_array = np.ascontiguousarray(np.array(final_image))
                final_image = Image.fromarray(final_array)

                logger.info("✅ Selected final 3D image (no additional files)")

            except Exception as e:
                logger.error(f"View selection failed: {e}")
                return jsonify({"error": f"View selection failed: {str(e)}"}), 500

            # Return ONLY the image
            try:
                buf = io.BytesIO()
                final_image.save(buf, "PNG", optimize=False, compress_level=1)
                buf.seek(0)

                result_cache.put(cache_key, io.BytesIO(buf.getvalue()))
                clear_gpu_memory()
                _flush()
                
                return send_file(
                    buf, 
                    mimetype="image/png", 
                    download_name="3d_render_image_only.png", 
                    as_attachment=True
                )
            except Exception as e:
                logger.error(f"Image saving failed: {e}")
                return jsonify({"error": f"Image saving failed: {str(e)}"}), 500

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM: {e}")
            clear_gpu_memory()
            return jsonify({
                "error": "GPU memory insufficient. Try reducing resolution."
            }), 500
        except Exception as e:
            logger.error("Unexpected error in /generate", exc_info=True)
            clear_gpu_memory()
            return jsonify({"error": f"Server error: {str(e)}"}), 500

    if __name__ == "__main__":
        logger.info("🚀 Starting IMAGE-ONLY TripoSR service")
        app.run(host="0.0.0.0", port=5000, debug=False)

except Exception as e:
    logger.error("❌ Error during initialization", exc_info=True)
    _flush()
    raise
