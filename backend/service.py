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

# FIXED: Enhanced preprocessing with stride-safe operations
def preprocess_image_for_triposr(input_image, do_remove_background=True, foreground_ratio=0.85):
    """Preprocess image with stride-safe operations to prevent negative stride errors"""
    def fill_background(image):
        """Enhanced background fill function with stride safety"""
        try:
            # Convert PIL to numpy with stride safety
            img_array = np.array(image).astype(np.float32)
            
            # FIXED: Ensure positive strides by copying array if needed
            if img_array.strides and any(s < 0 for s in img_array.strides):
                logger.info("🔧 Fixing negative strides with array.copy()")
                img_array = img_array.copy()
            
            # Normalize
            img_array = img_array / 255.0
            
            # Apply background fill
            if img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[:, :, :3] * img_array[:, :, 3:4] + (1 - img_array[:, :, 3:4]) * 0.5
            
            # Convert back to PIL
            img_array = (img_array * 255.0).astype(np.uint8)
            
            # FIXED: Create continuous array to prevent stride issues
            img_array = np.ascontiguousarray(img_array)
            
            image = Image.fromarray(img_array)
            return image
            
        except Exception as e:
            logger.error(f"Background fill failed: {e}")
            # Fallback to basic conversion
            if image.mode == "RGBA":
                white_bg = Image.new('RGB', image.size, (255, 255, 255))
                white_bg.paste(image, mask=image.split()[-1])
                return white_bg
            return image.convert('RGB')
    
    try:
        if do_remove_background:
            try:
                import rembg
                from tsr.utils import remove_background, resize_foreground
                
                # FIXED: Ensure input is stride-safe before processing
                if input_image.mode != "RGB":
                    input_image = input_image.convert("RGB")
                
                # Create a new image to ensure memory layout is clean
                clean_image = Image.new("RGB", input_image.size)
                clean_image.paste(input_image)
                
                rembg_session = rembg.new_session()
                image = remove_background(clean_image, rembg_session)
                image = resize_foreground(image, foreground_ratio)
                image = fill_background(image)
                logger.info("✅ Applied background removal with stride safety")
                
            except ImportError as e:
                logger.warning(f"Background removal not available: {e}, using stride-safe preprocessing")
                image = input_image
                if image.mode == "RGBA":
                    image = fill_background(image)
                elif image.mode != "RGB":
                    image = image.convert("RGB")
                    
        else:
            image = input_image
            if image.mode == "RGBA":
                image = fill_background(image)
            elif image.mode != "RGB":
                image = image.convert("RGB")
        
        # FIXED: Final stride safety check
        # Convert to numpy to check strides
        test_array = np.array(image)
        if test_array.strides and any(s < 0 for s in test_array.strides):
            logger.info("🔧 Final stride fix - creating clean image copy")
            # Create a new clean image
            clean_array = np.ascontiguousarray(test_array)
            image = Image.fromarray(clean_array)
        
        logger.info(f"✅ Image preprocessed with stride safety: {image.size}, mode: {image.mode}")
        return image
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        # Ultimate fallback - create completely new image
        if input_image.mode == "RGBA":
            white_background = Image.new('RGB', input_image.size, (255, 255, 255))
            white_background.paste(input_image, mask=input_image.split()[-1])
            return white_background
        else:
            # Create a clean copy to ensure no stride issues
            clean_array = np.ascontiguousarray(np.array(input_image.convert('RGB')))
            return Image.fromarray(clean_array)

class QualityOptimizedParameters:
    """Quality-optimized parameters for maximum output quality"""
    
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
            # FIXED: Ensure array is stride-safe before OpenCV processing
            img_array = np.array(i)
            if img_array.strides and any(s < 0 for s in img_array.strides):
                img_array = img_array.copy()
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(score)
            
        return img_list[int(np.argmax(scores))]
    except ImportError:
        logger.warning("OpenCV not available, returning first image")
        return img_list[0] if img_list else None

# Marching cubes fallback with stride safety
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
            # FIXED: Ensure stride safety in marching cubes
            vol_numpy = vol.detach().cpu().numpy()
            if vol_numpy.strides and any(s < 0 for s in vol_numpy.strides):
                vol_numpy = vol_numpy.copy()
            
            v, f = PyMCubes.marching_cubes(vol_numpy, thresh)
            return (_torch.from_numpy(v).to(vol.device, dtype=vol.dtype),
                    _torch.from_numpy(f.astype(_np.int64)).to(vol.device))
        
        stub = types.ModuleType("torchmcubes")
        stub.marching_cubes = _marching_cubes
        sys.modules["torchmcubes"] = stub
        logger.info("✅ Using PyMCubes fallback with stride safety")
    except ImportError:
        try:
            from skimage import measure
            import types
            import numpy as _np
            import torch as _torch
            
            def _marching_cubes(vol: _torch.Tensor, thresh: float = 0.0):
                # FIXED: Ensure stride safety in marching cubes
                vol_numpy = vol.detach().cpu().numpy()
                if vol_numpy.strides and any(s < 0 for s in vol_numpy.strides):
                    vol_numpy = vol_numpy.copy()
                
                verts, faces, _, _ = measure.marching_cubes(vol_numpy, level=thresh)
                return (_torch.from_numpy(verts).to(vol.device, dtype=vol.dtype),
                        _torch.from_numpy(faces.astype(_np.int64)).to(vol.device))
            
            stub = types.ModuleType("torchmcubes")
            stub.marching_cubes = _marching_cubes
            sys.modules["torchmcubes"] = stub
            logger.info("✅ Using scikit-image fallback with stride safety")
        except ImportError:
            logger.warning("No marching cubes implementation available")

print("Starting stride-safe, quality-optimized TripoSR service initialization...")
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
        from tsr.utils import resize_foreground, remove_background, to_gradio_3d_orientation
        logger.info("✅ TripoSR imported successfully")
    except ImportError as e:
        logger.error(f"❌ TripoSR import failed: {e}")
        raise

    # Load models with quality optimization
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading models on {DEVICE} with stride-safe quality optimization...")
    
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
    
    # Set optimal chunk size
    triposr_model.renderer.set_chunk_size(8192)
    
    clear_gpu_memory()
    logger.info("✅ TripoSR loaded with stride-safe float32 processing")

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
            "stride_safe": True
        })

    @app.route("/test", methods=["GET", "POST"])
    def test():
        return jsonify({"message": "Stride-safe TripoSR server is working!", "method": request.method})

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
    
    # Enable optimizations
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
    logger.info("✅ All models loaded with stride safety")
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
                logger.info("Returning cached result")
                cached_result.seek(0)
                return send_file(
                    cached_result, 
                    mimetype="image/png", 
                    download_name="3d_render.png", 
                    as_attachment=True
                )

            # Decode input image with stride safety
            try:
                sketch_data = data["sketch"]
                if "," in sketch_data:
                    png = base64.b64decode(sketch_data.split(",", 1)[1])
                else:
                    png = base64.b64decode(sketch_data)
                
                pil = Image.open(io.BytesIO(png)).convert("RGBA")
                
                # Keep higher resolution but ensure stride safety
                if pil.size[0] > 768 or pil.size[1] > 768:
                    pil = pil.resize((768, 768), Image.Resampling.LANCZOS)
                
                # FIXED: Create clean copy to prevent stride issues
                clean_array = np.ascontiguousarray(np.array(pil))
                pil = Image.fromarray(clean_array)
                    
            except Exception as e:
                logger.error(f"Image decode error: {e}")
                return jsonify({"error": f"Bad image data: {str(e)}"}), 400

            prompt = data.get("prompt", "a clean 3-D asset")
            params = QualityOptimizedParameters.get_quality_params(data)
            logger.info(f"Using stride-safe quality parameters: {params}")

            clear_gpu_memory()

            # Edge detection with stride safety
            try:
                edge = app.edge_det(pil)
                
                # FIXED: Ensure edge detection output is stride-safe
                edge_array = np.array(edge)
                if edge_array.strides and any(s < 0 for s in edge_array.strides):
                    edge_array = np.ascontiguousarray(edge_array)
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
                        
                        # FIXED: Ensure concept is stride-safe
                        concept_array = np.array(concept)
                        if concept_array.strides and any(s < 0 for s in concept_array.strides):
                            concept_array = np.ascontiguousarray(concept_array)
                            concept = Image.fromarray(concept_array)
                        
                        logger.info(f"Stride-safe concept: {type(concept)}, size: {concept.size if hasattr(concept, 'size') else 'unknown'}")
                        
                del edge, result
                clear_gpu_memory()
            except Exception as e:
                logger.error(f"Stable Diffusion failed: {e}")
                return jsonify({"error": f"Stable Diffusion failed: {str(e)}"}), 500

            # Resize with stride safety
            try:
                logger.info(f"About to resize concept: {type(concept)}")
                
                try:
                    concept = resize_foreground(concept, 1.0)
                    logger.info("✅ TripoSR resize_foreground successful")
                except Exception as resize_error:
                    logger.warning(f"TripoSR resize_foreground failed: {resize_error}")
                    concept = safe_resize_foreground(concept, 1.0)
                
                if concept is None:
                    raise ValueError("Concept is None after resize")
                
                # FIXED: Final stride safety check after resize
                concept_array = np.array(concept)
                if concept_array.strides and any(s < 0 for s in concept_array.strides):
                    concept_array = np.ascontiguousarray(concept_array)
                    concept = Image.fromarray(concept_array)
                    
                clear_gpu_memory()
                logger.info("✅ Stride-safe resize completed")
                
            except Exception as e:
                logger.error(f"Resize failed: {e}")
                return jsonify({"error": f"Resize failed: {str(e)}"}), 500

            # FIXED: TripoSR processing with comprehensive stride safety
            try:
                # Preprocess with stride safety
                processed_image = preprocess_image_for_triposr(
                    concept, 
                    do_remove_background=True, 
                    foreground_ratio=0.85
                )
                
                with torch.no_grad():
                    logger.info("Calling TripoSR with stride-safe float32 processing")
                    scene_codes = app.triposr(processed_image, device=DEVICE)
                    
                    # Extract mesh
                    mesh = app.triposr.extract_mesh(
                        scene_codes, 
                        resolution=params['mc_resolution']
                    )[0]
                    
                    # Apply orientation
                    mesh = to_gradio_3d_orientation(mesh)
                    
                    del scene_codes, processed_image
                    
                clear_gpu_memory()

                # Generate views with stride safety
                try:
                    with torch.no_grad():
                        temp_scene_codes = app.triposr(concept, device=DEVICE)
                        
                        rendered_views = app.triposr.render(
                            temp_scene_codes,
                            n_views=params['n_views'],
                            return_type="pil"
                        )[0]
                        
                        del temp_scene_codes
                        views = rendered_views
                    
                    clear_gpu_memory()
                    logger.info(f"✅ Generated {len(views)} stride-safe views")
                    
                except Exception as render_error:
                    logger.warning(f"Multi-view rendering failed: {render_error}, using concept")
                    views = [concept]
                    
            except Exception as e:
                logger.error(f"TripoSR processing failed: {e}")
                return jsonify({"error": f"TripoSR processing failed: {str(e)}"}), 500

            # Select best view with stride safety
            try:
                final_image = sharpest(views)
                del views
                
                if final_image is None:
                    return jsonify({"error": "Failed to generate final image"}), 500

                # FIXED: Final stride safety check
                final_array = np.array(final_image)
                if final_array.strides and any(s < 0 for s in final_array.strides):
                    final_array = np.ascontiguousarray(final_array)
                    final_image = Image.fromarray(final_array)

                logger.info("✅ Selected stride-safe highest quality view")

            except Exception as e:
                logger.error(f"View selection failed: {e}")
                return jsonify({"error": f"View selection failed: {str(e)}"}), 500

            # Return result
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
                    download_name="3d_render_stride_safe.png", 
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
        logger.info("🚀 Starting STRIDE-SAFE Quality-Optimized TripoSR service")
        app.run(host="0.0.0.0", port=5000, debug=False)

except Exception as e:
    logger.error("❌ Error during initialization", exc_info=True)
    _flush()
    raise
