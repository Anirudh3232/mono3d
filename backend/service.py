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

# Add TripoSR-main to Python path
TRIPOSR_PATH = os.path.join(os.path.dirname(__file__), "TripoSR-main")
sys.path.insert(0, TRIPOSR_PATH)

class MockCache:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

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
        # Only run GC occasionally to reduce CPU overhead
        if hasattr(clear_gpu_memory, '_gc_counter'):
            clear_gpu_memory._gc_counter += 1
        else:
            clear_gpu_memory._gc_counter = 0
        
        if clear_gpu_memory._gc_counter % 3 == 0:  # Run GC every 3rd call
            gc.collect()

def gpu_mem_mb():
    return (torch.cuda.memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0


class OptimizedParameters:
    """Optimized default parameters to reduce CPU usage"""
    
    # Reduced inference steps for faster generation
    DEFAULT_INFERENCE_STEPS = 30  # Reduced from 63
    DEFAULT_GUIDANCE_SCALE = 7.5  # Reduced from 9.96 for faster convergence
    
    # Render parameters for 3D image
    DEFAULT_RENDER_RESOLUTION = 512  # Resolution for rendering
    
    @classmethod
    def get_optimized_params(cls, data):
        """Get optimized parameters based on request data"""
        return {
            'num_inference_steps': int(data.get("num_inference_steps", cls.DEFAULT_INFERENCE_STEPS)),
            'guidance_scale': float(data.get("guidance_scale", cls.DEFAULT_GUIDANCE_SCALE)),
            'render_resolution': int(data.get("render_resolution", cls.DEFAULT_RENDER_RESOLUTION))
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
    from controlnet_aux import CannyDetector
    
    # Import TripoSR from local directory
    import rembg
    import numpy as np

    # Import optimization profiles
    try:
        from optimization_config import get_profile_parameters, list_profiles, get_recommended_profile
        OPTIMIZATION_AVAILABLE = True
        logger.info("Optimization profiles loaded successfully")   
    except ImportError:
        OPTIMIZATION_AVAILABLE = False
        logger.warning("Optimization profiles not available, using default parameters")

    # Import TripoSR from local directory
    logger.info(f"Loading TripoSR from local directory: {TRIPOSR_PATH}")
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
    
    # Initialize rembg session for background removal
    rembg_session = rembg.new_session()
    
    app = Flask(__name__)
    
    # Configure CORS for production deployment
    CORS(app, 
         origins=[
             "http://localhost:3000",  # Local development
             "https://*.vercel.app",   # Vercel deployments
             "https://your-vercel-app.vercel.app",  # Your specific Vercel app
             "https://mono3d.your-domain.com"  # Custom domain if you have one
         ],
         allow_headers=["Content-Type", "Authorization"],
         methods=["GET", "POST", "OPTIONS"])

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok", 
            "gpu_mb": gpu_mem_mb(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "models_loaded": all(hasattr(app, x) for x in ("edge_det", "cnet", "sd", "triposr")),
            "optimization_available": OPTIMIZATION_AVAILABLE,
            "triposr_path": TRIPOSR_PATH
        })

    @app.route("/test", methods=["GET", "POST"])
    def test():
        return jsonify({"message": "Optimized server is working!", "method": request.method})

    @app.get("/latest_concept_image")
    def get_latest_concept_image():
        if app.last_concept_image:
            buf = io.BytesIO()
            app.last_concept_image.save(buf, format="PNG")
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
        return jsonify({"error": "No concept image has been generated yet."}), 404

    @app.get("/profiles")
    def get_profiles():
        """Get available optimization profiles"""
        if not OPTIMIZATION_AVAILABLE:
            return jsonify({"error": "Optimization profiles not available"}), 503
        
        try:
            profiles = list_profiles()
            return jsonify({
                "profiles": profiles,
                "available": True
            })
        except Exception as e:
            logger.error(f"Error getting profiles: {e}")
            return jsonify({"error": str(e)}), 500

    @app.get("/recommend")
    def get_recommendation():
        """Get recommended profile based on system specs"""
        if not OPTIMIZATION_AVAILABLE:
            return jsonify({"error": "Optimization profiles not available"}), 503
        
        try:
            cpu_cores = psutil.cpu_count()
            gpu_memory_gb = 0
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            use_case = request.args.get("use_case", "general")
            recommended = get_recommended_profile(cpu_cores, gpu_memory_gb, use_case)
            
            return jsonify({
                "recommended_profile": recommended,
                "system_specs": {
                    "cpu_cores": cpu_cores,
                    "gpu_memory_gb": round(gpu_memory_gb, 1),
                    "use_case": use_case
                }
            })
        except Exception as e:
            logger.error(f"Error getting recommendation: {e}")
            return jsonify({"error": str(e)}), 500

    # ───────────── Load models with optimizations ─────────────
    device = "cuda" if torch.cuda.is_available() else "cpu";
    logger.info(f"Using {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Optimize CUDA memory allocation
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory

    logger.info("Loading edge detector …");
    _flush();
    app.edge_det = CannyDetector()
    
    logger.info("Loading ControlNet …");
    _flush();
    app.cnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
    
    logger.info("Loading Stable Diffusion …");
    _flush()
    app.sd = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=app.cnet,
        torch_dtype=torch.float16).to(device)
    
    # Optimized scheduler for faster inference
    logger.info("Using optimized scheduler for faster inference.")
    app.sd.scheduler = EulerAncestralDiscreteScheduler.from_config(app.sd.scheduler.config)
    
    try:
        app.sd.enable_xformers_memory_efficient_attention();
        logger.info("xformers enabled")
    except Exception:
        logger.warning("xformers not available — using plain attention")
    
    # Optimized memory management
    app.sd.enable_model_cpu_offload();
    app.sd.enable_attention_slicing()
    app.sd.enable_vae_slicing()  # Additional optimization

    logger.info("Loading TripoSR from local directory …");
    _flush()
    
    # Load TripoSR from HuggingFace
    app.triposr = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt"
    )
    
    # Set chunk size for memory optimization
    if hasattr(app.triposr, 'renderer'):
        app.triposr.renderer.set_chunk_size(8192)  # Optimize for memory usage
    
    app.triposr.to(device)
    app.triposr.eval()
    
    logger.info(f"TripoSR loaded on {device}");
    _flush()

    logger.info("✅ Optimized models ready");
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
                return send_file(
                    cached_result,
                    as_attachment=True,
                    download_name='3d_image.png',
                    mimetype='image/png'
                )

            # Decode input image
            try:
                png = base64.b64decode(data["sketch"].split(",", 1)[1])
                pil = Image.open(io.BytesIO(png)).convert("RGBA")
            except Exception as e:
                return jsonify({"error": f"Bad image data: {str(e)}"}), 400

            prompt = data.get("prompt", "a clean 3‑D asset")

            # Get parameters - use optimization profile if available, otherwise default to standard
            profile_name = data.get("profile", "standard")  # Default to standard
            custom_params = data.get("custom_params", {})

            if OPTIMIZATION_AVAILABLE:
                try:
                    params = get_profile_parameters(profile_name, custom_params)
                    logger.info(f"Using parameters from '{profile_name}' profile.")
                except Exception as e:
                    logger.warning(f"Failed to load profile '{profile_name}': {e}. Defaulting to 'standard'.")
                    params = get_profile_parameters("standard", custom_params)
            else:
                # Fallback if optimization_config.py is missing
                logger.warning("Optimization profiles not available. Using legacy parameter logic.")
                params = OptimizedParameters.get_optimized_params(data)

            logger.info(f"Using generation parameters: {params}")

            # A) Edge detection (optimized)
            edge = app.edge_det(pil);
            del pil

            # B) Stable Diffusion with optimized parameters
            with torch.no_grad():
                concept = app.sd(
                    prompt, image=edge,
                    num_inference_steps=params['num_inference_steps'], 
                    guidance_scale=params['guidance_scale']
                ).images[0]
            del edge;
            clear_gpu_memory()

            # C) Preprocess image for TripoSR (background removal and resizing)
            logger.info("Preprocessing image for TripoSR...")
            try:
                # Remove background and resize foreground
                processed_image = remove_background(concept, rembg_session)
                processed_image = resize_foreground(processed_image, 0.85)
                
                # Convert to proper format for TripoSR
                processed_image = np.array(processed_image).astype(np.float32) / 255.0
                processed_image = processed_image[:, :, :3] * processed_image[:, :, 3:4] + (1 - processed_image[:, :, 3:4]) * 0.5
                processed_image = Image.fromarray((processed_image * 255.0).astype(np.uint8))
                
                logger.info("Image preprocessing completed")
            except Exception as e:
                logger.warning(f"Background removal failed: {e}, using original image")
                processed_image = concept

            # D) Generate scene codes using TripoSR
            logger.info("Generating scene codes with TripoSR...")
            with torch.no_grad():
                with torch.cuda.amp.autocast() if device == "cuda" else nullcontext():
                    scene_codes = app.triposr([processed_image], device=device)
                    logger.info(f"Scene codes generated on device: {scene_codes.device}")
            clear_gpu_memory()

            # E) Render single 3D image
            render_resolution = params.get('render_resolution', 512)
            logger.info(f"Rendering 3D image with resolution {render_resolution}...")
            
            with torch.no_grad():
                with torch.cuda.amp.autocast() if device == "cuda" else nullcontext():
                    render_images = app.triposr.render(
                        scene_codes, 
                        n_views=1,  # Just one view
                        height=render_resolution, 
                        width=render_resolution,
                        return_type="pil"
                    )
            del scene_codes;
            clear_gpu_memory()

            # F) Get the single rendered image
            final_image = render_images[0][0]  # First (and only) image from first batch
            
            # Convert to base64 for response
            img_buffer = io.BytesIO()
            final_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            # Cache the result
            result_cache.put(cache_key, img_buffer)

            clear_gpu_memory();
            _flush()
            return send_file(
                img_buffer,
                as_attachment=True,
                download_name='3d_image.png',
                mimetype='image/png'
            )

        except Exception as e:
            logger.error("Error in /generate", exc_info=True);
            _flush()
            clear_gpu_memory()
            return jsonify({"error": str(e)}), 500

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000)

except Exception:
    logger.error("❌ Error during initialization", exc_info=True);
    _flush()
    raise
