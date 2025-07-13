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
    from controlnet_aux import CannyDetector
    from trimesh.smoothing import filter_laplacian
    
        # TripoSR imports - load from Hugging Face
    import rembg

    # Import optimization profiles
    try:
        from optimization_config import get_profile_parameters, list_profiles, get_recommended_profile
        OPTIMIZATION_AVAILABLE = True
        logger.info("Optimization profiles loaded successfully")   
    except ImportError:
        OPTIMIZATION_AVAILABLE = False
        logger.warning("Optimization profiles not available, using default parameters")

    # Load TripoSR model from Hugging Face
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"        
    logger.info(f"Loading TripoSR model locally on {DEVICE} ...")
    
    # Add the TripoSR directory to Python path for Colab
    triposr_path = os.path.join(os.getcwd(), "backend", "TripoSR-main")
    if triposr_path not in sys.path:
        sys.path.insert(0, triposr_path)
    
    # Import TSR from local TripoSR files
    try:
        from tsr.system import TSR
        logger.info("TSR module imported successfully from local files")
        # Initialize TSR without loading from Hugging Face
        triposr_model = TSR()
        triposr_model.to(DEVICE)
        triposr_model.eval()
        logger.info("TripoSR model loaded from local files.")
    except ImportError as e:
        logger.error(f"Failed to import TSR module: {e}")
        logger.error("Please ensure TripoSR-main directory is present in backend/")
        triposr_model = None
    
    rembg_session = rembg.new_session()

    
    app = Flask(__name__);
    CORS(app)

    # Helper: Check if running in Colab
    import importlib.util

    def in_colab():
        return importlib.util.find_spec('google.colab') is not None

    def download_triposr_if_needed():
        triposr_path = os.path.join(os.getcwd(), "backend", "TripoSR-main")
        if not os.path.exists(triposr_path) and in_colab():
            import subprocess
            print("TripoSR-main not found. Downloading from GitHub...")
            subprocess.run(["git", "clone", "https://github.com/stabilityai/TripoSR.git", triposr_path], check=True)
            print("TripoSR-main downloaded.")

    # Optionally download TripoSR-main if in Colab
    try:
        download_triposr_if_needed()
    except Exception as e:
        logger.warning(f"Could not download TripoSR-main: {e}")

    # Try to import TSR
    triposr_model = None
    try:
        triposr_path = os.path.join(os.getcwd(), "backend", "TripoSR-main")
        if triposr_path not in sys.path:
            sys.path.insert(0, triposr_path)
        from tsr.system import TSR
        logger.info("TSR module imported successfully from local files")
        triposr_model = TSR()
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        triposr_model.to(DEVICE)
        triposr_model.eval()
        logger.info("TripoSR model loaded from local files.")
    except Exception as e:
        logger.warning(f"TripoSR not available: {e}")
        triposr_model = None

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok", 
            "gpu_mb": gpu_mem_mb(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "triposr_available": triposr_model is not None,
            "optimization_available": OPTIMIZATION_AVAILABLE
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

    logger.info("Loading TripoSR locally…");
    _flush()
    app.last_concept_image = None

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

    # Use the already loaded TripoSR model
    app.triposr = triposr_model

    # If TripoSR is not available, disable endpoints that require it
    if app.triposr is None:
        logger.warning("TripoSR is not available. /generate endpoint will return an error.")

        @app.post("/generate")
        def generate_unavailable():
            return jsonify({"error": "TripoSR is not available. Please add backend/TripoSR-main or run in Colab to auto-download."}), 503
    else:
        # Ensure everything is on the correct device
        app.triposr = ensure_module_on_device(app.triposr, device)
        if hasattr(app.triposr, 'renderer'):
            app.triposr.renderer = ensure_module_on_device(app.triposr.renderer, device)
            if hasattr(app.triposr.renderer, 'triplane'):
                app.triposr.renderer.triplane = app.triposr.renderer.triplane.to(device)

        if device == "cuda":
            # Convert to half precision if using CUDA
            app.triposr = app.triposr.half()
            if hasattr(app.triposr, 'renderer'):
                app.triposr.renderer = app.triposr.renderer.half()

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
            cache_key = f"{data['sketch'][:100]}_{data.get('prompt', '')}_{data.get('preview', True)}"
            cached_result = result_cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                return send_file(
                    cached_result,
                    as_attachment=True,
                    download_name='3d_model.zip',
                    mimetype='application/zip'
                )

            # Decode input image
            try:
                png = base64.b64decode(data["sketch"].split(",", 1)[1])
                pil = Image.open(io.BytesIO(png)).convert("RGBA")
            except Exception as e:
                return jsonify({"error": f"Bad image data: {str(e)}"}), 400

            prompt = data.get("prompt", "a clean 3‑D asset")
            preview = data.get("preview", True)

            # Get parameters - use optimization profile if available, otherwise default to maximum quality
            profile_name = data.get("profile", "maximum_quality") # Default to maximum_quality
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

            # C) Scene codes - use same device as model
            with torch.cuda.amp.autocast() if device == "cuda" else nullcontext():
                codes = app.triposr([concept], device=device)
                logger.info(f"Codes device: {codes.device}")

                # Re-ensure renderer is on correct device
                if hasattr(app.triposr, 'renderer'):
                    app.triposr.renderer = ensure_module_on_device(app.triposr.renderer, codes.device)
            clear_gpu_memory()

            # D) Mesh extraction with optimized resolution
            res = params['preview_resolution'] if preview else params['full_resolution']
            with torch.no_grad():
                with torch.cuda.amp.autocast() if device == "cuda" else nullcontext():
                    # Final device check before mesh extraction
                    if hasattr(app.triposr, 'renderer'):
                        app.triposr.renderer = ensure_module_on_device(app.triposr.renderer, codes.device)
                    meshes = app.triposr.extract_mesh(codes, resolution=res, threshold=params['mesh_threshold'])
            del codes;
            clear_gpu_memory()

            # Export OBJ and texture
            mesh = meshes[0]

            # Conditional Laplacian smoothing (only if enabled)
            if params['smoothing_iterations'] > 0:
                logger.info(f"Applying Laplacian smoothing to the mesh (iterations={params['smoothing_iterations']})...")
                try:
                    if mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0:
                        filter_laplacian(mesh, iterations=params['smoothing_iterations'])
                        logger.info("Laplacian smoothing completed successfully")
                    else:
                        logger.warning("Mesh is empty, skipping smoothing")
                except Exception as e:
                    logger.warning(f"Laplacian smoothing failed: {e}, continuing without smoothing")

            # Process the mesh to fix potential issues before UV unwrapping
            logger.info("Processing mesh to fix potential issues...")
            try:
                mesh.process()
                logger.info("Mesh processing completed")
            except Exception as e:
                logger.warning(f"Mesh processing failed: {e}, continuing with original mesh")

            # Use xatlas to generate UVs
            import xatlas
            import trimesh

            vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

            # Create a new mesh with the unwrapped UVs
            uv_mesh = trimesh.Trimesh(
                vertices=mesh.vertices[vmapping],
                faces=indices,
                vertex_normals=mesh.vertex_normals[vmapping],
                visual=trimesh.visual.TextureVisuals(uv=uvs)
            )

            # Create a texture from the concept image
            texture = concept.resize((1024, 1024))

            # Create material
            material = trimesh.visual.material.SimpleMaterial(image=texture)

            # Assign material to the mesh
            uv_mesh.visual.material = material
            app.last_concept_image = concept.copy()

            # Export to a zip file in memory
            import zipfile
            zip_buffer = io.BytesIO()

            # Manually export each component for robustness
            obj_data = trimesh.exchange.obj.export_obj(uv_mesh, mtl_name="texture.mtl")

            # Manually create the MTL file content as a string
            mtl_data = f"""
newmtl material_0
Ka 1.000000 1.000000 1.000000
Kd 1.000000 1.000000 1.000000
Ks 0.000000 0.000000 0.000000
Tr 1.000000
illum 2
Ns 0.000000
map_Kd texture.png
"""

            # Save the texture image to a buffer
            texture_buffer = io.BytesIO()
            uv_mesh.visual.material.image.save(texture_buffer, format='PNG')
            texture_data = texture_buffer.getvalue()

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("model.obj", obj_data)
                zf.writestr("texture.mtl", mtl_data)
                zf.writestr("texture.png", texture_data)

            zip_buffer.seek(0)

            # Cache the result
            result_cache.put(cache_key, zip_buffer)

            clear_gpu_memory();
            _flush()
            return send_file(
                zip_buffer,
                as_attachment=True,
                download_name='3d_model.zip',
                mimetype='application/zip'
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
