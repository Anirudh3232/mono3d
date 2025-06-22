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

# ─────────────────────────── Hot‑patch dependencies ────────────────────────────
# 1) Create mock cache classes
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

# 2) Patch transformers main module
import transformers
for _n in ("Cache", "DynamicCache", "EncoderDecoderCache"):
    if not hasattr(transformers, _n):
        setattr(transformers, _n, MockEncoderDecoderCache)

# 3) Patch transformers.cache_utils
try:
    import transformers.cache_utils as _tcu
    for _n in ("Cache", "DynamicCache", "EncoderDecoderCache"):
        if not hasattr(_tcu, _n):
            setattr(_tcu, _n, MockEncoderDecoderCache)
except ImportError:
    pass

# 4) Patch transformers.models.encoder_decoder
try:
    import transformers.models.encoder_decoder as _ted
    if not hasattr(_ted, "EncoderDecoderCache"):
        setattr(_ted, "EncoderDecoderCache", MockEncoderDecoderCache)
except ImportError:
    pass

# 5) Fix diffusers/huggingface_hub mismatch
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "cached_download"):
    _hf_hub.cached_download = _hf_hub.hf_hub_download

# 6) Patch accelerate for peft compatibility
_acc_mem = importlib.import_module("accelerate.utils.memory")
if not hasattr(_acc_mem, "clear_device_cache"):
    _acc_mem.clear_device_cache = lambda *a, **k: None
# ───────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def _flush():
    for h in logger.handlers: h.flush()
atexit.register(_flush)

# ─────────────────────────── Helpers ───────────────────────────────────

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); gc.collect()

def gpu_mem_mb():
    return (torch.cuda.memory_allocated()/1024**2) if torch.cuda.is_available() else 0

def timing(fn):
    @wraps(fn)
    def _wrap(*a, **kw):
        t0=time.time(); out=fn(*a, **kw)
        logger.info(f"{fn.__name__} took {time.time()-t0:.2f}s"); _flush()
        return out
    return _wrap

print("Starting service initialization …")
try:
    logger.info("Importing diffusers …"); _flush()
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
    from controlnet_aux import CannyDetector
    from trimesh.smoothing import filter_taubin

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    TSR_PATH    = os.path.join(PROJECT_DIR, "TripoSR-main"); sys.path.append(TSR_PATH)
    from tsr.system import TSR

    # ───────────── Flask app setup ─────────────
    app = Flask(__name__); CORS(app)

    @app.get("/health")
    def health():
        return jsonify({"status":"ok","gpu_mb":gpu_mem_mb(),
                        "models_loaded":all(hasattr(app,x) for x in ("edge_det","cnet","sd","triposr"))})

    @app.route("/test", methods=["GET","POST"])
    def test():
        return jsonify({"message":"Server is working!","method":request.method})

    @app.get("/latest_concept_image")
    def get_latest_concept_image():
        if app.last_concept_image:
            buf = io.BytesIO()
            app.last_concept_image.save(buf, format="PNG")
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
        return jsonify({"error": "No concept image has been generated yet."}), 404

    # ───────────── Load models ─────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"; logger.info(f"Using {device}")
    if device=="cuda":
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True

    logger.info("Loading edge detector …");   _flush();   app.edge_det=CannyDetector()
    logger.info("Loading ControlNet …");     _flush();   app.cnet=ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
    logger.info("Loading Stable Diffusion …"); _flush()
    app.sd=StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=app.cnet,
        torch_dtype=torch.float16).to(device)
    # Replace scheduler for better detail preservation, per research guide
    logger.info("Replacing scheduler with EulerAncestralDiscreteScheduler for better detail preservation.")
    app.sd.scheduler = EulerAncestralDiscreteScheduler.from_config(app.sd.scheduler.config)
    try:
        app.sd.enable_xformers_memory_efficient_attention(); logger.info("xformers enabled")
    except Exception:
        logger.warning("xformers not available — using plain attention")
    app.sd.enable_model_cpu_offload(); app.sd.enable_attention_slicing()

    logger.info("Loading TripoSR …"); _flush()
    os.makedirs(os.path.join(TSR_PATH,"checkpoints"),exist_ok=True)
    app.last_concept_image = None # To store the last generated concept
    
    def ensure_module_on_device(module, target_device):
        """Helper function to ensure all tensors in a module are on the right device"""
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

    # Load TripoSR on GPU with mixed precision
    app.triposr = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt"
    )
    
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
    logger.info(f"TripoSR loaded on {device}"); _flush()

    logger.info("✅ Models ready"); _flush()

    # ───────────── /generate endpoint ─────────────
    @app.post("/generate")
    @timing
    def generate():
        try:
            if not request.is_json:
                return jsonify({"error":"Request must be JSON"}),400
            data=request.json
            if "sketch" not in data:
                return jsonify({"error":"Missing sketch"}),400

            # Decode input image
            try:
                png=base64.b64decode(data["sketch"].split(",",1)[1])
                pil=Image.open(io.BytesIO(png)).convert("RGB")
            except:
                return jsonify({"error":"Bad image data"}),400

            prompt     = data.get("prompt","a clean 3‑D asset")
            preview    = data.get("preview",True)

            # Allow overriding generation parameters for optimization
            num_inference_steps = int(data.get("num_inference_steps", 63))
            guidance_scale = float(data.get("guidance_scale", 9.96))
            smoothing_iterations = int(data.get("smoothing_iterations", 3))
            mesh_threshold = float(data.get("mesh_threshold", 25.0))

            # A) Edge detection
            edge = app.edge_det(pil); del pil

            # B) Stable Diffusion
            with torch.no_grad():
                concept = app.sd(
                    prompt, image=edge,
                    num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
                ).images[0]
            del edge; clear_gpu_memory()

            # C) Scene codes - use same device as model
            with torch.cuda.amp.autocast() if device == "cuda" else nullcontext():
                codes = app.triposr([concept], device=device)
                logger.info(f"Codes device: {codes.device}")
                
                # Re-ensure renderer is on correct device
                if hasattr(app.triposr, 'renderer'):
                    app.triposr.renderer = ensure_module_on_device(app.triposr.renderer, codes.device)
            # We no longer delete `concept` here, but we can still clear some memory
            clear_gpu_memory()

            # D) Mesh extraction
            res = 32 if preview else 128
            with torch.no_grad():
                with torch.cuda.amp.autocast() if device == "cuda" else nullcontext():
                    # Final device check before mesh extraction
                    if hasattr(app.triposr, 'renderer'):
                        app.triposr.renderer = ensure_module_on_device(app.triposr.renderer, codes.device)
                    meshes = app.triposr.extract_mesh(codes, resolution=res, threshold=mesh_threshold)
            del codes; clear_gpu_memory()

            # Export OBJ and texture
            mesh = meshes[0]
            
            # Post-process the mesh to reduce artifacts and improve quality
            logger.info(f"Applying Taubin smoothing to the mesh (iterations={smoothing_iterations})...")
            if smoothing_iterations > 0:
                filter_taubin(mesh, iterations=smoothing_iterations)
            
            # Process the mesh to fix potential issues before UV unwrapping
            logger.info("Processing mesh to fix potential issues...")
            mesh.process()

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
            
            clear_gpu_memory(); _flush()
            return send_file(
                zip_buffer,
                as_attachment=True,
                download_name='3d_model.zip',
                mimetype='application/zip'
            )

        except Exception as e:
            logger.error("Error in /generate", exc_info=True); _flush()
            clear_gpu_memory()
            return jsonify({"error":str(e)}),500

    if __name__=="__main__":
        app.run(host="0.0.0.0", port=5000)

except Exception:
    logger.error("❌ Error during initialization", exc_info=True); _flush()
    raise
