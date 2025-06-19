from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import sys, os, base64, io, gc, time, types, importlib, logging, atexit
from flask import Flask, request, jsonify
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
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from controlnet_aux import CannyDetector

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
    try:
        app.sd.enable_xformers_memory_efficient_attention(); logger.info("xformers enabled")
    except Exception:
        logger.warning("xformers not available — using plain attention")
    app.sd.enable_model_cpu_offload(); app.sd.enable_attention_slicing()

    logger.info("Loading TripoSR …"); _flush()
    os.makedirs(os.path.join(TSR_PATH,"checkpoints"),exist_ok=True)
    # Load TripoSR on GPU with mixed precision
    app.triposr = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt"
    ).to(device)
    
    # Ensure renderer and all submodules are on the correct device
    if hasattr(app.triposr, 'renderer'):
        app.triposr.renderer = app.triposr.renderer.to(device)
        if hasattr(app.triposr.renderer, 'triplane'):
            app.triposr.renderer.triplane = app.triposr.renderer.triplane.to(device)
    
    if device == "cuda":
        app.triposr = app.triposr.half()  # Use half precision on GPU
        # Also convert renderer to half precision if it exists
        if hasattr(app.triposr, 'renderer'):
            app.triposr.renderer = app.triposr.renderer.half()
    app.triposr.eval()

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

            # A) Edge detection
            edge = app.edge_det(pil); del pil

            # B) Stable Diffusion
            with torch.no_grad():
                concept = app.sd(
                    prompt, image=edge,
                    num_inference_steps=20, guidance_scale=7.5
                ).images[0]
            del edge; clear_gpu_memory()

            # C) Scene codes - use same device as model
            with torch.cuda.amp.autocast() if device == "cuda" else nullcontext():
                codes = app.triposr([concept], device=device)
                logger.info(f"Codes device: {codes.device}")
                # Ensure renderer is on the same device as codes
                if hasattr(app.triposr, 'renderer'):
                    app.triposr.renderer = app.triposr.renderer.to(codes.device)
            del concept; clear_gpu_memory()

            # D) Mesh extraction
            res = 32 if preview else 128
            with torch.no_grad():
                with torch.cuda.amp.autocast() if device == "cuda" else nullcontext():
                    # Double check device consistency before mesh extraction
                    if hasattr(app.triposr, 'renderer') and app.triposr.renderer.device != codes.device:
                        app.triposr.renderer = app.triposr.renderer.to(codes.device)
                    meshes = app.triposr.extract_mesh(codes, resolution=res)
            del codes; clear_gpu_memory()

            # Export OBJ
            mesh_bytes = meshes[0].export(file_type="obj")
            if isinstance(mesh_bytes, str): mesh_bytes = mesh_bytes.encode()
            clear_gpu_memory(); _flush()
            return jsonify({"mesh":base64.b64encode(mesh_bytes).decode()})

        except Exception as e:
            logger.error("Error in /generate", exc_info=True); _flush()
            clear_gpu_memory()
            return jsonify({"error":str(e)}),500

    if __name__=="__main__":
        app.run(host="0.0.0.0", port=5000)

except Exception:
    logger.error("❌ Error during initialization", exc_info=True); _flush()
    raise

# ───────────── TripoSR helpers ─────────────
class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float,float]=(0,1)
    @property
    def grid_vertices(self)->torch.FloatTensor: raise NotImplementedError

class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self,res:int)->None:
        super().__init__(); self.resolution=res; self._grid_vertices=None
    @property
    def grid_vertices(self)->torch.FloatTensor:
        if self._grid_vertices is None:
            x=y=z=torch.linspace(*self.points_range,self.resolution)
            x,y,z=torch.meshgrid(x,y,z,indexing="ij")
            self._grid_vertices=torch.cat([t.reshape(-1,1) for t in (x,y,z)],dim=-1)
        return self._grid_vertices
