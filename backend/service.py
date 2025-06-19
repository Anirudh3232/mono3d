from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import sys, os, base64, io, gc, time, types, importlib, logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from functools import wraps

# ─────────────────────────── Hot‑patch deps ────────────────────────────
import transformers as _tf
for _n in ("Cache", "DynamicCache", "EncoderDecoderCache"):
    if not hasattr(_tf, _n):
        setattr(_tf, _n, types.SimpleNamespace)

_acc_mem = importlib.import_module("accelerate.utils.memory")
if not hasattr(_acc_mem, "clear_device_cache"):
    _acc_mem.clear_device_cache = lambda *a, **k: None
# ───────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────── Helpers ───────────────────────────────────

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); gc.collect()

def gpu_mem_mb():
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

def timing(fn):
    @wraps(fn)
    def _wrap(*a, **kw):
        t0 = time.time(); out = fn(*a, **kw)
        logger.info(f"{fn.__name__} took {time.time()-t0:.2f}s")
        return out
    return _wrap

print("Starting service initialization …")
try:
    logger.info("Importing diffusers …")
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from controlnet_aux import CannyDetector

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    TSR_PATH    = os.path.join(PROJECT_DIR, "TripoSR-main"); sys.path.append(TSR_PATH)
    from tsr.system import TSR

    # ───────────── Flask ─────────────
    app = Flask(__name__); CORS(app)

    @app.get("/health")
    def health():
        return jsonify({"status":"ok","gpu_mb":gpu_mem_mb(),
                        "models_loaded":all(hasattr(app,x) for x in ("edge_det","cnet","sd","triposr"))})

    @app.route("/test",methods=["GET","POST"])
    def test():
        return jsonify({"message":"Server is working!","method":request.method})

    # ───────────── Load models ─────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"; logger.info(f"Using {device}")
    if device=="cuda":
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True

    logger.info("Loading edge detector …");   app.edge_det = CannyDetector()
    logger.info("Loading ControlNet …");     app.cnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
    logger.info("Loading Stable Diffusion …")
    app.sd = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=app.cnet,
            torch_dtype=torch.float16).to(device)
    try:
        app.sd.enable_xformers_memory_efficient_attention(); logger.info("xformers on")
    except Exception:
        logger.warning("xformers not available — using plain attention")
    app.sd.enable_model_cpu_offload(); app.sd.enable_attention_slicing()

    logger.info("Loading TripoSR …")
    os.makedirs(os.path.join(TSR_PATH, "checkpoints"), exist_ok=True)
    app.triposr = TSR.from_pretrained("stabilityai/TripoSR",
                                      config_name="config.yaml",
                                      weight_name="model.ckpt").cpu().eval()

    logger.info("✅ Models ready")

    # ───────────── /generate ─────────────
    @app.post("/generate")
    @timing
    def generate():
        if not request.is_json:
            return jsonify({"error":"Request must be JSON"}),400
        data = request.json
        if "sketch" not in data:
            return jsonify({"error":"Missing sketch"}),400
        try:
            png = base64.b64decode(data["sketch"].split(",",1)[1])
            pil = Image.open(io.BytesIO(png)).convert("RGB")
        except Exception:
            return jsonify({"error":"Bad image"}),400

        prompt  = data.get("prompt","a clean 3‑D asset")
        preview = data.get("preview",True)

        # A) Canny edges
        edge = app.edge_det(pil)
        # B) ControlNet‑guided SD → concept
        col  = app.sd(prompt, image=edge, num_inference_steps=30, guidance_scale=7.5).images[0]
        clear_gpu_memory()

        # C) concept → scene codes (Plain CPU call avoids device mismatch)
        codes_cpu = app.triposr([col], device="cpu")
        res = 64 if preview else 128
        meshes   = app.triposr.extract_mesh(codes_cpu, resolution=res)

        mesh_bytes = meshes[0].export(file_type="obj")
        if isinstance(mesh_bytes,str):
            mesh_bytes = mesh_bytes.encode()
        clear_gpu_memory()
        return jsonify({"mesh": base64.b64encode(mesh_bytes).decode()})

    if __name__=="__main__":
        app.run(host="0.0.0.0", port=5000)

except Exception:
    logger.error("❌ Error during initialization", exc_info=True)
    raise

# ───────────── TripoSR helpers (unchanged) ─────────────
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
