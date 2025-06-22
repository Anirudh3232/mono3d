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

    # ───────────── Flask app setup ─────────────
    app = Flask(__name__); CORS(app)

    @app.get("/health")
    def health():
        return jsonify({"status":"ok","gpu_mb":gpu_mem_mb(),
                        "models_loaded":all(hasattr(app,x) for x in ("edge_det","cnet","sd"))})

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
    logger.info("Replacing scheduler with EulerAncestralDiscreteScheduler for better detail preservation.")
    app.sd.scheduler = EulerAncestralDiscreteScheduler.from_config(app.sd.scheduler.config)
    try:
        app.sd.enable_xformers_memory_efficient_attention(); logger.info("xformers enabled")
    except Exception:
        logger.warning("xformers not available — using plain attention")
    app.sd.enable_model_cpu_offload(); app.sd.enable_attention_slicing()

    logger.info("✅ Models ready"); _flush()

    # ───────────── /generate endpoint (Simplified for 2D) ─────────────
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

            prompt = data.get("prompt","a clean 3‑D asset")

            # Allow overriding generation parameters for optimization
            num_inference_steps = int(data.get("num_inference_steps", 50))
            guidance_scale = float(data.get("guidance_scale", 9.0))

            # A) Edge detection
            edge = app.edge_det(pil); del pil

            # B) Stable Diffusion
            with torch.no_grad():
                concept = app.sd(
                    prompt, image=edge,
                    num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
                ).images[0]
            del edge; clear_gpu_memory()

            # C) Return the concept image directly
            buf = io.BytesIO()
            concept.save(buf, format="PNG")
            buf.seek(0)
            
            clear_gpu_memory(); _flush()
            return send_file(
                buf,
                mimetype='image/png'
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
