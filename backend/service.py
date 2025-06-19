from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import base64
import io
import gc
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from PIL import Image
import logging
import time
from functools import wraps
import types, importlib

# ─────────────────────────── Hot‑patch dependencies ───────────────────────────
# 1)  Transformers ≥4.41 moved cache classes; stub them if missing
import transformers as _tf
for _name in ("Cache", "DynamicCache", "EncoderDecoderCache"):
    if not hasattr(_tf, _name):
        setattr(_tf, _name, types.SimpleNamespace)

# 2)  accelerate <0.28 lacks clear_device_cache; stub for peft
_acc_mem = importlib.import_module("accelerate.utils.memory")
if not hasattr(_acc_mem, "clear_device_cache"):
    _acc_mem.clear_device_cache = lambda *a, **kw: None
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ───────────────────────────── Utility helpers ───────────────────────────────
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_gpu_memory_usage():
    return torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0

def timing_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = f(*args, **kwargs)
        logger.info(f"Request completed in {time.time() - t0:.2f}s")
        return out
    return wrapper

print("Starting service initialization…")

try:
    # ───────────────────────────── Import heavy libs ──────────────────────────
    logger.info("Importing diffusers…")
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from controlnet_aux import CannyDetector

    # Add TripoSR repo to path
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    TSR_PATH = os.path.join(PROJECT_DIR, "TripoSR-main")
    sys.path.append(TSR_PATH)
    logger.info(f"Added to path: {TSR_PATH}")
    from tsr.system import TSR

    # ────────────────────────────── Flask setup ───────────────────────────────
    app = Flask(__name__)
    CORS(app)

    @app.route("/test", methods=["GET", "POST"])
    def test():
        return jsonify({"message": "Server is working!", "method": request.method})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "gpu_mb": get_gpu_memory_usage(),
            "models_loaded": all(hasattr(app, attr) for attr in ("edge_det", "cnet", "sd", "triposr"))
        })

    # ───────────────────────────── Load models ────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.cuda.memory.set_per_process_memory_fraction(0.8)
        except AttributeError:
            pass

    logger.info("Loading edge detector…")
    app.edge_det = CannyDetector()

    logger.info("Loading ControlNet model…")
    app.cnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)

    logger.info("Loading Stable Diffusion model…")
    app.sd = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=app.cnet,
        torch_dtype=torch.float16
    ).to(device)

    # Memory‑efficient tweaks
    try:
        app.sd.enable_xformers_memory_efficient_attention()
        logger.info("xformers memory‑efficient attention enabled")
    except (ModuleNotFoundError, AttributeError, ValueError):
        logger.warning("xformers not available – running without it")

    app.sd.enable_model_cpu_offload()
    app.sd.enable_attention_slicing()

    logger.info("Loading TSR model…")
    os.makedirs(os.path.join(TSR_PATH, "checkpoints"), exist_ok=True)
    app.triposr = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt").to(device)
    app.triposr.eval()

    logger.info("✅ Models ready")

    # ──────────────────────────── Generate endpoint ───────────────────────────
    @app.route("/generate", methods=["POST"])
    @timing_decorator
    def generate():
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.json
        if "sketch" not in data:
            return jsonify({"error": "Missing sketch data"}), 400

        prompt = data.get("prompt", "a clean 3‑D asset")
        try:
            png_bytes = base64.b64decode(data["sketch"].split(",", 1)[1])
            pil_in = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid image data"}), 400

        clear_gpu_memory()

        edge_img = app.edge_det(pil_in)

        # B) ControlNet‑guided Stable Diffusion → color concept
        color_img = app.sd(
            prompt,
            image=edge_img,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]
        clear_gpu_memory()

        # C) Color concept → TripoSR scene codes
        scene_codes = app.triposr([color_img], device=device)

        # D) Extract mesh – keep all tensors on a single device
        preview = data.get("preview", True)
        if preview:
            scene_codes = scene_codes.to(device)
            meshes = app.triposr.extract_mesh(scene_codes, resolution=64, device=device)
        else:
            scene_codes_cpu = scene_codes.cpu()
            meshes = app.triposr.cpu().extract_mesh(scene_codes_cpu, resolution=128, device="cpu")

        mesh_bytes = meshes[0].export(file_type="obj")
        if isinstance(mesh_bytes, str):
            mesh_bytes = mesh_bytes.encode()

        clear_gpu_memory()
        return jsonify({"mesh": base64.b64encode(mesh_bytes).decode()})({"mesh": base64.b64encode(mesh_bytes).decode()})

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000)

except Exception as e:
    logger.error("❌ Error during initialization", exc_info=True)
    raise


# ───────────────────────────── Extra classes (TripoSR) ───────────────────────
class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)
    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError

class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            x = torch.linspace(*self.points_range, self.resolution)
            y = torch.linspace(*self.points_range, self.resolution)
            z = torch.linspace(*self.points_range, self.resolution)
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat([x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)], dim=-1).reshape(-1,3)
            self._grid_vertices = verts
        return self._grid_vertices
