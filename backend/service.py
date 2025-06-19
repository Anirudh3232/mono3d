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

import transformers as _tf, types
for _name in ("Cache", "DynamicCache", "EncoderDecoderCache"):
    if not hasattr(_tf, _name):
        setattr(_tf, _name, types.SimpleNamespace)

import importlib
_acc_mem = importlib.import_module("accelerate.utils.memory")
if not hasattr(_acc_mem, "clear_device_cache"):
    def _noop(*args, **kwargs):
        """Fallback no‑op for peft >=0.15 when running with accelerate<0.28"""
        pass
    _acc_mem.clear_device_cache = _noop



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0


def timing_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Request completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

print("Starting service initialization...")

try:
    
    logger.info("Importing diffusers...")
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from controlnet_aux import CannyDetector

   
    logger.info("Setting up TripoSR path...")
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(PROJECT_DIR, "TripoSR-main"))
    logger.info(f"Added to path: {os.path.join(PROJECT_DIR, 'TripoSR-main')}")

    logger.info("Importing TSR...")
    from tsr.system import TSR


    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    # Test endpoint
    @app.route("/test", methods=["GET", "POST"])
    def test():
        return jsonify({"message": "Server is working!", "method": request.method})

    # Health check endpoint
    @app.route("/health", methods=["GET"])
    def health_check():
        memory_usage = get_gpu_memory_usage()
        return jsonify({
            "status": "healthy",
            "gpu_memory_usage_mb": memory_usage,
            "models_loaded": all([
                hasattr(app, 'edge_det'),
                hasattr(app, 'cnet'),
                hasattr(app, 'sd'),
                hasattr(app, 'triposr')
            ])
        })

    logger.info("🔄 Loading models …")

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # CUDA performance tweaks
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.cuda.memory.set_per_process_memory_fraction(0.8)
        except AttributeError:
            pass

    try:
        # 1) Edge detector
        logger.info("Loading edge detector...")
        app.edge_det = CannyDetector()

        # 2) ControlNet model
        logger.info("Loading ControlNet model...")
        app.cnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        ).to(device)

        # 3) Stable Diffusion with ControlNet
        logger.info("Loading Stable Diffusion model...")
        app.sd = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=app.cnet,
            torch_dtype=torch.float16
        ).to(device)

        # Memory‑efficient tweaks
        if hasattr(app.sd, "enable_xformers_memory_efficient_attention"):
            app.sd.enable_xformers_memory_efficient_attention()
        app.sd.enable_model_cpu_offload()
        app.sd.enable_attention_slicing()

        # 4) TripoSR
        logger.info("Loading TSR model...")
        model_path = os.path.join(PROJECT_DIR, "TripoSR-main", "checkpoints")
        os.makedirs(model_path, exist_ok=True)

        app.triposr = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt"
        ).to(device)
        app.triposr.eval()

    except Exception as e:
        logger.error("Failed to load models", exc_info=True)
        raise

    logger.info("✅ Models ready")

  
    @app.route("/generate", methods=["POST"])
    @timing_decorator
    def generate():
        try:
            logger.info("Received /generate request")

            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400

            data = request.json
            if "sketch" not in data:
                return jsonify({"error": "Missing sketch data"}), 400

            prompt = data.get("prompt", "a clean 3-D asset")

            # Decode image
            try:
                png_bytes = base64.b64decode(data["sketch"].split(",", 1)[1])
                pil_in = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            except Exception:
                return jsonify({"error": "Invalid image data"}), 400

            clear_gpu_memory()

            # A) Edge detection
            edge_img = app.edge_det(pil_in)

            # B) Stable Diffusion generation
            color_img = app.sd(
                prompt,
                image=edge_img,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]

            clear_gpu_memory()

            # C) 3‑D mesh extraction
            preview = data.get("preview", True)
            with torch.cuda.device(device):
                scene_codes = app.triposr([color_img], device=device)

                if preview:
                    meshes = app.triposr.extract_mesh(scene_codes.to(device), resolution=64)
                else:
                    meshes = app.triposr.cpu().extract_mesh(scene_codes.cpu(), resolution=128)

            mesh_obj_bytes = meshes[0].export(file_type="obj")
            if isinstance(mesh_obj_bytes, str):
                mesh_obj_bytes = mesh_obj_bytes.encode()

            clear_gpu_memory()
            return jsonify({"mesh": base64.b64encode(mesh_obj_bytes).decode()})

        except Exception as e:
            logger.error("Error in /generate", exc_info=True)
            clear_gpu_memory()
            return jsonify({"error": "Failed to generate mesh"}), 500

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000)

except Exception as e:
    logger.error(f"❌ Error during initialization: {e}")
    raise



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
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat([
                x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)
            ], dim=-1).reshape(-1, 3)
            self._grid_vertices = verts
