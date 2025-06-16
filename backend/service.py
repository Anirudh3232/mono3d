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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Memory management utilities
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0

# Request timing decorator
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
    # ─────────────────────────  Stable Diffusion + ControlNet ─────────────────────
    logger.info("Importing diffusers...")
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from controlnet_aux import CannyDetector

    # ────────────────────────────  make TripoSR importable ────────────────────────
    logger.info("Setting up TripoSR path...")
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(PROJECT_DIR, "TripoSR-main"))
    logger.info(f"Added to path: {os.path.join(PROJECT_DIR, 'TripoSR-main')}")

    logger.info("Importing TSR...")
    from tsr.system import TSR

    # ───────────────────────────────  Flask service  ──────────────────────────────
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    # Add a test endpoint
    @app.route("/test", methods=["GET", "POST"])
    def test():
        return jsonify({"message": "Server is working!", "method": request.method})

    # Add health check endpoint with memory info
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

    # Set device and memory optimization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Configure PyTorch for optimal memory usage
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(0.8)

    try:
        # 1) Edge detector (runs on CPU, fast)
        logger.info("Loading edge detector...")
        app.edge_det = CannyDetector()

        # 2) ControlNet model that expects Canny edges
        logger.info("Loading ControlNet model...")
        app.cnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        ).to(device)

        # 3) Stable Diffusion v1‒5 with the Canny ControlNet
        logger.info("Loading Stable Diffusion model...")
        app.sd = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=app.cnet,
            torch_dtype=torch.float16
        ).to(device)
        
        # Enable memory optimizations
        app.sd.enable_xformers_memory_efficient_attention()
        app.sd.enable_model_cpu_offload()
        app.sd.enable_attention_slicing()

        # 4) Single-image-to-mesh model
        logger.info("Loading TSR model...")
        model_path = os.path.join(PROJECT_DIR, "TripoSR-main", "checkpoints")
        if not os.path.exists(model_path):
            logger.info(f"Creating model directory: {model_path}")
            os.makedirs(model_path, exist_ok=True)

        # Download model from HuggingFace if not present
        logger.info("Downloading TSR model from HuggingFace...")
        app.triposr = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt"
        )

        # Ensure model is on the correct device and in eval mode
        app.triposr = app.triposr.to(device)
        app.triposr.eval()

    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}", exc_info=True)
        raise

    logger.info("✅ Models ready")

    # ────────────────────────────  /generate endpoint  ────────────────────────────
    @app.route("/generate", methods=["POST"])
    @timing_decorator
    def generate():
        try:
            logger.info("Received request to /generate endpoint")
            
            # Validate request
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            
            data = request.json
            if "sketch" not in data:
                return jsonify({"error": "Missing sketch data"}), 400

            prompt = data.get("prompt", "a clean 3-D asset")
            logger.info(f"Processing request with prompt: {prompt}")

            # decode base64 PNG
            try:
                png_bytes = base64.b64decode(data["sketch"].split(",", 1)[1])
                pil_in = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                logger.info("Successfully decoded input image")
            except Exception as e:
                logger.error(f"Failed to decode image: {str(e)}")
                return jsonify({"error": "Invalid image data"}), 400

            # Clear GPU memory before processing
            clear_gpu_memory()
            logger.info(f"GPU memory before processing: {get_gpu_memory_usage():.2f} MB")

            # A) sketch → Canny edge map
            logger.info("Generating edge map...")
            edge_img = app.edge_det(pil_in)

            # B) edge map → colored concept
            logger.info("Generating colored concept...")
            color_img = app.sd(
                prompt, 
                image=edge_img, 
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]

            # Clear memory after SD generation
            clear_gpu_memory()
            logger.info(f"GPU memory after SD: {get_gpu_memory_usage():.2f} MB")

            # C) concept image → 3-D OBJ bytes
            logger.info("Generating 3D mesh...")
            preview = data.get("preview", True)

            try:
                # Ensure all operations are on the same device
                with torch.cuda.device(device):
                    # Move color_img to the correct device
                    if hasattr(color_img, 'to'):
                        color_img = color_img.to(device)

                    # Generate scene codes
                    scene_codes = app.triposr([color_img], device=device)

                    if preview:
                        logger.info("Using fast preview mode (GPU, low resolution)...")
                        # Ensure scene_codes is on GPU
                        scene_codes = scene_codes.to(device)
                        torch.cuda.synchronize()
                        meshes = app.triposr.extract_mesh(scene_codes, resolution=64)
                    else:
                        logger.info("Using high-quality export mode (CPU, medium resolution 128)...")
                        triposr_cpu = app.triposr.cpu()
                        scene_codes_cpu = scene_codes.cpu()
                        meshes = triposr_cpu.extract_mesh(scene_codes_cpu, resolution=128)

                    # Ensure mesh is on CPU before export
                    if hasattr(meshes[0], 'to'):
                        meshes[0] = meshes[0].to('cpu')

                    # Force CUDA synchronization before export
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    mesh_obj_bytes = meshes[0].export(file_type="obj")
                    if isinstance(mesh_obj_bytes, str):
                        mesh_obj_bytes = mesh_obj_bytes.encode("utf-8")

                    # Clear memory after processing
                    clear_gpu_memory()
                    logger.info(f"GPU memory after processing: {get_gpu_memory_usage():.2f} MB")

                    return jsonify({"mesh": base64.b64encode(mesh_obj_bytes).decode()})

            except Exception as e:
                logger.error(f"Error during mesh generation: {str(e)}", exc_info=True)
                clear_gpu_memory()
                return jsonify({"error": "Failed to generate mesh"}), 500

        except Exception as e:
            logger.error(f"Error in generate endpoint: {str(e)}", exc_info=True)
            clear_gpu_memory()
            return jsonify({"error": str(e)}), 500

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000)

except Exception as e:
    logger.error(f"❌ Error during initialization: {str(e)}")
    logger.error(f"Error type: {type(e).__name__}")
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
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
            self,
            level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)

        # Convert to numpy for processing
        level_np = level.detach().cpu().numpy()

        # Use scipy's marching cubes
        from scipy.spatial import Delaunay
        from scipy.interpolate import griddata

        # Create a simple mesh using the level set
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        z = np.linspace(0, 1, self.resolution)
        X, Y, Z = np.meshgrid(x, y, z)

        # Get points where level is close to 0
        mask = np.abs(level_np) < 0.1
        points = np.column_stack((X[mask], Y[mask], Z[mask]))
        values = level_np[mask]

        # Create a simple surface
        if len(points) > 4:  # Need at least 4 points for a 3D surface
            tri = Delaunay(points)
            v_pos = torch.from_numpy(points).float()
            t_pos_idx = torch.from_numpy(tri.simplices).long()
        else:
            # Fallback to a simple cube if not enough points
            v_pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float)
            t_pos_idx = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=torch.long)

        return v_pos, t_pos_idx
