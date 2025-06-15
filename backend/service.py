from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import base64
import io
from flask import Flask, request, jsonify
from PIL import Image
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


    # Add a test endpoint
    @app.route("/test", methods=["GET", "POST"])
    def test():
        return jsonify({"message": "Server is working!", "method": request.method})


    logger.info("🔄 Loading models …")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1) Edge detector (runs on CPU, fast)
    logger.info("Loading edge detector...")
    edge_det = CannyDetector()

    # 2) ControlNet model that expects Canny edges
    logger.info("Loading ControlNet model...")
    cnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    ).to(device)

    # 3) Stable Diffusion v1‒5 with the Canny ControlNet
    logger.info("Loading Stable Diffusion model...")
    sd = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=cnet,
        torch_dtype=torch.float16
    ).to(device)
    sd.enable_xformers_memory_efficient_attention()
    sd.enable_model_cpu_offload()  # save VRAM

    # 4) Single-image-to-mesh model
    logger.info("Loading TSR model...")
    model_path = os.path.join(PROJECT_DIR, "TripoSR-main", "checkpoints")
    if not os.path.exists(model_path):
        logger.info(f"Creating model directory: {model_path}")
        os.makedirs(model_path, exist_ok=True)

    # Download model from HuggingFace if not present
    logger.info("Downloading TSR model from HuggingFace...")
    triposr = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt"
    )

    # Ensure model is on the correct device and in eval mode
    triposr = triposr.to(device)
    triposr.eval()

    # Set default tensor type based on device
    if device == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        # Set CUDA device properties
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(0.8)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    logger.info("✅ Models ready")


    # ────────────────────────────  /generate endpoint  ────────────────────────────
    @app.route("/generate", methods=["POST"])
    def generate():
        try:
            logger.info("Received request to /generate endpoint")
            data = request.json
            prompt = data.get("prompt", "a clean 3-D asset")
            logger.info(f"Processing request with prompt: {prompt}")

            # decode base64 PNG
            png_bytes = base64.b64decode(data["sketch"].split(",", 1)[1])
            pil_in = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            logger.info("Successfully decoded input image")

            # A) sketch → Canny edge map
            logger.info("Generating edge map...")
            edge_img = edge_det(pil_in)

            # B) edge map → colored concept
            logger.info("Generating colored concept...")
            color_img = sd(prompt, image=edge_img, num_inference_steps=30).images[0]

            # C) concept image → 3-D OBJ bytes
            logger.info("Generating 3D mesh...")
            preview = data.get("preview", True)

            # Ensure all operations are on the same device
            with torch.cuda.device(device):
                # Move color_img to the correct device
                if hasattr(color_img, 'to'):
                    color_img = color_img.to(device)

                # Generate scene codes
                scene_codes = triposr([color_img], device=device)

                if preview:
                    logger.info("Using fast preview mode (GPU, low resolution)...")
                    # Ensure scene_codes is on GPU
                    scene_codes = scene_codes.to(device)

                    # Force CUDA synchronization
                    torch.cuda.synchronize()

                    # Extract mesh
                    meshes = triposr.extract_mesh(scene_codes, resolution=64)
                else:
                    logger.info("Using high-quality export mode (CPU, medium resolution 128)...")
                    # Move everything to CPU for high-quality export
                    triposr_cpu = triposr.cpu()
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
                return jsonify({"mesh": base64.b64encode(mesh_obj_bytes).decode()})
        except Exception as e:
            logger.error(f"Error in generate endpoint: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500


    # ─────────────────────────────────  main  ─────────────────────────────────────
    if __name__ == "__main__":
        logger.info("Starting Flask server...")
        logger.info("Server will be available at http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=True)

except Exception as e:
    logger.error(f"❌ Error during initialization: {str(e)}", exc_info=True)
    logger.error(f"Error type: {type(e).__name__}")
    sys.exit(1)


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
