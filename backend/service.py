from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import sys, os, base64, io, gc, time, types, importlib, logging, atexit, tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from functools import wraps
from torch.cuda.amp import autocast
from contextlib import nullcontext
import psutil
import subprocess
import trimesh
import zipfile

# [Previous compatibility patches remain the same...]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('service.log')
    ]
)
logger = logging.getLogger(__name__)

# [Previous helper functions remain the same...]

def create_high_quality_mesh_render(mesh, image_size=(1024, 1024)):
    """Create high-quality renders from mesh using trimesh"""
    try:
        # Create scene with proper lighting
        scene = trimesh.Scene([mesh])
        
        # Add multiple light sources for better quality
        scene.add_geometry(trimesh.creation.axis())
        
        # Configure high-quality rendering
        render_kwargs = {
            'resolution': image_size,
            'flags': {
                'shadows': True,
                'smooth': True,
                'wireframe': False
            }
        }
        
        # Generate multiple high-quality views
        views = []
        angles = [(0, 0), (45, 0), (0, 45), (-45, 0), (0, -45), (45, 45)]
        
        for azimuth, elevation in angles:
            # Set camera position
            camera_transform = trimesh.transformations.rotation_matrix(
                np.radians(elevation), [1, 0, 0]
            ) @ trimesh.transformations.rotation_matrix(
                np.radians(azimuth), [0, 0, 1]
            )
            
            # Render view
            try:
                rendered = scene.save_image(**render_kwargs)
                if rendered:
                    img = Image.open(io.BytesIO(rendered))
                    views.append(img)
            except Exception as e:
                logger.warning(f"Failed to render view {azimuth}, {elevation}: {e}")
                continue
        
        return views if views else None
        
    except Exception as e:
        logger.error(f"High-quality mesh rendering failed: {e}")
        return None

class HighQualityParameters:
    """High-quality parameters optimized for mesh generation"""
    
    DEFAULT_INFERENCE_STEPS = 30  # Increased for better quality
    DEFAULT_GUIDANCE_SCALE = 8.0  # Higher guidance for better detail
    DEFAULT_N_VIEWS = 6          # More views for better selection
    DEFAULT_HEIGHT = 1024        # High resolution
    DEFAULT_WIDTH = 1024         # High resolution
    DEFAULT_MC_RESOLUTION = 512  # High mesh resolution
    
    @classmethod
    def get_hq_params(cls, data):
        return {
            'num_inference_steps': int(data.get("num_inference_steps", cls.DEFAULT_INFERENCE_STEPS)),
            'guidance_scale': float(data.get("guidance_scale", cls.DEFAULT_GUIDANCE_SCALE)),
            'n_views': int(data.get("n_views", cls.DEFAULT_N_VIEWS)),
            'height': int(data.get("height", cls.DEFAULT_HEIGHT)),
            'width': int(data.get("width", cls.DEFAULT_WIDTH)),
            'mc_resolution': int(data.get("mc_resolution", cls.DEFAULT_MC_RESOLUTION)),
            'output_format': data.get("output_format", "mesh_and_image")  # New option
        }

# [Previous setup code remains the same until the generate endpoint...]

@app.route("/generate", methods=["POST"])
@timing
def generate():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        if not data or "sketch" not in data:
            return jsonify({"error": "Missing sketch in request"}), 400

        # [Previous image decoding and preprocessing remains the same...]

        prompt = data.get("prompt", "a high-quality detailed 3D object")
        params = HighQualityParameters.get_hq_params(data)
        logger.info(f"Using high-quality parameters: {params}")

        # [Previous edge detection and Stable Diffusion steps remain the same...]

        # ENHANCED: TripoSR processing with mesh extraction for quality
        try:
            processed_image = bulletproof_image_preprocessing(concept)
            
            with torch.no_grad():
                logger.info("Processing TripoSR for high-quality mesh generation")
                
                final_array = np.ascontiguousarray(np.array(processed_image))
                processed_image = Image.fromarray(final_array)
                
                # Generate scene codes
                scene_codes = app.triposr(processed_image, device=DEVICE)
                
                # ENHANCED: Extract high-quality mesh
                mesh = app.triposr.extract_mesh(
                    scene_codes, 
                    resolution=params['mc_resolution']  # High resolution mesh
                )[0]
                
                # Apply proper orientation
                from tsr.utils import to_gradio_3d_orientation
                mesh = to_gradio_3d_orientation(mesh)
                
                logger.info(f"✅ Generated high-quality mesh with {len(mesh.vertices)} vertices")
                
                # Generate high-quality renders from mesh
                hq_views = create_high_quality_mesh_render(mesh, (params['height'], params['width']))
                
                if hq_views:
                    views = hq_views
                    logger.info(f"✅ Generated {len(views)} high-quality mesh renders")
                else:
                    # Fallback to standard TripoSR rendering
                    rendered_views = app.triposr.render(
                        scene_codes,
                        n_views=params['n_views'],
                        return_type="pil"
                    )[0]
                    views = rendered_views
                    logger.info("Used fallback TripoSR rendering")
                
                del scene_codes, processed_image
                
            clear_gpu_memory()

            # Select best view
            final_image = sharpest(views)
            del views
            
            if final_image is None:
                return jsonify({"error": "Failed to generate final image"}), 500

            final_array = np.ascontiguousarray(np.array(final_image))
            final_image = Image.fromarray(final_array)

            logger.info("✅ Selected highest quality mesh-rendered image")

            # ENHANCED: Return mesh file alongside high-quality image
            output_format = params.get('output_format', 'mesh_and_image')
            
            if output_format == 'image_only':
                # Return only the high-quality image
                buf = io.BytesIO()
                final_image.save(buf, "PNG", optimize=False, compress_level=1)
                buf.seek(0)
                
                return send_file(
                    buf, 
                    mimetype="image/png", 
                    download_name="3d_render_hq.png", 
                    as_attachment=True
                )
            
            elif output_format == 'mesh_only':
                # Return only the mesh file
                mesh_buffer = io.BytesIO()
                mesh.export(mesh_buffer, file_type='obj')
                mesh_buffer.seek(0)
                
                return send_file(
                    mesh_buffer,
                    mimetype="application/octet-stream",
                    download_name="3d_model.obj",
                    as_attachment=True
                )
            
            else:  # mesh_and_image (default)
                # Create ZIP with both high-quality image and mesh
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add high-quality image
                    img_buffer = io.BytesIO()
                    final_image.save(img_buffer, "PNG", optimize=False, compress_level=1)
                    zip_file.writestr("3d_render_hq.png", img_buffer.getvalue())
                    
                    # Add mesh file
                    mesh_buffer = io.BytesIO()
                    mesh.export(mesh_buffer, file_type='obj')
                    zip_file.writestr("3d_model.obj", mesh_buffer.getvalue())
                    
                    # Add MTL file for materials
                    mtl_content = """# Material file for 3D model
newmtl default_material
Ka 0.2 0.2 0.2
Kd 0.8 0.8 0.8
Ks 0.5 0.5 0.5
Ns 32.0
"""
                    zip_file.writestr("3d_model.mtl", mtl_content)
                
                zip_buffer.seek(0)
                
                return send_file(
                    zip_buffer,
                    mimetype="application/zip",
                    download_name="3d_model_hq.zip",
                    as_attachment=True
                )
                
        except Exception as e:
            logger.error(f"High-quality mesh processing failed: {e}")
            return jsonify({"error": f"TripoSR processing failed: {str(e)}"}), 500

    except Exception as e:
        logger.error("Unexpected error in /generate", exc_info=True)
        clear_gpu_memory()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    logger.info("🚀 Starting HIGH-QUALITY TripoSR service with mesh support")
    app.run(host="0.0.0.0", port=5000, debug=False)
