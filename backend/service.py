from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import sys, os, base64, io, gc, time, logging, atexit
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from functools import wraps
from torch.cuda.amp import autocast
from contextlib import nullcontext
import psutil
import subprocess

# Minimal logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        logger.info(f"{f.__name__} took {end-start:.2f}s")
        return result
    return wrapper

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def gpu_mem_mb():
    return (torch.cuda.memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0

def ensure_tensor_from_output(data):
    if hasattr(data, 'images'):
        return data.images[0]
    elif isinstance(data, tuple):
        return data[0]
    return data

def safe_array_conversion(img):
    """Safe numpy array conversion without stride issues"""
    if isinstance(img, Image.Image):
        arr = np.array(img)
        if arr.strides and any(s < 0 for s in arr.strides):
            arr = np.ascontiguousarray(arr)
        return Image.fromarray(arr)
    return img

def simple_preprocessing(input_image):
    """Simple, reliable image preprocessing"""
    try:
        # Convert to RGB
        if input_image.mode != 'RGB':
            if input_image.mode == 'RGBA':
                # Simple alpha blend
                white_bg = Image.new('RGB', input_image.size, (255, 255, 255))
                white_bg.paste(input_image, mask=input_image.split()[-1])
                input_image = white_bg
            else:
                input_image = input_image.convert('RGB')
        
        # Ensure clean array
        return safe_array_conversion(input_image)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        # Ultimate fallback
        return input_image.convert('RGB')

class OptimizedParameters:
    DEFAULT_INFERENCE_STEPS = 25
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_N_VIEWS = 4
    DEFAULT_HEIGHT = 512
    DEFAULT_WIDTH = 512
    
    @classmethod
    def get_params(cls, data):
        return {
            'num_inference_steps': int(data.get("num_inference_steps", cls.DEFAULT_INFERENCE_STEPS)),
            'guidance_scale': float(data.get("guidance_scale", cls.DEFAULT_GUIDANCE_SCALE)),
            'n_views': int(data.get("n_views", cls.DEFAULT_N_VIEWS)),
            'height': int(data.get("height", cls.DEFAULT_HEIGHT)),
            'width': int(data.get("width", cls.DEFAULT_WIDTH))
        }

def sharpest(img_list):
    """Select sharpest image using simple variance method"""
    try:
        import cv2
        scores = []
        for img in img_list:
            arr = np.ascontiguousarray(np.array(img))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(score)
        return img_list[int(np.argmax(scores))]
    except ImportError:
        return img_list[0] if img_list else None

print("Starting clean TripoSR service...")
try:
    # Import dependencies cleanly
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
    from controlnet_aux import CannyDetector
    
    # TripoSR setup
    triposr_path = os.path.join(os.path.dirname(__file__), "TripoSR-main")
    if not os.path.exists(triposr_path):
        subprocess.run([
            "git", "clone", 
            "https://github.com/VAST-AI-Research/TripoSR.git", 
            triposr_path
        ], check=True)
    
    if triposr_path not in sys.path:
        sys.path.insert(0, triposr_path)
    
    # Import TripoSR without conflicts
    from tsr.system import TSR
    from tsr.utils import resize_foreground, remove_background
    
    # Load models
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading models on {DEVICE}...")
    
    if DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Load TripoSR
    triposr_model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml", 
        weight_name="model.ckpt"
    )
    triposr_model.to(DEVICE)
    triposr_model.eval()
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok", 
            "gpu_mb": gpu_mem_mb(),
            "triposr_available": True
        })
    
    # Load other models
    edge_detector = CannyDetector()
    
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=torch.float16
    ).to(DEVICE)
    
    sd_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to(DEVICE)
    
    sd_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        sd_pipeline.scheduler.config
    )
    
    # Enable optimizations
    try:
        sd_pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass
    
    sd_pipeline.enable_model_cpu_offload()
    sd_pipeline.enable_attention_slicing()
    
    logger.info("✅ All models loaded successfully")
    
    @app.route("/generate", methods=["POST"])
    @timing
    def generate():
        try:
            data = request.json
            if not data or "sketch" not in data:
                return jsonify({"error": "Missing sketch"}), 400
            
            # Decode image
            sketch_data = data["sketch"]
            if "," in sketch_data:
                png = base64.b64decode(sketch_data.split(",", 1)[1])
            else:
                png = base64.b64decode(sketch_data)
            
            pil_image = Image.open(io.BytesIO(png))
            pil_image = safe_array_conversion(pil_image)
            
            if pil_image.size[0] > 512 or pil_image.size[1] > 512:
                pil_image = pil_image.resize((512, 512), Image.Resampling.LANCZOS)
                pil_image = safe_array_conversion(pil_image)
            
            prompt = data.get("prompt", "a detailed 3D object")
            params = OptimizedParameters.get_params(data)
            
            clear_gpu_memory()
            
            # Edge detection
            edge = edge_detector(pil_image)
            edge = safe_array_conversion(edge)
            
            # Stable Diffusion
            with torch.no_grad():
                result = sd_pipeline(
                    prompt,
                    image=edge,
                    num_inference_steps=params['num_inference_steps'],
                    guidance_scale=params['guidance_scale'],
                    return_dict=True
                )
                
                concept = ensure_tensor_from_output(result)
                concept = safe_array_conversion(concept)
            
            clear_gpu_memory()
            
            # TripoSR processing
            processed_image = simple_preprocessing(concept)
            
            with torch.no_grad():
                scene_codes = triposr_model(processed_image, device=DEVICE)
                
                # Render views
                rendered_views = triposr_model.render(
                    scene_codes,
                    n_views=params['n_views'],
                    return_type="pil"
                )[0]
            
            clear_gpu_memory()
            
            # Select best view
            final_image = sharpest(rendered_views)
            final_image = safe_array_conversion(final_image)
            
            # Return result
            buf = io.BytesIO()
            final_image.save(buf, "PNG")
            buf.seek(0)
            
            return send_file(
                buf,
                mimetype="image/png",
                download_name="3d_render.png",
                as_attachment=True
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            clear_gpu_memory()
            return jsonify({"error": str(e)}), 500
    
    if __name__ == "__main__":
        logger.info("🚀 Starting clean TripoSR service")
        app.run(host="0.0.0.0", port=5000, debug=False)

except Exception as e:
    logger.error(f"Initialization failed: {e}")
    raise
