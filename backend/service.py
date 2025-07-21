import os
import sys
import gc
import io
import time
import base64
import atexit
import logging
import subprocess
import psutil
import numpy as np
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from functools import wraps

# ----- Compatibility patches -----
import transformers

# Mocked classes for compatibility with PEFT and HuggingFace
class MockCache:
    def __init__(self, *a, **k): pass
    def update(self, *a, **k): pass
    def get_decoder_cache(self, *a, **k): return self
    def get_encoder_cache(self, *a, **k): return self
    def __call__(self, *a, **k): return a[0] if a else None
    def forward(self, *a, **k): return a[0] if a else None

class MockAttentionProcessor(MockCache): pass
class MockEncoderDecoderCache(MockCache):
    @property
    def encoder(self): return self
    @property
    def decoder(self): return self
class MockHybridCache(MockCache): pass

for name, cls in [
    ('Cache', MockCache),
    ('DynamicCache', MockCache),
    ('EncoderDecoderCache', MockEncoderDecoderCache),
    ('HybridCache', MockHybridCache)
]:
    if not hasattr(transformers, name):
        setattr(transformers, name, cls)
transformers.__dict__.update({
    'Cache': MockCache, 'DynamicCache': MockCache,
    'EncoderDecoderCache': MockEncoderDecoderCache, 'HybridCache': MockHybridCache
})
try:
    import transformers.cache_utils as _tcu
    for n, c in [("Cache", MockCache), ("DynamicCache", MockCache), ("EncoderDecoderCache", MockEncoderDecoderCache)]:
        if not hasattr(_tcu, n): setattr(_tcu, n, c)
except Exception: pass
try:
    import transformers.models.encoder_decoder as _ted
    if not hasattr(_ted, "EncoderDecoderCache"):
        setattr(_ted, "EncoderDecoderCache", MockEncoderDecoderCache)
except Exception: pass
try:
    import diffusers.models.attention_processor
    diffusers.models.attention_processor.AttnProcessor2_0 = MockAttentionProcessor
except Exception: pass
try:
    import transformers.models.llama.modeling_llama
    transformers.models.llama.modeling_llama.AttnProcessor2_0 = MockAttentionProcessor
except Exception: pass

# ----- Logging -----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('service.log')
    ]
)
logger = logging.getLogger(__name__)

def _flush(): sys.stdout.flush(); sys.stderr.flush()
atexit.register(_flush)

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def gpu_mem_mb():
    return (torch.cuda.memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        cpu0 = psutil.cpu_percent()
        result = f(*args, **kwargs)
        t1 = time.time()
        cpu1 = psutil.cpu_percent()
        logger.info(f"{f.__name__} took {t1-t0:.2f}s [CPU: {cpu0:.1f}→{cpu1:.1f}%]")
        return result
    return wrapper

# ----- Quality/Speed Parameters -----
class OptimizedParameters:
    DEFAULT_INFER_STEPS = 20
    DEFAULT_GUIDANCE = 7.0
    DEFAULT_VIEWS = 2
    DEFAULT_HEIGHT = 256
    DEFAULT_WIDTH = 256
    DEFAULT_MC_RES = 192
    @classmethod
    def get(cls, data):
        return {
            "num_inference_steps": int(data.get("num_inference_steps", cls.DEFAULT_INFER_STEPS)),
            "guidance_scale": float(data.get("guidance_scale", cls.DEFAULT_GUIDANCE)),
            "n_views": int(data.get("n_views", cls.DEFAULT_VIEWS)),
            "height": int(data.get("height", cls.DEFAULT_HEIGHT)),
            "width": int(data.get("width", cls.DEFAULT_WIDTH)),
            "mc_resolution": int(data.get("mc_resolution", cls.DEFAULT_MC_RES)),
        }

# ----- Result Caching -----
class ResultCache:
    def __init__(self, max_size=2):
        self.cache = {}
        self.order = []
        self.max_size = max_size
    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            del self.cache[self.order.pop(0)]
        self.cache[key] = value
        self.order.append(key)
result_cache = ResultCache(max_size=2)

# ----- Sharpest View Selection -----
def sharpest(img_list):
    try:
        import cv2
        scores = [
            cv2.Laplacian(
                cv2.cvtColor(np.array(i), cv2.COLOR_RGB2GRAY), cv2.CV_64F
            ).var() for i in img_list
        ]
        return img_list[int(np.argmax(scores))]
    except Exception:
        return img_list[0] if img_list else None

# ----- Image Preprocessing for TripoSR -----
def preprocess_triposr_image(pil_img):
    """
    Official TripoSR RGBA→RGB background fill:
    https://github.com/VAST-AI-Research/TripoSR/blob/main/demo/ip_adapter_gradio.py
    """
    if pil_img.mode == "RGBA":
        arr = np.array(pil_img).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        pil_img = Image.fromarray((arr * 255.0).astype(np.uint8))
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return pil_img

# ----- Main API -----
print("Starting service and importing models...")
try:
    logger.info("Importing diffusers components...")
    _flush()
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
    from controlnet_aux import CannyDetector

    # TripoSR setup
    triposr_path = os.path.join(os.path.dirname(__file__), "TripoSR-main")
    if not os.path.exists(triposr_path):
        logger.info("Cloning TripoSR from GitHub...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/VAST-AI-Research/TripoSR.git", 
            triposr_path
        ], check=True)
        logger.info("✅ TripoSR cloned successfully")

    if triposr_path not in sys.path:
        sys.path.insert(0, triposr_path)

    from tsr.system import TSR
    from tsr.utils import resize_foreground

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading models on {DEVICE}")

    # TripoSR - full float32 for best quality/stability
    triposr_model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml", 
        weight_name="model.ckpt"
    )
    triposr_model.to(DEVICE)
    triposr_model.eval()
    triposr_model.renderer.set_chunk_size(8192)

    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "tripoSR_precision": "float32",
            "gpu_mb": gpu_mem_mb(),
            "cpu_percent": psutil.cpu_percent(),
        })

    logger.info("Loading edge detector, ControlNet, Stable Diffusion...")
    app.edge_det = CannyDetector()
    app.cnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
    ).to(DEVICE)
    app.sd = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=app.cnet, torch_dtype=torch.float16
    ).to(DEVICE)
    app.sd.scheduler = EulerAncestralDiscreteScheduler.from_config(app.sd.scheduler.config)
    try: app.sd.enable_xformers_memory_efficient_attention()
    except Exception: logger.warning("xformers not available")
    app.sd.enable_model_cpu_offload()
    app.sd.enable_attention_slicing()
    app.sd.enable_vae_slicing()
    clear_gpu_memory()

    def ensure_on_device(module):
        if module is not None:
            try:
                module.to(DEVICE)
            except Exception:
                pass
        return module

    app.triposr = triposr_model
    app.triposr = ensure_on_device(app.triposr)
    clear_gpu_memory()
    _flush()

    @app.route("/generate", methods=["POST"])
    @timing
    def generate():
        try:
            if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
            data = request.json
            if not data or "sketch" not in data:
                return jsonify({"error": "Missing sketch"}), 400

            cache_key = f"{data['sketch'][:50]}_{data.get('prompt','')}"
            cached_result = result_cache.get(cache_key)
            if cached_result:
                cached_result.seek(0)
                return send_file(
                    cached_result,
                    mimetype="image/png",
                    download_name="3d_render.png",
                    as_attachment=True
                )

            sketch_data = data["sketch"]
            try:
                png = base64.b64decode(sketch_data.split(",", 1)[1] if "," in sketch_data else sketch_data)
                pil = Image.open(io.BytesIO(png)).convert("RGBA")
                if pil.size[0] > 512 or pil.size[1] > 512:
                    pil = pil.resize((512, 512), Image.Resampling.LANCZOS)
            except Exception as e:
                logger.error(f"Image decode error: {e}")
                return jsonify({"error": f"Bad image data: {str(e)}"}), 400

            params = OptimizedParameters.get(data)
            logger.info(f"Params: {params}")
            prompt = data.get("prompt", "a clean 3-D asset")
            clear_gpu_memory()

            # Step 1: Edge detection for ControlNet
            try:
                edge = app.edge_det(pil)
                del pil
                clear_gpu_memory()
            except Exception as e:
                logger.error(f"Edge detection failed: {e}")
                return jsonify({"error": f"Edge detection failed: {str(e)}"}), 500

            # Step 2: Stable Diffusion with ControlNet (fast, float16)
            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext():
                        result = app.sd(
                            prompt,
                            image=edge,
                            num_inference_steps=params["num_inference_steps"],
                            guidance_scale=params["guidance_scale"],
                            return_dict=True
                        )
                        concept = result.images[0]
                del edge, result
                clear_gpu_memory()
            except Exception as e:
                logger.error(f"Stable Diffusion failed: {e}")
                return jsonify({"error": f"Stable Diffusion failed: {str(e)}"}), 500

            # Step 3: Preprocess for TripoSR
            try:
                concept = resize_foreground(concept, 1.0)
            except Exception:
                pass  # Fallback: use raw concept image
            concept = preprocess_triposr_image(concept)
            clear_gpu_memory()

            # Step 4: TripoSR forward pass—official, float32 precision
            try:
                with torch.no_grad():
                    scene_codes = app.triposr(concept, device=DEVICE)
                    mesh = app.triposr.extract_mesh(
                        scene_codes, resolution=params["mc_resolution"]
                    )[0]
            except Exception as e:
                logger.error(f"TripoSR processing failed: {e}")
                return jsonify({"error": f"TripoSR processing failed: {str(e)}"}), 500
            clear_gpu_memory()

            # Step 5: Render views for selection (speed: only n_views images)
            try:
                with torch.no_grad():
                    rendered = app.triposr.render(
                        [concept], n_views=params["n_views"],
                        height=params["height"], width=params["width"], return_type="pil"
                    )[0]
            except Exception as e:
                logger.error(f"TripoSR rendering failed: {e}")
                rendered = [concept]
            clear_gpu_memory()

            # Step 6: Select sharpest output and respond
            try:
                final_image = sharpest(rendered)
                buf = io.BytesIO()
                final_image.save(buf, "PNG")
                buf.seek(0)
                result_cache.put(cache_key, io.BytesIO(buf.getvalue()))
                clear_gpu_memory()
                _flush()
                return send_file(
                    buf,
                    mimetype="image/png",
                    download_name="3d_render.png",
                    as_attachment=True
                )
            except Exception as e:
                logger.error(f"Final image saving failed: {e}")
                return jsonify({"error": f"Image saving failed: {str(e)}"}), 500

        except torch.cuda.OutOfMemoryError:
            clear_gpu_memory()
            return jsonify({"error": "GPU OOM. Try smaller image or lower quality settings."}), 500
        except Exception as e:
            logger.error("Unexpected error", exc_info=True)
            clear_gpu_memory()
            return jsonify({"error": f"Server error: {str(e)}"}), 500

    if __name__ == "__main__":
        logger.info("🚀 Starting TripoSR service (optimized for speed & quality, float32)")
        app.run(host="0.0.0.0", port=5000, debug=False)

except Exception as e:
    logger.error("Initialization error", exc_info=True)
    _flush()
    raise
