import io, os, sys, time, base64, gc, logging, atexit
from functools import wraps
from contextlib import nullcontext

# ── third-party
import numpy as np
import psutil
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── paths ────────────────────────────────────────────────────────────────────
TRIPOSR_PATH = os.path.join(os.path.dirname(__file__), "TripoSR-main")
sys.path.insert(0, TRIPOSR_PATH)

# ── *Disable* CUDA-only torchmcubes (TripoSR will fall back to PyMCubes) ─────
os.environ["TSR_DISABLE_TORCHMCUBES"] = "1"

###############################################################################
### BEGIN ROBUST STUB – Multiple fallbacks for marching cubes ###############
###############################################################################
try:
    import torchmcubes
    print("✓ Using torchmcubes")
except ModuleNotFoundError:
    try:
        # First try PyMCubes
        import PyMCubes, numpy as _np, torch as _torch
        
        def _marching_cubes(vol: _torch.Tensor, thresh: float = 0.0):
            """PyMCubes fallback for marching-cubes."""
            v, f = PyMCubes.marching_cubes(vol.detach().cpu().numpy(), thresh)
            return (_torch.from_numpy(v).to(vol.device, dtype=vol.dtype),
                    _torch.from_numpy(f.astype(_np.int64)).to(vol.device))
        
        print("✓ Using PyMCubes fallback")
        
    except ImportError:
        try:
            # Second try mcubes
            import mcubes, numpy as _np, torch as _torch
            
            def _marching_cubes(vol: _torch.Tensor, thresh: float = 0.0):
                """mcubes fallback for marching-cubes."""
                v, f = mcubes.marching_cubes(vol.detach().cpu().numpy(), thresh)
                return (_torch.from_numpy(v).to(vol.device, dtype=vol.dtype),
                        _torch.from_numpy(f.astype(_np.int64)).to(vol.device))
            
            print("✓ Using mcubes fallback")
            
        except (ImportError, ValueError):
            # Final fallback to scikit-image (most reliable)
            from skimage import measure
            import numpy as _np, torch as _torch
            
            def _marching_cubes(vol: _torch.Tensor, thresh: float = 0.0):
                """Scikit-image fallback for marching-cubes."""
                verts, faces, _, _ = measure.marching_cubes(
                    vol.detach().cpu().numpy(), 
                    level=thresh
                )
                return (_torch.from_numpy(verts).to(vol.device, dtype=vol.dtype),
                        _torch.from_numpy(faces.astype(_np.int64)).to(vol.device))
            
            print("✓ Using scikit-image marching cubes fallback")
    
    # Create the stub module
    import types
    stub = types.ModuleType("torchmcubes")
    stub.marching_cubes = _marching_cubes
    sys.modules["torchmcubes"] = stub
###############################################################################
### END ROBUST STUB ###########################################################
###############################################################################

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("service.log")],
)
log = logging.getLogger(__name__)
atexit.register(lambda: (sys.stdout.flush(), sys.stderr.flush()))

# ── small timing decorator ───────────────────────────────────────────────────
def timing(fn):
    @wraps(fn)
    def wrap(*a, **k):
        t0, cpu0 = time.time(), psutil.cpu_percent(None)
        out = fn(*a, **k)
        t1, cpu1 = time.time(), psutil.cpu_percent(None)
        log.info(f"{fn.__name__}: {t1 - t0:.2f}s | CPU {cpu0:.1f}%→{cpu1:.1f}%")
        return out
    return wrap

# ── very small LRU – cache last 10 responses ────────────────────────────────
class LRU:
    def __init__(self, n=10):
        self.n, self.d, self.o = n, {}, []

    def get(self, k):
        if k in self.d:
            self.o.remove(k)
            self.o.append(k)
            return self.d[k]

    def put(self, k, v):
        if len(self.d) >= self.n:
            del self.d[self.o.pop(0)]
        self.d[k] = v
        self.o.append(k)

cache = LRU()

def clear_gpu():
    """Empty CUDA cache after big ops."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# ── load models with better error handling ──────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"▶ loading models on {DEVICE}")

try:
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        EulerAncestralDiscreteScheduler,
    )
    from controlnet_aux import CannyDetector
    from tsr.system import TSR
    from tsr.utils import resize_foreground, remove_background  # noqa: F401

    edge_det = CannyDetector()

    cnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16,
    ).to(DEVICE)

    sd = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=cnet,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    sd.scheduler = EulerAncestralDiscreteScheduler.from_config(sd.scheduler.config)
    
    try:
        sd.enable_xformers_memory_efficient_attention()
        log.info("✓ xformers memory optimization enabled")
    except Exception:
        log.warning("⚠ xformers not available, using default attention")
    
    sd.enable_model_cpu_offload()
    sd.enable_attention_slicing()

    tsr = (
        TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        .to(DEVICE)
        .eval()
    )
    if hasattr(tsr, "renderer"):
        tsr.renderer.set_chunk_size(8192)

    log.info("✔ models ready")

except Exception as e:
    log.error(f"✗ Failed to load models: {e}")
    raise

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.get("/health")
def health():
    """Lightweight health check."""
    return jsonify(
        status="ok",
        gpu_mb=(
            torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        ),
        cpu=psutil.cpu_percent(None),
        mem=psutil.virtual_memory().percent,
        device=DEVICE,
    )

# ── pick sharpest view of 4 renders ──────────────────────────────────────────
def sharpest(img_list):
    """Return PIL image with highest Laplacian variance."""
    import cv2

    scores = [
        cv2.Laplacian(cv2.cvtColor(np.array(i), cv2.COLOR_RGBA2GRAY), cv2.CV_64F).var()
        for i in img_list
    ]
    return img_list[int(np.argmax(scores))]

# ── /generate endpoint with better error handling ───────────────────────────
@app.post("/generate")
@timing
def generate():
    try:
        if not request.is_json:
            return jsonify(error="JSON body required"), 400
        data = request.json
        if "sketch" not in data:
            return jsonify(error="missing 'sketch'"), 400

        key = data["sketch"][:120] + data.get("prompt", "")
        if (buf := cache.get(key)) is not None:
            log.info("✓ Serving cached result")
            buf.seek(0)  # Reset buffer position
            return send_file(
                buf, mimetype="image/png", download_name="3d_image.png", as_attachment=True
            )

        # decode base64 PNG → RGBA PIL
        try:
            png_bytes = base64.b64decode(data["sketch"].split(",", 1)[1])
            sketch = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            log.info(f"✓ Decoded sketch: {sketch.size}")
        except Exception as e:
            log.error(f"✗ Image decode failed: {e}")
            return jsonify(error=f"bad image data: {e}"), 400

        prompt = data.get("prompt", "a clean 3-D asset")
        log.info(f"✓ Processing prompt: {prompt}")

        # 1) Canny edge map
        edge = edge_det(sketch)
        del sketch
        log.info("✓ Generated Canny edge map")

        # 2) Stable Diffusion + ControlNet
        with torch.no_grad():
            concept = sd(
                prompt,
                image=edge,
                num_inference_steps=63,
                guidance_scale=9.96,
            ).images[0]
        clear_gpu()
        log.info("✓ Generated concept image")

        # 3) Resize FG so TripoSR sees full object
        concept = resize_foreground(concept, 1.0)
        log.info("✓ Resized foreground")

        # 4) TripoSR → latent scene codes
        with torch.no_grad(), (
            torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext()
        ):
            codes = tsr([concept], device=DEVICE)
        clear_gpu()
        log.info("✓ Generated scene codes")

        # 5) Render 4 views, pick sharpest
        with torch.no_grad(), (
            torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext()
        ):
            views = tsr.render(codes, n_views=4, height=512, width=512, return_type="pil")[0]
        img = sharpest(views)
        log.info("✓ Rendered and selected sharpest view")

        # pack result
        buf = io.BytesIO()
        img.save(buf, "PNG")
        buf.seek(0)

        cache.put(key, buf)
        clear_gpu()
        
        return send_file(
            buf, mimetype="image/png", download_name="3d_image.png", as_attachment=True
        )

    except Exception as e:
        log.error(f"✗ Generation failed: {e}")
        clear_gpu()  # Clean up GPU memory on error
        return jsonify(error=f"generation failed: {str(e)}"), 500

# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("🚀 Starting TripoSR service")
    app.run(host="0.0.0.0", port=5000, debug=False)
