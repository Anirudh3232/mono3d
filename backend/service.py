"""
Memory-optimised TripoSR + Stable-Diffusion server
--------------------------------------------------
• Handles RGBA→RGB conversion.
• Converts PIL images to tensors before feeding TripoSR.
• Avoids channel-mismatch & dtype errors.
• Includes aggressive GPU-memory management and result caching.
"""

from __future__ import annotations
import os, sys, io, gc, time, base64, logging, atexit, subprocess
from functools import wraps
from typing import Dict

import numpy as np
import psutil
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import torch
from torch.cuda.amp import autocast
from contextlib import nullcontext

# ------------------------------------------------------------------
# 1.  Compatibility shims for HuggingFace / diffusers (MockCache)
# ------------------------------------------------------------------#
import transformers
class _MockCache:
    def __call__(self, *a, **k): return a[0] if a else None
    def update(self, *a, **k):   pass
    def forward(self, *a, **k):  return a[0] if a else None
for name in ("Cache", "DynamicCache", "EncoderDecoderCache", "HybridCache"):
    setattr(transformers, name, _MockCache)

try:
    import diffusers.models.attention_processor as _dap
    _dap.AttnProcessor2_0 = _MockCache
except ImportError:
    pass
# ------------------------------------------------------------------#
# 2.  Logging helpers
# ------------------------------------------------------------------#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s │ %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("service.log")]
)
log = logging.getLogger(__name__)

def _flush(): sys.stdout.flush(), sys.stderr.flush()
atexit.register(_flush)

def timed(fn):
    @wraps(fn)
    def _wrap(*a, **k):
        t0, cpu0 = time.time(), psutil.cpu_percent(None)
        out      = fn(*a, **k)
        log.info(f"{fn.__name__} › {time.time()-t0:.2f}s  CPU {cpu0:.1f}→{psutil.cpu_percent(None):.1f}%")
        return out
    return _wrap


# 3.  GPU helpers

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def gpu_mem_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

# ------------------------------------------------------------------#
# 4.  Parameter defaults
# ------------------------------------------------------------------#
class Opt:
    STEPS      = 20
    GUIDANCE   = 7.0
    N_VIEWS    = 2
    HEIGHT     = 256
    WIDTH      = 256

# ------------------------------------------------------------------#
# 5.  Tiny in-RAM result cache
# ------------------------------------------------------------------#
class _Cache:
    def __init__(self, max_items=5):
        self._data: Dict[str, bytes] = {}
        self._order: list[str] = []
        self.max = max_items

    def get(self, k): return self._data.get(k)
    def add(self, k, v: bytes):
        if k in self._data: self._order.remove(k)
        elif len(self._order) >= self.max:
            oldest = self._order.pop(0); self._data.pop(oldest, None)
        self._data[k] = v; self._order.append(k)

cache = _Cache()

# ------------------------------------------------------------------#
# 6.  Image utilities
# ------------------------------------------------------------------#
def rgba_to_rgb(im: Image.Image) -> Image.Image:
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB") if im.mode != "RGB" else im

def pil_to_tensor(im: Image.Image, device: str) -> torch.Tensor:
    im = rgba_to_rgb(im).resize((512, 512), Image.LANCZOS)
    arr = torch.from_numpy(np.array(im)).float() / 255.0       # HWC
    tensor = arr.permute(2, 0, 1).unsqueeze(0).to(device)      # BCHW
    return tensor

# ------------------------------------------------------------------#
# 7.  Model loading
# ------------------------------------------------------------------#
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
from controlnet_aux import CannyDetector

# --- clone / import TripoSR ---------------------------------------------------
ROOT = os.path.dirname(__file__)
TRIPOSR_DIR = os.path.join(ROOT, "TripoSR-main")
if not os.path.exists(TRIPOSR_DIR):
    log.info("Cloning TripoSR...")
    subprocess.run(["git", "clone", "https://github.com/VAST-AI-Research/TripoSR.git", TRIPOSR_DIR], check=True)
sys.path.insert(0, TRIPOSR_DIR)
from tsr.system import TSR

log.info("Loading models …")
canny           = CannyDetector()
controlnet      = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(DEVICE)
sd              = StableDiffusionControlNetPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    controlnet=controlnet,
                    torch_dtype=torch.float16).to(DEVICE)
sd.scheduler    = EulerAncestralDiscreteScheduler.from_config(sd.scheduler.config)
sd.enable_attention_slicing(); sd.enable_vae_slicing(); sd.enable_model_cpu_offload()

triposr         = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt").to(DEVICE)
if DEVICE == "cuda": triposr = triposr.half()
triposr.eval()
clear_gpu()
log.info("✅ All models ready")

# ------------------------------------------------------------------#
# 8.  Flask app
# ------------------------------------------------------------------#
app = Flask(__name__)
CORS(app)

@app.route("/health")
def _health():
    return jsonify(status="ok", gpu_mb=gpu_mem_mb(), cpu_pct=psutil.cpu_percent(None))

# ------------------------------------------------------------------#
# 9.  /generate endpoint
# ------------------------------------------------------------------#
@app.route("/generate", methods=["POST"])
@timed
def generate():
    try:
        if not request.is_json:         return jsonify(error="JSON body required"), 400
        data = request.get_json()
        if "sketch" not in data:        return jsonify(error="missing 'sketch'"), 400

        # ---- quick cache ----------------------------------------------------
        key = (data["sketch"][:80] + data.get("prompt", ""))[:256]
        if (hit := cache.get(key)):
            return send_file(io.BytesIO(hit), mimetype="image/png")

        # ---- decode input ---------------------------------------------------
        b64 = data["sketch"].split(",", 1)[1] if "," in data["sketch"] else data["sketch"]
        sketch = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
        prompt = data.get("prompt", "a clean 3-D asset")

        # ---- edge map -------------------------------------------------------
        edge = canny(sketch.resize((512, 512), Image.LANCZOS))

        # ---- stable-diffusion ----------------------------------------------
        with torch.no_grad(), (autocast() if DEVICE == "cuda" else nullcontext()):
            out   = sd(prompt, image=edge,
                       num_inference_steps=int(data.get("num_inference_steps", Opt.STEPS)),
                       guidance_scale=float(data.get("guidance_scale", Opt.GUIDANCE)),
                       return_dict=True)
            concept_img = out.images[0]

        # ---- TripoSR: PIL → tensor  ----------------------------------------
        tensor_input = pil_to_tensor(concept_img, DEVICE)
        with torch.no_grad(), (autocast() if DEVICE == "cuda" else nullcontext()):
            codes = triposr(tensor_input)
            views = triposr.render(
                        codes,
                        n_views=int(data.get("n_views", Opt.N_VIEWS)),
                        height=int(data.get("height", Opt.HEIGHT)),
                        width=int(data.get("width", Opt.WIDTH)),
                        return_type="pil")[0]

        # ---- pick sharpest --------------------------------------------------
        final = max(views, key=lambda im: np.var(cv2.Laplacian(np.array(im.convert("RGB")), cv2.CV_64F)))

        # ---- respond --------------------------------------------------------
        buf = io.BytesIO()
        final.save(buf, "PNG"); buf.seek(0)
        cache.add(key, buf.getvalue())
        clear_gpu()
        return send_file(buf, mimetype="image/png")

    except torch.cuda.OutOfMemoryError:
        clear_gpu()
        return jsonify(error="GPU-OOM: reduce parameters"), 500
    except Exception as e:
        clear_gpu()
        log.exception("generate failed")
        return jsonify(error=f"TripoSR processing failed: {e}"), 500

# ------------------------------------------------------------------#
if __name__ == "__main__":
    log.info("🚀  Server running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
