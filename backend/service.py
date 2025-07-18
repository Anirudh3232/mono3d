from typing import Tuple, Optional, Callable
import os, sys, io, time, gc, base64, logging, atexit, importlib
import psutil, torch, numpy as np
from PIL import Image, ImageFilter, ImageStat
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from functools import wraps
from contextlib import nullcontext

# ───────────── PATHS ─────────────
TRIPOSR_PATH = os.path.join(os.path.dirname(__file__), "TripoSR-main")
sys.path.insert(0, TRIPOSR_PATH)

# ───────────── MOCK CACHE ────────
class MockCache:
    def __init__(self, *_, **__):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __call__(self, *_, **__):           return None
    def dim(self):                          return 0
    def __getattr__(self, _):               return lambda *a, **k: None
    def size(self, dim=None):               return (0,) if dim is None else 0
    def to(self, d): self.device = d; return self
    def update(self,*_,**__):               ...
    def get_decoder_cache(self,*_,**__):    return self
    def get_encoder_cache(self,*_,**__):    return self

class MockEncoderDecoderCache(MockCache):
    @property
    def encoder(self): return self
    @property
    def decoder(self): return self

import transformers as _tf
for _n in ("Cache","DynamicCache","EncoderDecoderCache"):
    if not hasattr(_tf, _n):
        setattr(_tf, _n, MockEncoderDecoderCache)
try:
    import transformers.cache_utils as _tcu
    for _n in ("Cache","DynamicCache","EncoderDecoderCache"):
        if not hasattr(_tcu, _n):
            setattr(_tcu, _n, MockEncoderDecoderCache)
except ImportError: pass
try:
    import transformers.models.encoder_decoder as _ted
    if not hasattr(_ted,"EncoderDecoderCache"):
        setattr(_ted,"EncoderDecoderCache",MockEncoderDecoderCache)
except ImportError: pass
import huggingface_hub as _hf
if not hasattr(_hf,"cached_download"):
    _hf.cached_download=_hf.hf_hub_download
_acc_mem=importlib.import_module("accelerate.utils.memory")
if not hasattr(_acc_mem,"clear_device_cache"):
    _acc_mem.clear_device_cache=lambda *a,**k:None

# ───────────── LOGGING ───────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(),logging.FileHandler("service.log")])
logger=logging.getLogger(__name__)
def _flush(): sys.stdout.flush(); sys.stderr.flush()
atexit.register(_flush)

# ───────────── HELPERS ───────────
def timing(f):
    @wraps(f)
    def w(*a,**k):
        s,cpu_s=time.time(),psutil.cpu_percent(interval=None)
        r=f(*a,**k)
        e,cpu_e=time.time(),psutil.cpu_percent(interval=None)
        logger.info(f"{f.__name__} {e-s:.2f}s CPU {cpu_s:.1f}%→{cpu_e:.1f}%")
        return r
    return w
def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        clear_gpu.c=(getattr(clear_gpu,"c",0)+1)%3
        if clear_gpu.c==0: gc.collect()
def gpu_mb(): return torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0

def _best_view(imgs):
    """pick the sharpest image using variance-of-Laplacian"""
    def sharp(pil):
        lap = pil.convert("L").filter(ImageFilter.FIND_EDGES)
        return ImageStat.Stat(lap).var[0]
    scores=[sharp(i) for i in imgs]
    return imgs[int(np.argmax(scores))]

class OptimizedParameters:
    DEFAULTS=dict(num_inference_steps=63,   # ← high-quality floor
                  guidance_scale=9.96,
                  render_resolution=512)
    @classmethod
    def parse(cls,data):
        p=dict(cls.DEFAULTS)
        # allow override but never go below defaults
        for k,v in cls.DEFAULTS.items():
            if k in data:
                p[k]=max(type(v)(data[k]),v)
        return p

class LRU:
    def __init__(self,n=10): self.m,self.o,self.n={},[],n
    def get(self,k):
        if k in self.m:
            self.o.remove(k); self.o.append(k); return self.m[k]
    def put(self,k,v):
        if len(self.m)>=self.n: del self.m[self.o.pop(0)]
        self.m[k]=v; self.o.append(k)
cache=LRU()

# ───────────── STARTUP ───────────
print("Initializing service …")
from diffusers import StableDiffusionControlNetPipeline,ControlNetModel,EulerAncestralDiscreteScheduler
from controlnet_aux import CannyDetector
import rembg
try:
    from optimization_config import get_profile_parameters
    OPT=True
except ImportError:
    OPT=False
from tsr.system import TSR
from tsr.utils  import remove_background,resize_foreground

rembg_sess=rembg.new_session()
app=Flask(__name__); CORS(app)

# ── ROUTES ──
@app.get("/health")
def health():
    return jsonify(dict(status="ok",gpu_mb=gpu_mb(),
        cpu=psutil.cpu_percent(interval=None),
        mem=psutil.virtual_memory().percent,
        models=all(hasattr(app,x) for x in("edge","cnet","sd","tsr")),
        optimization=OPT))

# ── MODELS ──
device="cuda" if torch.cuda.is_available() else "cpu"
if device=="cuda":
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.cuda.set_per_process_memory_fraction(0.9)
logger.info("Loading models …")
app.edge=CannyDetector()
app.cnet=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16).to(device)
app.sd=StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",controlnet=app.cnet,
            torch_dtype=torch.float16).to(device)
app.sd.scheduler=EulerAncestralDiscreteScheduler.from_config(app.sd.scheduler.config)
try: app.sd.enable_xformers_memory_efficient_attention()
except Exception: logger.warning("xformers unavailable")
app.sd.enable_model_cpu_offload(); app.sd.enable_attention_slicing(); app.sd.enable_vae_slicing()
app.tsr=TSR.from_pretrained("stabilityai/TripoSR",
            config_name="config.yaml",weight_name="model.ckpt").to(device).eval()
if hasattr(app.tsr,"renderer"): app.tsr.renderer.set_chunk_size(8192)
logger.info("Models ready"); _flush()

# ── GENERATE ──
@app.post("/generate")
@timing
def generate():
    try:
        if not request.is_json: return jsonify(error="json required"),400
        d=request.json
        if "sketch" not in d:  return jsonify(error="missing sketch"),400
        key=f"{d['sketch'][:100]}_{d.get('prompt','')}"
        c=cache.get(key)
        if c: return send_file(c,as_attachment=True,download_name="3d.png",mimetype="image/png")

        png=base64.b64decode(d["sketch"].split(",",1)[1])
        pil=Image.open(io.BytesIO(png)).convert("RGBA").resize((768,768),Image.LANCZOS)
        prompt=d.get("prompt","a clean 3-D asset")
        params=(get_profile_parameters(d.get("profile","standard"),d.get("custom_params",{}))
                if OPT else OptimizedParameters.parse(d))

        edge=app.edge(pil); del pil
        with torch.no_grad():
            concept=app.sd(prompt,image=edge,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale']).images[0]
        clear_gpu()

        try:
            proc=remove_background(concept,rembg_sess)
            proc=resize_foreground(proc,1.0)  # keep full silhouette
            arr=np.asarray(proc).astype(np.float32)/255
            arr=arr[:,:,:3]*arr[:,:,3:4]+(1-arr[:,:,3:4])*0.5
            proc=Image.fromarray((arr*255).astype(np.uint8))
        except Exception: proc=concept

        with torch.no_grad(), (torch.cuda.amp.autocast() if device=="cuda" else nullcontext()):
            scene=app.tsr([proc],device=device)
        clear_gpu()

        h=params['render_resolution']
        with torch.no_grad(), (torch.cuda.amp.autocast() if device=="cuda" else nullcontext()):
            views=app.tsr.render(scene,n_views=4,height=h,width=h,return_type="pil")[0]
        img=_best_view(views)

        buf=io.BytesIO(); img.save(buf,format="PNG"); buf.seek(0)
        cache.put(key,buf); clear_gpu(); _flush()
        return send_file(buf,as_attachment=True,download_name="3d_image.png",mimetype="image/png")
    except Exception as e:
        logger.error("generate failed",exc_info=True); _flush(); clear_gpu()
        return jsonify(error=str(e)),500

# ── MAIN ──
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)
