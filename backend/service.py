from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import sys, os, base64, io, gc, time, types, importlib, logging, atexit
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from functools import wraps
from torch.cuda.amp import autocast
from contextlib import nullcontext
import psutil
import subprocess

# Mock classes for compatibility (unchanged from your code)
class MockCache:
    def __init__(self, *args, **kwargs):
        pass
    def update(self, *args, **kwargs):
        pass
    def get_decoder_cache(self, *args, **kwargs):
        return self
    def get_encoder_cache(self, *args, **kwargs):
        return self

class MockEncoderDecoderCache(MockCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    @property
    def encoder(self):
        return self
    @property
    def decoder(self):
        return self

# Compatibility patches (unchanged)
import transformers
for _n in ("Cache", "DynamicCache", "EncoderDecoderCache"):
    if not hasattr(transformers, _n):
        setattr(transformers, _n, MockEncoderDecoderCache)

try:
    import transformers.cache_utils as _tcu
    for _n in ("Cache", "DynamicCache", "EncoderDecoderCache"):
        if not hasattr(_tcu, _n):
            setattr(_tcu, _n, MockEncoderDecoderCache)
except ImportError:
    pass

try:
    import transformers.models.encoder_decoder as _ted
    if not hasattr(_ted, "EncoderDecoderCache"):
        setattr(_ted, "EncoderDecoderCache", MockEncoderDecoderCache)
except ImportError:
    pass

import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "cached_download"):
    _hf_hub.cached_download = _hf_hub.hf_hub_download

_acc_mem = importlib.import_module("accelerate.utils.memory")
if not hasattr(_acc_mem, "clear_device_cache"):
    _acc_mem.clear_device_cache = lambda *a, **k: None

import diffusers.models.attention_processor
diffusers.models.attention_processor.AttnProcessor2_0 = MockCache

import transformers.models.llama.modeling_llama
transformers.models.llama.modeling_llama.AttnProcessor2_0 = MockCache

# Logging setup (unchanged)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('service.log')
    ]
)
logger = logging.getLogger(__name__)

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        cpu_start = psutil.cpu_percent(interval=None)
        result = f(*args, **kwargs)
        end = time.time()
        cpu_end = psutil.cpu_percent(interval=None)
        logger.info(f"{f.__name__} took {end-start:.2f}s, CPU: {cpu_start:.1f}% -> {cpu_end:.1f}%")
        return result
    return wrapper

def _flush():
    """Force flush stdout/stderr to ensure logs are visible"""
    sys.stdout.flush()
    sys.stderr.flush()

atexit.register(_flush)

def clear_gpu_memory():
    """Optimized GPU memory clearing with reduced CPU overhead"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(clear_gpu_memory, '_gc_counter'):
            clear_gpu_memory._gc_counter += 1
        else:
            clear_gpu_memory._gc_counter = 0
        
        if clear_gpu_memory._gc_counter % 3 == 0:
            gc.collect()

def gpu_mem_mb():
    return (torch.cuda.memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0

class OptimizedParameters:
    """Optimized default parameters to reduce CPU usage"""
    
    DEFAULT_INFERENCE_STEPS = 30
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT
