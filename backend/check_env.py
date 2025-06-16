
import torch, torchvision, xformers

print(f"Torch {torch.__version__}   CUDA? {torch.cuda.is_available()}")
print("Torchvision", torchvision.__version__)
print("Xformers", xformers.__version__)

