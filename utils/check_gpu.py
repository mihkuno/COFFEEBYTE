import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    print("PyTorch is using the GPU!")
    print(f"GPU Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("PyTorch is using the CPU.")
