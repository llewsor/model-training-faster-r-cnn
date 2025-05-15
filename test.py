import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

if torch.backends.mps.is_available():
  print("✅ Using MPS (Apple Silicon)")
elif torch.cuda.is_available():
  print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
  print("⚠ Using CPU (Training will be slow!)")