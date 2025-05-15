import torch
import time
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision.transforms import functional as F

num_classes = 3

# Load model
model = fasterrcnn_resnet50_fpn()
model.transform.min_size = (640,)
model.transform.max_size = 640
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("checkpoints/best.pth"))
model.eval().to("cuda")

# Dummy input (same size as your validation images, e.g., 1280x720)
img = Image.open("clay_whole.jpg").convert("RGB")
img_tensor = F.to_tensor(img).unsqueeze(0).to("cuda")

# Warm-up
for _ in range(5):
    _ = model(img_tensor)

# Measure inference time
n_runs = 100
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(n_runs):
        _ = model(img_tensor)
torch.cuda.synchronize()
end = time.time()

avg_time = (end - start) / n_runs * 1000  # ms
fps = 1000 / avg_time

print(f"Inference Time: {avg_time:.2f} ms/frame")
print(f"Throughput: {fps:.2f} FPS")
