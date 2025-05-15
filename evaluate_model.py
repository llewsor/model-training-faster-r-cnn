import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.ops import box_iou
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import uuid
from tqdm import tqdm

# === Configuration ===
VAL_IMG_DIR = "C:/Projects/dataset/val/images"
VAL_ANN_FILE = "C:/Projects/dataset/val/annotations.json"
CHECKPOINT_PATH = "checkpoints/best.pth"
NUM_CLASSES = 3  # 0=Background, 1=Whole, 2=Broken
LABELS = [1, 2, 0]
LABEL_NAMES = ["Whole", "Broken", "Background"]
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transformation ===
class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

# === Dataset & Loader ===
val_dataset = CocoDetection(
    root=VAL_IMG_DIR,
    annFile=VAL_ANN_FILE,
    transforms=CocoTransform()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x)),
    pin_memory=True
)

# === Model Loader ===
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn()
    model.transform.min_size = (640,)
    model.transform.max_size = 640
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Evaluation Setup ===
y_true, y_pred, iou_list = [], [], []
coco_results, image_ids = [], []

# === Evaluation Loop ===
with torch.no_grad():
    for images, targets in tqdm(val_loader, desc="🔍 Running Confusion Matrix Eval"):
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            # Confidence filter
            keep = output["scores"] > CONF_THRESH
            output = {
                "boxes": output["boxes"][keep].cpu(),
                "labels": output["labels"][keep].cpu(),
                "scores": output["scores"][keep].cpu()
            }

            pred_boxes = output["boxes"]
            pred_labels = output["labels"]
            pred_scores = output["scores"]

            gt_boxes = torch.tensor([obj["bbox"] for obj in target], dtype=torch.float32)
            gt_boxes[:, 2] += gt_boxes[:, 0]  # x2 = x + w
            gt_boxes[:, 3] += gt_boxes[:, 1]  # y2 = y + h
            gt_labels = [obj["category_id"] for obj in target]
            img_id = target[0]["image_id"]
            image_ids.append(img_id)

            # Prepare for COCOeval
            for box, score, label in zip(pred_boxes.tolist(), pred_scores.tolist(), pred_labels.tolist()):
                x1, y1, x2, y2 = box
                coco_results.append({
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                    "id": uuid.uuid4().int >> 64
                })

            matched_gt, matched_pred = set(), set()

            if len(gt_boxes) and len(pred_boxes):
                ious = box_iou(gt_boxes, pred_boxes)
                gt_idx, pred_idx = torch.where(ious >= IOU_THRESHOLD)

                for g, p in zip(gt_idx.tolist(), pred_idx.tolist()):
                    if g not in matched_gt and p not in matched_pred:
                        y_true.append(gt_labels[g])
                        y_pred.append(pred_labels[p].item())
                        iou_list.append(ious[g, p].item())
                        matched_gt.add(g)
                        matched_pred.add(p)

            for g in range(len(gt_labels)):
                if g not in matched_gt:
                    y_true.append(gt_labels[g])
                    y_pred.append(0)  # background

            for p in range(len(pred_labels)):
                if p not in matched_pred:
                    y_true.append(0)  # background
                    y_pred.append(pred_labels[p].item())

# === COCO mAP Evaluation ===
coco_gt = COCO(VAL_ANN_FILE)
with open("logs/coco_results.json", "w") as f:
    json.dump(coco_results, f)

coco_dt = coco_gt.loadRes("logs/coco_results.json")
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

map_50 = coco_eval.stats[1]      # mAP@0.5
map_5095 = coco_eval.stats[0]    # mAP@0.5:0.95
avg_iou = np.mean(iou_list) if iou_list else 0.0

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)

plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Confusion Matrix Normalized")
plt.grid(False)
plt.tight_layout()
plt.savefig("logs/confusion_matrix_normalized.png")
plt.show()

# === Classification Report ===
print("\n📊 Classification Report:")
report = classification_report(y_true, y_pred, labels=LABELS, target_names=LABEL_NAMES, digits=3)
print(report)

print(f"\nmAP@0.5:       {map_50:.3f}")
print(f"mAP@0.5:0.95:   {map_5095:.3f}")
print(f"Average IoU:   {avg_iou:.3f}")

# === Save to file ===
with open("logs/classification_report.txt", "w") as f:
    f.write(report)
    f.write(f"\nmAP@0.5:       {map_50:.3f}")
    f.write(f"\nmAP@0.5:0.95:   {map_5095:.3f}")
    f.write(f"\nAverage IoU:   {avg_iou:.3f}")
