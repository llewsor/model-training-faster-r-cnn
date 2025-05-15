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

# === Constants ===
NUM_CLASSES = 3  # background + whole + broken
LABELS = [1, 2, 0]
LABEL_NAMES = ["Whole", "Broken", "Background"]
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best.pth"

# === Image transform ===
class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

# === Model definition ===
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn()
    model.transform.min_size = (640,)
    model.transform.max_size = 640
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# === Evaluation function ===
def evaluate_dataset(img_dir, ann_file, output_prefix):
    print(f"\n🔍 Evaluating: {output_prefix}")

    # Log directory
    log_dir = os.path.join("logs", output_prefix)
    os.makedirs(log_dir, exist_ok=True)
    results_file = os.path.join(log_dir, "coco_results.json")

    # Dataset and model setup
    dataset = CocoDetection(root=img_dir, annFile=ann_file, transforms=CocoTransform())
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), pin_memory=True)

    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    y_true, y_pred, iou_list = [], [], []
    coco_results, image_ids = [], []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f"Running Eval ({output_prefix})"):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
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
                gt_boxes[:, 2] += gt_boxes[:, 0]
                gt_boxes[:, 3] += gt_boxes[:, 1]
                gt_labels = [obj["category_id"] for obj in target]
                img_id = target[0]["image_id"]
                image_ids.append(img_id)

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
                        y_pred.append(0)
                for p in range(len(pred_labels)):
                    if p not in matched_pred:
                        y_true.append(0)
                        y_pred.append(pred_labels[p].item())

    # === COCO Evaluation ===
    coco_gt = COCO(ann_file)
    with open(results_file, "w") as f:
        json.dump(coco_results, f)
    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50 = coco_eval.stats[1]
    map_5095 = coco_eval.stats[0]
    avg_iou = np.mean(iou_list) if iou_list else 0.0

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title(f"{output_prefix} - Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "confusion_matrix.png"))
    plt.show()

    # === Classification Report ===
    report = classification_report(y_true, y_pred, labels=LABELS, target_names=LABEL_NAMES, digits=3)
    print("\n📊 Classification Report:")
    print(report)

    # === Save Summary ===
    with open(os.path.join(log_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nmAP@0.5:       {map_50:.3f}")
        f.write(f"\nmAP@0.5:0.95:   {map_5095:.3f}")
        f.write(f"\nAverage IoU:   {avg_iou:.3f}")

    print(f"\n✅ Finished: {output_prefix}")
    print(f"  mAP@0.5: {map_50:.3f}, mAP@0.5:0.95: {map_5095:.3f}, Avg IoU: {avg_iou:.3f}")
    print(f"  📁 Logs saved to: {log_dir}")

# === Run all conditions ===
if __name__ == "__main__":
    EVAL_SETS = {
        "eval_brightness": (
            "C:/Projects/dataset/dataset_test_brightness/images",
            "C:/Projects/dataset/dataset_test_brightness/annotations.json"
        ),
        "eval_motionblur": (
            "C:/Projects/dataset/dataset_test_motion_blur/images",
            "C:/Projects/dataset/dataset_test_motion_blur/annotations.json"
        ),
        "eval_background": (
            "C:/Projects/dataset/dataset_test_background_complexity/images",
            "C:/Projects/dataset/dataset_test_background_complexity/annotations.json"
        )
    }

    for name, (img_dir, ann_path) in EVAL_SETS.items():
        evaluate_dataset(img_dir, ann_path, output_prefix=name)
