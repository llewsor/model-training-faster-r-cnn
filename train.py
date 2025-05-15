import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
from datetime import datetime
from tqdm import tqdm
import multiprocessing

# Transformation class
class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

# Dataset loader
def get_coco_dataset(img_dir, ann_file):
    return CocoDetection(
        root=img_dir,
        annFile=ann_file,
        transforms=CocoTransform()
    )

# Paths	
train_img_dir = "C:/Projects/dataset/train/images"
train_ann_file = "C:/Projects/dataset/train/annotations.json"
val_img_dir = "C:/Projects/dataset/val/images"
val_ann_file = "C:/Projects/dataset/val/annotations.json"

# Create datasets and loaders
train_dataset = get_coco_dataset(train_img_dir, train_ann_file)
val_dataset   = get_coco_dataset(val_img_dir, val_ann_file)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x)),
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x)),
    pin_memory=True
)

# Model initialization
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.transform.min_size = (640,)
    model.transform.max_size = 640
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Device setup
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"✅ Using device: {device}")

# Prepare logging file
os.makedirs("logs", exist_ok=True)
tlog_path = f"logs/training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
with open(tlog_path, 'w') as f:
    f.write(f"Training Log/n")
    
def log_to_file(msg):
    with open(tlog_path, 'a') as f:
        f.write(msg + "/n")

# Instantiate model
num_classes = 3  # background + whole + broken
model = get_model(num_classes).to(device)

# Optimizer & scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Prepare checkpoints directory
os.makedirs("checkpoints", exist_ok=True)
best_val_map = -1.0
best_epoch = -1

# # Training function
# def train_one_epoch(model, optimizer, loader, device, epoch, num_epochs):
#     model.train()
#     total_loss = 0.0
#     for images, targets in loader:
#         images = [img.to(device) for img in images]
#         processed_targets = []
#         valid_images = []
#         for img, ann_list in zip(images, targets):
#             boxes, labels = [], []
#             for obj in ann_list:
#                 x, y, w, h = obj['bbox']
#                 if w > 0 and h > 0:
#                     boxes.append([x, y, x+w, y+h])
#                     labels.append(obj['category_id'])
#             if boxes:
#                 processed_targets.append({
#                     'boxes': torch.tensor(boxes, dtype=torch.float32).to(device),
#                     'labels': torch.tensor(labels, dtype=torch.int64).to(device)
#                 })
#                 valid_images.append(img)
#         if not processed_targets:
#             continue
#         loss_dict = model(valid_images, processed_targets)
#         loss = sum(loss for loss in loss_dict.values())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     avg_loss = total_loss / len(loader)
#     print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.4f}")
#     log_to_file(f"Epoch {epoch+1}/{num_epochs}/nAvg Loss = {avg_loss:.4f}")
#     return avg_loss

def train_one_epoch(model, optimizer, loader, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)
    for images, targets in loop:
        images = [img.to(device) for img in images]
        processed_targets = []
        valid_images = []

        for img, ann_list in zip(images, targets):
            boxes, labels = [], []
            for obj in ann_list:
                x, y, w, h = obj['bbox']
                if w > 0 and h > 0:
                    boxes.append([x, y, x+w, y+h])
                    labels.append(obj['category_id'])
            if boxes:
                processed_targets.append({
                    'boxes': torch.tensor(boxes, dtype=torch.float32).to(device),
                    'labels': torch.tensor(labels, dtype=torch.int64).to(device)
                })
                valid_images.append(img)
        if not processed_targets:
            continue
        loss_dict = model(valid_images, processed_targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())  # Show live loss value

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.4f}")
    log_to_file(f"Epoch {epoch+1}/{num_epochs}\nAvg Loss = {avg_loss:.4f}")
    return avg_loss

if __name__ == '__main__':
    # Main training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, num_epochs)
        lr_scheduler.step()

        # Validation
        model.eval()
        coco_gt = COCO(val_ann_file)
        results = []
        
        
        # with torch.no_grad():
        #     for images, targets in val_loader:
        #         images = [img.to(device) for img in images]
        #         outputs = model(images)
        #         outputs = [{k: v.cpu().numpy() for k, v in out.items()} for out in outputs]
        #         for ann_list, out in zip(targets, outputs):
        #             if not ann_list:
        #                 continue
        #             image_id = ann_list[0]['image_id']
        #             for (x1,y1,x2,y2), score, lbl in zip(
        #                     out['boxes'], out['scores'], out['labels']):
        #                 results.append({
        #                     'image_id': int(image_id),
        #                     'category_id': int(lbl),
        #                     'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
        #                     'score': float(score)
        #                 })
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc="🔍 Validating", leave=True)
            for images, targets in val_loop:
                images = [img.to(device) for img in images]
                outputs = model(images)
                outputs = [{k: v.cpu().numpy() for k, v in out.items()} for out in outputs]
                for ann_list, out in zip(targets, outputs):
                    if not ann_list:
                        continue
                    image_id = ann_list[0]['image_id']
                    for (x1, y1, x2, y2), score, lbl in zip(
                            out['boxes'], out['scores'], out['labels']):
                        results.append({
                            'image_id': int(image_id),
                            'category_id': int(lbl),
                            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            'score': float(score)
                        })
            
        
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        stats = coco_eval.stats
        templates = [
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {0:.3f}",
            "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {1:.3f}",
            "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {2:.3f}",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {3:.3f}",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {4:.3f}",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {5:.3f}",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {6:.3f}",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {7:.3f}",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {8:.3f}",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {9:.3f}",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {10:.3f}",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {11:.3f}"
        ]
        
        # Print and log each line
        for i, tmpl in enumerate(templates):
            line = tmpl.format(*stats)
            log_to_file(line)
    
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}_mAP_{stats[0]:.4f}.pth")
        
        # Save best model
        if stats[0] > best_val_map:
            best_val_map = stats[0]
            best_epoch = epoch+1
            best_path = f"checkpoints/best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"✅ New best VAL at epoch {best_epoch} (mAP={best_val_map:.4f}) saved to {best_path}")
            log_to_file(f"Saved best model: {best_path}")
            
        model.train()

    print(f"Training complete. Best epoch: {best_epoch}, mAP={best_val_map:.4f}")
    log_to_file(f"Training complete. Best epoch: {best_epoch}, mAP={best_val_map:.4f}")
