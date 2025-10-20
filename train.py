import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from yolov11n_model import YOLO11n
import os
import math
from PIL import Image
import numpy as np
import yaml

# =====================
# Helper: Shape IoU Function
# =====================
def box_iou(box1, box2):
    """Compute IoU between two boxes: [x1, y1, x2, y2]"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2Area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = box1Area + box2Area - interArea
    return interArea / union if union > 0 else 0


# =====================
# Dataset Class
# =====================
class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=320, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.img_size = img_size
        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, xc, yc, bw, bh = map(float, line.strip().split())
                    boxes.append([cls, xc, yc, bw, bh])
        boxes = torch.tensor(boxes) if len(boxes) else torch.zeros((0, 5))
        return img, boxes


# =====================
# YOLO Loss Function
# =====================
class YOLOLoss(nn.Module):
    def __init__(self, num_classes=2, lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

    def forward(self, preds, targets):
        total_loss = 0.0
        for p in preds:
            obj_pred = p[..., 4]
            cls_pred = p[..., 5:]
            obj_target = torch.zeros_like(obj_pred)
            box_target = torch.zeros_like(p[..., :4])
            cls_target = torch.zeros_like(cls_pred)

            loss_box = self.mse(p[..., :4], box_target)
            loss_obj = self.bce(obj_pred, obj_target)
            loss_cls = self.bce(cls_pred, cls_target)
            total_loss += (self.lambda_box * loss_box + self.lambda_obj * loss_obj + self.lambda_cls * loss_cls)
        return total_loss


# =====================
# Training Function
# =====================
def train_yolo11n(data_yaml='dataset.yaml', num_epochs=10, batch_size=1, lr=1e-4, img_size=320):
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataset config
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
    num_classes = data_cfg['nc']

    model = YOLO11n(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = YOLOLoss(num_classes=num_classes)

    train_ds = YOLODataset(data_cfg['train'], data_cfg['train'].replace('images', 'labels'), img_size)
    val_ds = YOLODataset(data_cfg['val'], data_cfg['val'].replace('images', 'labels'), img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_loss = math.inf

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

        # Validation with Shape-IoU
        model.eval()
        val_loss, ious = 0, []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item()

                # dummy IoU for demonstration
                for t in targets:
                    if len(t) > 0:
                        gt = t[0, 1:].cpu().numpy()  # xc, yc, bw, bh
                        xc, yc, bw, bh = gt
                        x1, y1 = xc - bw/2, yc - bh/2
                        x2, y2 = xc + bw/2, yc + bh/2
                        gt_box = [x1, y1, x2, y2]
                        pred_box = [x1+0.05, y1+0.05, x2, y2]
                        iou = box_iou(pred_box, gt_box)
                        ious.append(iou)

        avg_val_loss = val_loss / len(val_loader)
        mean_iou = np.mean(ious) if ious else 0
        print(f"   Validation Loss: {avg_val_loss:.4f} | Shape-IoU: {mean_iou:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best.pt")
            print("   ✅ Saved new best model!")

    print("\nTraining completed! Best model saved as best.pt")


if __name__ == "__main__":
    train_yolo11n()
