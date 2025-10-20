# ...existing code...
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from yolov11n_model import YOLO11n

# Minimal local Grad-CAM implementation (avoids external package)
class SimpleGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            # capture activation tensor and attach hook for gradients
            self.activations = output
            def backward_hook(grad):
                self.gradients = grad
            try:
                output.register_hook(backward_hook)
            except Exception:
                pass

        self.fh = self.target_layer.register_forward_hook(forward_hook)

    def __call__(self, input_tensor: torch.Tensor, target_category: int):
        """
        input_tensor: (bs=1, 3, H, W) on same device as model
        returns: numpy cam array shaped (H, W) normalized 0..1
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(True)
        outputs = self.model(input_tensor)  # expects (bs, num_classes)
        if outputs.ndim == 1:
            score = outputs[target_category]
        else:
            score = outputs[0, target_category]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM failed to collect activations/gradients.")

        # weights: global average pool of gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (bs, c, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (bs,1,h,w)
        cam = F.relu(cam)
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)

        # detach before converting to numpy to avoid "requires_grad" error
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = cam_np - cam_np.min()
        cam_max = cam_np.max() if cam_np.max() != 0 else 1e-8
        cam_np = cam_np / cam_max
        return cam_np

def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = True, colormap=cv2.COLORMAP_JET, alpha: float = 0.4):
    """
    img: HxWx3 float 0..1 (RGB)
    mask: HxW float 0..1
    returns: BGR uint8 image (same as cv2.imwrite expects)
    """
    heatmap = np.uint8(255 * mask)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0
    img_f = (img.astype(np.float32)) / 255.0
    cam = heatmap * alpha + img_f * (1 - alpha)
    cam = np.clip(cam, 0, 1)
    cam_bgr = cv2.cvtColor((cam * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cam_bgr

def overlay_gradcam_bgr(img_bgr: np.ndarray, cam_mask: np.ndarray, alpha: float = 0.4, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """
    img_bgr: HxWx3 uint8 (0..255) or float (0..1)
    cam_mask: HxW float normalized 0..1
    returns: HxWx3 BGR uint8 blended image
    """
    # ensure img float 0..1
    if img_bgr.dtype == np.uint8:
        img_f = img_bgr.astype(np.float32) / 255.0
    else:
        img_f = img_bgr.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0

    # prepare heatmap (BGR)
    heatmap = np.uint8(255 * cam_mask)
    heatmap = cv2.applyColorMap(heatmap, colormap)               # BGR
    heatmap_f = heatmap.astype(np.float32) / 255.0

    # blend
    blended = heatmap_f * alpha + img_f * (1.0 - alpha)
    blended = np.clip(blended, 0, 1)
    blended_bgr = (blended * 255).astype(np.uint8)
    return blended_bgr

def load_checkpoint(model: nn.Module, path: str = "best.pt", map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        new_k = re.sub(r"^module\.", "", k)
        new_state[new_k] = v

    try:
        model.load_state_dict(new_state, strict=True)
        print("Checkpoint loaded with strict=True")
    except Exception as e:
        print("Strict load failed:", e)
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"Loaded with strict=False. Missing keys: {len(missing)} Unexpected keys: {len(unexpected)}")
    return model

class DetectorAsClassifier(nn.Module):
    """
    Wrap detection model to produce per-class scores tensor of shape (bs, num_classes).
    Aggregates detection class scores across anchors / spatial positions / scales by max.
    """
    def __init__(self, det_model: nn.Module, num_classes: int):
        super().__init__()
        self.det_model = det_model
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = self.det_model(x)
        bs = x.shape[0]
        class_scores = []
        for cls in range(self.num_classes):
            per_scale_scores = []
            for p in preds:
                # p: (bs, na, h, w, no)
                score = p[..., 5 + cls]  # (bs, na, h, w)
                per_scale_scores.append(score.reshape(bs, -1))
            all_pos = torch.cat(per_scale_scores, dim=1) if len(per_scale_scores) > 1 else per_scale_scores[0]
            max_per_batch = all_pos.max(dim=1).values
            class_scores.append(max_per_batch)
        out = torch.stack(class_scores, dim=1)
        return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Load model --------
    model = YOLO11n(num_classes=2).to(device)
    model = load_checkpoint(model, "best.pt", map_location=device)
    model.eval()

    # -------- Load test image --------
    image_path = "D:/brain_tumour/test.jpg"  # change to your test image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    img_resized = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # -------- Inference (for detections) --------
    with torch.no_grad():
        preds = model(img_tensor)

    # -------- Process predictions --------
    detections = []
    for scale in preds:
        # scale: (bs, na, h, w, no)
        scale = scale[0].cpu()  # drop batch dim -> (na, h, w, no)
        conf = scale[..., 4]            # objectness (na, h, w)
        cls_scores = scale[..., 5:]     # (na, h, w, num_classes)
        total_conf = conf.unsqueeze(-1) * cls_scores  # (na, h, w, num_classes)
        flat = total_conf.reshape(-1, total_conf.shape[-1])  # (N, num_classes)
        max_conf_per_pos, cls_id = torch.max(flat, dim=1)    # per-position best class and score
        best_conf = max_conf_per_pos.max().item()
        best_class = cls_id[max_conf_per_pos.argmax()].item()
        detections.append((best_conf, best_class))

    best_conf, best_class = max(detections, key=lambda x: x[0])

    # -------- Print result --------
    if best_conf < 0.5:
        print("⚠️  No confident detection found.")
    elif best_class == 1:
        print(f"🧠 Tumor Detected! (Confidence: {best_conf:.2f})")
    else:
        print(f"✅ No Tumor Detected. (Confidence: {best_conf:.2f})")

    # -------- Grad-CAM visualization --------
    wrapped = DetectorAsClassifier(model, num_classes=2).to(device)
    wrapped.eval()

    # Prefer known neck / fusion / proj layers from your YOLO11n implementation
    prefer = ["c4_fuse", "c3_fuse", "proj_p3", "proj_p5", "proj_p2", "out_medium", "out_large", "sppf", "stage3"]
    target_layer = None
    chosen_name = None
    search_scopes = [getattr(wrapped, "det_model", None), model]
    for scope in search_scopes:
        if scope is None:
            continue
        for name in prefer:
            if hasattr(scope, name):
                target_layer = getattr(scope, name)
                chosen_name = f"{scope.__class__.__name__}.{name}"
                break
        if target_layer is not None:
            break

    if target_layer is None:
        for name, m in reversed(list(model.named_modules())):
            if isinstance(m, nn.Conv2d):
                target_layer = m
                chosen_name = name
                break

    if target_layer is None:
        print("Available model modules (first 100):")
        for i, (name, _) in enumerate(model.named_modules()):
            if i >= 100:
                break
            print(" ", name)
        raise AttributeError("Could not find a suitable target layer for Grad-CAM. Pick one of the printed module names and update the code.")
    print(f"Using target layer for Grad-CAM: {chosen_name}")

    rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    simple_cam = SimpleGradCAM(model=wrapped, target_layer=target_layer)
    input_tensor = img_tensor.clone().to(device).requires_grad_(True)

    try:
        cam_mask = simple_cam(input_tensor, target_category=1)  # HxW numpy mask
        # create overlay using the convenience function
        overlay = overlay_gradcam_bgr(img_resized, cam_mask, alpha=0.45)
        cv2.imwrite("gradcam_overlay.jpg", overlay)
        print("📸 Grad-CAM overlay saved as gradcam_overlay.jpg")
    except Exception as e:
        print("Grad-CAM failed:", e)

    # -------- Display input and overlay --------
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Input")
        ax[0].axis("off")
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        ax[1].imshow(overlay_rgb)
        ax[1].set_title("Grad-CAM Overlay")
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
# ...existing code...