# ...existing code...
"""
YOLO11n-StarNet: Lightweight version for medical image detection (e.g., brain tumor)
- Replaces heavy C3K2 backbone blocks with StarNet blocks
- Uses depthwise separable convs
- Reduces channels in neck and head for faster training
Author: ChatGPT (Optimized for Br35H dataset)
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Basic Conv (with depthwise separable option)
# -------------------------
class DWConv(nn.Module):
    """Depthwise + Pointwise convolution block"""
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        # depthwise conv can downsample via stride 's'
        self.dw = nn.Conv2d(c1, c1, k, s, k // 2, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

# -------------------------
# StarBlock (lightweight feature block)
# -------------------------
class StarBlock(nn.Module):
    """
    Inspired by StarNet: uses cross-branch connections and DW conv
    Accepts stride 's' to optionally downsample.
    """
    def __init__(self, c1, c2, s=1):
        super().__init__()
        mid = c2 // 2
        # pass stride to branches so the block can reduce spatial size when needed
        self.branch1 = DWConv(c1, mid, k=3, s=s)
        self.branch2 = DWConv(c1, mid, k=3, s=s)
        self.fuse = Conv(mid * 2, c2, k=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        return self.fuse(torch.cat((b1, b2), dim=1))

# -------------------------
# Simple Conv Block
# -------------------------
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# -------------------------
# SPPF remains same (lightweight pooling)
# -------------------------
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_ * 4, c2, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

# -------------------------
# Detection Head
# -------------------------
class DetectHead(nn.Module):
    def __init__(self, num_classes, channels: Tuple[int, int, int]):
        super().__init__()
        self.nc = num_classes
        self.no = num_classes + 5
        self.na = 3
        self.m = nn.ModuleList([nn.Conv2d(c, self.no * self.na, 1) for c in channels])

    def forward(self, feats):
        outs = []
        for i, x in enumerate(feats):
            bs, _, h, w = x.shape
            pred = self.m[i](x)
            pred = pred.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2).contiguous()
            outs.append(pred)
        return outs

# -------------------------
# YOLO11n-StarNet model
# -------------------------
class YOLO11n(nn.Module):
    def __init__(self, num_classes=2, width_mult=0.5):
        super().__init__()
        def c(ch): return max(1, int(ch * width_mult))

        # Backbone (StarNet) -- ensure progressive downsampling
        self.stem = Conv(3, c(16), 3, 2)       # 640 -> 320
        self.block1 = StarBlock(c(16), c(32), s=2)  # 320 -> 160
        self.block2 = StarBlock(c(32), c(64), s=2)  # 160 -> 80
        self.block3 = StarBlock(c(64), c(128), s=2) # 80 -> 40

        self.sppf = SPPF(c(128), c(128))

        # Neck
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')  # 40 -> 80
        self.neck1 = StarBlock(c(128) + c(64), c(64), s=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')  # 80 -> 160
        self.neck2 = StarBlock(c(64) + c(32), c(32), s=1)

        # Head (channels correspond to [small, medium, large] feature maps)
        self.detect = DetectHead(num_classes, channels=(c(32), c(64), c(128)))

    def forward(self, x):
        p1 = self.stem(x)     # 320
        p2 = self.block1(p1)  # 160
        p3 = self.block2(p2)  # 80
        p4 = self.block3(p3)  # 40
        p5 = self.sppf(p4)    # 40

        n1 = self.up1(p5)     # 80
        n1 = self.neck1(torch.cat([n1, p3], 1))
        n2 = self.up2(n1)     # 160
        n2 = self.neck2(torch.cat([n2, p2], 1))

        preds = self.detect([n2, n1, p5])
        return preds

# -------------------------
# Test run
# -------------------------
if __name__ == "__main__":
    model = YOLO11n(num_classes=2, width_mult=0.5)
    x = torch.randn(1, 3, 640, 640)
    outs = model(x)
    for i, o in enumerate(outs):
        print(f"Scale {i}: {o.shape}")# yolov11n_model.py
"""
YOLO11n + Star-style lightweight backbone + PAN-like neck implementation.
Architecture follows the diagram you provided:
  - Stem -> Stage1 -> Stage2 -> Stage3 -> SPPF (backbone)
  - Project p2/p3/p5 -> upsample p5 -> fuse with p3 -> upsample -> fuse with p2 (neck)
  - Head: three scales (small, medium, large) via DetectHead (same interface)
This is designed to be a near drop-in replacement for your previous file.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic Conv block (Conv + BN + SiLU)
# -------------------------
class Conv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int = None, g: int = 1, act: bool = True):
        super().__init__()
        p = (k // 2) if p is None else p
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------
# Depthwise separable conv (DWConv)
# -------------------------
class DWConv(nn.Module):
    """Depthwise + Pointwise conv (lightweight)"""
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, act: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=k, stride=s, padding=k // 2, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


# -------------------------
# StarBlock: small two-branch lightweight block
# -------------------------
class StarBlock(nn.Module):
    """
    Lightweight Star-style block: two DW branches fused then a pointwise conv.
    Optionally supports stride to downsample.
    """
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        mid = max(1, c_out // 2)
        # branches use stride to downsample when requested
        self.branch1 = DWConv(c_in, mid, k=3, s=stride)
        self.branch2 = DWConv(c_in, mid, k=3, s=stride)
        self.fuse = Conv(mid * 2, c_out, k=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        return self.fuse(torch.cat([b1, b2], dim=1))


# -------------------------
# SPPF (fast SPP)
# -------------------------
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (stacked maxpool)"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        hidden = max(1, in_ch // 2)
        self.cv1 = Conv(in_ch, hidden, k=1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(hidden * 4, out_ch, k=1)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.mp(x1)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        return self.cv2(torch.cat([x1, y1, y2, y3], dim=1))


# -------------------------
# Small C3-like block (used in neck fusion)
# -------------------------
class C3K2(nn.Module):
    """Small C3-like block used for fusion (keeps model lightweight)"""
    def __init__(self, in_ch: int, out_ch: int, n: int = 1):
        super().__init__()
        # simple design: project -> n convs -> project
        hidden = max(1, out_ch // 2)
        self.cv1 = Conv(in_ch, hidden, k=1)
        self.m = nn.Sequential(*[Conv(hidden, hidden, k=3) for _ in range(n)])
        self.cv2 = Conv(hidden, out_ch, k=1)

    def forward(self, x):
        x = self.cv1(x)
        x = self.m(x)
        return self.cv2(x)


# -------------------------
# Detect head (predictor) — remains same interface
# -------------------------
class DetectHead(nn.Module):
    def __init__(self, num_classes: int, channels: Tuple[int, int, int], anchors_per_scale: int = 3):
        """
        channels: number of channels in feature maps for [small, medium, large]
        """
        super().__init__()
        self.nc = num_classes
        self.no = num_classes + 5  # xywh + obj + classes
        self.na = anchors_per_scale
        self.m = nn.ModuleList([nn.Conv2d(c, self.no * self.na, kernel_size=1) for c in channels])

    def forward(self, feats: List[torch.Tensor]):
        outs = []
        for i, x in enumerate(feats):
            bs, _, h, w = x.shape
            pred = self.m[i](x)
            pred = pred.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2).contiguous()
            outs.append(pred)
        return outs


# -------------------------
# Full YOLO11n model following the diagram
# -------------------------
class YOLO11n(nn.Module):
    def __init__(self, num_classes: int = 2, width_mult: float = 0.5):
        """
        width_mult scales channels to create tiny variants.
        Architecture:
          - stem (downsample)
          - stage1_down -> stage1 (p2)
          - stage2_down -> stage2 (p3)
          - stage3_down -> stage3 (p4)
          - sppf on p4 -> p5
          - neck: p5 -> up -> fuse with p3 -> up -> fuse with p2
          - head: outputs from fused_p2 (small), fused_p3 (medium), p5 (large)
        """
        super().__init__()
        def c(ch): return max(1, int(ch * width_mult))

        # Backbone channel plan (before projection):
        ch1 = c(16)   # stem out
        ch2 = c(32)   # p2
        ch3 = c(64)   # p3
        ch4 = c(128)  # p4 (deep)

        # Stem
        self.stem = Conv(3, ch1, k=3, s=2)  # 640 -> 320

        # Stage1: downsample then block -> p2 (320->160)
        self.stage1_down = Conv(ch1, ch2, k=3, s=2)  # /2
        self.stage1 = StarBlock(ch2, ch2, stride=1)

        # Stage2: downsample then block -> p3 (160->80)
        self.stage2_down = Conv(ch2, ch3, k=3, s=2)
        self.stage2 = StarBlock(ch3, ch3, stride=1)

        # Stage3: downsample then block -> p4 (80->40)
        self.stage3_down = Conv(ch3, ch4, k=3, s=2)
        self.stage3 = StarBlock(ch4, ch4, stride=1)

        # SPPF at deepest level -> p5 (same spatial as p4)
        self.sppf = SPPF(ch4, ch4, k=5)

        # --- Projections to unify channels for neck ---
        neck_c = ch4  # unify to deep channel size for fusion simplicity
        self.proj_p2 = Conv(ch2, neck_c, k=1)  # project p2 (160) to neck_c
        self.proj_p3 = Conv(ch3, neck_c, k=1)  # project p3 (80) to neck_c
        self.proj_p5 = Conv(ch4, neck_c, k=1)  # project p5 (40) to neck_c

        # Neck fusion blocks (after concatenation)
        # concat(p5_up, p3) -> fused_mid (80)
        self.c3_fuse = C3K2(neck_c + neck_c, neck_c, n=1)
        # concat(fused_mid_up, p2) -> fused_shallow (160)
        self.c4_fuse = C3K2(neck_c + neck_c, neck_c, n=1)

        # Final converters to head channels (small, medium, large)
        self.out_small = Conv(neck_c, ch2, k=1)   # small-scale detection uses c(32)
        self.out_medium = Conv(neck_c, ch3, k=1)  # medium-scale => c(64)
        self.out_large = Conv(neck_c, ch4, k=1)   # large-scale => c(128)

        # Detection head: expects channels (small, medium, large)
        head_ch = (ch2, ch3, ch4)
        self.head = DetectHead(num_classes, channels=head_ch, anchors_per_scale=3)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Backbone
        x = self.stem(x)                     # /2 -> 320
        x = self.stage1_down(x)              # /2 -> 160
        p2 = self.stage1(x)                  # p2 (B, ch2, 160, 160)

        x = self.stage2_down(p2)             # /2 -> 80
        p3 = self.stage2(x)                  # p3 (B, ch3, 80, 80)

        x = self.stage3_down(p3)             # /2 -> 40
        p4 = self.stage3(x)                  # p4 (B, ch4, 40, 40)

        p5 = self.sppf(p4)                   # p5 (B, ch4, 40, 40)

        # Project to neck channels
        p2_proj = self.proj_p2(p2)           # (B, neck_c, 160,160)
        p3_proj = self.proj_p3(p3)           # (B, neck_c, 80,80)
        p5_proj = self.proj_p5(p5)           # (B, neck_c, 40,40)

        # Neck fusion 1: upsample p5_proj -> concat with p3_proj
        p5_up = F.interpolate(p5_proj, scale_factor=2, mode='nearest')  # -> (B, neck_c, 80,80)
        # if shapes mismatch, ensure same size (safety)
        if p5_up.shape[2:] != p3_proj.shape[2:]:
            p3_proj = F.interpolate(p3_proj, size=p5_up.shape[2:], mode='nearest')
        mid = torch.cat([p5_up, p3_proj], dim=1)  # (B, 2*neck_c, 80,80)
        fused_mid = self.c3_fuse(mid)             # (B, neck_c, 80,80)

        # Neck fusion 2: upsample fused_mid -> concat with p2_proj
        fused_mid_up = F.interpolate(fused_mid, scale_factor=2, mode='nearest')  # -> (B, neck_c, 160,160)
        if fused_mid_up.shape[2:] != p2_proj.shape[2:]:
            p2_proj = F.interpolate(p2_proj, size=fused_mid_up.shape[2:], mode='nearest')
        shallow = torch.cat([fused_mid_up, p2_proj], dim=1)  # (B, 2*neck_c, 160,160)
        fused_shallow = self.c4_fuse(shallow)                # (B, neck_c, 160,160)

        # Produce outputs tuned for head channels
        out_s = self.out_small(fused_shallow)  # small (B, ch2, 160,160)
        out_m = self.out_medium(fused_mid)     # medium (B, ch3, 80,80)
        out_l = self.out_large(p5_proj)        # large (B, ch4, 40,40)

        # Pass to detect head (will apply 1x1 convs and reshape)
        preds = self.head([out_s, out_m, out_l])
        return preds

    def _initialize_weights(self):
        # Kaiming init for convs, BN ones/zeros
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# -------------------------
# Quick sanity test (keeps same API)
# -------------------------
if __name__ == "__main__":
    model = YOLO11n(num_classes=2, width_mult=0.5)
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    outs = model(x)
    print("Outputs (3 scales):")
    for i, o in enumerate(outs):
        print(f" scale {i}: shape = {o.shape}  (bs, anchors, gh, gw, no)")

# ...existing code...