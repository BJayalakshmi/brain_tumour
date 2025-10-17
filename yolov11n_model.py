# yolov11n_model.py
"""
YOLO11-nano style model (PyTorch) with C3K2 blocks.
This is a lightweight, readable implementation intended for experimentation/modification.
Not a drop-in replacement for Ultralytics, but mirrors the overall structure:
  Stem -> Backbone (C3K2 stages) -> SPPF -> Neck (simple PAN-style fusion) -> Head (3-scale)
Author: ChatGPT (for educational / research use)
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic conv block (Conv + BN + SiLU)
# -------------------------
class Conv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int = None, g: int = 1, act: bool = True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------
# Simple Bottleneck (residual) used inside C3/C3K modules
# -------------------------
class Bottleneck(nn.Module):
    def __init__(self, c: int, hidden_ratio: float = 0.5, shortcut: bool = True):
        super().__init__()
        hidden = max(1, int(c * hidden_ratio))
        self.cv1 = Conv(c, hidden, k=1, s=1)
        self.cv2 = Conv(hidden, c, k=3, s=1)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


# -------------------------
# C3-like internal helper (small variant used by C3K)
# -------------------------
class C3KInner(nn.Module):
    """A small C3-like block used internally by C3K2"""
    def __init__(self, in_ch: int, out_ch: int, n: int = 1):
        super().__init__()
        hidden = out_ch // 2
        self.conv1 = Conv(in_ch, hidden, k=1)
        self.conv2 = Conv(in_ch, hidden, k=1)
        # use simple bottlenecks inside
        self.m = nn.Sequential(*[Bottleneck(hidden) for _ in range(n)])
        self.conv3 = Conv(hidden * 2, out_ch, k=1)

    def forward(self, x):
        y1 = self.m(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat((y1, y2), dim=1))


# -------------------------
# C3K2 block implementation
# -------------------------
class C3K2(nn.Module):
    """
    C3K2 block (one reasonable implementation variant).
    Structure:
      - 1x1 conv to reduce channels -> split into two halves
      - pass one half through n modules (residuals or small C3)
      - concat the halves + outputs -> 1x1 conv to restore channels
    This implementation is flexible: set use_c3inner to True to use C3KInner modules inside.
    """
    def __init__(self, in_ch: int, out_ch: int, n: int = 1, use_c3inner: bool = False, reduction: int = 2):
        """
        in_ch: input channels
        out_ch: output channels
        n: number of internal modules appended (the "K2" depth)
        use_c3inner: use C3KInner blocks inside instead of simple Bottleneck
        reduction: factor to compute hidden channels (typical 2)
        """
        super().__init__()
        # hidden channels after first 1x1 conv
        hidden = max(1, out_ch // reduction)  # may be half of out_ch
        # make sure hidden is divisible by 2 so we can chunk
        if hidden % 2 != 0:
            hidden += 1

        self.cv1 = Conv(in_ch, hidden, k=1)  # project to hidden
        # modules that will be applied to one chunk
        self.mods = nn.ModuleList()
        for _ in range(n):
            if use_c3inner:
                self.mods.append(C3KInner(hidden // 2, hidden // 2, n=1))
            else:
                # pure bottleneck on half-channels
                self.mods.append(Bottleneck(hidden // 2))
        # final conv to combine concatenated outputs
        # concatenation will be: [half_a, outputs_from_each_mod..., half_b]
        concat_len = (2 + n) * (hidden // 2)
        self.cv2 = Conv(concat_len, out_ch, k=1)

    def forward(self, x):
        x_proj = self.cv1(x)
        # split into two halves
        a, b = x_proj.chunk(2, dim=1)  # a: will be transformed by modules, b: passthrough
        outs = [a]
        last = a
        for m in self.mods:
            last = m(last)
            outs.append(last)
        outs.append(b)
        cat = torch.cat(outs, dim=1)
        return self.cv2(cat)


# -------------------------
# SPPF (fast SPP) block
# -------------------------
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (stacked maxpool)"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        hidden = in_ch // 2
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
# Simple PAN-like neck (lightweight)
# -------------------------
class SimplePAN(nn.Module):
    """
    A simplified PAN: combine features from 3 scales.
    We expect inputs: [p3 (small), p4 (medium), p5 (large)] (p5 deepest)
    We'll upsample & fuse to produce three fused outputs for the head.
    """
    def __init__(self, chs: Tuple[int, int, int]):
        """
        chs: channels for [p3, p4, p5] feature maps (from shallow->deep)
        We'll produce three outputs tuned for detection head.
        """
        super().__init__()
        c3, c4, c5 = chs
        # reduce convs
        self.reduce5 = Conv(c5, c4, k=1)
        self.c3_fuse = C3K2(c4 + c4, c4, n=1, use_c3inner=False)  # fuse upsampled p5 & p4
        self.downconv = Conv(c4, c4, k=3, s=2, p=1)
        self.c4_fuse = C3K2(c4 + c4, c4, n=1, use_c3inner=False)  # fuse back

        # final convs to standardize channels for detection
        self.out_small = Conv(c4, c3, k=1)   # for small-scale detection
        self.out_medium = Conv(c4, c4, k=1)  # for medium-scale detection
        self.out_large = Conv(c4, c5, k=1)   # for large-scale detection

    def forward(self, p3, p4, p5):
        # p3: shallow (e.g., 1/8), p4: mid (1/16), p5: deep (1/32)
        p5r = self.reduce5(p5)
        p5_up = F.interpolate(p5r, scale_factor=2, mode='nearest')  # up to p4 resolution
        p4_f = self.c3_fuse(torch.cat([p5_up, p4], dim=1))
        p4_down = self.downconv(p4_f)
        p5_back = self.c4_fuse(torch.cat([p4_down, p5], dim=1))

        # produce 3 outputs (small, medium, large)
        out_s = self.out_small(p4_f)   # will be used for smallest objects
        out_m = self.out_medium(p4_f)  # medium
        out_l = self.out_large(p5_back)  # large
        return out_s, out_m, out_l


# -------------------------
# Detect head (predictor)
# -------------------------
class DetectHead(nn.Module):
    def __init__(self, num_classes: int, channels: Tuple[int, int, int], anchors_per_scale: int = 3):
        """
        channels: number of channels in feature maps for [small, medium, large]
        anchors_per_scale: usually 3
        """
        super().__init__()
        self.nc = num_classes
        self.no = num_classes + 5  # (x, y, w, h, obj) + classes
        self.na = anchors_per_scale
        self.m = nn.ModuleList([nn.Conv2d(c, self.no * self.na, kernel_size=1) for c in channels])

    def forward(self, feats: List[torch.Tensor]):
        # feats: [small, medium, large] feature tensors
        outs = []
        for i, x in enumerate(feats):
            bs, _, h, w = x.shape
            pred = self.m[i](x)
            # reshape to (bs, anchors, grid_h, grid_w, no)
            pred = pred.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2).contiguous()
            outs.append(pred)  # (bs, na, gh, gw, no)
        return outs


# -------------------------
# Full YOLO11n model
# -------------------------
class YOLO11n(nn.Module):
    def __init__(self, num_classes: int = 80, width_mult: float = 1.0, use_c3inner: bool = False):
        """
        width_mult: scales channel numbers (useful to create tiny/nano variants)
        use_c3inner: whether to use the C3KInner inside C3K2 internal modules
        """
        super().__init__()
        # channel base (nano = smaller)
        def c(ch): return max(1, int(ch * width_mult))

        # --- Stem + Backbone (3 stages) ---
        self.stem = Conv(3, c(16), k=3, s=2)       # /2
        self.stage1_down = Conv(c(16), c(32), k=3, s=2)  # /4
        self.stage1 = C3K2(c(32), c(32), n=1, use_c3inner=use_c3inner)

        self.stage2_down = Conv(c(32), c(64), k=3, s=2)  # /8
        self.stage2 = C3K2(c(64), c(64), n=2, use_c3inner=use_c3inner)

        self.stage3_down = Conv(c(64), c(128), k=3, s=2)  # /16
        self.stage3 = C3K2(c(128), c(128), n=2, use_c3inner=use_c3inner)

        # SPPF (keep spatial reduced)
        self.sppf = SPPF(c(128), c(128), k=5)

        # We'll produce three feature maps:
        # p3: from stage1 (shallow)  -> outputs c(32)
        # p4: from stage2 (mid)      -> outputs c(64)
        # p5: from sppf / stage3     -> outputs c(128)
        # Project channels to unify neck channel size
        neck_c = c(128)
        # NOTE: use the actual channel counts produced by each stage for projection
        self.proj_p3 = Conv(c(32), neck_c, k=1)   # stage1 outputs 32 channels
        self.proj_p4 = Conv(c(64), neck_c, k=1)   # stage2 outputs 64 channels
        self.proj_p5 = Conv(c(128), neck_c, k=1)  # stage3/SPPF outputs 128 channels

        # neck expects channels tuple (p3, p4, p5) -> use unified neck_c for simplicity
        self.neck = SimplePAN((neck_c, neck_c, neck_c))

        # head channels should match the outputs of the neck (which are produced by out_small/out_medium/out_large)
        # Since SimplePAN was created with (neck_c, neck_c, neck_c), its outputs will have neck_c channels -> use that.
        head_ch = (neck_c, neck_c, neck_c)
        # instantiate detector
        self.head = DetectHead(num_classes, channels=head_ch, anchors_per_scale=3)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # stem + stage1
        x = self.stem(x)           # /2
        x = self.stage1_down(x)    # /4
        p3_in = self.stage1(x)     # p3 candidate (channels = c(32))

        # stage2
        x = self.stage2_down(p3_in)  # /8
        p4_in = self.stage2(x)       # channels = c(64)

        # stage3
        x = self.stage3_down(p4_in)  # /16
        p5_in = self.stage3(x)       # channels = c(128)

        p5 = self.sppf(p5_in)

        # project to unify channels
        p3 = self.proj_p3(p3_in)
        p4 = self.proj_p4(p4_in)
        p5 = self.proj_p5(p5)

        # neck outputs
        out_s, out_m, out_l = self.neck(p3, p4, p5)

        # head expects feature maps with channels matching head_ch defined earlier
        preds = self.head([out_s, out_m, out_l])
        return preds  # list of 3 tensors: (bs, na, gh, gw, no)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# -------------------------
# Quick local test (run this file)
# -------------------------
if __name__ == "__main__":
    model = YOLO11n(num_classes=20, width_mult=1.0, use_c3inner=False)  # example: 20 classes
    model.eval()
    # dummy input 640x640
    x = torch.randn(2, 3, 640, 640)
    outs = model(x)
    print("Outputs (3 scales):")
    for i, o in enumerate(outs):
        print(f" scale {i}: shape = {o.shape}  (bs, anchors, gh, gw, no)")
