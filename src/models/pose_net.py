"""SimpleBaseline heatmap network.  *** YOURS to implement (you asked to write the model) ***

Paper: Xiao et al. 2018, "Simple Baselines for Human Pose Estimation and Tracking" (arXiv 1804.06208),
Sec. 2 + Fig. 1(c). Read that first (~2 pages).

Data flow and shapes (B = batch, input 256x256, ResNet-50):

  image  (B, 3, 256, 256)   ImageNet-normalised, gray copied to 3 channels (speedplus.py does this)
    | backbone: ResNet without avgpool + fc       -> spatial size / 32
  feat   (B, 2048, 8, 8)    (ResNet-18/34: 512 channels instead of 2048)
    | deconv 1: ConvTranspose2d(k=4, s=2, p=1) + BN + ReLU   8 -> 16
    | deconv 2: same                                          16 -> 32
    | deconv 3: same                                          32 -> 64     (256 channels each)
  up     (B, 256, 64, 64)
    | final: Conv2d 1x1, 256 -> 11
  heatmaps (B, 11, 64, 64)   compared with make_heatmaps() targets via masked MSE

Total downsampling: /32 then x8 -> output = input / 4  (that's where STRIDE = 4 comes from).

Build order (check each with the shape test before the next):
  build_backbone -> make_deconv_layers -> PoseNet.__init__ -> PoseNet.forward -> init_head_weights
Check:  python -m pytest tests/test_model.py -q
"""
import torch
import torch.nn as nn
import torchvision


# --------------------------------------------------------------------------- #
# 1. Backbone (encoder)
# --------------------------------------------------------------------------- #
def build_backbone(name="resnet50", pretrained=True):
    """Return (backbone, out_channels).

    backbone: nn.Module mapping (B, 3, H, W) -> (B, C, H/32, W/32).
    out_channels: C  (2048 for resnet50/101, 512 for resnet18/34).

    Hints:
      - torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)
        builds the network and downloads the ImageNet weights (first time only, ~100 MB for r50).
      - print(model) once and look at the last two children: avgpool and fc. Those turn the
        8x8 feature map into ONE 1000-class vector. We need the spatial map, so drop them.
      - list(model.children())[:-2] then nn.Sequential(*...) does exactly that.
      - out_channels: model.fc.in_features (read it BEFORE you throw fc away).
    Think: why do we want ImageNet weights at all if Tango looks nothing like ImageNet?
    """
    weights = torchvision.models.get_model_weights(name).DEFAULT if pretrained else None
    model = torchvision.models.get_model(name, weights=weights)
    list_of_children = list(model.children())[:-2]
    backbone = nn.Sequential(*list_of_children)
    out_channels = model.fc.in_features
    return backbone, out_channels
    


# --------------------------------------------------------------------------- #
# 2. Deconv head (decoder)
# --------------------------------------------------------------------------- #
def make_deconv_layers(in_channels, num_layers=3, channels=256, kernel=4):
    """Return nn.Sequential of num_layers x [ConvTranspose2d -> BatchNorm2d -> ReLU], each doubling H, W.

    ConvTranspose2d output size: out = (in - 1) * stride - 2 * padding + kernel (+ output_padding)
      with kernel=4, stride=2, padding=1:  out = 2 * in   <- check this by hand
    Use bias=False in the ConvTranspose2d: BatchNorm right after it has its own shift (beta),
    so a conv bias would be redundant.
    The first layer takes in_channels (2048), the others take `channels` (256).
    Hint: build a Python list `layers`, append 3 modules per iteration, update in_channels = channels,
          return nn.Sequential(*layers).
    """
    layers = []
    for i in range(num_layers):
        layers.append(nn.ConvTranspose2d(in_channels, channels, kernel_size=kernel, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = channels
    return nn.Sequential(*layers)
    


def init_head_weights(module):
    """Initialise the NEW layers only (deconvs + final conv). Never call this on the backbone.

    SimpleBaseline recipe:
      ConvTranspose2d / Conv2d : weight ~ Normal(0, 0.001), bias (if any) = 0
      BatchNorm2d              : weight = 1, bias = 0
    Hint: for m in module.modules(): if isinstance(m, nn.ConvTranspose2d): nn.init.normal_(m.weight, std=0.001) ...
    Think: why such a tiny std? (what do the initial heatmaps look like, and what is the initial loss?)
    """
    for m in module.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    


# --------------------------------------------------------------------------- #
# 3. The full network
# --------------------------------------------------------------------------- #
class PoseNet(nn.Module):
    """backbone -> deconv head -> 1x1 conv -> (B, num_keypoints, H/4, W/4) heatmaps."""

    def __init__(self, backbone="resnet50", num_keypoints=11, pretrained=True,
                 deconv_layers=3, deconv_channels=256):
        super().__init__()
        self.backbone, out_channels = build_backbone(backbone, pretrained)
        self.deconv = make_deconv_layers(out_channels, deconv_layers, deconv_channels)
        self.final = nn.Conv2d(deconv_channels, num_keypoints, kernel_size=1)
        init_head_weights(self.deconv)
        init_head_weights(self.final)

    def forward(self, x):
        """x (B, 3, 256, 256) -> heatmaps (B, num_keypoints, 64, 64)."""
        x = self.backbone(x)
        x = self.deconv(x)
        x = self.final(x)
        return x
        
