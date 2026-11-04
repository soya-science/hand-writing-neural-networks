"""Frozen ImageNet ResNet-50 (torchvision IMAGENET1K_V2), cut after layer3. Nothing here is
trained or changed; the hand-typed head in head_weights.py sits on top of it.

Faces go in gray: the cartoon faces the head was wired on are gray, and the expression cues
the head reads are shapes, not colours. The 100x100 face is upsampled to 224x224 and
normalised with the ImageNet statistics. layer3 gives 1024 channels on a 14x14 grid; one cell
covers about 7x7 pixels of the 100x100 face.
"""
import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50

LUMA = np.array([0.299, 0.587, 0.114], np.float32)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
GRID = 14

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_net = None


def _network():
    global _net
    if _net is None:
        _net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().to(DEVICE)
        for p in _net.parameters():
            p.requires_grad_(False)
    return _net


@torch.no_grad()
def layer3(rgb, batch=128):
    """rgb: (N, 100, 100, 3) uint8 -> layer3 maps (N, 1024, 14, 14) float32."""
    net = _network()
    out = []
    for i in range(0, len(rgb), batch):
        gray = rgb[i:i + batch].astype(np.float32) @ LUMA / 255.0
        x = torch.from_numpy(gray)[:, None].repeat(1, 3, 1, 1)
        x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(DEVICE)
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        out.append(net.layer3(net.layer2(net.layer1(x))).cpu().numpy())
    return np.concatenate(out)
