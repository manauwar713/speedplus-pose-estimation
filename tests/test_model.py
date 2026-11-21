"""Shape / wiring tests for src/models/pose_net.py. CPU-only, no weight download (pretrained=False)."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.pose_net import (  # noqa: E402
    PoseNet, build_backbone, init_head_weights, make_deconv_layers,
)


@pytest.mark.parametrize("name, channels", [("resnet18", 512), ("resnet50", 2048)])
def test_backbone_downsamples_by_32(name, channels):
    bb, c = build_backbone(name, pretrained=False)
    assert c == channels
    with torch.no_grad():
        out = bb.eval()(torch.zeros(1, 3, 256, 256))
    assert out.shape == (1, channels, 8, 8)


def test_backbone_has_no_pool_or_fc():
    bb, _ = build_backbone("resnet18", pretrained=False)
    assert not any(isinstance(m, (nn.AdaptiveAvgPool2d, nn.Linear)) for m in bb.modules())


def test_deconv_doubles_three_times():
    head = make_deconv_layers(512, num_layers=3, channels=256)
    n_deconv = sum(isinstance(m, nn.ConvTranspose2d) for m in head.modules())
    n_bn = sum(isinstance(m, nn.BatchNorm2d) for m in head.modules())
    assert n_deconv == 3 and n_bn == 3
    with torch.no_grad():
        out = head.eval()(torch.zeros(2, 512, 8, 8))
    assert out.shape == (2, 256, 64, 64)


def test_deconv_has_no_bias():
    head = make_deconv_layers(512)
    assert all(m.bias is None for m in head.modules() if isinstance(m, nn.ConvTranspose2d))


def test_init_head_weights():
    head = nn.Sequential(make_deconv_layers(512), nn.Conv2d(256, 11, 1))
    init_head_weights(head)
    for m in head.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            assert m.weight.std().item() < 0.002
            if m.bias is not None:
                assert torch.all(m.bias == 0)
        if isinstance(m, nn.BatchNorm2d):
            assert torch.all(m.weight == 1) and torch.all(m.bias == 0)


@pytest.mark.parametrize("name, size", [("resnet18", 128), ("resnet50", 256)])
def test_posenet_output_shape(name, size):
    net = PoseNet(name, num_keypoints=11, pretrained=False).eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 3, size, size))
    assert out.shape == (2, 11, size // 4, size // 4)  # heatmap stride 4


def test_head_init_does_not_touch_backbone():
    # init_head_weights must only hit deconv + final. If it runs on the backbone, the (ImageNet)
    # weights get overwritten with N(0, 0.001). torchvision's own conv1 init has std ~0.025.
    net = PoseNet("resnet18", pretrained=False)
    assert net.backbone[0].weight.std().item() > 0.01
    assert net.final.weight.std().item() < 0.002


def test_initial_heatmaps_near_zero():
    # Tiny head init -> network starts by predicting ~flat zero maps.
    torch.manual_seed(0)
    net = PoseNet("resnet18", pretrained=False).eval()
    with torch.no_grad():
        out = net(torch.randn(1, 3, 128, 128))
    assert out.abs().max().item() < 0.1


def test_one_training_step_reduces_loss():
    # Overfit a single fake sample for a few steps: catches broken gradients / detached graph.
    torch.manual_seed(0)
    net = PoseNet("resnet18", pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    target = torch.zeros(2, 11, 32, 32)
    target[:, :, 16, 16] = 1.0
    # lr 1e-4: with 1e-3 Adam overshoots on this toy problem (loss goes UP for the first steps)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    losses = []
    for _ in range(10):
        loss = ((net(x) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        # gradient must reach the very first backbone layer, not only the head
        first = next(net.backbone.parameters())
        assert first.grad is not None and first.grad.abs().sum() > 0
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
