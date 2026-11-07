"""Segmenters. Each maps a normalised (B, 3, H, W) batch to {'out': (B, 4, H, W) logits}.

  mbv3          DeepLabV3-MobileNetV3-Large baseline, ImageNet backbone
  segformer_b2  SegFormer-B2 from NVIDIA's Cityscapes checkpoint with a new 4-class head, logits upsampled from 1/4 resolution
  segformer_b5  the same with the B5 encoder
SegFormer weights are under the NVIDIA Source Code License (non-commercial)."""
import hashlib
import os

import torch
import torch.nn.functional as F

SEGFORMER = {'segformer_b2': ('nvidia/segformer-b2-finetuned-cityscapes-1024-1024', 'd633b2072669ca68d8f8e309de9b52bfdbf6bf72'),
             'segformer_b5': ('nvidia/segformer-b5-finetuned-cityscapes-1024-1024', '2c6f153e4c23c229e2fa2b188eb250607e030cd8')}
ARCHS = ('mbv3',) + tuple(SEGFORMER)


class SegFormerWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        logits = self.net(pixel_values=x).logits
        return {'out': F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)}


def build(arch='segformer_b2', pretrained=True):
    """-> (model, provenance). pretrained=False builds the bare architecture for loading a checkpoint."""
    if arch == 'mbv3':
        from sliq.models.builders import deeplab_baseline
        return deeplab_baseline(pretrained=pretrained), dict(arch=arch, init='ImageNet MobileNetV3-Large')
    if arch not in SEGFORMER:
        raise ValueError(arch)
    from huggingface_hub import snapshot_download
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    repo, revision = SEGFORMER[arch]
    path = snapshot_download(repo, revision=revision, allow_patterns=None if pretrained else ['*.json'])
    if pretrained:
        net = SegformerForSemanticSegmentation.from_pretrained(path, num_labels=4, ignore_mismatched_sizes=True)
    else:
        net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(path, num_labels=4))
    weights = os.path.join(path, 'pytorch_model.bin')
    sha = hashlib.sha256(open(weights, 'rb').read()).hexdigest() if os.path.exists(weights) else None
    return SegFormerWrapper(net), dict(arch=arch, init=f'{repo}@{os.path.basename(path)} (Cityscapes)', init_sha256=sha,
                                       licence='NVIDIA Source Code License for SegFormer: research / non-commercial')


def load(checkpoint, arch='segformer_b2', device='cuda'):
    model, _ = build(arch, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


def load_run(run_dir, device='cuda'):
    import json
    arch = json.load(open(run_dir + '/run_record.json'))['args']['arch']
    return load(run_dir + '/selected.pt', arch, device)
