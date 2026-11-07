"""MobileNetV3-Small lamp-state classifier and the DeepLabV3-MobileNetV3 segmentation baseline, both with ImageNet backbones."""
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

STATES = ('On', 'Dim', 'Off')


class LampClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features, self.pool = base.features, base.avgpool
        self.head = nn.Linear(base.classifier[0].in_features, len(STATES))

    def get_extra_state(self):
        return {'model': 'mobilenet_v3_small_image', 'labels': STATES}

    def set_extra_state(self, state):
        if tuple(state['labels']) != STATES:
            raise ValueError('checkpoint labels do not match')

    def forward(self, crops):
        return self.head(self.pool(self.features(crops)).flatten(1))


def deeplab_baseline(num_classes=4, pretrained=True):
    return deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None,
                                        num_classes=num_classes, aux_loss=False)
