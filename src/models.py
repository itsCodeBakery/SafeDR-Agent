
import torch
import torch.nn as nn
import torchvision.models as models

def build_efficientnet_b0(num_classes=5, pretrained=True):
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(in_features, num_classes)
    )

    return model
