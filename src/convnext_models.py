
import torch.nn as nn
import torchvision.models as models

def build_convnext_tiny(num_classes=5, pretrained=True):
    if pretrained:
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.convnext_tiny(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model
