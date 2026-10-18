import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import Normalize
import timm
from typing import Dict


class ResizeWrapper(nn.Module):
    """
    Dynamically resizes 224x224 inputs to 256x256 for models like SwinV2
    that require a specific input resolution, preserving the strict 224x224 pipeline constraint.
    """
    def __init__(self, model: nn.Module, target_size=(256, 256)):
        super().__init__()
        self.model = model
        self.target_size = target_size

    def forward(self, x):
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return self.model(x)


class NormalizationWrapper(nn.Module):
    """
    CRITICAL: Applies ImageNet normalization internally.
    This guarantees the external generator operates entirely in un-normalised 
    pixel space [0, 1], ensuring ε = 10/255 is physically accurate.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    def forward(self, x):
        return self.model(self.normalize(x))


class ModelLoader:
    def __init__(self, device: torch.device):
        self.device = device

    def load_architecture_suite(self) -> Dict[str, nn.Module]:
        print("[*] Loading architectures with pinned weights...")

        # 1. Load CNNs
        raw_suite = {
            "VGG-16": models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
            "AlexNet": models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
            "ResNet-152": models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        }

        # 2. Load Transformers
        print("   [-] Loading Transformers (SwinV2, DeiT-III)...")
        swin = timm.create_model("swinv2_base_window12to16_192to256", pretrained=True)
        deit = timm.create_model("deit3_base_patch16_224", pretrained=True)

        # SwinV2 requires 256x256 inputs, so we wrap it
        raw_suite["Swin-V2"] = ResizeWrapper(swin, target_size=(256, 256))
        raw_suite["DeiT-III"] = deit

        # 3. Apply normalization and freeze weights
        wrapped_suite = {}
        for name, model in raw_suite.items():
            wrapped_model = NormalizationWrapper(model).to(self.device).eval()
            for param in wrapped_model.parameters():
                param.requires_grad = False
            wrapped_suite[name] = wrapped_model

        return wrapped_suite