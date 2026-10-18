import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class BaseObjective(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

class FFFObjective(BaseObjective):
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=features[0].device)
        for feat in features:
            mean_activation = torch.mean(torch.abs(feat)) + self.eps
            loss -= torch.log(mean_activation)
        return loss

class GDUAPObjective(BaseObjective):
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=features[0].device)
        for feat in features:
            l2_activation = torch.norm(feat.view(feat.size(0), -1), p=2, dim=1).mean()
            loss -= torch.log(l2_activation + self.eps)
        return loss

class CosineUAPObjective(BaseObjective):
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=features[0].device)
        for feat in features:
            flat = feat.view(feat.size(0), -1)
            norm_flat = F.normalize(flat, p=2, dim=1)
            cosine_sim = torch.mean(torch.mm(norm_flat, norm_flat.t()))
            loss += torch.log(cosine_sim.abs() + self.eps)
        return loss

class TRMUAPObjective(BaseObjective):
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=features[0].device)
        for feat in features:
            pos = torch.clamp(feat, min=0.0).mean()
            neg = torch.clamp(-feat, min=0.0).mean()
            ratio = (pos + self.eps) / (neg + self.eps)
            loss -= torch.log(ratio)
        return loss

class PSPUAPObjective(BaseObjective):
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=features[0].device)
        for feat in features:
            if len(feat.shape) == 4:
                b, c, h, w = feat.shape
                flat = feat.view(b * c, h * w)
                gram = torch.mm(flat, flat.t()) / (h * w)
                loss -= torch.log(torch.norm(gram, p='fro') + self.eps)
        return loss
