# import torch
# import torch.nn as nn
# from config import BenchmarkConfig
# from objectives import BaseObjective

# class FeatureHook:
#     def __init__(self):
#         self.features = []
#     def hook_fn(self, module, input, output):
#         self.features.append(output)
#     def clear(self):
#         self.features = []

# class AttackGenerator:
#     def __init__(self, surrogate: nn.Module, device: torch.device, config: BenchmarkConfig):
#         self.surrogate = surrogate
#         self.device = device
#         self.config = config
#         self.hook = FeatureHook()
#         self.handles = []
        
#         for module in self.surrogate.modules():
#             if isinstance(module, nn.Conv2d):
#                 self.handles.append(module.register_forward_hook(self.hook.hook_fn))

#     def generate(self, objective: BaseObjective, seed: int) -> torch.Tensor:
#         torch.manual_seed(seed)
#         shape = (1, 3, self.config.crop_size, self.config.crop_size)
        
#         delta = torch.zeros(shape, device=self.device)
#         canvas = torch.empty(shape, device=self.device).uniform_(0.0, 1.0)
        
#         for _ in range(self.config.iterations):
#             delta.requires_grad_(True)
#             self.hook.clear()
            
#             input_tensor = torch.clamp(canvas + delta, 0.0, 1.0)
#             _ = self.surrogate(input_tensor)
#             loss = objective(self.hook.features)
#             loss.backward()
            
#             with torch.no_grad():
#                 delta = delta - self.config.step_size * torch.sign(delta.grad)
#                 delta = torch.clamp(delta, -self.config.epsilon, self.config.epsilon)
                
#         return delta.detach()

#     def cleanup(self):
#         for handle in self.handles:
#             handle.remove()




import torch
import torch.nn as nn
from config import BenchmarkConfig
from objectives import BaseObjective

class FeatureHook:
    def __init__(self):
        self.features = []
        
    def hook_fn(self, module, input, output):
        self.features.append(output)
        
    def clear(self):
        self.features = []

class AttackGenerator:
    def __init__(self, surrogate: nn.Module, device: torch.device, config: BenchmarkConfig):
        self.surrogate = surrogate
        self.device = device
        self.config = config
        self.hook = FeatureHook()
        self.handles = []
        
    def _register_hooks(self):
        """Attaches hooks only when needed."""
        for module in self.surrogate.modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self.hook.hook_fn))
                
    def _remove_hooks(self):
        """Strips hooks off the model to prevent memory leaks during evaluation."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.hook.clear()

    def generate(self, objective: BaseObjective, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        
        # 1. Attach hooks right before generation
        self._register_hooks()
        
        shape = (1, 3, self.config.crop_size, self.config.crop_size)
        delta = torch.zeros(shape, device=self.device)
        canvas = torch.empty(shape, device=self.device).uniform_(0.0, 1.0)
        
        for _ in range(self.config.iterations):
            delta.requires_grad_(True)
            self.hook.clear()
            
            input_tensor = torch.clamp(canvas + delta, 0.0, 1.0)
            _ = self.surrogate(input_tensor)
            
            loss = objective(self.hook.features)
            loss.backward()
            
            with torch.no_grad():
                delta = delta - self.config.step_size * torch.sign(delta.grad)
                delta = torch.clamp(delta, -self.config.epsilon, self.config.epsilon)
                
        # 2. Clean up hooks to free VRAM for the evaluation phase
        self._remove_hooks()
        
        return delta.detach()

    def cleanup(self):
        self._remove_hooks()