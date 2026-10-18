import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import BenchmarkConfig

class AdaptUAPGenerator:
    """
    Generates the ADAPT-UAP perturbation using the 4 proposed components.
    Includes forward hooks to capture real feature maps for the TRM objective.
    """
    def __init__(self, surrogate: nn.Module, device: torch.device, config: BenchmarkConfig):
        self.model = surrogate
        self.device = device
        self.config = config
        self.ablation = config.ablation
        
        self.layer_weights = {}
        self.activations = {}
        self.hooks = []
        
        # Identify layers to hook (Conv2d) to extract intermediate feature maps
        self.target_layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.target_layers[name] = module

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output
            return hook

        for name, layer in self.target_layers.items():
            self.hooks.append(layer.register_forward_hook(get_activation(name)))

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()

    def _component_a_spectral_analysis(self):
        """Component A: Condition-number-weighted layer targeting."""
        if not self.ablation.use_component_a:
            # Fallback to uniform weighting
            weight_val = 1.0 / len(self.target_layers) if len(self.target_layers) > 0 else 1.0
            for name in self.target_layers.keys():
                self.layer_weights[name] = weight_val
            return 
            
        print("      [-] Running Component A: One-time SVD analysis...")
        log_k_sums = 0.0
        
        for name, module in self.target_layers.items():
            weight_matrix = module.weight.data.view(module.weight.size(0), -1)
            try:
                # Compute singular values
                _, S, _ = torch.svd(weight_matrix, compute_uv=False)
                k_l = (S[0] / (S[-1] + 1e-6)).item()
            except RuntimeError:
                k_l = 1.0 # Fallback if SVD fails to converge
                
            w_l = np.log(1 + k_l)
            self.layer_weights[name] = w_l
            log_k_sums += w_l
                
        # Normalize weights
        for name in self.layer_weights:
            self.layer_weights[name] /= (log_k_sums + 1e-8)

    def _component_c_multi_scale_priors(self, canvas: torch.Tensor, delta: torch.Tensor) -> list:
        """Component C: Multi-Scale Pseudo-Semantic Priors."""
        v = torch.clamp(canvas + delta, 0.0, 1.0)
        if not self.ablation.use_component_c:
            return [v]
            
        # Generates priors at 224x224, 64x64 (tiled), and 32x32 (tiled)
        v_64 = F.interpolate(v, size=(64, 64), mode='bilinear', align_corners=False)
        v_64_tiled = v_64.repeat(1, 1, 4, 4)[:, :, :224, :224]
        
        v_32 = F.interpolate(v, size=(32, 32), mode='bilinear', align_corners=False)
        v_32_tiled = v_32.repeat(1, 1, 7, 7)[:, :, :224, :224]
        
        return [v, v_64_tiled, v_32_tiled]

    def _component_d_budget_schedule(self, iteration: int, total_iterations: int) -> float:
        """Component D: Progressive Budget Scheduling (2 -> 4 -> 8 -> 10)."""
        if not self.ablation.use_component_d:
            return self.config.epsilon
            
        progress = iteration / max(1, total_iterations - 1)
        if progress < 0.25: return 2.0 / 255.0
        if progress < 0.50: return 4.0 / 255.0
        if progress < 0.75: return 8.0 / 255.0
        return self.config.epsilon

    def _compute_trm_loss(self, iteration: int, total_iterations: int) -> torch.Tensor:
        """Component B: Truncated Ratio Maximization loss scaled by Component A."""
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Schedule truncation curriculum k from 30% down to 5%
        progress = iteration / max(1, total_iterations - 1)
        k_ratio = 0.30 - (0.25 * progress)
        
        for name, activation in self.activations.items():
            A_flat = activation.view(-1)
            num_elements = A_flat.size(0)
            k_elements = max(1, int(k_ratio * num_elements))
            
            if not self.ablation.use_component_b:
                # Fallback to standard activation maximization
                layer_loss = torch.log(torch.norm(A_flat) + 1e-6)
            else:
                top_k_pos, _ = torch.topk(A_flat, k_elements, largest=True)
                top_k_neg, _ = torch.topk(A_flat, k_elements, largest=False)
                
                mean_pos = top_k_pos.mean()
                mean_neg_abs = top_k_neg.abs().mean()
                
                layer_loss = torch.log((mean_pos / (mean_neg_abs + 1e-6)) + 1e-6)
                
            weight = self.layer_weights.get(name, 1.0)
            total_loss += weight * layer_loss
            
        return total_loss

    def generate(self, seed: int) -> torch.Tensor:
        """Main training loop for the ADAPT-UAP perturbation."""
        torch.manual_seed(seed)
        self._component_a_spectral_analysis()
        
        shape = (1, 3, self.config.crop_size, self.config.crop_size)
        delta = torch.zeros(shape, device=self.device, requires_grad=True)
        canvas = torch.empty(shape, device=self.device).uniform_(0.0, 1.0)
        
        self._register_hooks()

        for i in range(self.config.iterations):
            current_eps = self._component_d_budget_schedule(i, self.config.iterations)
            priors = self._component_c_multi_scale_priors(canvas, delta)
            
            if delta.grad is not None:
                delta.grad.zero_()
                
            iteration_loss = torch.tensor(0.0, device=self.device)
            
            for prior in priors:
                self.activations.clear()
                _ = self.model(prior)
                
                # Maximize TRM ratio -> minimize negative loss
                scale_loss = self._compute_trm_loss(i, self.config.iterations)
                iteration_loss -= scale_loss 

            iteration_loss = iteration_loss / len(priors)
            iteration_loss.backward()
            
            with torch.no_grad():
                delta = delta - self.config.step_size * torch.sign(delta.grad)
                delta = torch.clamp(delta, -current_eps, current_eps)
            delta.requires_grad_(True)
            
        self._remove_hooks()
        return delta.detach()