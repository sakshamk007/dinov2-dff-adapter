"""DFF-Adapter: frozen DINOv2-with-registers plus routed low-rank adapters."""
from contextlib import contextmanager
import math

import torch
from torch import nn
from transformers import AutoModel


class RoutedLowRankAdapter(nn.Module):
    """Paper-style task-specific top-k routed LoRA plus a shared LoRA branch."""
    def __init__(self, hidden_size, rank=16, alpha=32, num_experts=6, num_heads=4, top_k=3):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if rank < num_heads or rank % num_heads:
            raise ValueError("rank must be divisible by num_heads")
        self.num_heads, self.num_experts = num_heads, num_experts
        self.top_k = min(top_k, num_experts)
        self.width, self.head_rank = hidden_size // num_heads, rank // num_heads
        self.scale = alpha / rank
        # A/B are shared expert pools; the two task branches only have separate routers.
        self.A = nn.Parameter(torch.empty(num_experts, self.width, self.head_rank))
        self.B = nn.Parameter(torch.zeros(num_experts, self.head_rank, self.width))
        self.task_router = nn.Parameter(torch.zeros(2, num_heads, num_experts))
        self.shared_router = nn.Parameter(torch.zeros(num_experts))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.normal_(self.task_router, std=0.02)
        nn.init.normal_(self.shared_router, std=0.02)

    def _expert_update(self, x, weights):
        # x: [B,L,H,W], weights: [H,E] or [E]; output: [B,L,H,W]
        low = torch.einsum("blhw,ewr->blher", x, self.A)
        updates = torch.einsum("blher,erw->blhew", low, self.B)
        if weights.ndim == 1:
            return torch.einsum("e,blhew->blhw", weights, updates)
        return torch.einsum("he,blhew->blhw", weights, updates)

    def forward(self, x, task_id: int):
        original_shape = x.shape
        x = x.reshape(*x.shape[:-1], self.num_heads, self.width)
        task_weights = torch.softmax(self.task_router[task_id], dim=-1)
        values, indices = torch.topk(task_weights, self.top_k, dim=-1)
        sparse = torch.zeros_like(task_weights).scatter_(-1, indices, values)
        sparse = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-8)
        # Shared branch intentionally uses every expert, as in Eq. 5 of the paper.
        shared = torch.softmax(self.shared_router, dim=-1)
        update = self._expert_update(x, sparse) + self._expert_update(x, shared)
        return (update * self.scale).reshape(original_shape)


class AdapterLinear(nn.Module):
    """Wrap one frozen projection and add an adapter update selected by model task."""
    def __init__(self, linear: nn.Linear, adapter: RoutedLowRankAdapter, task_getter):
        super().__init__()
        self.base = linear
        self.adapter = adapter
        self.task_getter = task_getter
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.adapter(x, self.task_getter())


class DFFAdapterDetector(nn.Module):
    def __init__(self, model_name, num_forgery_types, adapter_cfg):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        self._task_id = 0
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._inject_adapters(adapter_cfg)
        self.binary_head = nn.Sequential(nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, 1))
        self.type_head = nn.Sequential(nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, num_forgery_types))

    def _inject_adapters(self, cfg):
        # Verified against current Hugging Face Dinov2WithRegistersModel module names.
        count = 0
        for layer in self.backbone.encoder.layer:
            for parent, name in ((layer.attention.attention, "query"),
                                 (layer.attention.attention, "value"),
                                 (layer.attention.output, "dense")):
                linear = getattr(parent, name)
                setattr(parent, name, AdapterLinear(linear, RoutedLowRankAdapter(
                    self.hidden_size, **cfg), lambda: self._task_id))
                count += 1
        if count == 0:
            raise RuntimeError("No DINOv2 projections were wrapped; check transformers version.")

    @contextmanager
    def task(self, task_id):
        old = self._task_id
        self._task_id = task_id
        try:
            yield
        finally:
            self._task_id = old

    def forward(self, pixel_values, task_id=0):
        with self.task(task_id):
            cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.binary_head(cls).squeeze(-1) if task_id == 0 else self.type_head(cls)

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
