import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary

class MLP_LoRA(nn.Module):
    def __init__(self, hidden_dim, alpha, rank, dropout=0.1):
        super().__init__()

        self.scale = alpha / rank

        self.LoRA_A = nn.Linear(hidden_dim, rank, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.LoRA_B = nn.Linear(rank, hidden_dim, bias=False)

        nn.init.kaiming_uniform_(self.LoRA_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.LoRA_B.weight)

    def forward(self, hidden_state):
        dynamic_state = self.LoRA_B(self.dropout(self.LoRA_A(hidden_state)))

        return self.scale * dynamic_state



class MoE_Gate(nn.Module):
    def __init__(self, hidden_dim, n_experts, top_k):
        super().__init__()

        self.gate_linear = nn.Linear(in_features=hidden_dim, out_features=n_experts, bias=False)
        self.noise_linear = nn.Linear(in_features=hidden_dim, out_features=n_experts, bias=False)

        self.top_k = top_k

    def forward(self, hidden_state):
        logits = self.gate_linear(hidden_state)

        if self.training:
            # randn = uniform. like original implementation.
            logits += torch.randn_like(logits) * F.softplus(self.noise_linear(hidden_state))

        expert_vals, idx = torch.topk(logits, self.top_k, dim=-1)

        expert_probs = F.softmax(expert_vals, dim=-1)

        expert_mask = torch.zeros_like(logits)
        expert_mask.scatter_(-1, idx, expert_probs)

        return expert_mask



class MoE_LoRA(nn.Module):
    def __init__(self, n_experts, top_k, base_mlp, hidden_dim, alpha, rank, dropout=0.1):
        super().__init__()

        self.base = base_mlp
        for p in self.base.parameters(): # freeze base MLP parameters.
            p.requires_grad = False

        self.experts = nn.ModuleList([MLP_LoRA(hidden_dim, alpha, rank, dropout) for _ in range(n_experts)])

        self.gate = MoE_Gate(hidden_dim, n_experts, top_k)

    def forward(self, hidden_state):
        static_state = self.base(hidden_state)

        dynamic_state = torch.zeros_like(hidden_state)

        moe_probs = self.gate(hidden_state)
        for idx in range(len(self.experts)):
            selected = moe_probs[..., idx] > 0 # representes the selected samples.
            print(selected.shape)
            
            if not selected.any(): continue

            # selecting samples for every expert causes sparsity over zeroing tokens (dense multiplication). 
            weights = moe_probs[..., idx][selected].unsqueeze(-1)
            dynamic_state[selected] += self.experts[idx](hidden_state[selected]) * weights

        return static_state + dynamic_state


model = MoE_LoRA(
    n_experts=8,
    top_k=2,
    base_mlp=nn.Sequential(
        nn.Linear(100, 256),
        nn.GELU(),
        nn.Linear(256, 100)
    ),
    hidden_dim=100,
    alpha=8,
    rank=16,
    dropout=0.1
)

summary(model, input_size=(16, 1, 64, 100))
