import torch
import torch.nn as nn
import torch.nn.functional as F

phi = (1 + 5 ** 0.5) / 2  # golden ratio

class PsiGate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, context):
        phase = torch.cos((x * context).sum(dim=-1, keepdim=True))
        gate = torch.sigmoid(phase)
        return self.linear(x) * gate

class PhiPsiNet(nn.Module):
    def __init__(self, input_dim, output_dim, depth=5):
        super().__init__()
        dims = [input_dim]
        for _ in range(depth):
            dims.append(int(dims[-1] / phi))

        self.layers = nn.ModuleList([
            nn.Linear(int(dims[i]), int(dims[i+1]))
            for i in range(len(dims)-1)
        ])
        self.output = nn.Linear(int(dims[-1]), output_dim)

    def forward(self, x):
        gates = []
        for layer in self.layers:
            context = torch.randn_like(x)
            gated = PsiGate(x.size(-1), layer.out_features)(x, context)
            x = F.relu(layer(x) + gated)
            gates.append(gated.mean().item())
        return self.output(x), gates

def coherence_summary(gates):
    return {f"layer_{i}": g for i, g in enumerate(gates)}
