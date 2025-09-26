from phi_psi_core import PhiPsiNet, coherence_summary
import torch

model = PhiPsiNet(input_dim=256, output_dim=10, depth=5)
x = torch.randn(1, 256)
logits, gates = model(x)

print("logits:", logits)
print("gate summary:", coherence_summary(gates))
