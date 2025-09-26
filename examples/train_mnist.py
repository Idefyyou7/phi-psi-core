import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

from phi_psi_core import PhiPsiNet, coherence_summary

# Dataset
transform = transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Model
model = PhiPsiNet(input_dim=28*28, output_dim=10, depth=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

gate_history = []

# Training loop
for epoch in range(1):
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs, gates = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            gs = coherence_summary(gates)
            gate_history.append(gs)
            print(f"Epoch {epoch}, Step {i}, Loss {loss.item():.4f}, Gates {gs}")

# Plot ψ-gates
df = pd.DataFrame(gate_history)
df.plot(figsize=(12,6), title="ψ-Gate Coherence During Training")
plt.xlabel("Checkpoint")
plt.ylabel("Gate Strength")
plt.show()
