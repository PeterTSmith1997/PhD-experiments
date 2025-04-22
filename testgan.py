import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Filter dataset to only 0 and 1
class TwoClassMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        mnist = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        indices = (mnist.targets == 0) | (mnist.targets == 1)
        self.data = mnist.data[indices]
        self.targets = mnist.targets[indices]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        return self.transform(img), label.float()

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.label_embed = nn.Embedding(2, z_dim)
        self.model = nn.Sequential(
            nn.Linear(z_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embedding = self.label_embed(labels.long())
        x = torch.cat([z, label_embedding], dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(2, 28*28)
        self.model = nn.Sequential(
            nn.Linear(28*28 * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embedding = self.label_embed(labels.long())
        img_flat = img.view(img.size(0), -1)
        x = torch.cat([img_flat, label_embedding], dim=1)
        return self.model(x)

# Initialize
z_dim = 100
G = Generator(z_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002)
opt_D = optim.Adam(D.parameters(), lr=0.0002)

# Data
dataset = TwoClassMNIST()
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Training loop
epochs = 30
for epoch in range(epochs):
    for real_imgs, labels in loader:
        real_imgs, labels = real_imgs.to(device), labels.to(device)

        # Train Discriminator
        batch_size = real_imgs.size(0)
        z = torch.randn(batch_size, z_dim).to(device)
        fake_labels = torch.randint(0, 2, (batch_size,), dtype=torch.long).to(device)
        fake_imgs = G(z, fake_labels)

        real_validity = D(real_imgs, labels)
        fake_validity = D(fake_imgs.detach(), fake_labels)

        d_loss = criterion(real_validity, torch.ones_like(real_validity)) + \
                 criterion(fake_validity, torch.zeros_like(fake_validity))
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # Train Generator
        gen_validity = D(fake_imgs, fake_labels)
        g_loss = criterion(gen_validity, torch.ones_like(gen_validity))
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # Visualize and save outputs
    if (epoch+1) % 10 == 0:
        G.eval()
        with torch.no_grad():
            z = torch.randn(8, z_dim).to(device)
            sample_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).to(device)
            gen_imgs = G(z, sample_labels).cpu()
        
        fig, axes = plt.subplots(1, 8, figsize=(12, 2))
        for i in range(8):
            axes[i].imshow(gen_imgs[i][0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"Label {sample_labels[i].item()}")
        
        plt.tight_layout()
        plt.savefig(f"generated_digits_epoch_{epoch+1}.png")  # 🔥 Save the figure
        plt.show()
        G.train()
