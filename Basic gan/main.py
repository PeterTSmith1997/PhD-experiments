from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

file = "../dataset/flowFeatures.csv"
label_column = 'Label'

# Load with strings to avoid mixed dtype issues
data = pd.read_csv(file, dtype=str, low_memory=False)

# Convert all columns except label to numeric
for col in data.columns:
    if col != label_column:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with NaN or Inf
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# Separate label and features
labels = data[label_column]
data = data.drop(columns=[label_column])

# Normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Convert features to PyTorch tensors
data_tensor = torch.tensor(scaled_data, dtype=torch.float32)

# Dataset contains only features for now
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



#gen
class Generator(nn.Module):
    def __init__(self, noise_dim=10, output_dim=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 16),
            nn.LeakyReLU(0.2, output_dim),
            nn.Sigmoid()
        )

        def forward(self, z):
            return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(0.2, 2),
            nn.Sigmoid()
        )

        def forward(self, x):
            return self.model(x)

loss_fn = nn.BCELoss()

generator = Generator()
discriminator = Discriminator()

lr = 0.0001
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

num_epochs = 100
noise = 1

for epoch in range(num_epochs):
    for batch in dataloader:
        real_data = batch[label_column]
        batch_size = real_data.size[0]

        discriminator.zero_grad()
        # Real data labels = 1
        real_labels = torch.ones(batch_size, 1)
        real_outputs = discriminator(real_data)
        d_loss_real = loss_fn(real_outputs, real_labels)

        # Fake data labels = 0
        noise = torch.randn(batch_size, noise)
        fake_data = generator(noise)
        fake_labels = torch.zeros(batch_size, 1)
        fake_outputs = discriminator(fake_data.detach())
        d_loss_fake = loss_fn(fake_outputs, fake_labels)

        # Combine losses
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        ### === Train Generator === ###

        generator.zero_grad()

        # Try to fool the discriminator: want fake to be classified as real (label=1)
        noise = torch.randn(batch_size, noise)
        fake_data = generator(noise)
        fake_outputs = discriminator(fake_data)
        g_loss = loss_fn(fake_outputs, torch.ones(batch_size, 1))  # Want the fake to be classified as real

        # Backprop and update generator
        g_loss.backward()
        g_optimizer.step()



    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss / len(dataloader):.4f}, G Loss: {g_loss / len(dataloader):.4f}")
