import torch
from vae import VAE
from train import train_vae

from torch.utils.data import DataLoader, TensorDataset

# Create dummy dataset of shape (1000, 5) for x in R^5
dummy_data = torch.randn(1000, 5)
dataset = TensorDataset(dummy_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the VAE with input_dim=5 and latent_dim=2
vae = VAE(input_dim=5, latent_dim=2, hidden_dim=16)

# Train the VAE
train_vae(vae, dataloader, epochs=10)
