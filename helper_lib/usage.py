from data_loader import get_data_loader
from trainer import train_vae_model, vae_loss_function
from evaluator import evaluate_model
from model import get_model
from generator import generate_samples
from utils import save_model
import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = get_data_loader('data/train', batch_size=64)
test_loader = get_data_loader('data/test', batch_size=64, train=False)

vae = get_model("VAE")
criterion = vae_loss_function
optimizer = optim.Adam(vae.parameters(), lr=0.001)

train_vae_model(vae, train_loader, criterion, optimizer, device=device, epochs=5)
generate_samples(vae, device, num_samples=10)
