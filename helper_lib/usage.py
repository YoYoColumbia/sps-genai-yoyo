from data_loader import get_data_loader
from trainer import (
    train_vae_model,
    vae_loss_function,
    train_gan,
    train_model,
    train_diffusion,
)
from evaluator import evaluate_model
from model import get_model
from generator import generate_samples
from utils import save_model
import torch
import torch.nn as nn
import torch.optim as optim

model_name = "Energy" 
device = "cuda" if torch.cuda.is_available() else "cpu"

if model_name == "GAN":
    train_loader = get_data_loader('data', batch_size=64, train=True,  dataset="MNIST")
    test_loader  = get_data_loader('data', batch_size=64, train=False, dataset="MNIST")
else:
    train_loader = get_data_loader('data', batch_size=64, train=True,  dataset="CIFAR10")
    test_loader  = get_data_loader('data', batch_size=64, train=False, dataset="CIFAR10")


if model_name == "VAE":
    print("Training VAE...")
    model = get_model("VAE", in_channels=3, img_size=32, latent_dim=64)

    criterion = vae_loss_function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_vae_model(model, train_loader, criterion, optimizer,
                    device=device, epochs=5)

    save_model(model, path="vae.pth")
    generate_samples(model, device, num_samples=10)

elif model_name == "GAN":
    print("Training GAN...")
    model = get_model("GAN")

    criterion = nn.BCEWithLogitsLoss()

    optimizer_G = optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    train_gan(model, train_loader, criterion, optimizer_G, optimizer_D,
              device=device, epochs=10)

    save_model(model.generator, path="generator.pth")
    save_model(model.discriminator, path="discriminator.pth")

    generate_samples(model, device, num_samples=16, nrow=4)

elif model_name == "Energy":
    print("Training Energy Model (EnhancedCNN on CIFAR-10)...")
    model = get_model("EnhancedCNN", num_classes=10, in_channels=3)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader, criterion, optimizer,
                device=device, epochs=10)

    save_model(model, path="energy_cifar10.pth")
    print("Saved energy_cifar10.pth")

elif model_name == "Diffusion":
    print("Training Diffusion Model on CIFAR-10...")
    model = get_model("Diffusion", in_channels=3)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_diffusion(model, train_loader, criterion, optimizer,
                    device=device, epochs=10)

    save_model(model, path="diffusion_cifar10.pth")
    print("Saved diffusion_cifar10.pth")

else:
    raise ValueError("model_name must be 'VAE', 'GAN', 'Energy', or 'Diffusion'")

