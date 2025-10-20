import torch
import torch.nn as nn
from torchvision.models import resnet18

class VAE(nn.Module):
    """
    A simple convolutional VAE for images of shape (in_channels, img_size, img_size).
    Defaults are appropriate for CIFAR-10 (3x32x32).
    Forward returns: recon, mu, logvar
    """
    def __init__(self, in_channels=3, img_size=32, latent_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Encoder: downsample twice by stride=2 (32->16->8)
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # -> (64, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),          # -> (128, 8, 8)
            nn.ReLU(inplace=True),
        )

        # Infer flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            h = self.enc(dummy)
            self._enc_out_shape = h.shape[1:]          # (C, H, W)
            self._flat_dim = h.numel()                  # C*H*W

        # Latent parameters
        self.fc_mu = nn.Linear(self._flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, latent_dim)

        # Decoder: map z back to feature map, then upsample twice (8->16->32)
        self.fc_dec = nn.Linear(latent_dim, self._flat_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),       # (64, 8, 8)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),        # (32, 16, 16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # (C, 32, 32)
            nn.Sigmoid()  # outputs in [0,1] for BCE-style recon loss
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, *self._enc_out_shape)  # (B, 128, 8, 8)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def get_model(model_name, num_classes=10, in_channels=3, **kwargs):
    if model_name == "FCNN":
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * in_channels, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes)
        )
    elif model_name == "CNN":
        model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # -> (16, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> (16, 16, 16)
            nn.Flatten(),                                           # -> 16*16*16 = 4096
            nn.Linear(16 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)                             # logits for CE loss
        )
    elif model_name == "EnhancedCNN":
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        model = m
    elif model_name == "VAE":
        model = VAE(
            in_channels=kwargs.get("in_channels", in_channels),
            img_size=kwargs.get("img_size", 32),
            latent_dim=kwargs.get("latent_dim", 64),
        )
    else:
        raise ValueError("Unknown model name")

    return model