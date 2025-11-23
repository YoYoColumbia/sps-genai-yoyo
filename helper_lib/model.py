import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# Define simple GAN architectures
class Generator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 7 * 7 * 128),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z):
        x = self.fc(z)                       # (B, 7*7*128)
        x = x.view(-1, 128, 7, 7)           # (B, 128, 7, 7)
        x = self.deconv(x)                  # (B, 1, 28, 28)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        h = self.features(x)
        h = h.view(x.size(0), -1)
        logits = self.classifier(h)  # (B,1)
        return logits
    

class GAN(nn.Module):
    """Wrapper for Generator and Discriminator"""
    def __init__(self, latent_dim=100):
        super().__init__()
        self.generator = Generator(z_dim=latent_dim)
        self.discriminator = Discriminator()
        self.latent_dim = latent_dim

# Define simple VAE architectures
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


# Diffusion model components
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        exponents = torch.exp(torch.arange(half_dim, device=t.device) * -emb_factor)
        emb = t[:, None].float() * exponents[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        time_out = self.time_proj(t_emb)[:, :, None, None]
        h = h + time_out
        h = self.act(h)
        h = self.conv2(h)
        h = self.act(h)
        return h + self.skip(x)


class SimpleUNetDiffusion(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, time_emb_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        self.enc1 = ResidualBlock(in_channels, base_channels, time_emb_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)

        self.enc2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.mid = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualBlock(base_channels + in_channels, base_channels, time_emb_dim)

        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        h1 = self.enc1(x, t_emb)
        h = self.down1(h1)

        h2 = self.enc2(h, t_emb)
        h = self.down2(h2)

        h = self.mid(h, t_emb)

        h = self.up1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.dec1(h, t_emb)

        h = self.up2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec2(h, t_emb)

        out = self.out_conv(h)
        return out


class DiffusionModel(nn.Module):
    """
    Wrapper for the diffusion ε-predictor network.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64, time_emb_dim: int = 128):
        super().__init__()
        self.net = SimpleUNetDiffusion(
            in_channels=in_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x_t, t)


def get_model(model_name, num_classes: int | None = None, in_channels: int | None = None, **kwargs):
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
    elif model_name == "GAN":
        model = GAN(latent_dim=kwargs.get("latent_dim", 100))
    elif model_name == "Diffusion":
        model = DiffusionModel(
            in_channels=kwargs.get("in_channels", in_channels if in_channels is not None else 3),
            base_channels=kwargs.get("base_channels", 64),
            time_emb_dim=kwargs.get("time_emb_dim", 128),
        )
    else:
        raise ValueError("Unknown model name")

    return model