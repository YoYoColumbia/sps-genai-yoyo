import math
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# ---------- VAE sampling ----------
@torch.no_grad()
def generate_samples_vae(model, device='cpu', num_samples=10, nrow=5, save_path=None, denorm=None):
    """
    Sample z ~ N(0, I), decode via VAE.decode(z), and plot.
    Assumes decoder outputs in [0,1] (e.g., Sigmoid). If you trained with normalization,
    either pass a denorm function or consider using MSE recon during training.
    """
    model.eval()
    model.to(device)

    latent_dim = getattr(model, "latent_dim", 64)
    z = torch.randn(num_samples, latent_dim, device=device)
    imgs = model.decode(z).clamp(0, 1).cpu()

    if denorm is not None:
        imgs = denorm(imgs).clamp(0, 1)

    grid = make_grid(imgs, nrow=nrow, normalize=False)
    plt.figure(figsize=(nrow, max(1, (num_samples + nrow - 1) // nrow)))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).numpy())
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved samples to {save_path}")
    plt.show()


# ---------- GAN sampling ----------
@torch.no_grad()
def generate_samples_gan(model, device='cpu', num_samples=16, nrow=None, save_path=None, seed=None):
    """
    Samples z ~ N(0, I), runs model.generator(z) -> [-1,1] (Tanh),
    maps to [0,1], and shows/saves a grid.

    Expects `model.generator` and `model.latent_dim`.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.to(device)
    model.eval()

    G = getattr(model, "generator", None)
    latent_dim = getattr(model, "latent_dim", None)
    if G is None or latent_dim is None:
        raise ValueError("Expected a GAN wrapper with .generator and .latent_dim attributes.")

    # Our Generator expects (B, latent_dim)
    z = torch.randn(num_samples, latent_dim, device=device)
    imgs = G(z).cpu()              # (N,1,28,28), range [-1,1]
    imgs = (imgs + 1) / 2          # -> [0,1]
    imgs = imgs.clamp(0, 1)

    # Auto nrow if not provided: closest square
    if nrow is None:
        nrow = int(math.sqrt(num_samples)) or 1

    grid = make_grid(imgs, nrow=nrow, padding=2, normalize=False)

    plt.figure(figsize=(nrow, max(1, num_samples // nrow)))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).numpy(), interpolation="nearest")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved samples to {save_path}")
    plt.show()


# ---------- Convenience router ----------
@torch.no_grad()
def generate_samples(model, device='cpu', num_samples=16, nrow=None, save_path=None, denorm=None, seed=None):
    """
    Detects model type and calls the right sampler:
    - VAE: model has .decode and returns recon via decode(z).
    - GAN: model has .generator and .latent_dim; generator outputs with Tanh -> [-1,1].
    """
    if hasattr(model, "decode"):  # VAE
        return generate_samples_vae(
            model, device=device, num_samples=num_samples,
            nrow=nrow, save_path=save_path, denorm=denorm, seed=seed
        )
    if hasattr(model, "generator") and hasattr(model, "latent_dim"):  # GAN
        return generate_samples_gan(
            model, device=device, num_samples=num_samples,
            nrow=nrow, save_path=save_path, seed=seed
        )
    raise ValueError("Unknown model type: expected a VAE (.decode) or GAN (.generator, .latent_dim).")
