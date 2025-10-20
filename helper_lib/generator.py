import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

@torch.no_grad()
def generate_samples(model, device='cpu', num_samples=10, nrow=5, save_path=None, denorm=None):
    """
    Generate images by sampling from the VAE latent space.

    Args:
        model: Trained VAE model with .decode(z)
        device: 'cpu' or 'cuda'
        num_samples: how many images to generate
        nrow: images per row in the grid
        save_path: optional filepath to save the plotted grid
        denorm: optional function to reverse normalization before displaying
    """
    model.eval()
    model.to(device)

    # infer latent dimension
    latent_dim = getattr(model, "latent_dim", 64)
    z = torch.randn(num_samples, latent_dim, device=device)
    imgs = model.decode(z).clamp(0, 1).cpu()  # assume sigmoid output in [0,1]

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
