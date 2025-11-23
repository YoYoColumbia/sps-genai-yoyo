import os, io, math, base64, torch
from fastapi import FastAPI, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from helper_lib.model import Generator, get_model

# ---- config ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = os.getenv("GAN_GENERATOR_WEIGHTS", "generator.pth")
Z_DIM = 100
ENERGY_WEIGHTS_PATH = os.getenv("ENERGY_MODEL_WEIGHTS", "energy_cifar10.pth")
DIFFUSION_WEIGHTS_PATH = os.getenv("DIFFUSION_MODEL_WEIGHTS", "diffusion_cifar10.pth")

app = FastAPI(title="Module 6 API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# lazy singleton for the generator
_G = None          # GAN generator
_ENERGY = None     # Energy model
_DIFF = None       # Diffusion model

def _load_generator():
    global _G
    if _G is not None:
        return _G
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Generator weights not found at {WEIGHTS_PATH}")
    G = Generator(z_dim=Z_DIM).to(DEVICE)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    G.load_state_dict(state, strict=True)
    G.eval()
    _G = G
    return _G


def _load_energy_model():
    """
    Load the CIFAR-10 Energy Model
    """
    global _ENERGY
    if _ENERGY is not None:
        return _ENERGY
    if not os.path.exists(ENERGY_WEIGHTS_PATH):
        raise FileNotFoundError(f"Energy model weights not found at {ENERGY_WEIGHTS_PATH}")
    
    # Use EnhancedCNN as the energy model (10 CIFAR-10 classes, 3 channels)
    model = get_model("EnhancedCNN", num_classes=10, in_channels=3)
    state = torch.load(ENERGY_WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    _ENERGY = model
    return _ENERGY


def _load_diffusion_model():
    """
    Load the CIFAR-10 Diffusion Model
    """
    global _DIFF
    if _DIFF is not None:
        return _DIFF
    if not os.path.exists(DIFFUSION_WEIGHTS_PATH):
        raise FileNotFoundError(f"Diffusion model weights not found at {DIFFUSION_WEIGHTS_PATH}")
    
    model = get_model("Diffusion", in_channels=3)  
    state = torch.load(DIFFUSION_WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    _DIFF = model
    return _DIFF


def _sample_diffusion_grid(model, n: int, steps: int = 100,
                           img_size: int = 32, in_channels: int = 3):
    """
    Run reverse diffusion starting from Gaussian noise to get n samples.
    Uses the same linear beta schedule as in train_diffusion.
    Returns a single PIL image containing an n-image grid.
    """
    T = steps
    betas = torch.linspace(1e-4, 0.02, T, device=DEVICE)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Start from pure noise x_T ~ N(0, I)
    x_t = torch.randn(n, in_channels, img_size, img_size, device=DEVICE)

    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, device=DEVICE, dtype=torch.long)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]

        with torch.no_grad():
            eps_theta = model(x_t, t_batch)

        # DDPM mean step
        x_t = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
        )

        if t > 0:
            z = torch.randn_like(x_t)
            x_t = x_t + torch.sqrt(beta_t) * z

    # Map [-1, 1] -> [0, 1]
    imgs = (x_t + 1.0) / 2.0
    imgs = imgs.clamp(0, 1).cpu()

    nrow = int(math.sqrt(n)) or 1
    grid = make_grid(imgs, nrow=nrow, padding=2)
    pil = to_pil_image(grid)
    return pil


@app.get("/")
def root():
    return {"ok": True, "service": "module6-api", "device": DEVICE}

@app.get("/gan/health")
def gan_health():
    return {"ok": True, "weights_found": os.path.exists(WEIGHTS_PATH), "path": WEIGHTS_PATH}

@app.post("/gan/reload")
def gan_reload():
    global _G
    _G = None
    _load_generator()
    return {"reloaded": True}

@app.get("/gan/sample.png", response_class=Response)
def gan_sample_png(n: int = Query(16, ge=1, le=64), nrow: int | None = None, seed: int | None = None):
    G = _load_generator()
    if seed is not None:
        torch.manual_seed(seed)
    if nrow is None:
        nrow = int(math.sqrt(n)) or 1
    with torch.no_grad():
        z = torch.randn(n, Z_DIM, device=DEVICE)
        imgs = ((G(z) + 1) / 2).clamp(0, 1)
        grid = make_grid(imgs, nrow=nrow, padding=2)
        pil = to_pil_image(grid.cpu())
        buf = io.BytesIO(); pil.save(buf, format="PNG"); buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/gan/sample_b64")
def gan_sample_b64(n: int = Query(8, ge=1, le=32), seed: int | None = None):
    G = _load_generator()
    if seed is not None:
        torch.manual_seed(seed)
    outs = []
    with torch.no_grad():
        z = torch.randn(n, Z_DIM, device=DEVICE)
        imgs = ((G(z) + 1) / 2).clamp(0, 1).cpu()
        for i in range(n):
            pil = to_pil_image(imgs[i].squeeze(0))
            buf = io.BytesIO(); pil.save(buf, format="PNG")
            outs.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return JSONResponse({"images_b64": outs})

@app.get("/energy/health")
def energy_health():
    return {
        "ok": True,
        "weights_found": os.path.exists(ENERGY_WEIGHTS_PATH),
        "path": ENERGY_WEIGHTS_PATH,
    }

@app.post("/energy/reload")
def energy_reload():
    global _ENERGY
    _ENERGY = None
    _load_energy_model()
    return {"reloaded": True}

@app.get("/energy/random")
def energy_random(n: int = Query(8, ge=1, le=64)):
    model = _load_energy_model()
    with torch.no_grad():
        x = torch.randn(n, 3, 32, 32, device=DEVICE)
        logits = model(x)  # (n, 10)
        # Standard energy: -logsumexp over classes
        energies = -torch.logsumexp(logits, dim=1)
    return {"energies": energies.cpu().tolist()}

@app.get("/diffusion/health")
def diffusion_health():
    return {
        "ok": True,
        "weights_found": os.path.exists(DIFFUSION_WEIGHTS_PATH),
        "path": DIFFUSION_WEIGHTS_PATH,
    }


@app.post("/diffusion/reload")
def diffusion_reload():
    global _DIFF
    _DIFF = None
    _load_diffusion_model()
    return {"reloaded": True}


@app.get("/diffusion/sample.png", response_class=Response)
def diffusion_sample_png(
    n: int = Query(16, ge=1, le=64),
    steps: int = Query(100, ge=1, le=1000),
):
    """
    Generate n diffusion samples and return them as a PNG grid.
    """
    model = _load_diffusion_model()
    pil = _sample_diffusion_grid(model, n=n, steps=steps)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")