# main.py
import os, io, math, base64, torch
from fastapi import FastAPI, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from helper_lib.model import Generator

# ---- config ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = os.getenv("GAN_GENERATOR_WEIGHTS", "generator.pth")
Z_DIM = 100

app = FastAPI(title="Module 6 API", version="1.0")

# (optional) CORS if you’ll call from a browser frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# lazy singleton for the generator
_G = None
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
