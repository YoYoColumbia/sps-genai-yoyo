import torch
import torch.nn.functional as F
from tqdm import tqdm

def vae_loss_function(recon_x, x, mu, logvar):
    beta = 500
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * BCE + KLD

def train_model(model, data_loader, criterion, optimizer, device='cpu',
epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(data_loader):.4f}")
    return model

def train_vae_model(model, data_loader, criterion, optimizer, device, epochs=10):
    """
    Expects model(x) -> (recon, mu, logvar)
    Expects criterion(recon, x, mu, logvar) -> scalar loss
    """
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(data_loader, desc=f"[VAE] Epoch {epoch+1}/{epochs}"):
            # handle (x, y) or just x
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                x = batch[0]
            else:
                x = batch

            x = x.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(x)
            loss = criterion(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg = running_loss / len(data_loader)
        print(f"[VAE] Epoch {epoch+1}/{epochs} | Loss: {avg:.4f}")
    return model


def train_gan(model, data_loader, criterion, optimizer_G, optimizer_D, device='cpu', epochs=10):
    """
    Train a simple GAN consisting of model.generator and model.discriminator.
    - Use BCEWithLogitsLoss for `criterion` (since D returns logits).
    - Generator input noise shape: (batch_size, latent_dim)
    - Discriminator output shape: (batch_size, 1)
    """
    model.to(device)
    G, D = model.generator, model.discriminator
    latent_dim = model.latent_dim

    for epoch in range(epochs):
        G.train(); D.train()
        running_G, running_D = 0.0, 0.0

        pbar = tqdm(data_loader, desc=f"[GAN] Epoch {epoch+1}/{epochs}")
        for real, _ in pbar:
            real = real.to(device)
            bsz = real.size(0)

            # Labels shaped to match D output (B,1)
            real_labels = torch.ones(bsz, 1, device=device)
            fake_labels = torch.zeros(bsz, 1, device=device)

            # ----- Train Discriminator -----
            z = torch.randn(bsz, latent_dim, device=device)
            with torch.no_grad():
                fake = G(z)

            D_real = D(real)
            D_fake = D(fake)

            loss_real = criterion(D_real, real_labels)
            loss_fake = criterion(D_fake, fake_labels)
            loss_D = (loss_real + loss_fake) / 2

            optimizer_D.zero_grad(set_to_none=True)
            loss_D.backward()
            optimizer_D.step()

            # ----- Train Generator -----
            z = torch.randn(bsz, latent_dim, device=device)
            gen = G(z)
            D_gen = D(gen)
            loss_G = criterion(D_gen, real_labels)

            optimizer_G.zero_grad(set_to_none=True)
            loss_G.backward()
            optimizer_G.step()

            running_D += loss_D.item()
            running_G += loss_G.item()

            pbar.set_postfix(D=f"{loss_D.item():.4f}", G=f"{loss_G.item():.4f}")

        print(f"[GAN] Epoch {epoch+1}/{epochs} | D_loss: {running_D/len(data_loader):.4f} | G_loss: {running_G/len(data_loader):.4f}")

    return model