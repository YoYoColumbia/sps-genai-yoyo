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
# TODO: run several iterations of the training loop (based on epochs parameter) and return the model
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