import os
from pathlib import Path
import torch
from torch import nn, optim
from tqdm import tqdm

from .dataset import get_dataloaders
from .model import ConvAutoencoder

def train(
        latent_dim: int=1024,
        batch_size: int=32,
        lr: float=1e-3,
        num_epochs: int=200,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        out_dir: str="models"
):
    
    train_dl, val_dl, test_dl, x_unshuffled, texts_unshuffled = get_dataloaders(batch_size=batch_size, image_size=64, augment=True)
    model = ConvAutoencoder(in_channels=3, latent_dim=latent_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt = out_path/ f"autoencoder_latent{latent_dim}.pt"

    for epoch in range(num_epochs):


        # ===== TRAINING =====
        model.train()
        running_train_loss = 0.0

        for batch in train_dl:
            images = batch.to(device)
            recon = model(images)
            loss = criterion(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_dl.dataset)

        # ===== VALIDATION =====
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in val_dl:
                images = batch.to(device)
                recon = model(images)
                loss = criterion(recon, images)
                running_val_loss += loss.item() * images.size(0)

        epoch_val_loss = running_val_loss / len(val_dl.dataset)


    # ===== SAVE =====

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "latent_dim": latent_dim
            },
            f"model/autoencoder_latent{latent_dim}.pt"
        )



print("Training Complete.")