import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim = 256):
        super().__init__()
        self.in_channels = in_channels

        # ------------- Encoder -------------
        self.encoder = nn.Sequential(
            # Input: (B, in_channels, 64, 64)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),         # (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),                  # (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),                 # (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),                # (B, 256, 4, 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.encoder_out_dim = 256 * 4 * 4

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)                    # self.encoder_out_dim -> latent_dim
        self.unflatten = nn.Linear(latent_dim, self.encoder_out_dim)                # latent_dim -> self.encoder_out_dim

        # ------------- Decoder -------------
        self.decoder = nn.Sequential(
            # Input: (B, 256, 4, 4)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),       # (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),        # (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),         # (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1),    # (B, in_channel, 64, 64)
            nn.Sigmoid()
        )


    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.fc_mu(x)
        return z

    def decode(self, z):
        x = self.unflatten(z)
        x = x.view(-1, 256, 4, 4)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out
    

