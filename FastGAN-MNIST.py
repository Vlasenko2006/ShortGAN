
"""
Author: Andrey Vlasenko 

This software is an ilustrative example of how GAN neural network works. It perfectly 
fits for educational purposes.

It can be distributed under MIT general purpose license. The author of this code (me)
does not guarantee its functionality and possible damage of any kind to the potential
user. If you download it you take all possible risks from its usage on your own.

How to site:
Use this link to cite me: https://github.com/Vlasenko2006/FastGAN-MNIST


Advantages:

1. Easy to install
2. Fast converging.
3. You can run it on your home desctop and ge the first results in several minutes.
4. Works on CPU and GPU.

Disadvantages:

1. No waranties

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_generated_images(generator, epoch, device, examples=10):
    noise = torch.randn(examples, noise_dim).to(device)
    generated_images = generator(noise).cpu().detach()
    generated_images = generated_images.view(examples, 28, 28).numpy()
    
    plt.figure(figsize=(10, 1))
    for i in range(examples):
        plt.subplot(1, examples, i + 1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"gan3_generated_image_epoch_{epoch}.png")
    plt.show()

# Spectral Normalization Helper Function
def spectral_norm(layer):
    return nn.utils.spectral_norm(layer)


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU()
        )

        self.conv_block = nn.Sequential(
            nn.Unflatten(1, (1, 16, 16)),  # Adjust shape based on noise_dim
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (32, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 8, 8)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.fc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 512),  # Match dimensions
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()  # Better for [-1, 1] normalization
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.conv_block(x)
        x = self.fc2(x)
        return x.view(-1, 1, 28, 28)


# Define the Discriminator with Spectral Normalization
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            spectral_norm(nn.Linear(784, 256)),  # Apply spectral normalization
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(256, 128)),  # Apply spectral normalization
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Label Smoothing for Real Labels
def smooth_labels(labels, smoothing=0.1):
    return labels * (1 - smoothing) + 0.5 * smoothing


# Train the GAN
def train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion, dataloader, device, noise_dim, epochs=100, path="."):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as progress_bar:
            for real_images, _ in progress_bar:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)

                # Add Gaussian noise to real images
                noisy_real_images = real_images + 0.05 * torch.randn_like(real_images)

                # Generate fake images
                noise = torch.randn(batch_size, noise_dim).to(device)
                fake_images = generator(noise)

                # Train Discriminator
                real_labels = smooth_labels(torch.ones(batch_size, 1).to(device))
                fake_labels = torch.zeros(batch_size, 1).to(device)

                d_loss_real = criterion(discriminator(noisy_real_images), real_labels)
                d_loss_fake = criterion(discriminator(fake_images.detach()), fake_labels)
                d_loss = d_loss_real + d_loss_fake

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                # Train Generator (multiple times per epoch)
                for _ in range(2):  # Train the generator more frequently
                    noise = torch.randn(batch_size, noise_dim).to(device)
                    fake_images = generator(noise)
                    g_loss = criterion(discriminator(fake_images), real_labels)

                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()

                # Update progress bar with losses
                epoch_loss_d += d_loss.item()
                epoch_loss_g += g_loss.item()
                progress_bar.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

        # Log epoch-level progress
        print(f"Epoch {epoch+1}/{epochs} - D Loss: {epoch_loss_d/len(dataloader):.8f}, G Loss: {epoch_loss_g/len(dataloader):.4f}")

        # Save generated images and models
        if (epoch + 1) % 100 == 0:
            print(" ==== saving images ==== ")
            plot_generated_images(generator, epoch + 1, device)
            torch.save(generator.state_dict(), os.path.join(path, f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(path, f"discriminator_epoch_{epoch+1}.pth"))


def pretrain_generator(generator, dataloader, optimizer, criterion, device, noise_dim, epochs=20):
    generator.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        with tqdm(dataloader, desc=f"Pretraining Generator Epoch {epoch+1}/{epochs}", unit="batch") as progress_bar:
            for real_images, _ in progress_bar:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)

                # Generate random noise
                noise = torch.randn(batch_size, noise_dim).to(device)

                # Generate fake images
                fake_images = generator(noise)

                # Pretraining loss (Smooth L1 between fake and real images)
                loss = criterion(fake_images, real_images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(Loss=loss.item())

        print(f"Pretraining Generator Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.6f}")
      
# Main Function Update
if __name__ == "__main__":
    # Hyperparameters
    noise_dim = 128
    batch_size = 64
    learning_rate = 0.0002
    pretrain_epochs = 20  # Number of pre-training epochs
    gan_epochs = 30000
    path = "./gan_model/"
    os.makedirs(path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models
    generator = Generator(noise_dim).to(device)
    discriminator = Discriminator().to(device)

    # Optimizers and Loss
    g_pretrain_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    pretrain_criterion = nn.SmoothL1Loss()
    gan_criterion = nn.BCELoss()  # Loss for GAN training

    # DataLoader
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    # Uploads free MNIST dataset from web into ./data folder, if it is not on your computer. 
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Pretrain the generator
    print("Pretraining the generator...")
    pretrain_generator(generator, dataloader, g_pretrain_optimizer, pretrain_criterion, device, noise_dim, pretrain_epochs)

    # Train the GAN
    print("Starting GAN training...")
    train_gan(generator, discriminator, g_optimizer, d_optimizer, gan_criterion, dataloader, device, noise_dim, gan_epochs, path=path)

