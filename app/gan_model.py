"""
GAN-based AI text detection model.

This module implements a Generative Adversarial Network approach where:
- Generator: Creates synthetic AI-like text features
- Discriminator: Learns to distinguish between human and AI text features

The discriminator serves as the actual detection model, trained adversarially
to become robust against adversarial examples.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """
    Generator network that creates synthetic AI-like text features.

    Takes random noise and generates features that mimic AI-generated text,
    forcing the discriminator to learn more robust detection patterns.
    """

    def __init__(self, input_dim: int = 100, output_dim: int = 8, hidden_dim: int = 128):
        """
        Initialize generator.

        Args:
            input_dim: Dimension of random noise input (latent space)
            output_dim: Dimension of feature output (8 features)
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # LayerNorm works with batch_size=1
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # Normalize output to [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic features from noise.

        Args:
            z: Random noise tensor of shape (batch_size, input_dim)

        Returns:
            Generated features of shape (batch_size, output_dim)
        """
        return self.model(z)


class Discriminator(nn.Module):
    """
    Discriminator network that distinguishes between human and AI text features.

    This network serves as the actual detection model, learning robust features
    through adversarial training with the generator.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 128):
        """
        Initialize discriminator.

        Args:
            input_dim: Dimension of feature input (8 features)
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify features as human (0) or AI (1) text.

        Args:
            x: Feature tensor of shape (batch_size, input_dim)

        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        features = self.feature_extractor(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate features for analysis.

        Args:
            x: Feature tensor of shape (batch_size, input_dim)

        Returns:
            Feature tensor from intermediate layer
        """
        return self.feature_extractor(x)


class GANDetector:
    """
    Complete GAN-based AI text detection system.

    Combines generator and discriminator in an adversarial training setup
    to create a robust detection model.
    """

    def __init__(
        self,
        feature_dim: int = 8,
        latent_dim: int = 100,
        hidden_dim: int = 128,
        lr_generator: float = 0.0002,
        lr_discriminator: float = 0.0001,
        device: Optional[str] = None,
    ):
        """
        Initialize GAN detector.

        Args:
            feature_dim: Number of input features
            latent_dim: Dimension of generator's latent space
            hidden_dim: Hidden layer dimension for both networks
            lr_generator: Learning rate for generator
            lr_discriminator: Learning rate for discriminator
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing GAN detector on {self.device}")

        # Initialize networks
        self.generator = Generator(
            input_dim=latent_dim, output_dim=feature_dim, hidden_dim=hidden_dim
        ).to(self.device)

        self.discriminator = Discriminator(input_dim=feature_dim, hidden_dim=hidden_dim).to(
            self.device
        )

        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(), lr=lr_generator, betas=(0.5, 0.999)
        )

        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999)
        )

        # Loss function
        self.criterion = nn.BCELoss()

        self.latent_dim = latent_dim

    def train_step(
        self, real_features: torch.Tensor, real_labels: torch.Tensor, train_generator: bool = True
    ) -> Dict[str, float]:
        """
        Perform one training step of adversarial training.

        Args:
            real_features: Real feature tensor (batch_size, feature_dim)
            real_labels: Real labels (batch_size,) - 0 for human, 1 for AI
            train_generator: Whether to train generator this step

        Returns:
            Dictionary containing discriminator and generator losses
        """
        batch_size = real_features.size(0)
        real_features = real_features.to(self.device)
        real_labels = real_labels.to(self.device)

        # Labels for real and fake data
        real_label = torch.ones(batch_size, 1, device=self.device)
        fake_label = torch.zeros(batch_size, 1, device=self.device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        self.optimizer_d.zero_grad()

        # Real data loss
        output_real = self.discriminator(real_features)
        loss_real = self.criterion(output_real, real_labels.unsqueeze(1))

        # Fake data loss (generated features)
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_features = self.generator(z).detach()
        output_fake = self.discriminator(fake_features)
        loss_fake = self.criterion(output_fake, fake_label)

        # Combined discriminator loss
        loss_d = (loss_real + loss_fake) / 2
        loss_d.backward()
        self.optimizer_d.step()

        # ---------------------
        # Train Generator
        # ---------------------
        loss_g = torch.tensor(0.0)

        if train_generator:
            self.optimizer_g.zero_grad()

            # Generate fake features
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_features = self.generator(z)

            # Try to fool discriminator
            output_fake = self.discriminator(fake_features)
            loss_g = self.criterion(output_fake, real_label)

            loss_g.backward()
            self.optimizer_g.step()

        return {"discriminator_loss": loss_d.item(), "generator_loss": loss_g.item()}

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of AI-generated text.

        Args:
            features: Feature tensor (batch_size, feature_dim)

        Returns:
            Probability tensor (batch_size,) of AI class
        """
        self.discriminator.eval()

        with torch.no_grad():
            features = features.to(self.device)
            output = self.discriminator(features)

        # Handle both single and batch predictions
        # squeeze() with dim=-1 removes only the last dimension
        output = output.squeeze(-1).cpu()

        # Ensure we always return 1D tensor, even for single sample
        if output.dim() == 0:
            output = output.unsqueeze(0)

        return output

    def extract_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract learned features from discriminator.

        Args:
            features: Feature tensor (batch_size, feature_dim)

        Returns:
            Learned feature representation
        """
        self.discriminator.eval()

        with torch.no_grad():
            features = features.to(self.device)
            learned_features = self.discriminator.get_features(features)

        return learned_features.cpu()

    def save(self, path: str):
        """
        Save model state to file.

        Args:
            path: Path to save model
        """
        save_dict = {
            "generator_state": self.generator.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "optimizer_g_state": self.optimizer_g.state_dict(),
            "optimizer_d_state": self.optimizer_d.state_dict(),
        }

        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load model state from file.

        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)  # nosec B614

        self.generator.load_state_dict(checkpoint["generator_state"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state"])

        logger.info(f"Model loaded from {path}")

    def to_device(self, device: str):
        """
        Move model to specified device.

        Args:
            device: Target device ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

    def train(self):
        """Set models to training mode."""
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        """Set models to evaluation mode."""
        self.generator.eval()
        self.discriminator.eval()
