"""Tests for GAN-based AI text detection model."""

import pytest
import torch
import torch.nn as nn

from app.gan_model import Discriminator, GANDetector, Generator


@pytest.fixture
def sample_features():
    """Sample feature tensor for testing."""
    # 8 features: 7 entropy + 1 RoBERTa
    return torch.randn(32, 8)  # batch_size=32, features=8


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return torch.randint(0, 2, (32,)).float()  # binary labels


class TestGenerator:
    """Test suite for the Generator network."""

    def test_generator_initialization(self):
        """Test that generator initializes correctly."""
        gen = Generator(input_dim=100, output_dim=8)
        assert gen is not None
        assert isinstance(gen, nn.Module)

    def test_generator_forward_pass(self):
        """Test generator produces correct output shape."""
        gen = Generator(input_dim=100, output_dim=8)
        noise = torch.randn(32, 100)  # batch_size=32, latent_dim=100

        output = gen(noise)

        assert output.shape == (32, 8)
        assert torch.isfinite(output).all()

    def test_generator_output_range(self):
        """Test that generator output is in reasonable range."""
        gen = Generator(input_dim=100, output_dim=8)
        noise = torch.randn(100, 100)

        output = gen(noise)

        # Features should be normalized (roughly between -3 and 3)
        assert output.min() > -5.0
        assert output.max() < 5.0


class TestDiscriminator:
    """Test suite for the Discriminator network."""

    def test_discriminator_initialization(self):
        """Test that discriminator initializes correctly."""
        disc = Discriminator(input_dim=8)
        assert disc is not None
        assert isinstance(disc, nn.Module)

    def test_discriminator_forward_pass(self, sample_features):
        """Test discriminator produces correct output shape."""
        disc = Discriminator(input_dim=8)

        output = disc(sample_features)

        assert output.shape == (32, 1)  # binary classification
        assert torch.isfinite(output).all()

    def test_discriminator_output_range(self, sample_features):
        """Test that discriminator output is probability."""
        disc = Discriminator(input_dim=8)

        output = disc(sample_features)

        # After sigmoid, should be between 0 and 1
        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_discriminator_gradient_flow(self, sample_features, sample_labels):
        """Test that gradients flow through discriminator."""
        disc = Discriminator(input_dim=8)
        criterion = nn.BCELoss()

        output = disc(sample_features)
        loss = criterion(output.squeeze(), sample_labels)
        loss.backward()

        # Check that gradients exist
        for param in disc.parameters():
            assert param.grad is not None


class TestGANDetector:
    """Test suite for the complete GAN-based detector."""

    def test_gan_detector_initialization(self):
        """Test that GAN detector initializes correctly."""
        detector = GANDetector(feature_dim=8, latent_dim=100)
        assert detector is not None
        assert detector.generator is not None
        assert detector.discriminator is not None

    def test_gan_detector_train_step(self, sample_features, sample_labels):
        """Test that training step executes without errors."""
        detector = GANDetector(feature_dim=8, latent_dim=100)

        losses = detector.train_step(sample_features, sample_labels)

        assert "discriminator_loss" in losses
        assert "generator_loss" in losses
        assert isinstance(losses["discriminator_loss"], float)
        assert isinstance(losses["generator_loss"], float)
        assert losses["discriminator_loss"] >= 0
        assert losses["generator_loss"] >= 0

    def test_gan_detector_predict(self, sample_features):
        """Test that detector can make predictions."""
        detector = GANDetector(feature_dim=8, latent_dim=100)

        predictions = detector.predict(sample_features)

        assert predictions.shape == (32,)
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()

    def test_gan_detector_save_load(self, tmp_path):
        """Test that detector can be saved and loaded."""
        detector = GANDetector(feature_dim=8, latent_dim=100)
        model_path = tmp_path / "gan_model.pt"

        # Save model
        detector.save(str(model_path))
        assert model_path.exists()

        # Load model
        loaded_detector = GANDetector(feature_dim=8, latent_dim=100)
        loaded_detector.load(str(model_path))

        # Test that loaded model works
        sample_input = torch.randn(10, 8)
        original_pred = detector.predict(sample_input)
        loaded_pred = loaded_detector.predict(sample_input)

        # Predictions should be identical
        torch.testing.assert_close(original_pred, loaded_pred)

    def test_gan_detector_convergence(self):
        """Test that GAN can learn simple pattern."""
        detector = GANDetector(feature_dim=8, latent_dim=100)

        # Create simple synthetic data: class 0 has mean -1, class 1 has mean +1
        n_samples = 100
        real_features_0 = torch.randn(n_samples, 8) - 1.0  # Human text features
        real_features_1 = torch.randn(n_samples, 8) + 1.0  # AI text features

        features = torch.cat([real_features_0, real_features_1], dim=0)
        labels = torch.cat([torch.zeros(n_samples), torch.ones(n_samples)])

        # Train for a few steps
        initial_loss: float = 0.0
        final_loss: float = 0.0

        for i in range(20):
            losses = detector.train_step(features, labels)
            if i == 0:
                initial_loss = losses["discriminator_loss"]
            final_loss = losses["discriminator_loss"]

        # Loss should decrease (model is learning)
        assert final_loss < initial_loss

    def test_gan_detector_feature_extraction(self, sample_features):
        """Test that detector can extract learned features."""
        detector = GANDetector(feature_dim=8, latent_dim=100)

        features = detector.extract_features(sample_features)

        # Should return features from an intermediate layer
        assert features.shape[0] == 32
        assert features.shape[1] > 0  # Has some feature dimension


@pytest.mark.parametrize("batch_size", [1, 16, 64])
def test_gan_detector_batch_sizes(batch_size):
    """Test that GAN detector handles different batch sizes."""
    detector = GANDetector(feature_dim=8, latent_dim=100)
    features = torch.randn(batch_size, 8)
    labels = torch.randint(0, 2, (batch_size,)).float()

    # Train step should work
    losses = detector.train_step(features, labels)
    assert losses["discriminator_loss"] >= 0

    # Predict should work
    predictions = detector.predict(features)
    assert predictions.shape == (batch_size,)


def test_gan_detector_gpu_compatibility():
    """Test that GAN detector can move to GPU if available."""
    detector = GANDetector(feature_dim=8, latent_dim=100)

    if torch.cuda.is_available():
        detector.to_device("cuda")
        sample_input = torch.randn(10, 8)

        # Should work on GPU
        predictions = detector.predict(sample_input)
        assert predictions.shape == (10,)


def test_gan_detector_evaluation_mode():
    """Test that detector properly switches between train and eval modes."""
    detector = GANDetector(feature_dim=8, latent_dim=100)

    # Should start in train mode
    assert detector.discriminator.training

    # Switch to eval
    detector.eval()
    assert not detector.discriminator.training

    # Switch back to train
    detector.train()
    assert detector.discriminator.training
