import torch


def make_offmanifold(encoder, discriminator, disc_factor=5.0):
    """
    Build a lightweight off-manifold embedding that augments latent codes with a
    discriminator-driven distance. Higher discriminator confidence on fake points
    increases the added dimension, pushing the curve off the data manifold.
    """

    def _fn(x):
        z = encoder(x)
        penalty = torch.exp(disc_factor * (1 - discriminator.positive_prob(z)))
        return torch.cat([z, penalty.unsqueeze(1)], dim=1)

    return _fn

