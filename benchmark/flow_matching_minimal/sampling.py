import numpy as np
import scipy.spatial as spatial


def neg_sample_additive(x, noise_levels, seed=42):
    """Add Gaussian noise at multiple scales."""
    np.random.seed(seed)
    noisy = []
    for i, level in enumerate(noise_levels):
        noise = np.random.randn(*x.shape)
        noisy.append(x + noise * level)
    return np.vstack(noisy)


def compute_kernel(X, Y, sigma=1.0):
    D = spatial.distance.cdist(X, Y)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-D**2) / (2 * sigma**2))


def sampling_rejection(x, x_noisy, method="density", k=20, threshold=0.01):
    """
    Reject negative samples that stay too close to the data manifold.
    Returns a boolean mask (True = reject).
    """
    if method == "density":
        distances = spatial.distance.cdist(x_noisy, x)
        dist_closest = np.partition(distances, k, axis=1)[:, :k]
        dist_mean = np.mean(dist_closest, axis=1)
        return dist_mean <= threshold
    if method == "sugar":
        G_TN = compute_kernel(x, x_noisy)
        P_TN = G_TN / np.sum(G_TN, axis=0, keepdims=True)
        G_NT = G_TN.T
        P_NT = G_NT / np.sum(G_NT, axis=1, keepdims=True)
        x_noisy_bar = P_NT @ (P_TN @ x_noisy)
        change = np.linalg.norm(x_noisy - x_noisy_bar, axis=1)
        return change <= threshold
    raise ValueError(f"Invalid sampling rejection method: {method}")

