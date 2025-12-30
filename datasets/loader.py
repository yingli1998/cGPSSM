"""============================================================================
Dataset loading functions.
============================================================================"""

"""============================================================================
Dataset loading functions.
============================================================================"""

from datasets.dataset import Dataset
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import RBFKernel
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

BASE_DIR = '/Users/lynn/Desktop/code/SDE-GPSSM/'

# -----------------------------------------------------------------------------

def load_dataset(rng, name, test_split=0):
    """Given a dataset string, returns data and possibly true generative
    parameters.
    """

    loaders = {
        "regular_toy_example": regular_toy_example,
        "irregular_toy_example": irregular_toy_example,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset name: {name}")
    return loaders[name](rng, test_split)


def generate_gp_observation(X, M, noise_std=0.05):
    X_torch = torch.tensor(X, dtype=torch.float32)
    N, D = X.shape
    Y = []
    kernel = RBFKernel()
    for _ in range(M):
        K = kernel(X_torch).evaluate() + noise_std**2 * torch.eye(N)
        dist = MultivariateNormal(torch.zeros(N), K)
        y = dist.sample()
        Y.append(y)
    Y = torch.stack(Y, dim=1)
    return Y.numpy()


# 不同的latent生成的轨迹
def circular_latent_trajectory(T=1.0, N=100, t=None, radius=1.0, omega=2*np.pi):
    if t is None:
        t = np.linspace(0, T, N)
    else:
        t = np.sort(t)  # ensure monotonic time

    x1 = radius * np.cos(omega * t)
    x2 = radius * np.sin(omega * t)
    X = np.stack([x1, x2], axis=1)
    return X, t

def sweeping_curve_latent_trajectory(T=1.0, N=100, t=None):
    if t is None:
        t = np.linspace(0, T, N)
    else:
        t = np.sort(t)
    x1 = t
    x2 = np.sin(2 * np.pi * t)
    X = np.stack([x1, x2], axis=1)
    return X, t

def spiral_latent_trajectory(T=1.0, N=100, t=None, n_loops=2):
    if t is None:
        t = np.linspace(0, T, N)
    else:
        t = np.sort(t)

    theta = 2 * np.pi * n_loops * t
    r = t  # radius increases linearly
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    X = np.stack([x1, x2], axis=1)
    return X, t

def lissajous_latent_trajectory(T=1.0, N=100, t=None, a=3, b=2, delta=np.pi/2):
    if t is None:
        t = np.linspace(0, T, N)
    else:
        t = np.sort(t)

    x1 = np.sin(a * 2 * np.pi * t + delta)
    x2 = np.sin(b * 2 * np.pi * t)
    X = np.stack([x1, x2], axis=1)
    return X, t


def regular_toy_example(rng, test_split):
    X, t = circular_latent_trajectory(T=1.0, N=1000)
    Y = generate_gp_observation(X, M=10)
    return Dataset(rng, "regular_circular_example", is_categorical=False, Y=Y, t=t, test_split=test_split, X=X)


def irregular_toy_example(rng, test_split):
    t_irregular = np.random.uniform(0, 1.0, size=1000)
    X, t = circular_latent_trajectory(t=t_irregular)
    Y = generate_gp_observation(X, M=10)
    return Dataset(rng, "irregular_circular_example", is_categorical=False, Y=Y, t=t, test_split=test_split, X=X)



if __name__ == "__main__":
    # 固定随机种子
    rng = np.random.default_rng(seed=42)

    # 加载不规则 toy 数据
    dataset = load_dataset(rng, name="regular_toy_example", test_split=0)

    # 画 latent trajectory
    plt.figure(figsize=(12, 4))
    for d in range(dataset.X.shape[1]):
        plt.plot(dataset.t, dataset.X[:, d], label=f'Latent dim {d}')
    plt.title("Latent Trajectories (Irregular Time)")
    plt.xlabel("Time")
    plt.ylabel("Latent State")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 画 observed trajectory
    plt.figure(figsize=(12, 4))
    for m in range(dataset.Y.shape[1]):
        plt.plot(dataset.t, dataset.Y[:, m], label=f'Observed dim {m}')
    plt.title("Observed GP Observations")
    plt.xlabel("Time")
    plt.ylabel("Observed Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()