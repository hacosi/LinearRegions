import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset
from torchvision import datasets, transforms


def get_dataset(task_name, train_fraction=1.0, seed=42, noise=0.0):
    """Returns (train_dataset, test_dataset, task_info)."""
    rng = np.random.default_rng(seed)

    loaders = {
        "simple_classification": _simple_classification,
        "simple_regression": _simple_regression,
        "bullseye": _bullseye,
        "mnist": _mnist,
        "cifar10": _cifar10,
    }
    if task_name not in loaders:
        raise ValueError(f"Unknown task: {task_name}")
    return loaders[task_name](rng, train_fraction, seed, noise)


def _subsample(dataset, fraction, seed):
    if fraction >= 1.0:
        return dataset
    n = len(dataset)
    k = max(1, int(n * fraction))
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False).tolist()
    return Subset(dataset, indices)


# ---------------------------------------------------------------------------
# Synthetic tasks
# ---------------------------------------------------------------------------

def _simple_classification(rng, train_fraction, seed, noise=0.0):
    """Binary classification with two small Gaussian clusters."""
    n_train, n_test = 20, 10

    def _make(n, rng):
        half = n // 2
        x0 = rng.normal(loc=[-1, -1], scale=0.5, size=(half, 2))
        x1 = rng.normal(loc=[1, 1], scale=0.5, size=(n - half, 2))
        X = np.vstack([x0, x1]).astype(np.float32)
        y = np.array([0] * half + [1] * (n - half), dtype=np.int64)
        if noise > 0:
            X += rng.normal(0, noise, size=X.shape).astype(np.float32)
        return X, y

    X_tr, y_tr = _make(n_train, rng)
    X_te, y_te = _make(n_test, rng)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    train_ds = _subsample(train_ds, train_fraction, seed)

    task_info = {
        "type": "classification", "input_dim": 2, "output_dim": 2, "num_classes": 2,
        "input_range": [[-3.0, 3.0], [-3.0, 3.0]],
    }
    return train_ds, test_ds, task_info


def _simple_regression(rng, train_fraction, seed, noise=0.0):
    """1-D regression on y = sin(x) + noise."""
    n_train, n_test = 200, 50

    def _make(n, rng):
        X = rng.uniform(-3, 3, size=(n, 1)).astype(np.float32)
        y = (np.sin(X) + rng.normal(0, 0.1, size=(n, 1))).astype(np.float32)
        if noise > 0:
            X += rng.normal(0, noise, size=X.shape).astype(np.float32)
        return X, y

    X_tr, y_tr = _make(n_train, rng)
    X_te, y_te = _make(n_test, rng)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    train_ds = _subsample(train_ds, train_fraction, seed)

    task_info = {
        "type": "regression", "input_dim": 1, "output_dim": 1,
        "input_range": [[-3.0, 3.0]],
    }
    return train_ds, test_ds, task_info


def _bullseye(rng, train_fraction, seed, noise=0.0):
    """2-D concentric-circles classification.

    Class 0: inside the inner circle (r < r_inner) OR outside the outer circle
              (r >= r_outer).
    Class 1: in the ring between the two circles (r_inner <= r < r_outer).
    """
    n_train, n_test = 500, 200
    r_inner, r_outer, r_max = 1.0, 2.0, 3.0

    def _sample_annulus(n, r_min, r_max, rng):
        """Uniform sampling inside an annulus via inverse-CDF on r^2."""
        r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2, size=n))
        theta = rng.uniform(0, 2 * np.pi, size=n)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    def _make(n, rng):
        half = n // 2
        # Class 0: inner disk + outer ring
        n_inner = half // 2
        n_outer = half - n_inner
        inner = _sample_annulus(n_inner, 0, r_inner, rng)
        outer = _sample_annulus(n_outer, r_outer, r_max, rng)
        class0 = np.vstack([inner, outer])

        # Class 1: middle ring
        class1 = _sample_annulus(n - half, r_inner, r_outer, rng)

        X = np.vstack([class0, class1]).astype(np.float32)
        y = np.array([0] * half + [1] * (n - half), dtype=np.int64)

        if noise > 0:
            X += rng.normal(0, noise, size=X.shape).astype(np.float32)

        perm = rng.permutation(len(X))
        return X[perm], y[perm]

    X_tr, y_tr = _make(n_train, rng)
    X_te, y_te = _make(n_test, rng)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    train_ds = _subsample(train_ds, train_fraction, seed)

    task_info = {
        "type": "classification", "input_dim": 2, "output_dim": 2, "num_classes": 2,
        "input_range": [[-3.5, 3.5], [-3.5, 3.5]],
    }
    return train_ds, test_ds, task_info


# ---------------------------------------------------------------------------
# Standard benchmarks
# ---------------------------------------------------------------------------

def _mnist(rng, train_fraction, seed, noise=0.0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_ds = _subsample(train_ds, train_fraction, seed)

    task_info = {"type": "classification", "input_dim": 784, "output_dim": 10, "num_classes": 10}
    return train_ds, test_ds, task_info


def _cifar10(rng, train_fraction, seed, noise=0.0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    train_ds = _subsample(train_ds, train_fraction, seed)

    task_info = {"type": "classification", "input_dim": 3072, "output_dim": 10, "num_classes": 10}
    return train_ds, test_ds, task_info
