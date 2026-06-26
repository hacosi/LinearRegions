import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_activation_patterns(model, inputs):
    """Forward pass that records the binary activation pattern at each ReLU.

    Returns a (N, total_relu_units) boolean tensor, or None if the model
    has no hidden layers.
    """
    patterns = []
    x = inputs.view(inputs.size(0), -1)

    # Depth-0 model is just a single Linear — no ReLUs, one linear region.
    if isinstance(model.network, nn.Linear):
        return None

    for layer in model.network:
        if isinstance(layer, nn.Linear):
            x = layer(x)
        elif isinstance(layer, nn.ReLU):
            patterns.append(x > 0)
            x = torch.relu(x)

    if not patterns:
        return None

    return torch.cat(patterns, dim=1)


def _rms_std_from_dataset(train_dataset, batch_size=512):
    """RMS of per-feature standard deviations across the training set.

    Uses a single batched pass to avoid an O(N) Python loop.
    """
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    feat_sum = None
    feat_sq_sum = None
    n_total = 0
    for X, *_ in loader:
        X = X.view(X.size(0), -1).float()
        if feat_sum is None:
            feat_sum = X.sum(dim=0)
            feat_sq_sum = X.pow(2).sum(dim=0)
        else:
            feat_sum += X.sum(dim=0)
            feat_sq_sum += X.pow(2).sum(dim=0)
        n_total += X.size(0)
    mean = feat_sum / n_total
    var = (feat_sq_sum / n_total - mean.pow(2)).clamp(min=0)
    return var.mean().sqrt().item()  # sqrt(mean(sigma_i^2))


def _count_unique_packed(patterns_np):
    """Pack a 2-D bool/uint8 array into bytes and count unique rows."""
    packed = np.packbits(patterns_np.astype(np.uint8), axis=1)
    return len(np.unique(packed, axis=0))


# ---------------------------------------------------------------------------
# Method 1: grid  (1-D / 2-D tasks)
# ---------------------------------------------------------------------------

def _build_grid(input_range, n_points):
    """Return a (N, D) tensor of grid points over the input space."""
    dim = len(input_range)

    if dim == 1:
        lo, hi = input_range[0]
        return torch.linspace(lo, hi, n_points).unsqueeze(1)

    if dim == 2:
        per_dim = int(np.sqrt(n_points))
        lo0, hi0 = input_range[0]
        lo1, hi1 = input_range[1]
        g0, g1 = torch.meshgrid(
            torch.linspace(lo0, hi0, per_dim),
            torch.linspace(lo1, hi1, per_dim),
            indexing="ij",
        )
        return torch.stack([g0.flatten(), g1.flatten()], dim=1)

    raise ValueError(f"Grid sampling not supported for {dim}-D input")


def count_linear_regions_grid(model, task_info, n_points, device,
                              batch_size=4096):
    """Count unique activation patterns over a uniform grid."""
    input_range = task_info.get("input_range")
    if input_range is None:
        return None

    grid = _build_grid(input_range, n_points).to(device)

    model.eval()
    all_patterns = []

    with torch.no_grad():
        for i in range(0, len(grid), batch_size):
            batch = grid[i : i + batch_size]
            patterns = get_activation_patterns(model, batch)
            if patterns is None:
                return 1
            all_patterns.append(patterns.cpu())

    all_patterns = torch.cat(all_patterns, dim=0)
    return _count_unique_packed(all_patterns.numpy())


# ---------------------------------------------------------------------------
# Method 2: pairwise  (any dimensionality)
# ---------------------------------------------------------------------------

def count_linear_regions_pairwise(model, train_dataset, n_pairs,
                                  n_line_samples, device, seed=42,
                                  batch_size=4096):
    """Approximate region count by averaging unique patterns along random lines.

    For each of *n_pairs* randomly chosen pairs of training points, sample
    *n_line_samples* equally spaced points on the connecting segment, record
    activation patterns, and count unique patterns.  Return the average count
    across all pairs.
    """
    rng = np.random.default_rng(seed)
    n = len(train_dataset)

    # Sample distinct pairs.
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    mask = idx_a == idx_b
    while mask.any():
        idx_b[mask] = rng.integers(0, n, size=int(mask.sum()))
        mask = idx_a == idx_b

    # Load pair endpoints and flatten to (P, D).
    points_a = torch.stack([train_dataset[int(i)][0].view(-1) for i in idx_a])
    points_b = torch.stack([train_dataset[int(i)][0].view(-1) for i in idx_b])

    # Interpolate: shape (P, S, D).
    t = torch.linspace(0, 1, n_line_samples).view(1, -1, 1)
    line_points = points_a.unsqueeze(1) + t * (points_b - points_a).unsqueeze(1)

    P, S, D = line_points.shape
    flat = line_points.reshape(P * S, D)

    # Batched forward pass.
    model.eval()
    all_patterns = []
    with torch.no_grad():
        for i in range(0, len(flat), batch_size):
            batch = flat[i : i + batch_size].to(device)
            pat = get_activation_patterns(model, batch)
            if pat is None:
                return 1.0
            all_patterns.append(pat.cpu())

    all_patterns = torch.cat(all_patterns, dim=0).numpy().astype(np.uint8)
    all_patterns = all_patterns.reshape(P, S, -1)

    # Pack once, then count unique rows per pair.
    total = 0
    for i in range(P):
        total += _count_unique_packed(all_patterns[i])

    return total / P


# ---------------------------------------------------------------------------
# Method 3: local  (any dimensionality)
# ---------------------------------------------------------------------------

def count_linear_regions_local(model, train_dataset, n_anchors, n_directions,
                                scale, device, seed=42, batch_size=4096):
    """Approximate region count using orthonormal local probes.

    For each of *n_anchors* training points, sample *n_directions* orthonormal
    vectors in input space, form probe points as (anchor + actual_scale * v_i)
    for each direction, and count unique activation patterns across those points
    plus the anchor itself.  Return the average count across all anchors.

    *scale* is a fraction of the RMS per-feature standard deviation of the
    training data, so the probes are always in sensible units regardless of
    how the inputs are normalised.
    """
    rng = np.random.default_rng(seed)
    n = len(train_dataset)
    D = train_dataset[0][0].view(-1).shape[0]
    k = min(n_directions, D)
    pts_per_anchor = k + 1  # anchor + k probes

    rms_std = _rms_std_from_dataset(train_dataset)
    actual_scale = scale * rms_std

    anchor_idx = rng.integers(0, n, size=n_anchors)
    anchors = torch.stack([train_dataset[int(i)][0].view(-1) for i in anchor_idx])

    all_pts = torch.empty(n_anchors, pts_per_anchor, D)
    for a in range(n_anchors):
        Q, _ = np.linalg.qr(rng.standard_normal((D, k)))   # Q: (D, k)
        directions = torch.from_numpy(Q.T).float()           # (k, D)
        anchor = anchors[a]                                  # (D,)
        all_pts[a, 0] = anchor
        all_pts[a, 1:] = anchor.unsqueeze(0) + actual_scale * directions

    flat = all_pts.reshape(n_anchors * pts_per_anchor, D)

    model.eval()
    all_patterns = []
    with torch.no_grad():
        for i in range(0, len(flat), batch_size):
            batch = flat[i : i + batch_size].to(device)
            pat = get_activation_patterns(model, batch)
            if pat is None:
                return 1.0
            all_patterns.append(pat.cpu())

    all_patterns = torch.cat(all_patterns, dim=0).numpy().astype(np.uint8)
    all_patterns = all_patterns.reshape(n_anchors, pts_per_anchor, -1)

    total = 0
    for i in range(n_anchors):
        total += _count_unique_packed(all_patterns[i])

    return total / n_anchors


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def count_regions(model, task_info, train_dataset, device,
                  method="auto", grid_points=10000,
                  n_pairs=100, n_line_samples=100,
                  n_anchors=100, n_directions=10, local_scale=1.0,
                  seed=42):
    """Count (or approximate) the number of linear regions.

    Parameters
    ----------
    method : str
        ``"grid"``     – exact grid count (1-D/2-D only).
        ``"pairwise"`` – average unique patterns along random inter-point lines.
        ``"local"``    – average unique patterns in orthonormal local probes.
        ``"auto"``     – grid when ``input_range`` exists, pairwise otherwise.
    """
    if method == "auto":
        method = "grid" if task_info.get("input_range") is not None else "pairwise"

    if method == "grid":
        return count_linear_regions_grid(model, task_info, grid_points, device)

    if method == "local":
        return count_linear_regions_local(
            model, train_dataset, n_anchors, n_directions, local_scale,
            device, seed=seed,
        )

    return count_linear_regions_pairwise(
        model, train_dataset, n_pairs, n_line_samples, device, seed=seed,
    )
