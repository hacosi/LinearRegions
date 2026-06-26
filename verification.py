"""Interval-bound-propagation (IBP) verification utilities.

Implements the Phase-1/Phase-2 metrics for the optimizer-vs-linear-regions
verification experiment (see applications.md, idea #1):

  * count_unstable_neurons -- the "bridge" metric: number of ReLU units whose
    pre-activation can be both positive and negative over an L-inf eps-ball.
    This is the *local linear-region density* readout, expressed in the exact
    units a verifier branches on.
  * certified_robust_ibp   -- sound (incomplete) robustness certificate via IBP
    bound propagation on the logit margin.
  * certified_radius       -- largest IBP-certifiable eps (binary search).
  * lipschitz_product      -- product of layer spectral norms (weight-scale
    control, to separate a region-count effect from a pure rescaling effect).

IBP is the dependency-free baseline.  A complete verifier (alpha,beta-CROWN /
auto_LiRPA) is the Phase-2 upgrade and plugs in at the same call sites: it would
add `verification_time` and `num_branchings` per instance, and tighten the
`verified_rate` / `certified_radius` numbers IBP reports here.

All eps values are L-inf radii in the model's (post-normalisation) input space.
"""

import torch
import torch.nn as nn


def _layers(model):
    """Return the ordered Linear/ReLU modules of an MLP."""
    net = model.network
    if isinstance(net, nn.Linear):
        return [net]
    return list(net)


def _linear_ibp(lower, upper, layer):
    """Propagate an interval [lower, upper] through a Linear layer (IBP)."""
    W = layer.weight                       # (out, in)
    b = layer.bias                         # (out,)
    center = (lower + upper) / 2
    radius = (upper - lower) / 2
    out_center = center @ W.t() + b
    out_radius = radius @ W.abs().t()
    return out_center - out_radius, out_center + out_radius


def _input_interval(x, eps, input_lo, input_hi):
    x = x.view(-1)
    lower = x - eps
    upper = x + eps
    if input_lo is not None:
        lower = torch.maximum(lower, input_lo)
    if input_hi is not None:
        upper = torch.minimum(upper, input_hi)
    return lower, upper


@torch.no_grad()
def count_unstable_neurons(model, x, eps, input_lo=None, input_hi=None):
    """Count ReLU units whose pre-activation straddles 0 over the eps-ball.

    Returns (total_unstable, per_layer_counts).  A depth-0 model (no ReLUs)
    returns (0, []).  This is the bridge metric: it equals the number of binary
    case-splits a complete verifier may have to branch on inside this ball.
    """
    lower, upper = _input_interval(x, eps, input_lo, input_hi)

    per_layer = []
    layers = _layers(model)
    i = 0
    while i < len(layers):
        layer = layers[i]
        if isinstance(layer, nn.Linear):
            pre_lower, pre_upper = _linear_ibp(lower, upper, layer)
            followed_by_relu = (i + 1 < len(layers)
                                and isinstance(layers[i + 1], nn.ReLU))
            if followed_by_relu:
                unstable = ((pre_lower < 0) & (pre_upper > 0)).sum().item()
                per_layer.append(int(unstable))
                lower, upper = torch.relu(pre_lower), torch.relu(pre_upper)
                i += 2
                continue
            lower, upper = pre_lower, pre_upper      # final (logit) layer
        i += 1

    return sum(per_layer), per_layer


@torch.no_grad()
def logit_bounds(model, x, eps, input_lo=None, input_hi=None):
    """IBP lower/upper bounds on the network logits over the eps-ball."""
    lower, upper = _input_interval(x, eps, input_lo, input_hi)
    for layer in _layers(model):
        if isinstance(layer, nn.Linear):
            lower, upper = _linear_ibp(lower, upper, layer)
        elif isinstance(layer, nn.ReLU):
            lower, upper = torch.relu(lower), torch.relu(upper)
    return lower, upper


@torch.no_grad()
def certified_robust_ibp(model, x, y, eps, input_lo=None, input_hi=None):
    """Sound (incomplete) certificate that class y is the argmax over the ball.

    Certified iff the worst-case logit of y still exceeds every other class's
    best-case logit: lower[y] > max_{j != y} upper[j].
    """
    lo, hi = logit_bounds(model, x, eps, input_lo, input_hi)
    hi_others = hi.clone()
    hi_others[y] = float("-inf")
    return bool(lo[y] > hi_others.max())


@torch.no_grad()
def certified_radius(model, x, y, eps_max=1.0, iters=20,
                     input_lo=None, input_hi=None):
    """Largest eps for which IBP certifies class y, via binary search.

    Assumes x is correctly classified (eps=0 trivially certifies).  Returns a
    sound *lower bound* on the true certified radius (IBP is conservative).
    """
    if certified_robust_ibp(model, x, y, eps_max, input_lo, input_hi):
        return float(eps_max)
    lo, hi = 0.0, float(eps_max)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if certified_robust_ibp(model, x, y, mid, input_lo, input_hi):
            lo = mid
        else:
            hi = mid
    return lo


@torch.no_grad()
def lipschitz_product(model):
    """Product of Linear-layer spectral norms -- a global Lipschitz upper bound.

    Reported as the weight-scale control: the region-count story must survive
    conditioning on this quantity.
    """
    prod = 1.0
    for layer in _layers(model):
        if isinstance(layer, nn.Linear):
            prod *= torch.linalg.matrix_norm(layer.weight, ord=2).item()
    return prod


# ---------------------------------------------------------------------------
# Tier 2: auto_LiRPA-backed tight certifier (CROWN / alpha-CROWN)
# ---------------------------------------------------------------------------
#
# Tighter than IBP, so it certifies more instances and reports larger (still
# sound) certified radii.  auto_LiRPA is imported lazily so the IBP tier above
# stays dependency-free.  Soundness ordering, for cross-checking a run:
#     radius_IBP  <=  radius_CROWN  <=  radius_complete (alpha,beta-CROWN)

def _margin_spec(num_classes, y, device):
    """Specification matrix C of shape (1, num_classes-1, num_classes).

    Each row is e_y - e_j for j != y, so C @ logits gives the margins
    (logit_y - logit_j).  The input is certified robust iff every margin has a
    positive lower bound over the eps-ball.
    """
    rows = []
    for j in range(num_classes):
        if j == y:
            continue
        c = torch.zeros(num_classes, device=device)
        c[y] = 1.0
        c[j] = -1.0
        rows.append(c)
    return torch.stack(rows).unsqueeze(0)


def build_bounded_model(model, input_dim, device="cpu"):
    """Wrap an MLP in an auto_LiRPA BoundedModule, reusable across points/eps."""
    from auto_LiRPA import BoundedModule
    model = model.to(device).eval()
    example = torch.zeros(1, input_dim, device=device)
    return BoundedModule(model, example, device=device)


def _crown_margin_lb(bounded_model, x, y, eps, num_classes, method, device):
    """Lower bound on min_j (logit_y - logit_j) over the L-inf eps-ball.

    Note: 'CROWN-Optimized' (alpha-CROWN) optimizes relaxation parameters with
    gradient steps internally, so this must NOT run under torch.no_grad().
    """
    import numpy as np
    from auto_LiRPA import BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    bx = BoundedTensor(x.view(1, -1).to(device), ptb)
    C = _margin_spec(num_classes, y, device)
    lb, _ = bounded_model.compute_bounds(x=(bx,), C=C, method=method,
                                         bound_upper=False)
    return lb.min().item()


def certified_robust_crown(model, x, y, eps, num_classes,
                           method="CROWN-Optimized", device="cpu",
                           bounded_model=None, input_dim=None):
    """Sound (incomplete) CROWN/alpha-CROWN certificate for class y over the ball."""
    if input_dim is None:
        input_dim = x.view(-1).shape[0]
    if bounded_model is None:
        bounded_model = build_bounded_model(model, input_dim, device)
    return bool(_crown_margin_lb(bounded_model, x, y, eps, num_classes,
                                 method, device) > 0)


def certified_radius_crown(model, x, y, num_classes, eps_max=1.0, iters=20,
                           method="CROWN", device="cpu", bounded_model=None,
                           input_dim=None):
    """Largest CROWN-certifiable eps via binary search (sound lower bound).

    Defaults to plain 'CROWN' (backward mode, no per-call optimization) because
    the binary search calls this many times; pass method='CROWN-Optimized' for
    the tightest (slower) radius.
    """
    if input_dim is None:
        input_dim = x.view(-1).shape[0]
    if bounded_model is None:
        bounded_model = build_bounded_model(model, input_dim, device)

    def certifies(e):
        return _crown_margin_lb(bounded_model, x, y, e, num_classes,
                                method, device) > 0

    if certifies(eps_max):
        return float(eps_max)
    lo, hi = 0.0, float(eps_max)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if certifies(mid):
            lo = mid
        else:
            hi = mid
    return lo
