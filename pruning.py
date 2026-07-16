"""Compression toolkit: global magnitude pruning + uniform quantization.

Implements the metrics for the optimizer-vs-linear-regions *compressibility*
experiment (see applications.md, idea #3):

  * global_magnitude_mask / apply_mask_ -- one-shot global magnitude pruning to
    a target sparsity, the standard "how many weights can we throw away" probe.
  * quantize_weights_                   -- per-tensor symmetric uniform weight
    quantization to n_bits, the low-precision counterpart.
  * finetune_with_masks                 -- optional post-pruning fine-tuning that
    keeps pruned weights pinned at zero, for the recovered frontier.
  * evaluate                            -- test accuracy of a (possibly pruned /
    quantized) model.

The hypothesis: Adam-trained models, carving fewer realized linear regions, use
fewer effective degrees of freedom and so should prune to higher sparsity and
tolerate fewer bits at matched accuracy than SGD-trained ones.  As in the
verification experiment, `lipschitz_product` is reported as the weight-scale
control so a region-count effect can be separated from a pure rescaling effect.

Dependency-free (torch only).
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _layers(model):
    """Return the ordered Linear/ReLU modules of an MLP (mirrors verification._layers)."""
    net = model.network
    if isinstance(net, nn.Linear):
        return [net]
    return list(net)


def prunable_layers(model):
    """The Linear layers whose weights we prune / quantize (biases untouched)."""
    return [m for m in _layers(model) if isinstance(m, nn.Linear)]


@torch.no_grad()
def global_magnitude_mask(model, sparsity):
    """Return {layer: bool_mask} for global magnitude pruning to `sparsity`.

    A single threshold is chosen over the |weights| of *all* Linear layers at
    once (global, not per-layer), so capacity is removed where it matters least
    across the whole network.  mask == True marks weights to KEEP.
    """
    layers = prunable_layers(model)
    if not layers or sparsity <= 0.0:
        return {l: torch.ones_like(l.weight, dtype=torch.bool) for l in layers}

    all_w = torch.cat([l.weight.abs().flatten() for l in layers])
    # kth-smallest magnitude that we prune away; keep strictly-larger weights.
    k = int(round(sparsity * all_w.numel()))
    k = min(max(k, 0), all_w.numel() - 1)
    threshold = torch.kthvalue(all_w, k + 1).values  # (k+1)-th smallest == first kept
    return {l: (l.weight.abs() >= threshold) for l in layers}


@torch.no_grad()
def per_layer_magnitude_mask(model, sparsity):
    """Per-layer magnitude pruning: each Linear hits `sparsity` independently.

    Immune to cross-layer scale imbalance (unlike the global threshold), so it
    isolates within-layer redundancy from how magnitude is spread across layers.
    """
    masks = {}
    for layer in prunable_layers(model):
        w = layer.weight.abs().flatten()
        if sparsity <= 0.0 or w.numel() == 0:
            masks[layer] = torch.ones_like(layer.weight, dtype=torch.bool)
            continue
        k = min(max(int(round(sparsity * w.numel())), 0), w.numel() - 1)
        threshold = torch.kthvalue(w, k + 1).values
        masks[layer] = (layer.weight.abs() >= threshold)
    return masks


def _global_threshold_masks(model, scores, sparsity):
    """Given a per-layer score dict, keep the globally-largest (1-sparsity) fraction."""
    layers = prunable_layers(model)
    if not layers or sparsity <= 0.0:
        return {l: torch.ones_like(l.weight, dtype=torch.bool) for l in layers}
    flat = torch.cat([scores[l].flatten() for l in layers])
    k = min(max(int(round(sparsity * flat.numel())), 0), flat.numel() - 1)
    threshold = torch.kthvalue(flat, k + 1).values
    return {l: (scores[l] >= threshold) for l in layers}


def sensitivity_mask(model, sparsity, train_ds, task_info, device,
                     n_batches=10, batch_size=64):
    """Global pruning by |w * dL/dw| -- a first-order Taylor importance score.

    Data-driven and (unlike magnitude) largely scale-invariant, because dL/dw
    scales inversely with w.  Accumulates gradients over `n_batches` training
    batches; leaves the model's weights and grads untouched on return.
    """
    was_training = model.training
    model.eval()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    layers = prunable_layers(model)

    grad_accum = {l: torch.zeros_like(l.weight) for l in layers}
    model.zero_grad(set_to_none=True)
    for i, (X, y) in enumerate(loader):
        if i >= n_batches:
            break
        X, y = X.to(device), y.to(device)
        loss = criterion(model(X), y)
        grads = torch.autograd.grad(loss, [l.weight for l in layers])
        for l, g in zip(layers, grads):
            grad_accum[l] += g.detach()
    with torch.no_grad():
        scores = {l: (l.weight * grad_accum[l]).abs() for l in layers}
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return _global_threshold_masks(model, scores, sparsity)


def synflow_mask(model, sparsity, input_dim, device):
    """Global pruning by SynFlow saliency |w * dR/dw|, R = 1^T (prod |W_l|) 1.

    Computed on the *trained* weights as a scale-invariant importance score
    (SynFlow is provably invariant to per-layer rescaling).  Data-free: one
    all-ones input through the all-positive linearised network.  Restores the
    original weights before returning.
    """
    layers = prunable_layers(model)
    snapshot = copy.deepcopy(model.state_dict())
    try:
        with torch.no_grad():
            for l in layers:
                l.weight.abs_()
                if l.bias is not None:
                    l.bias.abs_()
        model.eval()
        model.zero_grad(set_to_none=True)
        x = torch.ones(1, input_dim, device=device)
        R = model(x).sum()
        grads = torch.autograd.grad(R, [l.weight for l in layers])
        with torch.no_grad():
            # weights are |w| here; snapshot holds the signed originals.
            scores = {l: (l.weight * g).abs() for l, g in zip(layers, grads)}
        masks = _global_threshold_masks(model, scores, sparsity)
    finally:
        model.load_state_dict(snapshot)
        model.zero_grad(set_to_none=True)
    return masks


@torch.no_grad()
def structured_neuron_mask(model, sparsity):
    """Structured pruning: remove whole hidden neurons (deletes ReLU hinges).

    Hidden neurons are the outputs of every Linear except the last.  Each is
    scored by the L2 norm of its incoming row (W_k[j]) plus outgoing column
    (W_{k+1}[:, j]); the globally-lowest `sparsity` fraction are removed by
    zeroing that row and that column (zeroing the outgoing column removes the
    neuron regardless of its bias).  `sparsity` is a *neuron* fraction.
    """
    layers = prunable_layers(model)
    masks = {l: torch.ones_like(l.weight, dtype=torch.bool) for l in layers}
    if len(layers) < 2 or sparsity <= 0.0:
        return masks  # depth-0/1 has no prunable hidden neurons

    # (layer_k, neuron_index, score) for every hidden neuron.
    entries = []
    for k in range(len(layers) - 1):
        Wk, Wk1 = layers[k].weight, layers[k + 1].weight
        incoming = Wk.norm(dim=1)          # per output neuron (rows of W_k)
        outgoing = Wk1.norm(dim=0)         # per input neuron (cols of W_{k+1})
        score = torch.sqrt(incoming ** 2 + outgoing ** 2)
        for j in range(Wk.shape[0]):
            entries.append((k, j, score[j].item()))

    n_prune = int(round(sparsity * len(entries)))
    n_prune = min(max(n_prune, 0), len(entries))
    entries.sort(key=lambda e: e[2])
    for k, j, _ in entries[:n_prune]:
        masks[layers[k]][j, :] = False          # incoming row
        masks[layers[k + 1]][:, j] = False      # outgoing column
    return masks


def build_mask(model, method, sparsity, ctx):
    """Dispatch to a pruning criterion. `ctx` carries data for data-driven ones."""
    if method == "magnitude":
        return global_magnitude_mask(model, sparsity)
    if method == "magnitude_local":
        return per_layer_magnitude_mask(model, sparsity)
    if method == "sensitivity":
        return sensitivity_mask(model, sparsity, ctx["train_ds"], ctx["task_info"],
                                ctx["device"], ctx["n_batches"], ctx["batch_size"])
    if method == "synflow":
        return synflow_mask(model, sparsity, ctx["input_dim"], ctx["device"])
    if method == "structured":
        return structured_neuron_mask(model, sparsity)
    raise ValueError(f"unknown pruning method: {method}")


@torch.no_grad()
def balance_spectral_norms_(model):
    """Function-preserving equalization of per-layer spectral norms (in place).

    Uses ReLU positive-homogeneity (relu(s*z) = s*relu(z), s>0): scaling a
    hidden layer's output by s and the next layer's input by 1/s leaves the
    computed function unchanged.  We choose per-layer weight factors that set
    every Linear's spectral norm to their geometric mean, which removes the
    cross-layer scale imbalance that global magnitude pruning is fooled by --
    without changing the function, accuracy, or the Lipschitz product.
    """
    layers = prunable_layers(model)
    if len(layers) < 2:
        return model  # nothing to redistribute

    # SVD (spectral norm) is unimplemented on MPS; compute it on CPU copies.
    sigmas = [torch.linalg.matrix_norm(l.weight.detach().cpu(), ord=2).item()
              for l in layers]
    log_target = sum(torch.log(torch.tensor(s)) for s in sigmas) / len(sigmas)
    target = float(torch.exp(log_target))
    f = [target / s for s in sigmas]                 # desired weight factor per layer

    # Boundary scalars s_i = prod(f[:i+1]); apply function-preservingly.
    s = []
    acc = 1.0
    for fi in f[:-1]:
        acc *= fi
        s.append(acc)

    # Boundary factor s[i] scales hidden layer i's OUTPUT.  A hidden layer's
    # weight then scales by (output / input) = s[i]/s[i-1], but its bias scales
    # by the output factor s[i] alone (the input rescaling is undone by the
    # weight, not the bias).  The last layer only undoes its input scaling.
    layers[0].weight.mul_(s[0])
    if layers[0].bias is not None:
        layers[0].bias.mul_(s[0])
    for i in range(1, len(layers) - 1):
        layers[i].weight.mul_(s[i] / s[i - 1])
        if layers[i].bias is not None:
            layers[i].bias.mul_(s[i])
    layers[-1].weight.mul_(1.0 / s[-1])              # last weight only; bias unchanged
    return model


@torch.no_grad()
def apply_mask_(model, masks):
    """Zero the pruned weights in place (mask == True keeps)."""
    for layer, mask in masks.items():
        layer.weight.mul_(mask.to(layer.weight.dtype))


@torch.no_grad()
def count_nonzero(model):
    """(nonzero, total) over prunable weights -- the realized sparsity readout."""
    nonzero = total = 0
    for layer in prunable_layers(model):
        nonzero += int((layer.weight != 0).sum().item())
        total += layer.weight.numel()
    return nonzero, total


@torch.no_grad()
def quantize_weights_(model, n_bits):
    """Per-tensor symmetric uniform quantization of each Linear weight, in place.

    Levels = 2**n_bits.  scale = max|w| / (2^(n_bits-1) - 1); round to the grid
    and clamp to the signed range.  Biases are left untouched (standard).
    """
    qmax = 2 ** (n_bits - 1) - 1
    for layer in prunable_layers(model):
        w = layer.weight
        max_abs = w.abs().max()
        if max_abs == 0 or qmax == 0:
            continue
        scale = max_abs / qmax
        q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
        layer.weight.copy_(q * scale)


@torch.no_grad()
def evaluate(model, dataset, task_info, device, batch_size=256):
    """Test accuracy of `model` (classification only)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X).argmax(1)
        correct += int((pred == y).sum().item())
        total += X.size(0)
    return correct / total if total else 0.0


def finetune_with_masks(model, masks, optimizer, train_ds, task_info, device,
                        epochs, batch_size):
    """Fine-tune a pruned model, re-pinning pruned weights to zero each step."""
    model.train()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    apply_mask_(model, masks)  # ensure we start from the pruned network
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            apply_mask_(model, masks)  # keep pruned weights at zero
    return model


@torch.no_grad()
def lipschitz_product(model):
    """Product of Linear-layer spectral norms -- global Lipschitz upper bound.

    The weight-scale control (same definition as verification.lipschitz_product):
    the compressibility story must survive conditioning on this quantity.
    """
    prod = 1.0
    for layer in prunable_layers(model):
        prod *= torch.linalg.matrix_norm(layer.weight, ord=2).item()
    return prod


# ---------------------------------------------------------------------------
# Self-test (dependency-free): `python pruning.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models import MLP

    torch.manual_seed(0)
    net = MLP(input_dim=8, output_dim=3, hidden_width=16, num_hidden_layers=2)

    # 1. mask hits the requested sparsity (within rounding).
    for s in (0.5, 0.8, 0.9, 0.99):
        masks = global_magnitude_mask(net, s)
        m2 = MLP(input_dim=8, output_dim=3, hidden_width=16, num_hidden_layers=2)
        m2.load_state_dict(copy.deepcopy(net.state_dict()))
        apply_mask_(m2, global_magnitude_mask(m2, s))
        nz, tot = count_nonzero(m2)
        realized = 1 - nz / tot
        assert abs(realized - s) < 0.02, f"sparsity {s}: realized {realized:.3f}"
    print("[ok] global_magnitude_mask reaches target sparsity")

    # 2. quantization collapses the number of distinct weight values.
    m3 = MLP(input_dim=8, output_dim=3, hidden_width=16, num_hidden_layers=2)
    m3.load_state_dict(copy.deepcopy(net.state_dict()))
    before = sum(l.weight.unique().numel() for l in prunable_layers(m3))
    quantize_weights_(m3, n_bits=3)
    after = sum(l.weight.unique().numel() for l in prunable_layers(m3))
    assert after < before, f"quant did not reduce distinct values: {before} -> {after}"
    # each 3-bit tensor should have at most 2**3 distinct levels.
    for l in prunable_layers(m3):
        assert l.weight.unique().numel() <= 2 ** 3 + 1
    print(f"[ok] quantize_weights_ collapses distinct values ({before} -> {after})")

    # 3. evaluate matches a hand-computed accuracy on a tiny dataset.
    from torch.utils.data import TensorDataset
    X = torch.randn(20, 8)
    y = net(X).argmax(1)  # labels == model's own predictions => accuracy 1.0
    ds = TensorDataset(X, y)
    ti = {"type": "classification", "input_dim": 8, "output_dim": 3}
    acc = evaluate(net, ds, ti, device=torch.device("cpu"))
    assert abs(acc - 1.0) < 1e-9, f"evaluate returned {acc}"
    print("[ok] evaluate matches hand-computed accuracy")

    dev = torch.device("cpu")

    # 4. per-layer magnitude hits the target sparsity in EVERY layer.
    m4 = MLP(input_dim=8, output_dim=3, hidden_width=16, num_hidden_layers=2)
    m4.load_state_dict(copy.deepcopy(net.state_dict()))
    apply_mask_(m4, per_layer_magnitude_mask(m4, 0.5))
    for l in prunable_layers(m4):
        realized = 1 - int((l.weight != 0).sum()) / l.weight.numel()
        assert abs(realized - 0.5) < 0.05, f"per-layer sparsity {realized:.3f}"
    print("[ok] per_layer_magnitude_mask hits target sparsity per layer")

    # 5. sensitivity + synflow reach target sparsity; synflow restores weights.
    for name, fn in (("sensitivity",
                      lambda m: sensitivity_mask(m, 0.8, ds, ti, dev, n_batches=2,
                                                 batch_size=8)),
                     ("synflow",
                      lambda m: synflow_mask(m, 0.8, input_dim=8, device=dev))):
        m = MLP(input_dim=8, output_dim=3, hidden_width=16, num_hidden_layers=2)
        m.load_state_dict(copy.deepcopy(net.state_dict()))
        before_w = copy.deepcopy(m.state_dict())
        masks = fn(m)
        # weights untouched by scoring
        for kk in before_w:
            assert torch.equal(before_w[kk], m.state_dict()[kk]), f"{name} mutated {kk}"
        apply_mask_(m, masks)
        nz, tot = count_nonzero(m)
        assert abs((1 - nz / tot) - 0.8) < 0.02, f"{name} sparsity {1 - nz/tot:.3f}"
        print(f"[ok] {name}_mask reaches target sparsity and leaves weights intact")

    # 6. structured pruning zeros whole rows/cols at the target neuron fraction.
    m6 = MLP(input_dim=8, output_dim=3, hidden_width=16, num_hidden_layers=2)
    m6.load_state_dict(copy.deepcopy(net.state_dict()))
    masks = structured_neuron_mask(m6, 0.5)
    # hidden neurons = rows of every Linear except the last (32 total here);
    # global ranking removes 0.5 * 32 = 16, distributed across both layers.
    hidden_layers = prunable_layers(m6)[:-1]
    dead = sum(int((~masks[l]).all(dim=1).sum()) for l in hidden_layers)
    assert dead == 16, f"expected 16/32 dead neurons, got {dead}"
    apply_mask_(m6, masks)
    print(f"[ok] structured_neuron_mask removed {dead}/32 whole neurons")

    # 7. spectral-norm balancing preserves the function AND the Lipschitz product,
    #    while equalizing per-layer spectral norms.
    m7 = MLP(input_dim=8, output_dim=3, hidden_width=16, num_hidden_layers=2)
    m7.load_state_dict(copy.deepcopy(net.state_dict()))
    with torch.no_grad():
        before_logits = m7(X).clone()
    lip_before = lipschitz_product(m7)
    balance_spectral_norms_(m7)
    with torch.no_grad():
        after_logits = m7(X)
    assert torch.allclose(before_logits, after_logits, atol=1e-4), "balance changed function"
    assert abs(lipschitz_product(m7) - lip_before) / lip_before < 1e-4, "balance changed Lipschitz"
    sig = [torch.linalg.matrix_norm(l.weight, ord=2).item() for l in prunable_layers(m7)]
    assert max(sig) - min(sig) < 1e-3, f"spectral norms not equalized: {sig}"
    print("[ok] balance_spectral_norms_ preserves function/Lipschitz, equalizes sigma")

    print("All pruning.py self-tests passed.")
