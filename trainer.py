import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from linear_regions import count_regions


def _compute_loss(output, target, criterion, loss_type, task_info):
    """Compute loss, handling the MSE-on-classification case (one-hot targets)."""
    if loss_type == "mse" and task_info["type"] == "classification":
        target = F.one_hot(target, task_info["output_dim"]).float()
    return criterion(output, target)


def train(model, optimizer, criterion, train_dataset, test_dataset,
          task_info, loss_type, batch_size, epochs, device,
          count_regions_every=0, region_methods=None,
          grid_points=10000, n_pairs=100, n_line_samples=100,
          n_anchors=100, n_directions=10, local_scale=1.0, seed=42):
    if region_methods is None:
        region_methods = ["auto"]
    if isinstance(region_methods, str):
        region_methods = [region_methods]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    is_cls = task_info["type"] == "classification"
    track_regions = count_regions_every > 0

    history = {
        "train_loss": [],
        "test_loss": [],
        "epoch_times": [],
    }
    if is_cls:
        history["train_accuracy"] = []
        history["test_accuracy"] = []
    if track_regions:
        history["linear_regions"] = {
            "methods": region_methods,
            "epochs": [],
            "counts": {m: [] for m in region_methods},
        }

    _APPROX_LABEL = {"pairwise": "pairwise avg", "local": "local avg"}

    def _do_count_regions(epoch):
        if not track_regions:
            return
        history["linear_regions"]["epochs"].append(epoch)
        for method in region_methods:
            n = count_regions(
                model, task_info, train_dataset, device,
                method=method, grid_points=grid_points,
                n_pairs=n_pairs, n_line_samples=n_line_samples,
                n_anchors=n_anchors, n_directions=n_directions, local_scale=local_scale,
                seed=seed + epoch,
            )
            history["linear_regions"]["counts"][method].append(n)
            if isinstance(n, float):
                label = _APPROX_LABEL.get(method, "approx avg")
                print(f"  -> [{method}] linear regions ~ {n:.1f}  ({label})")
            else:
                print(f"  -> [{method}] linear regions = {n}")

    # Count at epoch 0 (before any training).
    _do_count_regions(0)

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ---- training ----
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = _compute_loss(out, y, criterion, loss_type, task_info)
            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item() * X.size(0)
            tr_total += X.size(0)
            if is_cls:
                tr_correct += (out.argmax(1) == y).sum().item()

        avg_tr_loss = tr_loss_sum / tr_total

        # ---- evaluation ----
        model.eval()
        te_loss_sum, te_correct, te_total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = _compute_loss(out, y, criterion, loss_type, task_info)
                te_loss_sum += loss.item() * X.size(0)
                te_total += X.size(0)
                if is_cls:
                    te_correct += (out.argmax(1) == y).sum().item()

        avg_te_loss = te_loss_sum / te_total
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_tr_loss)
        history["test_loss"].append(avg_te_loss)
        history["epoch_times"].append(epoch_time)

        if is_cls:
            tr_acc = tr_correct / tr_total
            te_acc = te_correct / te_total
            history["train_accuracy"].append(tr_acc)
            history["test_accuracy"].append(te_acc)
            print(f"Epoch {epoch:>{len(str(epochs))}}/{epochs}  "
                  f"train_loss={avg_tr_loss:.4f}  test_loss={avg_te_loss:.4f}  "
                  f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}  "
                  f"time={epoch_time:.2f}s")
        else:
            print(f"Epoch {epoch:>{len(str(epochs))}}/{epochs}  "
                  f"train_loss={avg_tr_loss:.6f}  test_loss={avg_te_loss:.6f}  "
                  f"time={epoch_time:.2f}s")

        # ---- linear region count ----
        if track_regions and epoch % count_regions_every == 0:
            _do_count_regions(epoch)

    history["total_time"] = time.time() - total_start
    return history
