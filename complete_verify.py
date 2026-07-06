"""Tier 3: complete verification via the alpha,beta-CROWN CLI.

This is the source of the *true* Phase-2 cost metrics in the experiment:
verification status (verified / falsified / timeout), wall-clock verification
time, and the number of branch-and-bound domains explored (the direct count of
linear-region case-splits the verifier had to reason about).

alpha,beta-CROWN is a standalone tool (https://github.com/Verified-Intelligence/
alpha-beta-CROWN), not a pip import.  We drive it by:

  1. exporting the trained MLP to ONNX,
  2. writing a per-instance VNNLIB local-robustness specification,
  3. generating a YAML config and invoking the `abcrown` CLI via subprocess,
  4. parsing its output for status / time / branchings.

The integration is *guarded*: if the CLI cannot be located, `run_complete`
returns ``{"status": "skipped", ...}`` so an experiment run still completes on
the IBP + CROWN tiers.  Point the integration at an install with either:

    export ABCROWN_PATH=/path/to/alpha-beta-CROWN/complete_verifier/abcrown.py
    export ABCROWN_PYTHON=/path/to/that/env/bin/python   # optional

or by putting an `abcrown` executable on PATH.

ONNX export and VNNLIB generation have no dependency on the CLI and are unit-
testable on their own (see `_selftest`).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import torch


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model, input_dim, path, device="cpu"):
    """Export an MLP to ONNX with a fixed (1, input_dim) input.

    We export the underlying Linear/ReLU stack (`model.network`) directly rather
    than the wrapper, so the graph contains no Flatten/Reshape from the model's
    internal view().  A pure Gemm/Relu chain imports cleanly into the
    onnx2pytorch frontend that alpha,beta-CROWN uses; a leading Reshape there
    triggers shape-inference errors.
    """
    net = model.network.to(device).eval()  # nn.Linear (depth 0) or nn.Sequential
    example = torch.zeros(1, input_dim, device=device)
    torch.onnx.export(
        net, example, path,
        input_names=["input"], output_names=["output"],
        opset_version=13, dynamo=False,
    )
    return path


# ---------------------------------------------------------------------------
# VNNLIB specification (local robustness)
# ---------------------------------------------------------------------------

def write_vnnlib(path, x, y, eps, num_classes, input_lo=None, input_hi=None):
    """Write a local-robustness VNNLIB spec for one input.

    Encodes the *unsafe* set: there exists an input in the L-inf eps-ball whose
    logit for some class j != y is >= the logit for the true class y.  If the
    verifier proves this set unreachable, the network is robust at x.
    """
    x = x.view(-1).tolist()
    D = len(x)

    lines = []
    for i in range(D):
        lines.append(f"(declare-const X_{i} Real)")
    for k in range(num_classes):
        lines.append(f"(declare-const Y_{k} Real)")
    lines.append("")

    # Input box: [x_i - eps, x_i + eps], optionally clamped to a valid domain.
    for i in range(D):
        lo = x[i] - eps
        hi = x[i] + eps
        if input_lo is not None:
            lo = max(lo, input_lo if isinstance(input_lo, float) else input_lo[i])
        if input_hi is not None:
            hi = min(hi, input_hi if isinstance(input_hi, float) else input_hi[i])
        lines.append(f"(assert (>= X_{i} {lo:.8f}))")
        lines.append(f"(assert (<= X_{i} {hi:.8f}))")
    lines.append("")

    # Unsafe output condition: OR_j (Y_j >= Y_y), j != y.  Emitted in the
    # disjunctive-normal form abcrown's read_vnnlib expects, i.e. each disjunct
    # wrapped in (and ...):  (assert (or (and (>= Y_0 Y_y))(and (>= Y_1 Y_y))...))
    others = [k for k in range(num_classes) if k != y]
    if len(others) == 1:
        j = others[0]
        lines.append(f"(assert (>= Y_{j} Y_{y}))")
    else:
        clause = "".join(f"(and (>= Y_{j} Y_{y}))" for j in others)
        lines.append(f"(assert (or {clause}))")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# YAML config
# ---------------------------------------------------------------------------

def write_config(path, onnx_path, vnnlib_path, timeout=120, device="cpu"):
    """Minimal alpha,beta-CROWN YAML config for a single (onnx, vnnlib) pair."""
    cfg = (
        "general:\n"
        f"  device: {device}\n"
        "  enable_incomplete_verification: true\n"
        "model:\n"
        f"  onnx_path: {onnx_path}\n"
        "specification:\n"
        f"  vnnlib_path: {vnnlib_path}\n"
        "solver:\n"
        "  batch_size: 2048\n"
        "bab:\n"
        f"  timeout: {timeout}\n"
        "attack:\n"
        "  pgd_order: before\n"
    )
    with open(path, "w") as f:
        f.write(cfg)
    return path


# ---------------------------------------------------------------------------
# CLI discovery + invocation
# ---------------------------------------------------------------------------

def find_abcrown():
    """Locate the abcrown CLI. Returns (python_or_None, target) or (None, None)."""
    script = os.environ.get("ABCROWN_PATH")
    if script and os.path.exists(script):
        return os.environ.get("ABCROWN_PYTHON", sys.executable), script
    exe = shutil.which("abcrown")
    if exe:
        return None, exe
    return None, None


def is_available():
    return find_abcrown()[1] is not None


# abcrown prints a final "Result: <verified_status>" line (abcrown.py) and a
# "<N> domains visited" line per branch-and-bound run (bab.py).
_RESULT_RE = re.compile(r"Result:\s*(\S+)", re.I)
_TIME_RE = re.compile(r"\bTime:\s*([0-9.]+)", re.I)
_BRANCH_RE = re.compile(r"(\d+)\s+domains\s+visited", re.I)
_INIT_SOLVED_RE = re.compile(r"initial CROWN|init bound", re.I)

# Map abcrown's verified_status tokens to our three outcomes.
_VERIFIED = {"unsat", "safe", "holds", "verified"}
_FALSIFIED = {"sat", "unsafe", "violated", "falsified"}


def _parse_output(text):
    """Extract (status, num_branchings, verification_time) from CLI output.

    Reads the *final* "Result:" line (intermediate logs contain a misleading
    'verified_status unknown'); branch count is the max "<N> domains visited".
    """
    results = _RESULT_RE.findall(text)
    token = results[-1].lower() if results else ""
    if token in _VERIFIED:
        status = "verified"
    elif token in _FALSIFIED:
        status = "falsified"
    elif token in ("timeout", "unknown"):
        status = token
    else:
        status = "unknown"

    branches = [int(n) for n in _BRANCH_RE.findall(text)]
    if branches:
        num_branchings = max(branches)
    elif status == "verified" and _INIT_SOLVED_RE.search(text):
        num_branchings = 0          # solved at the initial CROWN bound, no BaB
    else:
        num_branchings = None

    times = _TIME_RE.findall(text)
    return {
        "status": status,
        "num_branchings": num_branchings,
        "verification_time": float(times[-1]) if times else None,
    }


def run_complete(model, x, y, eps, num_classes, input_dim=None,
                 input_lo=None, input_hi=None, timeout=120, device="cpu",
                 workdir=None):
    """Run alpha,beta-CROWN on one robustness instance.

    Returns a dict with at least {"status": ...}.  Status is "skipped" if the
    CLI is unavailable.  On success it also carries "num_branchings" and
    "verification_time" (either may be None if not found in the output).
    """
    python, target = find_abcrown()
    if target is None:
        return {"status": "skipped", "reason": "abcrown CLI not found "
                "(set ABCROWN_PATH or put abcrown on PATH)"}

    if input_dim is None:
        input_dim = x.view(-1).shape[0]

    tmp = workdir or tempfile.mkdtemp(prefix="abcrown_")
    onnx_path = os.path.join(tmp, "model.onnx")
    vnnlib_path = os.path.join(tmp, "spec.vnnlib")
    cfg_path = os.path.join(tmp, "config.yaml")

    export_onnx(model, input_dim, onnx_path, device=device)
    write_vnnlib(vnnlib_path, x, y, eps, num_classes, input_lo, input_hi)
    write_config(cfg_path, onnx_path, vnnlib_path, timeout=timeout, device=device)

    cmd = ([python, target] if python else [target]) + ["--config", cfg_path]

    # The abcrown install ships a pinned auto_LiRPA submodule; prefer it over any
    # auto_LiRPA installed in the calling interpreter via ABCROWN_PYTHONPATH.
    env = os.environ.copy()
    extra_pp = os.environ.get("ABCROWN_PYTHONPATH")
    if extra_pp:
        env["PYTHONPATH"] = extra_pp + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout + 60, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "num_branchings": None,
                "verification_time": float(timeout)}

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    result = _parse_output(out)
    result["returncode"] = proc.returncode
    return result


# ---------------------------------------------------------------------------
# Self-test for the dependency-free parts (ONNX export + VNNLIB + parsing)
# ---------------------------------------------------------------------------

def _selftest():
    import onnxruntime as ort
    from models import MLP

    torch.manual_seed(0)
    D, K = 6, 3
    model = MLP(input_dim=D, output_dim=K, hidden_width=8, num_hidden_layers=2)
    x = torch.randn(D)
    y = int(model(x.unsqueeze(0)).argmax(1))

    tmp = tempfile.mkdtemp(prefix="abcrown_selftest_")
    onnx_path = os.path.join(tmp, "m.onnx")
    vnnlib_path = os.path.join(tmp, "s.vnnlib")
    export_onnx(model, D, onnx_path)
    write_vnnlib(vnnlib_path, x, y, 0.05, K)

    # ONNX numerically matches torch.
    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {"input": x.view(1, -1).numpy()})[0]
    torch_out = model(x.unsqueeze(0)).detach().numpy()
    max_diff = float(abs(onnx_out - torch_out).max())
    assert max_diff < 1e-4, f"ONNX/torch mismatch: {max_diff}"

    spec = open(vnnlib_path).read()
    assert f"declare-const X_{D-1}" in spec and f"Y_{K-1}" in spec

    # Output parser on a synthetic log in abcrown's real format.
    parsed = _parse_output("verified_status unknown\n42 domains visited\n"
                           "Result: unsat\nTime: 1.23")
    assert parsed["status"] == "verified", parsed
    assert parsed["num_branchings"] == 42, parsed
    assert abs(parsed["verification_time"] - 1.23) < 1e-6, parsed
    # Falsified ('sat') must not be confused with 'unsat'.
    assert _parse_output("Result: sat\nTime: 0.5")["status"] == "falsified"

    print(f"selftest OK | onnx/torch max_diff={max_diff:.2e} | "
          f"abcrown available={is_available()}")


if __name__ == "__main__":
    _selftest()
