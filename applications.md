# Applications of Optimizers Reducing Linear Regions

**Context:** Our experiments with this repo found that Adam partitions the input space into
*fewer* linear regions than (stochastic) gradient descent over the course of training. Since
the number of linear regions is a concrete, measurable complexity statement about a ReLU
network, most applications reduce to: *what does lower realized function complexity buy you?*
The ideas below are ordered roughly from most concrete/defensible to most speculative.

## Strongest, most concrete applications

### 1. Formal verification & certification

**Summary:** Complete ReLU verifiers have cost that scales with the number of activation
patterns / linear regions they must reason about — region boundaries are exactly the case
splits. If Adam produces fewer regions, **Adam-trained networks should be cheaper and faster to
verify** (e.g., for local robustness).

#### What "verifying a network" actually means
Neural-network verification asks a yes/no question of the form: *for every input in some set X,
does the output satisfy property P?* The canonical case is **local robustness**: given a
correctly classified point x₀, take the set X = {x : ‖x − x₀‖∞ ≤ ε} (an ε-ball around it), and
prove that the predicted class doesn't change anywhere in that ball — equivalently, that the
margin between the correct logit and every other logit stays positive over all of X.

A verifier that answers definitively (proving the property, or returning a genuine
counterexample) is **complete**. One that is **sound** but may give up ("can't prove it") is
**incomplete**. The gap between them is exactly where linear regions enter.

#### Why this is hard, and where the cost comes from
A ReLU MLP is a continuous **piecewise-linear** function: on each linear region — i.e., each
fixed activation pattern (every ReLU pinned active or inactive) — the network is a single affine
map, and verifying an affine map over a box is a trivial linear program. If the whole ε-ball X
fell inside one linear region, verification would be instant.

The difficulty is that X generally **straddles many regions**. The combinatorial culprit is
*unstable* neurons:

- A ReLU is **stable** over X if its pre-activation has a fixed sign for all x ∈ X — always ≥ 0
  (acts as identity) or always ≤ 0 (acts as zero). It contributes no branching; it's effectively
  linear there.
- A ReLU is **unstable** over X if its pre-activation can be both positive and negative as x
  ranges over X. *That* neuron forces the verifier to consider both linear pieces.

The number of distinct activation patterns realizable within X — i.e., **the number of linear
regions X intersects** — is governed by how many neurons are unstable over X. Worst case, k
unstable neurons give up to 2^k pieces. Exact ReLU verification is NP-hard (Katz et al.,
Reluplex), and that exponential is precisely the source.

#### How verifiers cope, and why region count drives their cost
- **Incomplete / bound-propagation methods** (IBP, CROWN/DeepPoly, etc.) propagate interval or
  linear bounds through the network. For each *unstable* ReLU they insert a convex relaxation (a
  triangle/linear over-approximation of the kink). Every relaxation loses a little tightness, and
  the looseness **accumulates with the number of unstable neurons**. Fewer unstable neurons →
  tighter bounds → the verifier proves the property outright more often without any branching.
- **Complete methods** — MILP/SMT (Marabou, Reluplex) or modern **branch-and-bound** (α,β-CROWN,
  which dominates VNN-COMP). Branch-and-bound repeatedly **case-splits on an unstable ReLU's
  sign**, partitioning X into subregions and bounding each, until every piece is conclusively
  settled. Each split is a region boundary. So the size of the search tree — the dominant cost —
  scales with the number of unstable neurons / linear regions inside X.

**The connection to our finding:** if Adam carves the space into fewer linear regions, then
within a given ε-ball there should be **fewer unstable neurons**, which means:

1. tighter bound propagation → more instances verified by the cheap incomplete pass,
2. smaller branch-and-bound trees → faster complete verification and more instances solved
   within a time budget,
3. plausibly **larger certified radii** (the property holds verifiably out to larger ε).

So "Adam-trained networks are easier to formally verify" is a direct, testable consequence.

#### The crucial precision: it's *local* region density
Verification cost depends on the regions intersecting the **ε-ball around each data point**, not
the global region count over the whole input space. This ties straight back to the caveat at the
end of this document: the relevant quantity is region density *near the data manifold*, which the
`local`/`pairwise` estimators capture and the `grid` method does not. In fact, the cleanest single
proxy for "verification difficulty" is the **count of unstable neurons over the ε-ball**,
computable cheaply by one interval-bound-propagation pass (a neuron is unstable iff its
pre-activation lower bound < 0 < upper bound). That metric is essentially "local linear-region
density" expressed in the exact units a verifier cares about.

#### A concrete experiment with this repo
**Implemented** as `run_verification.py` (+ `verification.py`, `complete_verify.py`,
`analysis_verification.py`); see CLAUDE.md → "Verifiability experiment". Quick start:
`uv run python run_verification.py --task bullseye --seeds 2 --epochs 5 --n_points 10`.
The steps it automates:

1. Train matched Adam vs. SGD models on `mnist` at fixed `--width`/`--depth` (small ReLU MLPs are
   *ideal* here — verification scales poorly, so this repo's scale is a feature). **Control for
   confounds:** match test accuracy, and watch weight norms — verification is sensitive to
   Lipschitz constant / weight magnitude, so you must show the effect isn't just Adam vs. SGD
   producing different-scale weights.
2. Pick a set of correctly classified test points and a range of ε.
3. For each (point, ε): run one interval-bound pass and record the **number of unstable neurons**
   over the ball — the cheap proxy, and the most direct link to our central quantity.
4. Run a real complete verifier (auto_LiRPA / α,β-CROWN, or Marabou) and record **verification
   time, number of branchings, % verified within timeout, and certified radius**.
5. Correlate all of these with the repo's local region-density measurement.

**Hypothesis:** Adam models have fewer unstable neurons per ε-ball → fewer branches → faster
verification, higher solve rate, and possibly larger certified radii — with the verifier metrics
tracking the local-region-density measurements.

#### Caveats to address head-on
- **Confounds:** differences could come from weight scale or accuracy rather than region count
  per se. Match accuracy and report weight norms / Lipschitz estimates so the region-count
  explanation is isolated.
- **Regions ≠ unstable neurons exactly**, but they're tightly linked; state the relationship
  cleanly (regions intersecting X are enumerated by patterns of the unstable neurons over X).
- **Computational ease ≠ the property holding:** a point sitting near a decision boundary may be
  genuinely non-robust at a given ε even if it's *cheap* to analyze. Keep "easy to verify" (the
  computational claim) separate from "verifiably robust" (a property of the model), and report
  both.

### 2. Adversarial robustness & decision-boundary smoothness
Region density near the data is tied to how quickly the function can bend — i.e., decision
boundary complexity and local Lipschitz behavior. Fewer regions → smoother local behavior →
potentially **larger margins and better robustness**.

*How to test:* measure region density near test points (the `local` method does roughly this)
and correlate with empirical adversarial accuracy / certified radii.

### 3. Compression, pruning, and quantization
Fewer realized regions suggests the network uses fewer effective degrees of freedom. This
predicts Adam-trained nets may be **more compressible** — pruning to higher sparsity at fixed
accuracy, or tolerating lower-bit quantization.

*How to test:* run matched models through standard pruning/quantization pipelines and compare
the accuracy/compression frontier.

### 4. Interpretability
A piecewise-linear net *is* a collection of local linear models, one per region. Fewer regions
= fewer distinct local behaviors = local linear explanations (LIME/saliency-style) that are
**more stable across nearby inputs**.

*How to test:* quantify explanation stability across neighborhoods; expect higher stability for
the lower-region (Adam) models.

## Broader / theory-flavored applications

### 5. Implicit regularization framing — but mind the tension
The natural story is "Adam has an implicit bias toward lower-complexity functions → better
generalization bound." But empirically the folklore is often the *opposite* — SGD frequently
generalizes as well or better than Adam. A naive "fewer regions ⇒ better generalization" claim
will draw pushback. Lean into the tension instead: it suggests **region count is not
monotonically tied to generalization**, and what matters is region count *relative to task
complexity* or *where regions are placed*, not the raw total.

### 6. Realized vs. potential expressivity
The classic region-counting literature (Montúfar et al., Telgarsky, Serra et al.) bounds the
*maximum* regions an architecture *can* express. Our finding concerns *realized* regions under
a given optimizer at fixed architecture. The reframing — **the optimizer, not just the
architecture, determines effective complexity** — is a clean conceptual contribution. It also
gives practical guidance: choose the optimizer for the complexity regime you want (Adam for
simple/verifiable/robust, SGD to exploit more capacity).

### 7. Extrapolation / out-of-distribution behavior
Far from the data, ReLU nets become linear (you sit in one unbounded region). Fewer regions
implies more controlled, predictable extrapolation away from the training manifold — relevant
to OOD detection and to applications (control, scientific ML) where behavior outside the data
envelope matters.

### 8. Training diagnostics / early stopping
Region count as a cheap, architecture-agnostic **progress / overfitting signal** monitored
during training (already logged per epoch by the trainer). Does a sudden acceleration in region
growth predict overfitting onset? Could become a lightweight diagnostic.

## Key caveat to build into all of these

**Total region count is coarse — *location and density* matter more than the global number.**
A network could have few regions globally but pack them densely right at the decision boundary
(bad for robustness/verification), or vice versa. Most of the applications above (robustness,
verification, interpretability) actually depend on region density *near the data manifold*,
which is exactly what the `pairwise`/`local` methods estimate and the `grid` method does not.
Making the global-vs-local distinction a first-class part of the analysis both strengthens the
applied claims and pre-empts the obvious objection that "fewer regions overall doesn't
necessarily mean simpler where it counts."
