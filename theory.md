# Theoretical Foundations of Adaptive Depth Reasoning

This document presents the Recurrent-Depth Transformer (RDT) research that
inspired ADR and maps architectural insights to prompt-level reasoning. All
such mappings are labeled as pedagogical metaphors. They are useful for
intuition but carry no mathematical content.

## Category Distinction

RDTs operate on continuous R^d hidden state vectors through differentiable
recurrence with learned parameters. Prompt-level reasoning operates on discrete
token sequences through autoregressive generation. These are fundamentally
different computational substrates. No claim in ADR follows deductively from
RDT mathematics. The analogy is: both systems exhibit degradation from excessive
iteration, and both benefit from adaptive halting. The mechanisms differ.

## Core Concept: Looped Computation

A Recurrent-Depth Transformer divides its layers into three blocks:

- **Prelude**: standard layers, run once (initial encoding)
- **Recurrent Block**: a subset of layers run T times (iterative reasoning)
- **Coda**: standard layers, run once (output synthesis)

The recurrence update at each loop step t:

```
h_{t+1} = A * h_t + B * e + Transformer(h_t, e)
```

Where h_t is the hidden state, e is the encoded input (re-injected every step),
and A, B are learned injection parameters.

The re-injection of e at every step prevents the hidden state from drifting
away from the original input signal.

**Metaphor for ADR:** The re-injection of e motivates ADR's Step 1 (re-reading
the original question each pass). The prompt-level mechanism is different --
literal text re-reading vs. continuous vector injection -- but the intent is
the same: prevent the reasoning process from wandering away from the problem.
This mapping is motivational, not mathematical.

## Why Adaptive Depth Works

### Systematic Generalization

Looped transformers exhibit a three-phase learning process [1]:
1. Memorization of training distribution
2. In-distribution generalization
3. Systematic (out-of-distribution) generalization -- emerges abruptly

The third phase makes looped models qualitatively different on novel
compositional problems. It phase-transitions in rather than emerging gradually.

**Metaphor for ADR:** Iterative reasoning can unlock solutions to novel
compositional problems that single-pass reasoning misses. This is a loose
parallel, not a proven mechanism at the prompt level.

### Depth Extrapolation

Models trained on k-hop reasoning chains and tested on 2k-hop chains succeed
when looped, fail when not [1]. More inference-time loops enable deeper
reasoning.

**Metaphor for ADR:** Allocating more reasoning passes to harder problems
should improve outcomes. This is the most direct and least controversial
mapping, though it remains unverified empirically.

### Latent Thoughts as Implicit Chain-of-Thought

Each loop iteration functions as one step of chain-of-thought, but in
continuous space rather than token space [2]. Critically, continuous latent
states can encode multiple alternative paths simultaneously, enabling something
closer to breadth-first search rather than single-path reasoning.

**Metaphor for ADR:** This motivates the breadth scan. However, prompt-level
breadth scanning is sequential (enumerate angles, then select), not parallel
(simultaneous encoding of alternatives). The prompt-level mechanism is
structurally weaker.

## The Stability Constraint

### The Spectral Radius Condition

Recast as a linear time-invariant system (ignoring nonlinear terms):

```
h_{t+1} = A * h_t + B * e
```

Stability requires the spectral radius rho(A) < 1. When rho(A) >= 1, the
hidden state grows without bound (residual explosion) and training diverges [3].

The Parcae architecture [3] enforces stability by construction:
1. Parameterize A as a continuous negative diagonal
2. Discretize via ZOH/Euler: A_discrete = exp(dt * A_continuous)
3. Enforce negativity: A := Diag(-exp(log_A))
4. This guarantees rho(A) < 1 regardless of hyperparameters

**Metaphor for ADR (explicit category error warning):** There is no matrix A
in prompt-level reasoning. There is no spectral radius. The concept that
motivates ADR's convergence testing is looser: "each reasoning pass should
bring you closer to the answer, not further away, and re-grounding in the
original question helps ensure this." This is a heuristic principle, not a
mathematical constraint. Stating that "rho(A) effectively exceeds 1" in
prompt-level reasoning is a metaphor. It should not be interpreted as a
mathematical claim.

## The Overthinking Failure Mode

Beyond a certain depth, excessive recurrence degrades predictions [1,4]. The
hidden state drifts past the solution into noise. The Universal Transformer [4]
addressed this with Adaptive Computation Time (ACT): a learned halting
mechanism that decides per-position when to stop looping.

In the RDT context, overthinking manifests as:
- Answer oscillation (the state orbits rather than converging)
- Increasing noise (the hidden state accumulates perturbations)
- Drift from the original problem (the re-injection signal is overwhelmed)

**Metaphor for ADR and key difference:** ADR's overthinking signals are
heuristic checks inspired by these failure modes. However, ACT is a
differentiable mechanism learned through backpropagation -- it predicts when
to stop before the next loop. ADR's checks are reactive -- they detect
overthinking after it has occurred. ADR is therefore fundamentally weaker
than ACT as a halting mechanism. This limitation is inherent to prompt-level
heuristics and is not solvable within the ADR framework.

## Mixture of Experts: Breadth Alongside Depth

The MoE component in looped transformers provides breadth. Each FFN is split
into many small experts, a router selects a subset per token, and shared
experts handle common cross-domain knowledge [5].

As the hidden state evolves across iterations, the router may select different
expert subsets at each depth, making each loop computationally distinct despite
shared weights.

**Metaphor for ADR:** Different "expert perspectives" (technical,
organizational, temporal, risk) are evaluated in the breadth scan, and the most
relevant are selected for deep reasoning. This is a loose structural parallel.
The MoE routing is automatic and learned; ADR's angle selection is manual and
heuristic.

## The Memorization-Reasoning Tradeoff

Looped models improve reasoning but can hurt memorization [1,6]. The recurrent
structure optimizes for iterative composition, not rote storage.

**Metaphor for ADR:** Deep reasoning passes improve novel problem-solving but
do not improve factual recall. If the answer is a known fact, classify at
Depth 0 and skip the loop. Applying reasoning to a factual question can
introduce errors (overthinking applied to recall).

## Scaling Laws

Parcae establishes predictable scaling for looped training [3]:
- For fixed FLOPs, increasing mean recurrence and reducing token count yields
  lower loss than minimal loops on more data
- More test-time loops improve quality following a saturating exponential decay

At 770M parameters, a looped model matches a 1.3B fixed-depth model on the
same data. Roughly half the parameters for the same quality.

**Metaphor for ADR:** A few well-calibrated reasoning passes should outperform
many uncalibrated passes. Quality follows diminishing returns. However, the
specific shape of the diminishing-returns curve at the prompt level has not
been measured. ADR's default of 4 max passes is a guess informed by the
saturating exponential shape, not a value derived from this data.

---

## References

[1] S. Gontijo Lopes, K. Haas, R. Nayak, and D. J. Rezende, "Loop, think, and
    generalize: implicit reasoning in recurrent depth transformers," arXiv preprint
    arXiv:2604.07822, 2025.

[2] N. Saunshi, S. Deshpande, H. Yun, S. Kakade, and S. Arora, "Reasoning with
    latent thoughts: on the power of looped transformers," arXiv preprint
    arXiv:2502.17416, 2025.

[3] H. Prairie, R. Du, and D. Schwab, "Parcae: scaling laws for stable looped
    language models," arXiv preprint arXiv:2604.12946, 2026.

[4] M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and L. Kaiser, "Universal
    transformers," arXiv preprint arXiv:1807.03819, 2018.

[5] DeepSeek-AI, "DeepSeek-MoE: towards ultimate expert specialization in mixture-
    of-experts language models," arXiv preprint arXiv:2401.06066, 2024.

[6] S. Gao, A. A. Tong, and B. Wu, "Training large language models to reason in a
    continuous latent space," arXiv preprint arXiv:2412.06769, 2024.

[7] S. Bae, J. Kim, and S. Yun, "Relaxed recursive transformers: effective parameter
    sharing with layer-wise LoRA," arXiv preprint arXiv:2410.20672, 2024.

[8] K. Gomez, "OpenMythos: a theoretical reconstruction of the Claude Mythos
    architecture," GitHub repository, 2026. Available:
    https://github.com/kyegomez/OpenMythos