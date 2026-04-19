# Theoretical Foundations of Adaptive Depth Reasoning

This document presents the research that inspired ADR. It draws on two
complementary lines of work: theoretical results on Recurrent-Depth
Transformers (RDTs), and empirical results from LLM Neuroanatomy / RYS
(Repeat Your Self) experiments on production models.

All mappings from architecture-level findings to prompt-level reasoning are
labeled as pedagogical metaphors. They are useful for intuition but carry no
mathematical content.

## Category Distinction

RDTs operate on continuous R^d hidden state vectors through differentiable
recurrence with learned parameters. RYS operates on actual model layer
duplication at inference time. Prompt-level reasoning operates on discrete
token sequences through autoregressive generation. These are three distinct
computational substrates. ADR draws inspiration from all three but is
reducible to none of them.

## Empirical Evidence: LLM Neuroanatomy and RYS

The strongest empirical support for ADR's motivating principles comes from
David Noel Ng's RYS (Repeat Your Self) work [9,10]. Ng took Qwen2-72B,
duplicated seven middle layers (45-51) without changing any weights, and
produced the #1 model on the HuggingFace Open LLM Leaderboard. The method
was discovered by sweeping all 3,241 possible layer-duplication configurations
using two orthogonal probes (hard math and emotional quotient).

### The Three-Zone Functional Anatomy

Ng's heatmaps revealed three functional regions in the transformer stack:

- Early layers (encoding): duplicating these breaks the model. They translate
  input tokens into an abstract internal representation.
- Middle layers (reasoning): duplicating these improves performance. They
  operate in a format-agnostic "thinking space."
- Late layers (decoding): duplicating these degrades output. They translate
  abstract representations back into token predictions.

In Part II [10], this was directly confirmed via cosine similarity analysis.
Four inputs (English fact, English poem, Chinese fact, Chinese poem) were fed
through Qwen3.5-27B, and pairwise hidden-state similarity was measured at
every layer. The results showed that cross-language same-content pairs
(English fact vs Chinese fact) were more similar in the mid-stack than
same-language different-content pairs (English fact vs English poem). The
model's internal representation cares more about what you are saying than
what language you are saying it in. This "universal thinking space" exists
in the middle layers and corresponds exactly to the region where layer
duplication improves reasoning.

**Metaphor for ADR:** ADR's three-stage structure (Classify/Loop/Commit)
mirrors this anatomy. Classification is encoding (understanding the problem).
The reasoning loop is the mid-stack (thinking in abstract space). Committing
is decoding (translating back to a concrete answer). Re-injecting the
original question each pass is the prompt-level analog of input injection
in recurrent architectures, keeping the reasoning grounded.

### Circuit Structure

Single-layer duplication almost never helps. You must duplicate a complete
multi-layer circuit for benefits to appear [9]. In Qwen2-72B, the optimal
circuit was 7 layers. In Qwen3.5-27B, the core circuit was as small as 1-3
layers for EQ tasks and 11 layers for math [10]. Different cognitive tasks
activate different circuits with different boundaries.

Ng describes this as a recipe: layers 46-52 are not seven workers doing the
same job but seven steps in a sequence. Duplicating one step does nothing.
Duplicating the full recipe gives the model a second pass through the
complete reasoning operation.

When the wrong layers are duplicated, models exhibit what Ng calls "brain
damage": one configuration produced a model that announced "Let's act like
cowboys! Yeehaw!" and descended into unrecoverable giggling. This is not
"slightly worse" performance. It is catastrophic disruption of specific
functional circuits [9].

**Metaphor for ADR:** Each reasoning pass in ADR should be a complete unit
of work, not a partial step. The convergence check tests whether a full
pass changed the answer, not whether individual sub-steps progressed. The
overthinking failure mode (answer oscillation, drift, degradation) maps to
what happens when layer duplication crosses circuit boundaries.

### Diminishing Returns and the Pareto Frontier

Part II [10] measured the efficiency frontier across 397 configurations of
Qwen3.5-27B. The results showed steeply diminishing returns:

- Duplicating layer 33 alone: +0.0945 EQ delta at 1.6% overhead
- Expanding to layers 31-33: +0.0972 EQ delta at 4.7% overhead
- Expanding to layers 26-33: +0.1009 EQ delta at 12.5% overhead

A 10x increase in overhead bought only a 7% marginal improvement. The Pareto
frontier was dominated by simple contiguous blocks in the mid-stack. Complex
multi-block compositions from beam search (3,024 candidates) and surrogate
model screening (2 million candidates) did not beat simple blocks on the
efficiency frontier.

**Metaphor for ADR:** This empirically validates ADR's diminishing returns
assumption. The first 1-2 reasoning passes contribute most of the value.
Additional passes help but with rapidly declining marginal benefit. ADR's
default of 4 max passes is consistent with this curve, though the specific
value remains a tunable parameter, not a derived constant.

### Generality Across Architectures

RYS has been tested on Qwen2-72B, Qwen3.5-27B, Llama-3-70B, Phi-3-medium,
and GLM-4.7 [9,10]. All show the same general pattern: a mid-stack reasoning
region where layer duplication improves performance, bounded by encoding and
decoding regions where duplication degrades it. The specific boundaries
differ per architecture, but the structural principle holds.

## Theoretical Foundations: Recurrent-Depth Transformers

### Core Concept: Looped Computation

A Recurrent-Depth Transformer divides its layers into three blocks:

- **Prelude**: standard layers, run once (initial encoding)
- **Recurrent Block**: a subset of layers run T times (iterative reasoning)
- **Coda**: standard layers, run once (output synthesis)

The recurrence update at each loop step t:

```
h_{t+1} = A * h_t + B * e + Transformer(h_t, e)
```

Where h_t is the hidden state, e is the encoded input (re-injected every
step), and A, B are learned injection parameters. The re-injection of e at
every step prevents the hidden state from drifting [1,3].

**Metaphor for ADR:** Re-reading the original question each pass serves the
same grounding function. The mechanism is different (text re-reading vs
continuous vector injection) but the intent is identical: prevent reasoning
from wandering.

### Systematic Generalization

Looped transformers exhibit a three-phase learning process [1]: memorization,
in-distribution generalization, then systematic out-of-distribution
generalization that emerges abruptly as a phase transition.

### Depth Extrapolation

Models trained on k-hop reasoning chains and tested on 2k-hop chains succeed
when looped, fail when not [1]. More inference-time loops enable deeper
reasoning.

### Latent Thoughts as Implicit Chain-of-Thought

Each loop iteration functions as one step of chain-of-thought in continuous
space rather than token space [2]. Continuous latent states can encode
multiple alternative paths simultaneously.

## The Stability Constraint

### The Spectral Radius Condition

Recast as a linear time-invariant system (ignoring nonlinear terms):

```
h_{t+1} = A * h_t + B * e
```

Stability requires rho(A) < 1. When rho(A) >= 1, the hidden state grows
without bound [3]. The Parcae architecture [3] enforces stability by
construction via A := Diag(-exp(log_A)), guaranteeing rho(A) < 1.

**Metaphor for ADR (explicit category warning):** There is no matrix A in
prompt-level reasoning. The concept is looser: each reasoning pass should
bring you closer to the answer, and re-grounding in the original question
helps ensure this. This is a heuristic principle, not a mathematical
constraint.

## The Overthinking Failure Mode

Beyond a certain depth, excessive recurrence degrades predictions [1,4]. The
Universal Transformer [4] addressed this with Adaptive Computation Time
(ACT): a differentiable halting mechanism that decides per-position when to
stop looping.

RYS provides independent empirical confirmation: Ng's heatmaps show blue
(degraded) regions where duplicating the wrong layers or too many layers
actively hurts performance [9]. The degradation is not gradual. It can be
catastrophic, producing incoherent or unhinged outputs.

**Metaphor for ADR and key difference:** ADR's overthinking signals are
heuristic checks inspired by these failure modes. ACT is predictive (decides
before the next loop). ADR is reactive (detects after the fact). ADR is
fundamentally weaker than ACT as a halting mechanism.

## Mixture of Experts: Breadth Alongside Depth

The MoE component in looped transformers provides breadth [5]. As the hidden
state evolves across iterations, the router may select different expert
subsets at each depth.

**Metaphor for ADR:** Different "expert perspectives" in the breadth scan
mirror MoE routing. The MoE routing is automatic and learned; ADR's angle
selection is manual and heuristic.

## The Memorization-Reasoning Tradeoff

Looped models improve reasoning but can hurt memorization [1,6]. If the
answer is a known fact, classify at Depth 0 and skip the loop.

## Scaling Laws

Parcae establishes predictable scaling for looped training [3]:
- Increasing mean recurrence and reducing token count yields lower loss
- More test-time loops improve quality following a saturating exponential

RYS's Pareto frontier [10] independently confirms the saturating shape at
the architecture level: most value is in the first duplication, with steeply
diminishing returns thereafter.

---

## References

[1] S. Gontijo Lopes, K. Haas, R. Nayak, and D. J. Rezende, "Loop, think,
    and generalize: implicit reasoning in recurrent depth transformers,"
    arXiv preprint arXiv:2604.07822, 2025.

[2] N. Saunshi, S. Deshpande, H. Yun, S. Kakade, and S. Arora, "Reasoning
    with latent thoughts: on the power of looped transformers," arXiv
    preprint arXiv:2502.17416, 2025.

[3] H. Prairie, R. Du, and D. Schwab, "Parcae: scaling laws for stable
    looped language models," arXiv preprint arXiv:2604.12946, 2026.

[4] M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and L. Kaiser,
    "Universal transformers," arXiv preprint arXiv:1807.03819, 2018.

[5] DeepSeek-AI, "DeepSeek-MoE: towards ultimate expert specialization in
    mixture-of-experts language models," arXiv preprint arXiv:2401.06066,
    2024.

[6] S. Gao, A. A. Tong, and B. Wu, "Training large language models to
    reason in a continuous latent space," arXiv preprint arXiv:2412.06769,
    2024.

[7] S. Bae, J. Kim, and S. Yun, "Relaxed recursive transformers: effective
    parameter sharing with layer-wise LoRA," arXiv preprint
    arXiv:2410.20672, 2024.

[8] K. Gomez, "OpenMythos: a theoretical reconstruction of the Claude Mythos
    architecture," GitHub repository, 2026. Available:
    https://github.com/kyegomez/OpenMythos

[9] D. N. Ng, "LLM neuroanatomy: how I topped the LLM leaderboard without
    changing a single weight," 2026. [Online]. Available:
    https://dnhkng.github.io/posts/rys/

[10] D. N. Ng, "LLM neuroanatomy II: modern LLM hacking and hints of a
     universal language?," 2026. [Online]. Available:
     https://dnhkng.github.io/posts/rys-ii/