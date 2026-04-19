---
name: adaptive-depth-reasoning
description: >
  A meta-cognitive reasoning calibration framework inspired by Recurrent-Depth Transformer
  theory. Use this skill whenever a problem requires careful reasoning about HOW DEEPLY to
  think, when extended thinking or chain-of-thought is involved, when you notice yourself
  overthinking or going in circles, when tackling novel or compositional problems, or when
  the user asks for calibrated/adaptive reasoning. Also trigger when the user mentions
  "thinking skill", "reasoning depth", "overthinking", "adaptive computation",
  "calibrated thinking", or references OpenMythos/looped transformer concepts.
  Requires models with extended thinking and basic self-monitoring capability.
---

# Adaptive Depth Reasoning (ADR)

A heuristic framework for calibrating reasoning depth. Core principle:
**not all problems deserve equal thought, and more thinking is not always better.**

Inspired by (not derived from) Recurrent-Depth Transformer theory. The mapping
from neural architecture to prompt-level reasoning is analogical, not deductive.
ADR is experimentally unvalidated. Use as a structured starting point for depth
calibration, not as a proven methodology.

## The Three Stages

### Stage 1: Classify (Problem Depth)

Assign an initial depth level. This classification is provisional -- it can
change during reasoning (see reclassification in Stage 2).

**Depth Level 0 -- Recall**
Answer is a stored fact. No inference required.
Action: answer immediately. Do not enter the reasoning loop.

**Depth Level 1 -- Single-hop**
Answer follows from one inference step or rule application.
Action: one reasoning pass, then answer.

**Depth Level 2 -- Multi-hop**
Multiple facts or rules must be composed. Answer not obvious after one pass.
Action: enter the reasoning loop with convergence testing.

**Depth Level 3 -- Adversarial**
Problem has known failure modes, common misconceptions, or the obvious answer
is likely wrong. The reasoner can articulate the specific traps expected.
Action: reasoning loop with mandatory challenge pass.

**Classification limitations:** Depth classification requires estimating difficulty
before solving the problem. This is inherently imperfect. Misclassification is
handled through reclassification in Stage 2. Under-classification is caught when
pass 1 produces a low-confidence answer. Over-classification costs extra passes
but converges normally via the convergence test.

**Depth 3 requires articulable concern:** Assign Depth 3 only when you can name
the specific failure modes you expect. "This domain has gotchas" without specifics
is not sufficient -- default to Depth 2 and let the loop discover adversarial
features if they exist.

### Stage 2: Reasoning Loop

For Depth 2+ problems, reason iteratively. Each pass has five steps.

```
FOR each pass (1 to max_passes, default 4, tunable):

  1. RE-INJECT the original question verbatim (prevents drift)
  2. REASON one step (must produce at least one new insight)
  3. CONVERGENCE TEST (see definition below)
  4. OVERTHINKING CHECK (see signals below)
  5. RECLASSIFY depth if evidence warrants
```

**Step 1: Re-injection.** Re-read the user's exact question, not your
reformulation. This is the prompt-level analog of input injection in RDT
recurrence [1,3], where the original signal is re-injected at every loop step
to prevent hidden state drift. The analogy is motivational, not mathematical.

**Step 2: Reason.** Apply one round of analysis. Each pass must contribute at
least one new fact, constraint, or logical step. A pass that produces no new
insight is evidence of convergence.

**Step 3: Convergence test.**

Definition: extract the actionable recommendation from the current pass (what
should the user do?). Compare it to the previous pass's recommendation. If the
recommended action is unchanged, reasoning has converged. Exit.

This criterion tests whether the decision changed, not whether the wording
changed. It is deliberately coarse.

Limitations: (a) Collapses multi-dimensional answers into a single action
comparison. For problems with no clear action (pure analysis), fall back to:
"Would I give the user a substantively different answer?" This fallback is
weaker. (b) Can false-positive: the same action recommended for different reasons
may shift in later passes. The max-pass cap partially mitigates this.

**Step 4: Overthinking detection.** See the dedicated section below.

**Step 5: Reclassification.** If a pass reveals the problem is harder or simpler
than initially classified, adjust. Evidence of higher depth: uncovered hidden
constraint, domain gotcha, or contradiction. Evidence of lower depth: first-pass
answer confirmed with no new complexity.

**Maximum passes:** Default 4. This is a tunable parameter, not a derived
constant. The rationale is diminishing returns [3], but the specific value is
not derived from any cited result. Adjust per domain.

**Max passes reached without convergence:** The problem needs decomposition into
sub-problems, each with its own depth classification. Do not add more passes.

### Stage 3: Commit

After exiting the loop:

1. State the answer directly. Lead with the conclusion.
2. Report confidence. If you cannot meaningfully estimate it, say "confidence
   uncertain" rather than producing a number with false precision.
3. Flag what could change the answer. Name the assumption, missing data, or
   condition.
4. Low confidence does NOT mean withhold the answer. Commit to the best
   available answer AND state what would resolve the uncertainty.

## The Overthinking Problem

Overthinking is a specific failure mode where additional reasoning actively
degrades the answer. It is not "being thorough."

### Self-Monitoring Limitation

ADR's overthinking detection relies on the reasoning agent evaluating its own
reasoning. This is fundamentally limited -- self-monitoring in LLMs is known to
be unreliable. ACT in Universal Transformers [4] is a differentiable, trained
mechanism. ADR's heuristic checks are weaker. In production, external monitoring
(pass count logging, answer stability tracking, human review) is the only
reliable mitigation.

### Six Heuristic Signals

These are reactive (detected after the fact), not predictive. They are not
exhaustive. Each includes a testable criterion.

**1. Answer Oscillation**
Conclusion flips between alternatives across passes.
Test: does the actionable recommendation reverse between consecutive passes?
Action: stop. Report the tension. Let the user choose or supply constraints.

**2. Hedging Without New Information**
Each pass adds qualifiers but no new facts or logical steps.
Test: does this pass contain at least one fact, constraint, or logical step
not present in the previous pass? If no, this is hedging.
Action: stop. Use the previous pass's more direct answer.

**3. Circular Reasoning**
Current pass re-examines a point already considered, possibly rephrased.
Test: can you identify a specific earlier pass that covered the same ground?
Action: stop. The answer was reached before the loop began.

**4. Drift From Original Question**
Reasoning now addresses a related but different question.
Test: does the conclusion answer the literal question the user asked?
Compare against the re-injected original (Step 1).
Action: roll back to last pass that answered the original question.

**5. Phantom Edge Cases**
Reasoning invents increasingly unlikely scenarios.
Test: can you state the probability or cite evidence that this edge case
applies? If neither, it is a phantom.
Limitation: distinguishing phantom from genuine edge cases requires domain
expertise. When uncertain, include but flag as unverified.
Action: stop adding edge cases. State the common-case answer.

**6. Comfort-Seeking Convergence**
A correct-but-uncomfortable conclusion softens across passes without new evidence.
Test: did the tone shift without new evidence or logic justifying it?
Limitation: requires distinguishing legitimate nuance from social softening.
This is a judgment call with a known failure mode: rationalizing softening
as "adding nuance."
Action: roll back to the more direct earlier assessment.

### Recovery Protocol

Do NOT try to reason your way out of overthinking.

1. Roll back to the last stable, direct pass
2. State that answer
3. Note the unresolved uncertainty honestly
4. Let the user decide whether to explore further

## Breadth Before Depth

For Depth 2+ problems, before entering the reasoning loop, enumerate 2-4
distinct angles on the problem. Select 1-2 for deep reasoning.

Selection criterion: pick angles that most directly address the user's stated
constraints. If constraints are unstated, pick angles with the highest potential
to change the recommended action. Discard angles that produce the same
recommendation regardless of analysis.

Failure mode: the breadth scan can itself become overthinking if treated as
exhaustive. Spend at most one pass on it. If you cannot quickly identify
distinct angles, skip to the reasoning loop.

## Anti-Patterns

**Full CoT on everything.** Depth 0-1 problems do not benefit from reasoning
chains. Elaborate CoT on a factual lookup wastes compute.

**Performative thoroughness.** Every qualifier must earn its place by pointing
to a specific condition that would change the recommended action.

**Perpetual non-commitment.** After the loop, commit. Low confidence is a
property of the answer stated alongside it, not a reason to withhold.

**Symmetric false balance.** If evidence strongly favors one conclusion,
"valid points on both sides" is comfort-seeking (Signal 6), not analysis.

## When NOT to Use ADR

Do not use when: the task is pure generation (not reasoning); the answer is a
confident lookup; the user asks for a quick answer; the reasoning follows a
fixed external protocol; or token budget precludes multi-pass reasoning.

## ADR Failure Modes

| Failure mode | Impact | Mitigation |
|-------------|--------|------------|
| Under-classification | False confidence in wrong answer | Reclassification in Stage 2 |
| Over-classification | Wasted compute | Convergence test exits early |
| False convergence | Wrong answer delivered as stable | Adversarial pass (Depth 3) partially mitigates |
| Missed overthinking | Silent quality degradation | Max-pass cap as backstop |
| Self-monitoring failure | All detections less reliable | External monitoring required |
| Framework overhead | Tokens spent on classification not reasoning | "When NOT to Use" guidelines |

## Integration

ADR wraps other frameworks by governing depth. Use CoT within each pass. Use
DECOMPOSE to break problems apart, then apply ADR depth classification to each
sub-problem independently -- sub-problems may have different depths. ReAct
observes external tool output; ADR observes internal reasoning state. They are
complementary.

## Quick Reference

```
1. CLASSIFY    -> Depth 0/1/2/3 (provisional)
2. BREADTH     -> If depth >= 2: 2-4 angles, select 1-2 by relevance
3. LOOP        -> Max 4 passes (tunable):
                   a. Re-inject original question verbatim
                   b. Reason one step (must produce new insight)
                   c. Convergence: actionable recommendation unchanged? -> exit
                   d. Overthinking signal? -> rollback + exit
                   e. Reclassify if evidence warrants
4. COMMIT      -> Answer + confidence + what would change it
5. Max hit without convergence -> decompose, do not add passes
```

## Theoretical Background

For RDT foundations, spectral stability, and Adaptive Computation Time theory,
see `theory.md`. Note: the theory document uses "ADR analog" labels
to map neural mechanisms to prompt-level reasoning. These mappings are
pedagogical metaphors, not mathematical equivalences.