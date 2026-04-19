# Adaptive Depth Reasoning (ADR)

> **Not all problems deserve equal thought, and more thinking is not always better.**

ADR is a heuristic framework for calibrating reasoning depth in LLMs and structured human reasoning. It is **inspired by** (not derived from) Recurrent-Depth Transformer (RDT) theory, where a neural network runs the same reasoning block multiple times with convergence detection and a learned halting mechanism. ADR translates the motivating intuition -- that reasoning depth should be adaptive, not fixed -- into a set of practical, testable rules for prompt-level reasoning.

**Status:** Experimental. This framework has no empirical validation. It is a structured hypothesis about how to improve reasoning quality through depth calibration. Use it as a starting point for experimentation, not as a proven methodology.

## Epistemic Honesty Notice

The relationship between RDT architecture and prompt-level reasoning is **analogical, not deductive**. RDTs operate on continuous R^d vectors through differentiable recurrence. Prompt-level reasoning operates on discrete token sequences through autoregressive generation. These are fundamentally different computational substrates. The analogy is useful for motivation and intuition, but no claim in this framework follows necessarily from the RDT mathematics. Where the source documents use terms like "spectral radius" or "convergence" in the context of ADR, these are pedagogical metaphors, not mathematical statements. See the [Assumptions](#assumptions) section for the complete inventory.

## Why This Exists

Frontier reasoning models exhibit a specific failure mode: given the ability to think longer, they often do -- even when the extra computation degrades the answer. Hedging accumulates. Answers oscillate. The model drifts from the original question. The response gets longer but not better.

In RDT research, excessive loop iterations cause the hidden state to drift past the solution into noise [1]. The Universal Transformer addressed this with Adaptive Computation Time (ACT), a learned halting mechanism [4]. ADR adapts the same motivating principle -- halt when continued reasoning degrades rather than improves -- to prompt-level reasoning, using heuristic checks rather than learned scalars.

**Key difference from ACT:** ACT is a differentiable mechanism trained via backpropagation. ADR's halting is a set of heuristic checks applied by the same system doing the reasoning. ADR is therefore weaker than ACT: it relies on self-monitoring, which is known to be unreliable in LLMs. This limitation is acknowledged, not solved.

## The Framework

ADR structures reasoning into three stages:

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   INPUT (user's question)                                │
│       │                                                  │
│       ▼                                                  │
│   ┌────────────────────┐                                 │
│   │  STAGE 1: CLASSIFY │  Assign initial depth level     │
│   │  (run once)        │  Depth 0/1 -> answer now        │
│   └────────┬───────────┘  Depth 2/3 -> enter loop        │
│            │                                             │
│            ▼                                             │
│   ┌────────────────────┐                                 │
│   │  STAGE 2: LOOP     │<-- re-inject original question  │
│   │  (1 to N passes)   │                                 │
│   │                    │    Convergence test:            │
│   │  Per pass:         ├--> actionable answer unchanged? │
│   │  reason -> test    │    -> EXIT                      │
│   │                    │    Overthinking signal?         │
│   │                    │    -> ROLLBACK + EXIT           │
│   │                    │    Pass N reached?              │
│   │                    │    -> DECOMPOSE instead         │
│   └────────┬───────────┘                                 │
│            │                                             │
│            ▼                                             │
│   ┌────────────────────┐                                 │
│   │  STAGE 3: COMMIT   │  State answer directly          │
│   │  (run once)        │  Report confidence + gaps       │
│   └────────┬───────────┘  Flag what could change answer  │
│            │                                             │
│            ▼                                             │
│   OUTPUT                                                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Formal Specification

```
FUNCTION adr_reason(question: str, context: dict) -> (answer, confidence, uncertainties):

    // Stage 1: Classify
    depth := classify_depth(question, context)
    IF depth <= 1:
        RETURN (direct_answer(question), confidence, [])

    // Breadth scan for depth >= 2
    angles := enumerate_angles(question, context)  // 2-4 framings
    selected := select_by_relevance(angles, context, max=2)

    // Stage 2: Reasoning loop
    prev_action := NULL
    prev_answer := NULL
    max_passes := N  // Default 4, tunable per domain

    FOR pass_i IN 1..max_passes:

        // Step 1: Re-inject original question (prevents drift)
        current_question := question  // NOT a reformulation

        // Step 2: Reason on selected angles
        answer_i := reason(current_question, selected, context)

        // Step 3: Convergence test (defined below)
        action_i := extract_actionable_recommendation(answer_i)
        IF pass_i > 1 AND action_i == prev_action:
            RETURN commit(answer_i, pass_i, "converged")

        // Step 4: Overthinking detection
        IF detect_overthinking(answer_i, prev_answer, pass_i):
            RETURN commit(prev_answer, pass_i - 1, "rollback")

        // Step 5: Reclassify if needed
        IF evidence_of_higher_depth(answer_i):
            depth := min(depth + 1, 3)
        IF evidence_of_lower_depth(answer_i):
            RETURN commit(answer_i, pass_i, "reclassified_down")

        prev_action := action_i
        prev_answer := answer_i

    // Hit max passes without convergence
    RETURN commit(prev_answer, max_passes, "decompose_recommended")
```

### Stage 1: Classification

Assign an initial depth level. This classification is provisional and can change during reasoning (see reclassification in Stage 2, step 5).

| Level | Name | Decision criterion | Action |
|-------|------|--------------------|--------|
| 0 | Recall | Answer is a stored fact; no inference required | Answer immediately |
| 1 | Single-hop | Answer follows from one inference step | One reasoning pass, then answer |
| 2 | Multi-hop | Multiple facts or rules must be composed; answer is not obvious after one pass | Reasoning loop with convergence testing |
| 3 | Adversarial | Problem has known failure modes, common misconceptions, or the obvious answer is likely wrong | Reasoning loop with mandatory challenge pass |

**The classification problem:** Depth classification requires estimating problem difficulty before solving the problem. This is inherently imperfect. The framework handles misclassification through reclassification during Stage 2: if pass 1 reveals the problem is harder or simpler than initially assessed, the depth level adjusts. Under-classification (starting too shallow) is caught when the first pass produces a low-confidence answer. Over-classification (starting too deep) costs extra passes but converges normally.

**Depth 3 circularity acknowledged:** "Known gotchas" requires domain knowledge the reasoner may lack. When domain expertise is uncertain, default to Depth 2 and let the reasoning loop discover adversarial features if they exist. Depth 3 should only be assigned when the reasoner can articulate the specific failure modes they expect.

### Stage 2: Reasoning Loop

For Depth 2+ problems, reason iteratively. Each pass has five steps.

**Step 1: Re-inject the original question.** Re-read the user's exact question, not your reformulation. In RDT theory, the input signal `e` is injected at every recurrence step to prevent the hidden state from drifting [1,3]. The prompt-level analog is re-grounding in the actual question to prevent reasoning from wandering to adjacent but different questions. This mapping is analogical; its effectiveness at the prompt level is plausible but unproven.

**Step 2: Reason on the current state.** Apply one round of analysis on the selected angles. Each pass should produce at least one new insight -- a new fact, a new logical step, or a new constraint. If a pass produces no new insight, it is evidence of convergence.

**Step 3: Convergence test.**

Convergence criterion: extract the actionable recommendation from the current pass's answer (what should the user do?). Compare it to the previous pass's actionable recommendation. If the recommended action is unchanged, reasoning has converged. Exit the loop.

This criterion is deliberately coarse: it tests whether the decision changed, not whether the wording changed. A pass that adds nuance without changing the recommended action is useful context but does not prevent exit.

Limitations of this criterion: (a) It collapses multi-dimensional answers into a single action comparison. For problems with no clear action (pure analysis, explanation), fall back to: "Would I give the user a substantively different answer than last pass?" This fallback is weaker. (b) It can false-positive: the same action may be recommended for different reasons, and a shift in reasoning that does not yet change the action may eventually do so in a later pass. The max-pass cap partially mitigates this.

**Step 4: Overthinking detection.** See the dedicated section below.

**Step 5: Reclassification.** If a pass reveals the problem is simpler or harder than initially classified, adjust the depth level. Evidence of higher depth: the pass uncovered a hidden constraint, a domain-specific gotcha, or a contradiction. Evidence of lower depth: the pass confirmed the first-pass answer with no new complexity.

**Maximum passes:** Default is 4. This is a tunable parameter, not a derived constant. The rationale is diminishing returns: RDT scaling laws show quality improvement per additional loop follows a saturating exponential [3], meaning most value concentrates in early passes. The specific value 4 is not derived from any cited result. Adjust per domain.

**When max passes are reached without convergence:** The problem likely needs decomposition into sub-problems, each with its own depth classification. Do not add more passes; restructure the problem.

### Stage 3: Commit

After exiting the loop:

1. State the answer directly. Lead with the conclusion.
2. Report confidence as a calibrated estimate. If you cannot meaningfully estimate confidence, say "confidence uncertain" rather than producing a number with false precision.
3. Flag what could change the answer. Be specific: name the assumption, the missing data, or the condition.
4. Conflict resolution for low confidence: low confidence does NOT mean "refuse to commit." It means commit to the best available answer AND explicitly state what additional information or analysis would resolve the uncertainty. Low confidence is a property of the answer, not a reason to withhold it.

## The Overthinking Problem

Overthinking is a specific failure mode where additional reasoning actively degrades the answer. It is not "being thorough." In RDT research, it corresponds to the hidden state drifting past the solution into noise [1,4].

### Six Heuristic Signals

These are heuristic indicators, not formal detectors. They are reactive (detected after the fact), not predictive (detected before the next pass). This is a known limitation relative to ACT, which is predictive by design. The signals are not exhaustive -- there may be additional overthinking patterns not captured here.

Each signal includes a testable criterion so it can be evaluated, not just invoked by intuition.

**Signal 1: Answer Oscillation**
The conclusion flips between alternatives across passes (X, Y, X, Y...).
Test: does the actionable recommendation reverse direction between consecutive passes?
Indicates: the available evidence does not discriminate between options. More passes will not resolve this.
Action: stop. Report the tension explicitly. Let the user choose or supply additional constraints.

**Signal 2: Increasing Hedging Without New Information**
Each pass adds qualifiers but introduces no new facts, constraints, or logical steps.
Test: does this pass contain at least one fact, constraint, or logical step not present in the previous pass? If no, this is hedging.
Action: stop. Use the previous pass's more direct answer.

**Signal 3: Circular Reasoning**
The current pass re-examines a point already considered, possibly rephrased.
Test: can you identify a specific earlier pass that covered the same ground? If yes, you are looping without progress.
Action: stop. The answer was reached before the loop began.

**Signal 4: Drift From Original Question**
The reasoning now addresses a related but different question.
Test: does the current pass's conclusion answer the literal question the user asked? Compare against the re-injected original question (Step 1). If the subject has shifted, drift has occurred.
Action: roll back to the last pass that answered the original question.

**Signal 5: Phantom Edge Cases**
The reasoning invents increasingly unlikely scenarios that would change the answer.
Test: can you state the probability or cite evidence that this edge case applies to the user's situation? If neither, it is a phantom.
Limitation: distinguishing phantom edge cases from genuinely relevant ones requires domain expertise. When domain expertise is uncertain, err toward inclusion but flag the edge case as unverified rather than discarding it silently.
Action: stop adding edge cases. State the answer for the common case and note unverified edge cases separately.

**Signal 6: Comfort-Seeking Convergence**
A correct-but-uncomfortable conclusion softens across passes without new evidence.
Test: did the tone shift from "this is flawed" toward "this has tradeoffs" between passes? Was the shift justified by new evidence or new logical steps? If not, the earlier, more direct assessment was more honest.
Limitation: this signal requires the reasoner to distinguish between legitimate nuance and social softening. This is a judgment call, not a mechanical test. Known failure mode: the reasoner may rationalize comfort-seeking as "adding nuance."
Action: roll back to the more direct earlier assessment.

### Recovery Protocol

Do NOT attempt to reason your way out of overthinking. That is more overthinking.

1. Roll back to the last pass where the answer was stable and direct
2. State that answer as the conclusion
3. Note the unresolved uncertainty: "I considered X but could not resolve it with the available information."
4. Let the user decide whether to explore the tension, supply constraints, or accept the answer

## Breadth Before Depth

For Depth 2+ problems, before entering the reasoning loop, enumerate 2-4 distinct framings or angles on the problem. Then select the 1-2 most relevant for deep reasoning.

Selection criterion: select the angle(s) that most directly address the user's stated constraints or objectives. If the user has not stated constraints, select the angle(s) with the highest potential to change the recommended action. Discard angles that would produce the same recommendation regardless of analysis.

Failure mode: the breadth scan itself can become a source of overthinking if treated as exhaustive. It is a quick triage, not a complete survey. Spend at most one reasoning pass on it. If you cannot identify distinct angles quickly, the problem may not benefit from breadth scanning -- proceed directly to the reasoning loop.

## Anti-Patterns

**"Let me think step by step" on everything.** Depth 0-1 problems do not benefit from reasoning chains. Elaborate CoT on a factual lookup wastes compute and can introduce errors through overthinking.

**Performative thoroughness.** Caveats and disclaimers without substance are noise. Every qualifier must earn its place by pointing to a specific condition that would change the recommended action.

**Perpetual non-commitment.** After the reasoning loop, commit. "On the other hand" in the output stage is a failure of synthesis, not a sign of rigor. Note: this does not conflict with reporting low confidence. Low confidence is a property of the answer stated alongside it, not a reason to withhold the answer.

**Symmetric false balance.** Not all considerations deserve equal weight. If evidence strongly favors one conclusion, "there are valid points on both sides" is a comfort-seeking pattern (Signal 6), not analysis.

## When NOT to Use ADR

ADR adds overhead. It is not appropriate for all reasoning tasks.

Do not use ADR when: the task is pure generation (creative writing, drafting) not reasoning; the answer is a straightforward lookup you are confident in; the user has explicitly asked for a quick direct answer; the reasoning is already structured by an external protocol (a fixed checklist, a regulatory procedure); or the token budget or latency constraint makes multi-pass reasoning impractical.

Use ADR when: the problem requires composing multiple facts or rules; the domain has known failure modes or common misconceptions; you notice yourself hedging or oscillating during a first-pass answer; the user has asked for careful, validated reasoning; or the problem is novel and you have not seen this specific combination before.

## ADR Failure Modes

ADR itself can fail. These are the framework's own failure modes, distinct from the overthinking signals it is designed to detect.

| Failure mode | Description | Est. frequency | Impact | Mitigation |
|-------------|-------------|----------------|--------|------------|
| Under-classification | Assigning Depth 0-1 to a Depth 2-3 problem | Medium | Incorrect answer delivered with false confidence | Reclassification in Stage 2 catches some cases; user feedback catches others |
| Over-classification | Assigning Depth 2-3 to a Depth 0-1 problem | Medium | Wasted compute; possible overthinking of a simple problem | Convergence test exits early |
| False convergence | Convergence test triggers when the answer is wrong but stable | Low-Medium | Incorrect answer delivered as converged | Adversarial pass (Depth 3) partially mitigates; not fully solvable |
| Missed overthinking | Overthinking signals not detected; reasoning degrades silently | Medium | Degraded answer quality | Signals are heuristic; max-pass cap provides backstop |
| Self-monitoring failure | The reasoner cannot accurately assess its own reasoning | High | All detection mechanisms less reliable than assumed | Fundamental limitation; external validation is the only real mitigation |
| Framework overhead | Classification and checking consume tokens without improving quality | Medium | Wasted compute | "When NOT to Use" guidelines; monitoring to tune |

The most important entry is self-monitoring failure. ADR relies on the reasoning agent to detect its own overthinking, which is asking the system to evaluate itself. This is inherently limited. In production, external monitoring -- logging pass counts, tracking answer stability, human review of flagged cases -- is the only reliable mitigation.

## Cost Model

Each reasoning pass has a cost in tokens consumed and latency added.

Overhead per pass: re-injection (Step 1) and convergence test (Step 3) add minimal token overhead. The breadth scan adds roughly one pass equivalent for Depth 2+ problems. Total framework overhead is approximately 1 additional pass equivalent for Depth 2+ problems.

Break-even estimate: if uncalibrated reasoning averages 5-7 passes with degradation in later passes, ADR's overhead is justified if it consistently exits at 2-4 passes with equal or better quality. This has not been empirically verified.

## Validation Methodology

ADR is unvalidated. The following methodology would test its claims. This section exists to make the framework's untested status transparent and to provide a path toward validation.

**Test 1: Depth Classification Consistency.** Present the same problem set to the framework multiple times. Measure agreement on depth level across runs. Expected: greater than 80% agreement for unambiguous problems. Automatable.

**Test 2: Convergence Test Accuracy.** For problems with known correct answers, measure how often the convergence test exits at a pass where the answer is correct. Requires a ground-truth benchmark. Expected: convergence-on-correct rate greater than 90% for Depth 2 problems.

**Test 3: Overthinking Detection Sensitivity and Specificity.** Construct problem sets designed to trigger each overthinking signal. Measure detection rate (sensitivity) and false alarm rate (specificity). Expected thresholds: sensitivity greater than 70%, specificity greater than 85%. These are estimates; real-world calibration is needed.

**Test 4: End-to-End Quality Comparison.** Compare ADR-guided reasoning against unguided reasoning on a diverse benchmark. Measure answer correctness, token consumption, and latency. This is the definitive test. Without it, ADR remains a hypothesis.

**Test 5: Domain-Specific Calibration.** Run Tests 1-4 on domain-specific problem sets (mathematics, code review, financial analysis, legal reasoning). ADR's parameters may need per-domain tuning.

**Finance-specific note:** In financial reasoning, synthetic test problems may not capture the distributional properties of real financial data (fat tails, regime changes, correlated risks). Validation in financial domains should use historical scenarios with known outcomes, not synthetic constructions. End-to-end validation -- from problem intake through to decision recommendation -- is required, not just pass/fail on intermediate steps.

## Assumptions

Every assumption in this framework, with status and disposition:

| # | Assumption | Status | Disposition |
|---|-----------|--------|-------------|
| 1 | RDT theory validly informs prompt-level reasoning | Weak | Labeled as inspiration, not derivation |
| 2 | Problems are classifiable into discrete depth levels | Weak | Acknowledged as provisional; reclassification added |
| 3 | 4 passes is the right default cap | Unverified | Flagged as tunable, not derived |
| 4 | Agents can reliably self-detect overthinking | Weak | Acknowledged as fundamental limitation |
| 5 | Previous pass states are recoverable | Moderate | Implementation-dependent; noted |
| 6 | 2-4 breadth angles is sufficient | Moderate | Domain-dependent default; noted |
| 7 | ADR improves reasoning quality | Unverified | No claim of proven effectiveness; validation methodology provided |
| 8 | Re-injecting the original question prevents drift | Moderate | Plausible by analogy; unproven at prompt level |
| 9 | Framework is model-agnostic | Weak | Restricted: requires extended thinking and self-monitoring |
| 10 | Overthinking signals are exhaustive | Weak | Explicitly stated as non-exhaustive |
| 11 | Framework overhead is justified | Unverified | Cost model added; break-even not proven |

## Calibration Heuristics

Starting-point depth assignments. These are surface-level heuristics, not true calibration (which requires feedback loops and per-domain tuning). They will be wrong in specific cases.

| Signal | Default depth | Rationale | Known exceptions |
|--------|--------------|-----------|-----------------|
| Factual question | 0 | Recall only | Facts the model is uncertain about should be 1 |
| "Explain X" | 1 | One-pass explanation | Complex systems or unfamiliar domains may be 2 |
| "Compare X and Y" | 1-2 | Depends on dimensions | Comparisons with hidden dependencies are 2-3 |
| "Design/architect X" | 2-3 | Compositional | Well-trodden patterns may be 1 |
| "Is this correct/safe/secure?" | 2-3 | Verification needs adversarial thinking | Trivially correct/incorrect cases are 0-1 |
| "Why does X fail?" | 2 | Multi-hop root cause | Single-cause failures may be 1 |
| Novel concept combination | 2 | Compositional reasoning | Very familiar concepts may be 1 |
| Domain with known gotchas | 3 | Adversarial challenge pass needed | Only if the reasoner can articulate the specific gotchas |
| User says "think carefully" | 2+ | Explicit depth signal | -- |
| User says "quick answer" | 0-1 | Explicit shallowness signal | -- |

## Integration With Other Frameworks

ADR wraps existing reasoning frameworks by governing their depth. It does not replace them.

| Framework | Integration point | Note |
|-----------|------------------|------|
| Chain-of-Thought | Use CoT within each ADR pass; ADR governs pass count | ADR adds convergence testing CoT lacks |
| Tree-of-Thought | Breadth scan is a shallow version; full ToT runs within Stage 2 | ADR's max-pass cap constrains tree expansion |
| DECOMPOSE/SOLVE/VERIFY/SYNTHESIZE | Apply ADR depth classification to each sub-problem after decomposition | Sub-problems may have different depths |
| ReAct | ReAct observes external tool output; ADR observes the agent's own reasoning state; they are complementary | Use ReAct for tool-augmented reasoning, ADR for internal calibration |

## Installation

### Claude

**Claude.ai (web/mobile/desktop):**
Navigate to Settings > Customize > Skills, click the "+" button, then "+ Create skill." Upload the `adaptive-depth-reasoning/` folder (containing `SKILL.md` and `theory.md`). Toggle the skill on. Requires Code execution to be enabled in Settings > Capabilities.

**Claude Code (terminal agent):**
Copy the skill folder to your personal skills directory. The skill activates automatically when Claude Code detects a matching task, or invoke it explicitly with `/adaptive-depth-reasoning`.

```bash
# Personal install (available across all projects)
cp -r adaptive-depth-reasoning/ ~/.claude/skills/adaptive-depth-reasoning/

# Project install (available only in this repo, shared via git)
cp -r adaptive-depth-reasoning/ .claude/skills/adaptive-depth-reasoning/
```

**Claude API:**
Copy the content of `SKILL.md` (without the YAML frontmatter between `---` markers) into the `system` parameter of your API request.

### Gemini CLI

Gemini CLI uses the same SKILL.md format. Copy the skill folder to one of Gemini's skill discovery directories. Gemini auto-discovers skills at session start and activates them via the `activate_skill` tool when a task matches the description.

```bash
# User-level (available across all workspaces)
cp -r adaptive-depth-reasoning/ ~/.gemini/skills/adaptive-depth-reasoning/
# OR use the cross-agent alias path:
cp -r adaptive-depth-reasoning/ ~/.agents/skills/adaptive-depth-reasoning/

# Workspace-level (available only in this project)
cp -r adaptive-depth-reasoning/ .gemini/skills/adaptive-depth-reasoning/
```

Verify installation with `/skills list` inside a Gemini CLI session. The skill should appear as `adaptive-depth-reasoning`.

**Gemini API / Google AI Studio:**
Copy the content of `SKILL.md` (without the YAML frontmatter) into the `system_instruction` parameter.

### OpenAI Codex

Codex CLI, the VS Code extension, and the Codex desktop app all share the same skill discovery. Codex scans `.agents/skills/` directories from your current working directory up to the repository root. Skills use the same SKILL.md format.

```bash
# Repository-level (shared via git, available to all Codex users of this repo)
cp -r adaptive-depth-reasoning/ .agents/skills/adaptive-depth-reasoning/

# User-level (available across all repos)
# Add this path to your ~/.codex/config.toml if not already configured
cp -r adaptive-depth-reasoning/ ~/.codex/skills/adaptive-depth-reasoning/
```

Invoke with `/skills` or `$adaptive-depth-reasoning` in the Codex CLI, or let Codex activate it implicitly when a task matches the description.

**ChatGPT (web):**
Copy the content of `SKILL.md` (without YAML frontmatter) into Settings > Personalization > Custom Instructions, or create a custom GPT with the content as its instruction set.

**OpenAI API:**
Include the content of `SKILL.md` (without YAML frontmatter) as a `system` message in your messages array.

### Cross-Agent Compatibility

The SKILL.md format is an emerging open standard. The same skill folder works across Claude Code, Gemini CLI, and Codex CLI without modification. For environments that share a single skill directory across agents, the `~/.agents/skills/` path is recognized by both Gemini CLI and Codex CLI natively, and Claude Code can be configured to read it via the `--add-dir` flag.

```bash
# Shared cross-agent installation
cp -r adaptive-depth-reasoning/ ~/.agents/skills/adaptive-depth-reasoning/
```

### Any Other LLM

Copy the content of `SKILL.md` (without the YAML frontmatter between `---` markers) into your model's system prompt or custom instructions. The framework is plain text and requires only that the model supports extended reasoning and basic self-monitoring.

## Quick Reference

```
1. CLASSIFY    -> Depth 0/1/2/3 (provisional; reclassify if needed)
2. BREADTH     -> If depth >= 2: enumerate 2-4 angles, select 1-2 by relevance
3. LOOP        -> Default max 4 passes (tunable):
                   a. Re-inject original question (verbatim)
                   b. Reason one step (must produce new insight)
                   c. Convergence test: actionable recommendation unchanged? -> exit
                   d. Overthinking signals? -> rollback to previous pass + exit
                   e. Reclassify depth if evidence warrants
4. COMMIT      -> Answer + confidence + what would change it
5. If max passes hit without convergence -> decompose, do not add passes
```

## Project Structure

```
adaptive-depth-reasoning/
├── README.md         # This file
├── SKILL.md          # Skill definition (usable as system prompt)
└── theory.md         # Theoretical foundations and citations
```

## Acknowledgments

This framework builds on ideas from the OpenMythos project [8], the RYS (Repeat Your Self) empirical work on LLM Neuroanatomy [9,10], and the broader Recurrent-Depth Transformer research community. The RYS work provides the strongest empirical evidence for the architectural principles ADR translates to prompt-level reasoning: three-zone functional anatomy, circuit-level reasoning structures, and measurable diminishing returns from additional computation depth.

For the full theoretical treatment -- including RDT recurrence equations, spectral stability, ACT halting, RYS heatmaps, and how each maps (as metaphor) to ADR -- see [`theory.md`](theory.md).

## References

[1] S. Gontijo Lopes, K. Haas, R. Nayak, and D. J. Rezende, "Loop, think, and generalize: implicit reasoning in recurrent depth transformers," arXiv preprint arXiv:2604.07822, 2025.

[2] N. Saunshi, S. Deshpande, H. Yun, S. Kakade, and S. Arora, "Reasoning with latent thoughts: on the power of looped transformers," arXiv preprint arXiv:2502.17416, 2025.

[3] H. Prairie, R. Du, and D. Schwab, "Parcae: scaling laws for stable looped language models," arXiv preprint arXiv:2604.12946, 2026.

[4] M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and L. Kaiser, "Universal transformers," arXiv preprint arXiv:1807.03819, 2018.

[5] DeepSeek-AI, "DeepSeek-MoE: towards ultimate expert specialization in mixture-of-experts language models," arXiv preprint arXiv:2401.06066, 2024.

[6] S. Gao, A. A. Tong, and B. Wu, "Training large language models to reason in a continuous latent space," arXiv preprint arXiv:2412.06769, 2024.

[7] S. Bae, J. Kim, and S. Yun, "Relaxed recursive transformers: effective parameter sharing with layer-wise LoRA," arXiv preprint arXiv:2410.20672, 2024.

[8] K. Gomez, "OpenMythos: a theoretical reconstruction of the Claude Mythos architecture," GitHub repository, 2026. [Online]. Available: https://github.com/kyegomez/OpenMythos

[9] D. N. Ng, "LLM neuroanatomy: how I topped the LLM leaderboard without changing a single weight," 2026. [Online]. Available: https://dnhkng.github.io/posts/rys/

[10] D. N. Ng, "LLM neuroanatomy II: modern LLM hacking and hints of a universal language?," 2026. [Online]. Available: https://dnhkng.github.io/posts/rys-ii/
