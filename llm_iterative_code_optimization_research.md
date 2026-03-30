# LLM-Driven Iterative Code Optimization: Research Findings

Research compiled 2026-03-29 from published papers, blog posts, and open-source implementations.

---

## 1. FunSearch (Google DeepMind, Nature 2023)

**What it is:** An LLM-powered evolutionary search over *programs* (not solutions). Pairs an LLM with an automated evaluator to iteratively evolve functions that solve hard mathematical and combinatorial problems.

**Architecture:**
- **Programs Database** with island-based evolutionary model
- **Sampler** selects k high-performing programs as few-shot exemplars
- **LLM** (PaLM 2) generates new candidate programs
- **Evaluator** executes and scores candidates; only correct ones enter database

**Key design choices that make it work:**

1. **Search over programs, not solutions.** FunSearch evolves a *priority function* within a fixed program skeleton (boilerplate + known structure). The LLM only modifies the critical decision logic. This constrains the search space dramatically.

2. **Island model for diversity.** Multiple subpopulations (optimally ~10 islands) evolve independently. Periodically, the worst half of programs in an island are replaced by clones from another island (migration). This prevents premature convergence.

3. **Programs clustered by signature.** Within each island, programs are grouped by their performance *profile* across test cases (not just aggregate score). This preserves functionally diverse approaches even when scores are similar.

4. **Compact programs favored.** Shorter code is preferred (low Kolmogorov complexity), which improves both interpretability and the LLM's ability to reason about the code.

5. **Millions of LLM samples.** FunSearch generates millions of candidates with fast evaluation (<20 min per candidate on single CPU). Volume matters -- most candidates are bad, but the evolutionary process surfaces rare good ones.

6. **Best-shot prompting.** The LLM receives 1-2 exemplar programs (the best from the database), possibly annotated with performance statistics, and is asked to improve upon them. This implicitly performs crossover and mutation.

**Results:** Discovered the largest known cap sets (biggest advance in 20 years), and bin-packing heuristics that outperformed established algorithms.

**Source:** [FunSearch - Google DeepMind Blog](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/) | [Nature paper](https://www.nature.com/articles/s41586-023-06924-6)

---

## 2. AlphaEvolve (Google DeepMind, 2025)

**What it is:** The successor to FunSearch. A general-purpose evolutionary coding agent that evolves entire codebases (not just single functions), using an ensemble of Gemini models.

**Architecture:**
- **Prompt Sampler** -- assembles prompts with previously discovered solutions
- **LLM Ensemble** -- Gemini Flash (fast, breadth) + Gemini Pro (deep, quality)
- **Evaluator Pool** -- deterministic evaluators with cascading filters
- **Program Database** -- evolutionary database combining MAP-Elites + island models

**Key improvements over FunSearch:**

1. **Ensemble of models.** Flash handles exploration (high throughput, many candidates). Pro handles exploitation (deep reasoning on promising candidates). This is a crucial insight: *use cheap fast models for breadth, expensive models for depth*.

2. **Cascading evaluation.** Lightweight semantic checks eliminate nonviable candidates early. Only promising ones get full evaluation. This saves massive compute.

3. **Evolves entire codebases.** Can modify hundreds of lines across multiple files, not just a single function. Supports any programming language.

4. **Thousands (not millions) of samples.** The ensemble approach + cascading evaluation makes it ~1000x more sample-efficient than FunSearch.

5. **Multi-objective optimization.** Can optimize for multiple metrics simultaneously.

**Results:** Improved 4x4 matrix multiplication (beyond Strassen's 1969 algorithm), improved solutions on 20% of 50+ open math problems, improved Google data center efficiency.

**Source:** [AlphaEvolve - Google DeepMind Blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | [Paper](https://arxiv.org/abs/2506.13131)

---

## 3. OpenEvolve / CodeEvolve (Open Source Implementations)

**Practical implementation details from open-source AlphaEvolve replicas:**

**OpenEvolve configuration:**
```yaml
database:
  population_size: 500
  num_islands: 5
  exploitation_ratio: 0.7  # 70% refine best, 30% explore
llm:
  primary_model: "gemini-2.0-flash-lite"  # fast, cheap
  secondary_model: "gemini-2.0-flash"     # deeper
  temperature: 0.7
```

**CodeEvolve findings:**
- **LLM weighting:** 80% Flash (cheap/fast), 20% Pro (expensive/deep)
- **Migration:** Ring topology across 5 islands, 10% elite exchange every 40 epochs
- **Population:** Max 40 individuals per island, starting from trivial solutions
- **Failed solutions tracked for instructive feedback** -- not discarded

**Two-phase strategy for breaking plateaus:**
- Phase 1 (initial exploration): Higher exploitation ratio (0.7), smaller population
- Phase 2 (breakthrough): Lower exploitation ratio (0.6), updated prompts encouraging "radical innovation," larger population

**Critical practical finding:** Different problems benefit from different operator combinations. No single configuration is universally optimal. This suggests dynamic scheduling between exploration and exploitation.

**Source:** [OpenEvolve on HuggingFace](https://huggingface.co/blog/codelion/openevolve) | [CodeEvolve paper](https://arxiv.org/html/2510.14150v1)

---

## 4. OPRO -- Optimization by PROmpting (Google DeepMind, ICLR 2024)

**What it is:** Uses an LLM as a general-purpose optimizer by describing the optimization task in natural language and including previous solution-score pairs in the prompt.

**How the meta-prompt works:**
1. Describe the optimization goal in natural language
2. Include previously generated solutions paired with their scores
3. **Crucially: order solutions by ascending score** (worst to best)
4. Ask the LLM to generate improved solutions
5. Evaluate new solutions, add to the history, repeat

**Why ascending order matters:** LLMs have recency bias in in-context learning -- they pay more attention to the last examples. Putting the best solutions last makes the LLM more likely to generate similar-but-improved variants.

**Key findings:**
- Temperature controls exploration vs exploitation (higher = more exploration)
- Generate multiple solutions per step to reduce instability
- Works well for discrete optimization with manageable search spaces
- Fails on highly complex combinatorial problems (e.g., TSP with >10 nodes)
- Optimized prompts outperformed human-designed ones by up to 8% (GSM8K) and 50% (Big-Bench Hard)
- **Requires capable models** -- small LLMs show limited optimization ability

**Source:** [OPRO paper](https://arxiv.org/abs/2309.03409) | [Summary](https://fanpu.io/summaries/2024-07-13-large-language-models-as-optimizers/)

---

## 5. REx -- Refine, Explore, Exploit (NeurIPS 2024)

**What it is:** Frames iterative LLM code repair as an arm-acquiring bandit problem, solved with Thompson Sampling.

**The core insight:** When you have multiple code versions to potentially refine, which one should you pick? Greedy (always refine the best) wastes compute on local optima. Breadth-first (try everything equally) wastes compute on bad code. REx adapts dynamically.

**Thompson Sampling formula:**
```
Beta(1 + C*h, 1 + C*(1-h) + N)
```
Where:
- `h` = heuristic value (e.g., test case pass rate)
- `N` = number of failed refinement attempts for this code
- `C` = hyperparameter controlling exploration intensity

**Two principles:**
1. **Focus on Promising Code** -- prioritize versions with higher test pass rates
2. **Avoid Over-Focusing** -- as failed refinement attempts accumulate (N increases), the selection probability decreases automatically

**Results:** Saves 10-80% of LLM queries vs baselines. State-of-the-art on loop invariant synthesis, visual reasoning puzzles, and competition programming. Consistent across GPT-4, GPT-3.5-turbo, Claude-3.5-Sonnet, and Llama-3.1-405B.

**Implementation:** ~10 lines of code. Trivial to add to existing systems.

**Source:** [REx project page](https://haotang1995.github.io/projects/rex) | [Paper](https://arxiv.org/abs/2405.17503)

---

## 6. EvoPrompting (ICLR 2024)

**What it is:** Uses LLMs as mutation and crossover operators in an evolutionary neural architecture search.

**Key findings:**
- **LLM as crossover operator:** Concatenate multiple parent programs into a prompt. The LLM synthesizes offspring that integrate successful patterns from all parents -- *semantic crossover*, not just code splicing.
- **LLM as mutation operator:** The LLM can make both fine-grained tweaks and radical structural changes, unlike traditional mutation operators.
- **Quality-Diversity:** Combine LLM mutation with MAP-Elites style diversity maintenance for broad coverage of the solution space.

**Source:** [EvoPrompting - OpenReview](https://openreview.net/forum?id=ifbF4WdT8f) | [MIT paper](https://dspace.mit.edu/bitstream/handle/1721.1/156878/10710_2024_Article_9494.pdf)

---

## 7. AlphaCodium -- Flow Engineering (2024)

**What it is:** A multi-stage, test-driven approach to code generation that replaces prompt engineering with "flow engineering."

**The flow (pre-processing then iteration):**

Pre-processing (all natural language, no code yet):
1. Problem reflection -- identify goals, inputs, outputs, rules, constraints
2. Public test reasoning -- explain why each test produces its output
3. Generate 2-3 solution approaches in natural language
4. Rank and select the most robust approach

Iterative code phase:
1. Generate initial code
2. Run against tests
3. Fix failures iteratively
4. Generate 6-8 additional AI tests covering edge cases
5. Continue refining against expanded test suite

**Key techniques:**

1. **Test Anchors.** Public tests become "anchors" (known correct). When fixing against AI-generated tests, the code must still pass all anchors. This prevents cascading regressions.

2. **Test enrichment.** "Generating more tests is easier than generating a full solution." Create diverse tests (edge cases, large inputs) to surface bugs the LLM can then fix.

3. **YAML over JSON** for structured output -- handles special characters in code better.

4. **Modular code generation** -- multiple small functions, not monolithic implementations.

5. **Soft decision-making** -- avoid yes/no correctness questions. Use double validation. Postpone irreversible decisions.

6. **Bullet-point reasoning** -- encourages semantic segmentation of problems.

**Results:** GPT-4 pass@5 jumped from 19% to 44% on CodeContests. Uses ~15-20 LLM calls per solution (vs AlphaCode's 1 million). 10,000x more sample-efficient.

**Source:** [AlphaCodium paper](https://ar5iv.labs.arxiv.org/html/2401.08500) | [Qodo blog](https://www.qodo.ai/blog/qodoflow-state-of-the-art-code-generation-for-code-contests/)

---

## 8. SBLLM -- Search-Based LLMs for Code Optimization (ICSE 2025)

**What it is:** Integrates LLMs with evolutionary search for code performance optimization (execution speed), using three synergistic components.

**Three components:**

1. **Execution-based selection.** Evaluate candidates by accuracy (pass tests) + speedup rate. Abstract code into ASTs to avoid redundant solutions. Include *incorrect* samples ranked by edit distance -- LLMs can learn from mistakes.

2. **Adaptive pattern retrieval.** Two types:
   - *Similar patterns*: Examples with same input characteristics but correct optimization
   - *Different patterns*: Examples with comparable structure but distinct strategies
   Uses BM25 similarity on both input code and optimization deltas (what was deleted/added).

3. **Genetic Operator Chain-of-Thought (GO-COT).** Prompt the LLM to:
   - *Crossover*: combine advantages from multiple representative samples
   - *Mutation*: incorporate novel optimization patterns
   - *Synthesis*: generate improved code integrating both

**Results:** Up to 209% execution speed improvement. Outperforms baselines by 8-28%.

**Source:** [SBLLM paper](https://arxiv.org/abs/2408.12159)

---

## 9. Self-Refine (2023) and LLMLOOP (ICSME 2025)

**Self-Refine:** A single-LLM iterative loop: GENERATE -> FEEDBACK -> REFINE -> FEEDBACK -> REFINE...

Key: Feedback must be *actionable* -- not just "this is wrong" but "this is wrong because X, fix it by Y." Localizes problems and provides improvement instructions. Maintains history of previous outputs to avoid repeating mistakes. 5-40% improvement across tasks.

**LLMLOOP:** Five specialized feedback loops for code:
1. Resolving compilation errors
2. Addressing static analysis issues
3. Fixing test case failures
4. Improving test quality via mutation analysis
5. General code quality refinement

**Source:** [Self-Refine](https://selfrefine.info/) | [LLMLOOP](https://www.researchgate.net/publication/394085087)

---

## 10. LangProp (Oxford, 2024)

**What it is:** Treats code as trainable parameters and the LLM as the optimizer. Iteratively optimizes code against a dataset of input-output pairs.

**Training loop:**
1. LLM generates multiple code variations
2. Execute all against training data, catch exceptions
3. Policy tracker records inputs, outputs, exceptions, metric scores
4. Rank implementations by objective function
5. Feed top performers + their failure modes back to LLM
6. LLM generates improved variants
7. Repeat

**Key insight:** The policy tracker maintains full records of *what failed and why*, giving the LLM rich semantic signals for improvement -- not just scores.

**Source:** [LangProp blog (Oxford VGG)](https://www.robots.ox.ac.uk/~vgg/blog/langprop-a-code-optimization-framework-using-large-language-models-applied-to-driving.html)

---

## 11. OpenAI's Approach (Codex / o3)

OpenAI's published work focuses less on iterative optimization frameworks and more on scaling reasoning at inference time:

- **Codex** (2025) is powered by codex-1, a version of o3 optimized for software engineering. Trained with RL on real-world coding tasks. Can iteratively run tests until passing.
- **o3** uses extended chain-of-thought reasoning + test-time search during inference. Achieved ~2727 ELO on Codeforces.
- The core approach: generate long internal reasoning chains, then generate code. Repeat with test execution feedback.

No public equivalent of FunSearch/AlphaEvolve from OpenAI, but the reasoning model approach (o1/o3) effectively does internal iterative refinement within a single generation.

**Source:** [OpenAI Codex](https://openai.com/index/introducing-codex/) | [Competitive Programming with Reasoning Models](https://arxiv.org/html/2502.06807v1)

---

## Synthesized Principles: What Makes the Feedback Loop Work

### A. Feedback Signal Design

1. **Execution results are essential.** Every successful system uses actual code execution (not just LLM self-evaluation). Scores, test results, exceptions, and runtime metrics provide ground truth.

2. **Combine scores with natural language.** Scores alone tell the LLM *how much* to change, not *what* to change. Natural language feedback about failure modes, exceptions, and specific test cases gives actionable direction. The best systems provide both.

3. **Include failure context.** LangProp and SBLLM both track *what* failed and *how* -- not just pass/fail. Exceptions, incorrect outputs, and failing test cases are all valuable signal.

4. **Test anchors prevent regression.** AlphaCodium's key insight: when iterating, lock in known-passing tests as anchors. New changes must not break them.

### B. Exploration vs Exploitation

5. **Island models prevent premature convergence.** FunSearch, AlphaEvolve, CodeEvolve all use multiple independent subpopulations. This is the single most consistent architectural pattern across successful systems.

6. **Use Thompson Sampling (REx) for cheap adaptive exploration.** ~10 lines of code, saves 10-80% of LLM calls. Track failed refinement attempts per candidate; reduce probability of selecting repeatedly-failing code.

7. **Two-phase strategy for plateaus.** Start with higher exploitation (refine what works). When progress stalls, shift to higher exploration (encourage radical changes, update prompts to request innovation).

8. **Cheap model for breadth, expensive model for depth.** AlphaEvolve's Flash+Pro pattern. Generate many candidates cheaply, then use the best model on the most promising ones.

### C. Getting Fundamentally Different Approaches

9. **Semantic crossover via LLM.** Give the LLM 2+ parent programs and ask it to combine their strengths. LLMs can perform semantic crossover that traditional genetic operators cannot -- understanding *what* each parent does well and synthesizing.

10. **Show diverse exemplars, not just the best.** FunSearch clusters programs by their test-case signature (behavioral fingerprint). Sampling exemplars from different clusters exposes the LLM to fundamentally different strategies.

11. **Adaptive pattern retrieval (SBLLM).** Retrieve optimization patterns that are *different* from the current approach, not just similar ones. This actively pushes toward unexplored strategies.

12. **Evolve the skeleton, not the whole program.** FunSearch constrains the LLM to modifying only the critical decision function within a fixed skeleton. This focuses creativity on where it matters.

### D. Preventing Repeated Mistakes

13. **Track and show failure history.** Self-Refine maintains history of previous outputs. LangProp logs all exceptions. SBLLM includes incorrect samples ranked by edit distance. Showing the LLM what has already been tried and failed prevents repetition.

14. **AST-based deduplication (SBLLM).** Abstract code to syntax trees before comparing. This catches semantically identical solutions with superficial differences.

15. **Worst-half replacement (FunSearch).** Periodically purge the worst half of an island's population. Prevents dead weight from consuming selection probability.

16. **Failed attempts penalize future selection (REx).** The more times you've tried to improve a candidate and failed, the less likely it should be selected again.

### E. Compute Budget Efficiency

17. **Cascading evaluation.** Cheap filters first (syntax check, basic execution), expensive evaluation only for candidates that pass. AlphaEvolve and CodeEvolve both do this.

18. **Generate tests, not just solutions.** AlphaCodium: "generating more tests is easier than generating a full solution." Use the LLM to create a richer test suite, then iterate the code against it.

19. **Speed of the inner loop matters enormously.** OpenEvolve found that model inference latency was the primary bottleneck. Fast models + many generations beats slow models + few generations.

20. **~15-20 LLM calls per solution** is the sweet spot for focused code generation (AlphaCodium). Evolutionary search over populations needs thousands of calls. Plan accordingly.

### F. Prompt Construction

21. **Order previous solutions by ascending score (OPRO).** Best solutions last. Exploits LLM recency bias.

22. **Include performance statistics with exemplars.** Not just the code -- also its score, which tests it passed, and where it failed.

23. **Use structured output (YAML > JSON for code).** YAML handles special characters in code better than JSON.

24. **Natural language reasoning before code.** AlphaCodium's pre-processing phase (problem reflection, test reasoning, approach ranking) significantly outperforms direct code generation.

25. **Bullet-point analysis encourages systematic thinking.** Rather than prose, use structured bullet points to break down the problem.

### G. System Architecture Patterns

26. **The universal loop:** SELECT parents -> PROMPT LLM -> EVALUATE output -> UPDATE population -> repeat
27. **Population-based > single-trajectory.** Every top-performing system maintains a *population* of solutions, not just the single best.
28. **Separate the skeleton from the evolved code.** Mark evolution targets explicitly (OpenEvolve uses `EVOLVE-BLOCK-START/END` comments).
29. **Checkpoint and resume.** Long-running optimization needs persistence. Store population + evaluation history.
30. **Self-generated exemplars compound.** Storing successful trajectories as future in-context examples lifted performance from 73% to 89%+ in agent settings.
