# Phase 0 results — 2026-10-20

Setup: Dream-v0-Instruct-7B (base, zero fine-tuning), 8-bit, Colab **T4 GPU**
(load cell confirmed "ready on cuda"; an earlier stalled attempt in the same
session had run on CPU and was discarded). 8 candidates/turn,
32 denoising steps, confidence threshold 0.45. 6 scripted turns + 20-turn benchmark.

## Numbers

- Latency (T4, 8-bit): **p50 58,569 ms | p95 58,828 ms** per turn (8 candidates) —
  ~7.3 s per candidate line. Not interactive yet, but the near-zero variance shows
  fixed step count = predictable latency, as theorized. The gap to interactive
  (~15x) is engineering, not research: steps 32→16 (~2x), fp16 on a 24GB card
  (~2-3x), Fast-dLLM block cache (~2-3x).
- Mean confidence 0.84; fallbacks **0/20** (all candidates ≥0.72, threshold never fired).

## Quality (the actual Phase 0 questions)

- **In-character: yes.** Borin uses "lad/lass", short and gruff; Mira is warm, chatty,
  food-and-gossip focused; Captain Voss is terse ("Show me your papers." — conf 0.90,
  the best line of the run).
- **World-state grounding: yes.** Each NPC referenced the facts relevant to them —
  miller's wheel + bandits (Borin, Voss), market day + the robbery (Mira) — from the
  world state + their persona knowledge, unprompted.
- **Candidate diversity: yes.** Alternatives are genuinely different lines, not copies.
- **Confidence ranking: directionally honest.** The cleanest lines scored highest
  (0.90), the rambling alt scored lowest (0.72) and wasn't shipped. But confidence
  measures token plausibility, not correctness: Borin's shipped line "Work for you,
  lad, I'm afraid." scored 0.86 despite a missing negation. Ranking works; absolute
  values are optimistic.
- **Grammar warts (base model, expected):** "heard bandits gone by that road",
  "Oh, the the travelers!" — this is what the fine-tune is for.

## Verdict

- Quality thesis: **proven at base-model level** (persona adherence, grounding,
  diversity, ranking) with zero training.
- Speed thesis: **measured on T4/8-bit — 58.6 s/turn, ~7.3 s/line; not interactive
  yet**, with a concrete lever path (fewer steps, fp16/bigger GPU, Fast-dLLM cache).
- Fallback guard: unit-tested, not yet exercised by real data.
- Next levers: LoRA fine-tune on the 73,930 agentlans triples for grammar/format;
  steps 32→16 and Fast-dLLM caching for speed.
