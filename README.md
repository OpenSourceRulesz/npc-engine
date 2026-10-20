# NPC Engine

A tiny text-diffusion NPC dialogue engine, **built for indie and small devs**:
free, open source, runs locally — no API keys, no per-player costs, no server.
Proves that a diffusion LM (Dream 7B Instruct) can generate in-character NPC
lines with **parallel candidate generation + confidence ranking**, at
predictable latency, on a single GPU.

Phase 0 (base model, zero training) is validated — see `PHASE0_RESULTS.md`.
Phase 1 is a LoRA fine-tune on 74k openly licensed dialogue triples —
`colab_finetune.ipynb` runs the whole thing free on Colab.

**Playable demo:** `colab_game.ipynb` runs *Base of Three*, a tiny survivor-
camp game (3 survivors, 3 days, state-driven dialogue) on the base model —
no fine-tune needed.

## The diffusion-native design

1. The persona card + world state + player line form the prompt, kept fully
   clean (never masked). Only the NPC's line is denoised.
2. N masked canvases are denoised in **one batched pass** — N candidate lines
   for roughly the wall-clock of one. (AR would need N sequential generations.)
   Diversity comes from temperature/top-p sampling.
3. A single scoring forward pass computes per-candidate **confidence** (mean
   max-probability over the generated span). Candidates are ranked; the best
   ships.
4. If the best candidate scores below threshold, the engine returns one of the
   persona's canned **fallback lines** instead — a low-confidence generation
   never reaches the player. This is the anti-gameslop guard, and it's free.

## Layout

- `npc_engine/personas.py` — persona cards (6 starters) with fallback lines
- `npc_engine/engine.py` — `NPCEngine.respond(...)` → ranked `NPCResponse`
- `npc_engine/demo.py` — interactive text adventure + `--bench` latency mode
- `colab_phase0.ipynb` — self-contained Colab notebook (package rides along
  as base64; no GitHub push needed to run it)

## Run it

On Colab (T4, 8-bit): open `colab_phase0.ipynb`, Runtime → GPU, run all.
First run downloads ~14GB of weights.

Locally with a 24GB+ GPU:

```
PYTHONPATH=. python -m npc_engine.demo
PYTHONPATH=. python -m npc_engine.demo --bench
```

On a 16GB GPU add `--8bit`.

## What Phase 0 proves (before any training spend)

- p50 turn latency with 8 candidates on one GPU
- whether the confidence score is honest (bad lines score low)
- whether candidates are actually diverse

## Next steps

1. Push this repo to GitHub (needs owner's approval)
2. Fine-tune Dream 7B (LoRA) on persona-conditioned dialogue (LIGHT + synthetic)
3. Unity plugin: thin client sending persona + state + player line over localhost
