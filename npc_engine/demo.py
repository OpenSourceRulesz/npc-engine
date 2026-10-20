"""Interactive text-adventure demo for the NPC Engine.

Two modes:
  python -m npc_engine.demo            # talk to the NPCs, see candidates + confidences
  python -m npc_engine.demo --bench    # latency benchmark: 20 turns, reports p50/p95
"""

import argparse
import statistics
import sys

from .engine import NPCEngine
from .personas import STARTER_PERSONAS

WORLD_STATES = [
    "Evening. The miller's cart threw a wheel on the north road; bandits were seen there last week.",
    "Morning market day. The square is busy; a traveler was robbed yesterday.",
    "Late night. Rain. The tavern is nearly empty; a stranger watches from the corner.",
]


def interactive(engine: NPCEngine):
    print("=== NPC Engine — text adventure demo ===\n")
    for i, p in enumerate(STARTER_PERSONAS):
        print(f"  [{i}] {p.name} — {p.role}")
    try:
        choice = int(input("\nPick an NPC (0-5): ").strip() or "0")
        persona = STARTER_PERSONAS[choice % len(STARTER_PERSONAS)]
    except (ValueError, IndexError):
        persona = STARTER_PERSONAS[0]
    world = WORLD_STATES[0]
    print(f"\nYou approach {persona.name}. ({world})")
    print("Type to them. 'quit' to leave, 'npc' to switch character.\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() == "quit":
            break
        if line.lower() == "npc":
            return interactive(engine)
        r = engine.respond(persona, world, line)
        flag = " [FALLBACK]" if r.used_fallback else ""
        print(f"\n{persona.name}> {r.text}{flag}")
        print(f"  (confidence {r.confidence:.2f} | {r.latency_ms:.0f} ms | "
              f"{len(r.candidates)} unique candidates)")
        if len(r.candidates) > 1:
            alts = ", ".join(f'"{t}" ({c:.2f})' for t, c in r.candidates[1:4])
            print(f"  alternatives: {alts}")
        print()


def benchmark(engine: NPCEngine):
    persona = STARTER_PERSONAS[0]
    world = WORLD_STATES[0]
    probes = [
        "Any work for me?",
        "What happened on the north road?",
        "I need a sword, fast.",
        "Tell me about the miller.",
        "Are the bandits still out there?",
    ] * 4
    lat, conf, fb = [], [], 0
    print(f"running {len(probes)} turns, 8 candidates each...")
    for p in probes:
        r = engine.respond(persona, world, p)
        lat.append(r.latency_ms)
        conf.append(r.confidence)
        fb += r.used_fallback
    lat.sort()
    p50 = statistics.median(lat)
    p95 = lat[int(0.95 * len(lat)) - 1]
    print(f"\np50 latency: {p50:.0f} ms   p95 latency: {p95:.0f} ms")
    print(f"mean confidence: {statistics.mean(conf):.2f}   fallbacks: {fb}/{len(probes)}")
    print(f"throughput: {1000 / p50 * 8:.0f} candidate-lines/sec (8 per turn)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--8bit", dest="eightbit", action="store_true",
                    help="load in 8-bit (for 16GB GPUs)")
    args = ap.parse_args()
    print("loading model (first run downloads ~14GB)...", flush=True)
    engine = NPCEngine(model_id=args.model, load_in_8bit=args.eightbit)
    if args.bench:
        benchmark(engine)
    else:
        interactive(engine)


if __name__ == "__main__":
    sys.exit(main())
