"""Build SFT examples from NPC triples.

Triple -> {system, user, assistant} where:
- system = the canonical engine prompt (persona card + history + situation
  + rules), built by personas.system_prompt_for so train == inference format
- user = the player line (the last turn of the triple's context)
- assistant = the NPC's line
- history = the context turns BEFORE the player line, rendered "Name: line"
"""

import argparse
import json
import pathlib

from npc_engine.personas import system_prompt_for


def triple_to_example(t: dict) -> dict:
    context_lines = [ln for ln in (t.get("context") or "").split("\n") if ln.strip()]
    history = context_lines[:-1] if len(context_lines) > 1 else None
    system = system_prompt_for(
        t["persona"]["name"],
        t["persona"]["description"],
        t["world_state"],
        history=history,
    )
    return {"system": system, "user": t["player_line"], "assistant": t["npc_line"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="data/npc_triples.jsonl")
    ap.add_argument("--out", default="data/sft_examples.jsonl")
    args = ap.parse_args()

    n = 0
    with open(args.triples, encoding="utf-8") as f, \
            open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = triple_to_example(json.loads(line))
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} SFT examples to {args.out}")
    with open(args.out, encoding="utf-8") as f:
        sample = json.loads(f.readline())
    print("\nSAMPLE:")
    print("SYSTEM:", sample["system"][:400])
    print("USER:", sample["user"])
    print("ASSISTANT:", sample["assistant"])


if __name__ == "__main__":
    main()
