"""NPC training-triple converter.

agentlans/multi-character-dialogue (CC-BY-4.0) rows look like:
  {setting, characters: {name: description}, conversation: [{from, message}],
   'setting after interaction'}

Each turn i>=1 becomes one training triple:
  persona      = {name, description} of the speaker (their character entry)
  world_state  = setting
  context      = the last `context_turns` prior turns, rendered "Name: message"
  player_line  = turn i-1's message (whoever spoke just before)
  npc_line     = turn i's message (the training target)

Filters: unknown speakers dropped, npc_line outside [min_words, max_words]
dropped, empty messages dropped.
"""

import argparse
import gzip
import json
import pathlib
import urllib.request

HF_URL = (
    "https://huggingface.co/datasets/agentlans/multi-character-dialogue"
    "/resolve/main/train.jsonl.gz"
)


def _words(text: str) -> int:
    return len(text.split())


def convert_row(row: dict, context_turns: int = 6, min_words: int = 2,
                max_words: int = 120):
    """Return (triples, stats) for one scene row."""
    triples, skipped = [], {"unknown_speaker": 0, "bad_length": 0}
    setting = (row.get("setting") or "").strip()
    characters = row.get("characters") or {}
    conversation = row.get("conversation") or []
    if not isinstance(conversation, list) or len(conversation) < 2:
        return triples, skipped
    for i in range(1, len(conversation)):
        turn = conversation[i]
        speaker = turn.get("from")
        message = (turn.get("message") or "").strip()
        if not message or not speaker:
            continue
        description = characters.get(speaker)
        if not description:
            skipped["unknown_speaker"] += 1
            continue
        nw = _words(message)
        if nw < min_words or nw > max_words:
            skipped["bad_length"] += 1
            continue
        prev = conversation[i - 1]
        prior = conversation[max(0, i - context_turns):i]
        context = "\n".join(
            f"{t.get('from')}: {(t.get('message') or '').strip()}" for t in prior
        )
        triples.append({
            "persona": {"name": speaker, "description": description.strip()},
            "world_state": setting,
            "context": context,
            "player_line": (prev.get("message") or "").strip(),
            "npc_line": message,
        })
    return triples, skipped


def convert_file(path: pathlib.Path, out_path: pathlib.Path,
                 context_turns=6, min_words=2, max_words=120, limit=None):
    total_rows, total_triples = 0, 0
    skipped = {"unknown_speaker": 0, "bad_length": 0}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f, \
            open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            triples, sk = convert_row(json.loads(line), context_turns,
                                      min_words, max_words)
            for s, v in sk.items():
                skipped[s] += v
            for t in triples:
                out.write(json.dumps(t, ensure_ascii=False) + "\n")
            total_triples += len(triples)
            if limit and total_rows >= limit:
                break
    return {"rows": total_rows, "triples": total_triples, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--max-words", type=int, default=120)
    args = ap.parse_args()

    ddir = pathlib.Path(args.data_dir)
    ddir.mkdir(parents=True, exist_ok=True)
    src = ddir / "agentlans-train.jsonl.gz"
    if not src.exists():
        print("downloading agentlans/multi-character-dialogue (CC-BY-4.0)...", flush=True)
        urllib.request.urlretrieve(HF_URL, src)
    out = ddir / "npc_triples.jsonl"
    stats = convert_file(src, out, min_words=args.min_words,
                         max_words=args.max_words, limit=args.limit)
    print(json.dumps(stats, indent=2))
    print("wrote", out)
    # show a sample
    with open(out, encoding="utf-8") as f:
        sample = json.loads(f.readline())
    print("\nSAMPLE TRIPLE:")
    print(json.dumps(sample, indent=2, ensure_ascii=False)[:900])


if __name__ == "__main__":
    main()
