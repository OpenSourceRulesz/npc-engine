"""Generate colab_game.ipynb — playable Base of Three on the base model.

If an npc_adapter folder exists in the Colab working directory (from the
fine-tune notebook), the game uses it automatically; otherwise the base
Dream 7B plays. Lines use steps=16 / 4 candidates to keep turns snappy.
"""

import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

PACKAGE_FILES = [
    "npc_engine/__init__.py",
    "npc_engine/personas.py",
    "npc_engine/engine.py",
    "npc_engine/games/__init__.py",
    "npc_engine/games/base_of_three.py",
]


def restore_cell() -> str:
    lines = [
        "# --- restore the npc_engine package (embedded in this notebook) ---",
        "import base64, pathlib",
        "FILES = {",
    ]
    for rel in PACKAGE_FILES:
        data = (ROOT / rel).read_bytes()
        b64 = base64.b64encode(data).decode()
        lines.append(f"    {rel!r}: {b64!r},")
    lines += [
        "}",
        "for rel, b64 in FILES.items():",
        "    p = pathlib.Path(rel)",
        "    p.parent.mkdir(parents=True, exist_ok=True)",
        "    p.write_bytes(base64.b64decode(b64))",
        "import npc_engine",
        'print("npc_engine version:", npc_engine.__version__, " (must say 0.3.0)")',
    ]
    return "\n".join(lines)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


def build():
    cells = [
        md("# Base of Three — a tiny survivor game on the NPC engine\n\n"
           "Three survivors, three days. Assign tasks each morning; every "
           "evening each survivor says a line generated from their live state "
           "(morale, injuries, infection). Keep everyone alive, fed, and from "
           "giving up until day 4.\n\n"
           "**Runtime must be a T4 GPU.** Run the cells top to bottom, then "
           "type your task choices (scavenge / guard / rest) when asked."),
        md("## 1. Install (pinned — Dream's code breaks on newer transformers)"),
        code("!pip install -q transformers==4.53.3 accelerate bitsandbytes"),
        md("## 2. Restore the npc_engine package"),
        code(restore_cell()),
        md("## 3. Play\n\n"
           "Uses your fine-tuned `npc_adapter` automatically if it exists in "
           "this runtime; otherwise the base model plays."),
        code(
            "import os\n"
            "from npc_engine import NPCEngine\n"
            "from npc_engine.games.base_of_three import play, BARK_PROMPT\n\n"
            "adapter = 'npc_adapter' if os.path.exists('npc_adapter/adapter_config.json') else None\n"
            "print('engine:', 'fine-tuned adapter' if adapter else 'base model')\n"
            "eng = NPCEngine(load_in_8bit=True, adapter_path=adapter)\n\n"
            "def speak(persona, world):\n"
            "    r = eng.respond(persona, world, BARK_PROMPT, steps=16,\n"
            "                    n_candidates=4, max_new_tokens=48)\n"
            "    return r.line\n\n"
            "play(speak)"
        ),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = ROOT / "colab_game.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, {len(cells)} cells)")


if __name__ == "__main__":
    build()
