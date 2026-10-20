"""Generate colab_finetune.ipynb — self-contained LoRA fine-tune notebook.

Embeds the whole npc_engine package (base64) so the notebook works from a
bare Colab runtime: data build (agentlans convert + SFT build) -> train ->
base-vs-tuned eval -> speed check at steps 32 vs 16.
"""

import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

PACKAGE_FILES = [
    "npc_engine/__init__.py",
    "npc_engine/personas.py",
    "npc_engine/engine.py",
    "npc_engine/masking.py",
    "npc_engine/finetune.py",
    "npc_engine/demo.py",
    "npc_engine/data/__init__.py",
    "npc_engine/data/convert_agentlans.py",
    "npc_engine/data/build_sft.py",
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
        'print("npc_engine version:", npc_engine.__version__, " (must say 0.2.0)")',
    ]
    return "\n".join(lines)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


def build():
    cells = [
        md("# NPC Engine — LoRA fine-tune (Dream 7B)\n\n"
           "Trains the NPC specialist on 74k dialogue triples (agentlans, CC-BY-4.0): "
           "persona + world state stay clean, only the NPC's line is masked — the same "
           "recipe the big diffusion labs use.\n\n"
           "**Runtime must be a T4 GPU** (Runtime → Change runtime type → T4). "
           "Run the cells top to bottom."),
        md("## 1. Install (pinned — Dream's code breaks on newer transformers)"),
        code("!pip install -q transformers==4.53.3 peft accelerate bitsandbytes"),
        md("## 2. Restore the npc_engine package"),
        code(restore_cell()),
        md("## 3. Build the training data"),
        code("!python -m npc_engine.data.convert_agentlans\n"
             "!python -m npc_engine.data.build_sft"),
        md("## 4. Train (LoRA on the 8-bit base)\n\n"
           "12,000 examples, 1 epoch, effective batch 16. Watch the loss printouts — "
           "it should trend down. If Colab disconnects, rerun from cell 2 (training "
           "restarts; data rebuild is fast)."),
        code("from npc_engine.finetune import train\n"
             "train(data='data/sft_examples.jsonl', out='npc_adapter',\n"
             "      max_examples=12000, epochs=1)"),
        md("## 5. Base vs fine-tuned, same scenes\n\n"
           "Compare the lines: the tuned model's should be cleaner grammatically and "
           "stick to 1-2 short lines."),
        code(
            "import gc, torch\n"
            "from npc_engine import NPCEngine, STARTER_PERSONAS\n\n"
            "SCENES = [\n"
            "    ('Borin Ironfist', 'The market square at noon. A traveler\\'s cart has a cracked wheel; bandits were seen on the north road.', 'My cart wheel cracked just outside town. Can you fix it?'),\n"
            "    ('Mira Thistledown', 'The inn common room at dusk. Market day was busy; a robbery occurred at the miller\\'s stall.', 'What\\'s the word around town today?'),\n"
            "    ('Captain Sera Voss', 'The town gate at shift change. A curfew begins at sundown; strangers need a pass.', 'Let me through, I have business inside.'),\n"
            "]\n\n"
            "def run(engine, label):\n"
            "    print(f'===== {label} =====')\n"
            "    for name, world, line in SCENES:\n"
            "        p = STARTER_PERSONAS[name]\n"
            "        r = engine.respond(p, world, line)\n"
            "        print(f'{name}: \"{r.line}\"  (conf {r.confidence:.2f}, {r.latency_ms:.0f} ms)')\n"
            "    print()\n\n"
            "base = NPCEngine(load_in_8bit=True)\n"
            "run(base, 'BASE MODEL')\n"
            "del base; gc.collect(); torch.cuda.empty_cache()\n\n"
            "tuned = NPCEngine(load_in_8bit=True, adapter_path='npc_adapter')\n"
            "run(tuned, 'FINE-TUNED')"
        ),
        md("## 6. Speed check: steps 32 vs 16 (tuned model)\n\n"
           "The Phase 0 T4 number was ~58s per turn at 32 steps. Halving the steps "
           "should roughly halve that — check the trade against line quality above."),
        code(
            "import time\n"
            "p = STARTER_PERSONAS['Borin Ironfist']\n"
            "world = 'The market square at noon. A traveler\\'s cart has a cracked wheel; bandits were seen on the north road.'\n"
            "for steps in (32, 16):\n"
            "    ts = []\n"
            "    for _ in range(5):\n"
            "        t0 = time.perf_counter()\n"
            "        r = tuned.respond(p, world, 'Can you fix a cracked wheel?', steps=steps)\n"
            "        ts.append(time.perf_counter() - t0)\n"
            "    ts.sort()\n"
            "    print(f'steps={steps}: p50 {ts[2]*1000:.0f} ms/turn, line: \"{r.line}\"')"
        ),
        md("## 7. Keep the adapter\n\n"
           "The adapter is ~100MB in the `npc_adapter` folder. To keep it past this "
           "session, mount Drive and copy it:\n\n"
           "```python\n"
           "# from google.colab import drive; drive.mount('/content/drive')\n"
           "# !cp -r npc_adapter /content/drive/MyDrive/\n"
           "```"),
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
    out = ROOT / "colab_finetune.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, {len(cells)} cells)")


if __name__ == "__main__":
    build()
