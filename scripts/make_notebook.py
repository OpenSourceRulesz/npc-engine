"""Regenerate colab_phase0.ipynb with the current npc_engine package embedded
as base64, so the notebook runs in Colab with no GitHub clone needed."""

import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    files = {}
    for name in ["__init__.py", "personas.py", "engine.py", "demo.py"]:
        p = ROOT / "npc_engine" / name
        files[f"npc_engine/{name}"] = base64.b64encode(p.read_bytes()).decode()

    restore = ["import base64, pathlib\n", "files = {\n"]
    for path, b64 in files.items():
        restore.append(f'    "{path}": "{b64}",\n')
    restore += [
        "}\n",
        "for path, b64 in files.items():\n",
        "    p = pathlib.Path(path)\n",
        "    p.parent.mkdir(parents=True, exist_ok=True)\n",
        "    p.write_bytes(base64.b64decode(b64))\n",
        'print("npc_engine package restored:", sorted(str(p) for p in pathlib.Path("npc_engine").glob("*.py")))\n',
    ]

    def md(*lines):
        return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

    def code(*lines):
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": list(lines)}

    cells = [
        md("# NPC Engine — Phase 0\n",
           "\n",
           "**Goal:** prove a diffusion LM (Dream 7B Instruct) can do in-character NPC dialogue with "
           "parallel candidate generation + confidence ranking, at interactive latency, on one Colab GPU.\n",
           "\n",
           "What to watch for:\n",
           "- **Latency**: p50 per turn (8 candidates denoised in one batched pass)\n",
           "- **Confidence**: does the top candidate's score feel honest? Do weak lines score low?\n",
           "- **Fallbacks**: low-confidence turns ship a canned line instead of a broken one\n",
           "- **Diversity**: are the 8 candidates actually different lines?\n"),
        code("!pip install -q transformers accelerate bitsandbytes"),
        md("## Restore the `npc_engine` package (embedded in this notebook)"),
        code(*restore),
        md("## Load the model (8-bit fits a T4 16GB)"),
        code("from npc_engine.engine import NPCEngine\n",
             "from npc_engine.personas import STARTER_PERSONAS\n",
             "\n",
             "engine = NPCEngine(load_in_8bit=True)\n",
             "print('ready on', engine.device)\n"),
        md("## Scripted conversation — 3 NPCs, watch candidates + confidences"),
        code("scenes = [\n",
             "    (0, \"Evening. The miller's cart threw a wheel on the north road; bandits were seen there last week.\",\n",
             "     [\"Any work for me?\", \"What happened on the north road?\"]),\n",
             "    (1, \"Morning market day. The square is busy; a traveler was robbed yesterday.\",\n",
             "     [\"What's good today?\", \"Heard any gossip?\"]),\n",
             "    (2, \"Late night. Rain. The tavern is nearly empty.\",\n",
             "     [\"I need to pass the north gate.\", \"Are the bandits still out there?\"]),\n",
             "]\n",
             "for idx, world, lines in scenes:\n",
             "    p = STARTER_PERSONAS[idx]\n",
             "    print(f\"\\n===== {p.name} ({p.role}) =====\")\n",
             "    for line in lines:\n",
             "        r = engine.respond(p, world, line)\n",
             "        flag = ' [FALLBACK]' if r.used_fallback else ''\n",
             "        print(f\"\\nyou> {line}\")\n",
             "        print(f\"{p.name}> {r.text}{flag}  (conf {r.confidence:.2f}, {r.latency_ms:.0f} ms)\")\n",
             "        for t, c in r.candidates[1:3]:\n",
             "            print(f\"    alt ({c:.2f}): {t}\")\n"),
        md("## Benchmark — 20 turns, latency distribution"),
        code("import statistics\n",
             "p = STARTER_PERSONAS[0]\n",
             "world = \"Evening. The miller's cart threw a wheel on the north road.\"\n",
             "probes = [\"Any work for me?\", \"What happened on the north road?\", \"I need a sword, fast.\",\n",
             "          \"Tell me about the miller.\", \"Are the bandits still out there?\"] * 4\n",
             "lat, conf, fb = [], [], 0\n",
             "for q in probes:\n",
             "    r = engine.respond(p, world, q)\n",
             "    lat.append(r.latency_ms); conf.append(r.confidence); fb += r.used_fallback\n",
             "lat.sort()\n",
             "print(f\"p50 {statistics.median(lat):.0f} ms | p95 {lat[int(0.95*len(lat))-1]:.0f} ms | \"\n",
             "      f\"mean conf {statistics.mean(conf):.2f} | fallbacks {fb}/{len(probes)}\")\n"),
        md("## Reading the results\n",
           "\n",
           "- **p50 under ~1s** on a T4 with 8-bit = the concept works; a 5090/4090 at fp16 will be several times faster.\n",
           "- **Confidence vs quality**: if lines that read wrong score low and good lines score high, the ranking signal is honest — that's the anti-gameslop guard.\n",
           "- **Next steps after this**: fine-tune Dream 7B (LoRA) on persona-conditioned dialogue (agentlans triples + LIGHT + synthetic), then the Unity plugin.\n"),
    ]

    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "accelerator": "GPU"},
          "cells": cells}
    out = ROOT / "colab_phase0.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print("notebook written:", out, "| cells:", len(cells))


if __name__ == "__main__":
    main()
