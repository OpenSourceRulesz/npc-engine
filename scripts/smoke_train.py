"""Smoke test for the fine-tune path — REAL Dream code, tiny config, CPU.

Downloads Dream's actual modeling files from its HF repo, builds a package,
instantiates a TINY DreamModel from them, wraps it in LoRA with the exact
target modules finetune.py uses, and runs 3 real training steps through
npc_engine.masking (encode + collate) with the real Dream tokenizer.

This exists because mocked tests missed two real load-path crashes in a row.
Run under the transformers version Dream was built for (4.53.3):

    PYTHONPATH=.venvs/tf453 python scripts/smoke_train.py
"""

import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / ".smoke_cache"
REPO = "Dream-org/Dream-v0-Instruct-7B"
CODE_FILES = ["configuration_dream.py", "modeling_dream.py", "generation_utils.py"]


def fetch_code():
    from huggingface_hub import hf_hub_download

    pkg = CACHE / "dreampkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for f in CODE_FILES:
        src = hf_hub_download(REPO, f)
        shutil.copy(src, pkg / f)
    sys.path.insert(0, str(CACHE))


def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    fetch_code()
    from dreampkg.configuration_dream import DreamConfig
    from dreampkg.modeling_dream import DreamModel

    sys.path.insert(0, str(ROOT))
    from npc_engine.finetune import LORA_TARGETS
    from npc_engine.masking import (MaskingCollator, encode_example,
                                    to_bidirectional_mask)

    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    print("tokenizer ok; mask id:", tok.convert_tokens_to_ids("<|mask|>"))

    cfg = DreamConfig(hidden_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=256, vocab_size=152064,
                      max_position_embeddings=1024)
    model = DreamModel(cfg)

    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, target_modules=LORA_TARGETS))
    model.print_trainable_parameters()

    examples = [
        {"system": "You are Borin. A gruff blacksmith.\n\nCurrent situation: A forge.",
         "user": "Can you fix my wheel?",
         "assistant": "Aye. Bring it here, lad."},
        {"system": "You are Mira. A warm innkeeper.\n\nCurrent situation: An inn.",
         "user": "What is the word around town?",
         "assistant": "Bandits on the north road, they say."},
    ]
    encoded = [encode_example(tok, ex) for ex in examples]
    assert all(e is not None for e in encoded)
    collate = MaskingCollator(tok, seed=0)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=1e-3)
    model.train()
    losses = []
    for step in range(3):
        feats = collate(encoded)
        logits = model(
            input_ids=feats["input_ids"],
            attention_mask=to_bidirectional_mask(feats["attention_mask"]),
        ).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                               feats["labels"].view(-1), ignore_index=-100)
        loss.backward()
        if step == 0:
            grads = [p.grad.abs().sum().item() for n, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None]
            assert grads and max(grads) > 0, "no LoRA gradients flowed"
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        print(f"step {step}: loss {loss.item():.4f}")
    assert all(l == l and abs(l) != float("inf") for l in losses), "non-finite loss"
    print("SMOKE OK — real Dream code + LoRA + masking collator train end to end")


if __name__ == "__main__":
    main()
