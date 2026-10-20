"""LoRA fine-tune Dream 7B for NPC dialogue (masked-diffusion SFT).

Recipe (from the labs' playbook): prompt stays clean, response is masked at a
per-example ratio t ~ U(0.15, 1), loss = plain cross-entropy on masked
response positions only, computed on UNSHIFTED logits (diffusion predicts the
token at the same position - do NOT use the model's built-in causal-LM loss,
which shifts labels).

Base model loads in 8-bit (QLoRA-style) so it fits a T4 16GB. Requires
transformers==4.53.3 (Dream's custom code breaks on 5.x) + peft.
"""

import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from .masking import MaskingCollator, encode_example, to_bidirectional_mask

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def load_examples(path, tokenizer, max_len, max_examples, seed=0):
    raw = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    random.Random(seed).shuffle(raw)
    encoded, dropped = [], 0
    for ex in raw:
        enc = encode_example(tokenizer, ex, max_len=max_len)
        if enc is None:
            dropped += 1
            continue
        encoded.append(enc)
        if len(encoded) >= max_examples:
            break
    print(f"examples: {len(encoded)} usable, {dropped} dropped (>{max_len} tokens)")
    return encoded


def train(base_model="Dream-org/Dream-v0-Instruct-7B", data="data/sft_examples.jsonl",
          out="npc_adapter", epochs=1, lr=1e-4, lora_r=16, batch=4, accum=4,
          max_len=512, max_examples=12000, seed=0, log_every=20):
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModel, AutoTokenizer, BitsAndBytesConfig,
                              get_cosine_schedule_with_warmup)

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    encoded = load_examples(data, tokenizer, max_len, max_examples, seed)

    model = AutoModel.from_pretrained(
        base_model, trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
        target_modules=LORA_TARGETS,
    ))
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr)
    steps_per_epoch = math.ceil(len(encoded) / (batch * accum))
    total_steps = steps_per_epoch * epochs
    sched = get_cosine_schedule_with_warmup(
        opt, int(0.03 * total_steps), total_steps)
    collate = MaskingCollator(tokenizer, max_len=max_len, seed=seed)
    device = next(model.parameters()).device
    # attention mask must match the model's compute dtype (bf16 on GPU) —
    # SDPA rejects mismatched mask/query dtypes on CUDA backends
    mask_dtype = model.get_input_embeddings().weight.dtype

    rng = random.Random(seed)
    step, micro, window_loss, t0 = 0, 0, 0.0, time.time()
    model.train()
    done = False
    for epoch in range(epochs):
        if done:
            break
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            feats = collate([encoded[i] for i in idx])
            feats = {k: v.to(device) for k, v in feats.items()}
            logits = model(
                input_ids=feats["input_ids"],
                attention_mask=to_bidirectional_mask(
                    feats["attention_mask"], dtype=mask_dtype),
            ).logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                feats["labels"].view(-1), ignore_index=-100) / accum
            loss.backward()
            window_loss += loss.item()
            micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0 or step == 1:
                    print(f"step {step}/{total_steps} "
                          f"loss {window_loss * accum / micro:.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                    window_loss, micro = 0.0, 0
                if step >= total_steps:
                    done = True
                    break
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"adapter saved to {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_examples.jsonl")
    ap.add_argument("--out", default="npc_adapter")
    ap.add_argument("--base", default="Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-examples", type=int, default=12000)
    args = ap.parse_args()
    train(base_model=args.base, data=args.data, out=args.out, epochs=args.epochs,
          lr=args.lr, lora_r=args.lora_r, batch=args.batch, accum=args.accum,
          max_len=args.max_len, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
