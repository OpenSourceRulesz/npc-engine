"""Masked-diffusion SFT encoding + collator.

The labs' recipe (LLaDA / Dream): the prompt stays fully clean; only the
response is noised. Per example we sample a mask ratio t ~ U(min_ratio, 1),
mask each response token independently with probability t, and compute loss
ONLY on masked response positions (labels = -100 everywhere else).

At least one response token is always masked, so every example contributes.
"""

import random

import torch

MASK_TOKEN = "<|mask|>"
IM_END_TOKEN = "<|im_end|>"


def to_bidirectional_mask(attention_mask: torch.Tensor,
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """2D padding mask (1=real, 0=pad) -> 4D additive mask (B, 1, L, L).

    Dream's attention layers pass the mask straight into SDPA / eager
    attention as an additive bias and expect it 4D — they do NOT convert a
    2D long mask (their generation code builds 4D masks itself). Diffusion
    is bidirectional: every real token attends to every real token; padded
    positions are masked out as keys only.
    """
    neg = torch.finfo(dtype).min
    key_ok = attention_mask.bool()
    mask = torch.zeros(
        (attention_mask.size(0), 1, attention_mask.size(1), attention_mask.size(1)),
        dtype=dtype, device=attention_mask.device,
    )
    mask.masked_fill_(~key_ok[:, None, None, :], neg)
    return mask


def encode_example(tokenizer, example: dict, max_len: int = 512):
    """(prompt_ids, resp_ids) or None if the pair exceeds max_len.
    resp_ids ends with <|im_end|> so the model learns to stop."""
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    resp_ids = tokenizer.encode(example["assistant"], add_special_tokens=False)
    resp_ids = resp_ids + [tokenizer.convert_tokens_to_ids(IM_END_TOKEN)]
    if len(prompt_ids) + len(resp_ids) > max_len:
        return None
    return prompt_ids, resp_ids


class MaskingCollator:
    def __init__(self, tokenizer, max_len: int = 512, min_mask_ratio: float = 0.15,
                 seed: int | None = None):
        self.max_len = max_len
        self.min_mask_ratio = min_mask_ratio
        self.mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        pad = tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else tokenizer.eos_token_id
        self.rng = random.Random(seed)

    def __call__(self, batch):
        """batch: list of (prompt_ids, resp_ids). Returns input_ids,
        attention_mask, labels tensors (right-padded)."""
        seqs, labs, attns = [], [], []
        for prompt_ids, resp_ids in batch:
            ids = list(prompt_ids) + list(resp_ids)
            labels = [-100] * len(ids)
            t = self.rng.uniform(self.min_mask_ratio, 1.0)
            resp_start = len(prompt_ids)
            masked_any = False
            for j in range(resp_start, len(ids)):
                if self.rng.random() < t:
                    labels[j] = ids[j]
                    ids[j] = self.mask_id
                    masked_any = True
            if not masked_any and resp_ids:
                j = self.rng.randrange(resp_start, len(ids))
                labels[j] = ids[j]
                ids[j] = self.mask_id
            seqs.append(ids)
            labs.append(labels)
            attns.append([1] * len(ids))
        width = max(len(s) for s in seqs)
        for ids, labels, attn in zip(seqs, labs, attns):
            pad = width - len(ids)
            ids.extend([self.pad_id] * pad)
            labels.extend([-100] * pad)
            attn.extend([0] * pad)
        return {
            "input_ids": torch.tensor(seqs, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
            "attention_mask": torch.tensor(attns, dtype=torch.long),
        }
