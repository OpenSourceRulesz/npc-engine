"""Tests for masked-diffusion SFT encoding + collator (stub tokenizer)."""

from npc_engine.data.build_sft import triple_to_example
from npc_engine.masking import (MaskingCollator, encode_example,
                                to_bidirectional_mask)

MASK = 999
IM_END = 998


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def convert_tokens_to_ids(self, tok):
        return {"<|mask|>": MASK, "<|im_end|>": IM_END}[tok]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        # fixed 4-token prompt regardless of content
        return [1, 10, 11, 12]

    def encode(self, text, add_special_tokens=False):
        return [50 + (sum(map(ord, w)) % 40) for w in text.split()]


def test_encode_appends_im_end():
    tok = StubTokenizer()
    enc = encode_example(tok, {"system": "s", "user": "u", "assistant": "hello world"})
    assert enc is not None
    prompt_ids, resp_ids = enc
    assert prompt_ids == [1, 10, 11, 12]
    assert len(resp_ids) == 3 and resp_ids[-1] == IM_END


def test_encode_too_long_returns_none():
    tok = StubTokenizer()
    enc = encode_example(tok, {"system": "s", "user": "u",
                               "assistant": " ".join(["w"] * 600)}, max_len=64)
    assert enc is None


def test_collator_prompt_never_masked():
    tok = StubTokenizer()
    col = MaskingCollator(tok, seed=3)
    prompt, resp = [1, 10, 11, 12], [50, 51, 52, IM_END]
    out = col([(prompt, resp)])
    ids, labels = out["input_ids"][0].tolist(), out["labels"][0].tolist()
    assert ids[:4] == prompt and labels[:4] == [-100] * 4
    # every masked position: input is MASK, label is the original token
    masked = [j for j in range(4, 8) if labels[j] != -100]
    assert len(masked) >= 1
    for j in masked:
        assert ids[j] == MASK and labels[j] == resp[j - 4]
    # unmasked response positions keep their token and -100
    for j in range(4, 8):
        if labels[j] == -100:
            assert ids[j] == resp[j - 4]


def test_collator_always_masks_something():
    tok = StubTokenizer()
    for seed in range(30):
        col = MaskingCollator(tok, seed=seed, min_mask_ratio=0.15)
        out = col([([1, 2], [60, 61])])
        labels = out["labels"][0].tolist()
        assert any(l != -100 for l in labels), f"seed {seed} masked nothing"


def test_collator_padding():
    tok = StubTokenizer()
    col = MaskingCollator(tok, seed=0)
    out = col([([1, 2], [50, IM_END]), ([1], [50, 51, 52, IM_END])])
    assert out["input_ids"].shape == (2, 5)
    # shorter row: 4 real tokens + 1 pad at the end
    assert out["attention_mask"][0].tolist() == [1, 1, 1, 1, 0]
    assert out["input_ids"][0].tolist()[-1] == 0
    assert out["labels"][0].tolist()[-1] == -100
    assert out["attention_mask"][1].tolist() == [1, 1, 1, 1, 1]


def test_bidirectional_mask():
    import torch

    attn = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    m = to_bidirectional_mask(attn)
    assert m.shape == (2, 1, 4, 4)
    neg = torch.finfo(torch.float32).min
    # padded key columns are -inf for every query row
    assert (m[0, 0, :, 3] == neg).all()
    assert (m[1, 0, :, 2:] == neg).all()
    # real key columns are 0 (attend freely, both directions)
    assert (m[0, 0, :, :3] == 0).all()
    assert (m[1, 0, :, :2] == 0).all()
    # no row is fully masked (would NaN the softmax)
    assert (m > neg).any(dim=-1).all()
    # output lives on the INPUT's device (regression: on GPU runs the mask
    # was built on CPU while the batch was on cuda -> RuntimeError)
    assert m.device == attn.device


def test_triple_to_example_history_split():
    t = {
        "persona": {"name": "Renn", "description": "A jaded ex-soldier."},
        "world_state": "A hot spring.",
        "context": "Sarisha: I give up.\nRenn: Never give up.\nSarisha: Why not?",
        "player_line": "Why not?",
        "npc_line": "Because the road home matters.",
    }
    ex = triple_to_example(t)
    assert ex["user"] == "Why not?"
    assert ex["assistant"] == "Because the road home matters."
    assert "Recent conversation:" in ex["system"]
    assert "Sarisha: I give up." in ex["system"]
    # the player line turn itself is not duplicated into history
    assert ex["system"].count("Why not?") == 0
    assert "You are Renn. A jaded ex-soldier." in ex["system"]


def test_triple_to_example_no_history():
    t = {
        "persona": {"name": "Renn", "description": "A jaded ex-soldier."},
        "world_state": "A hot spring.",
        "context": "Sarisha: Hello.",
        "player_line": "Hello.",
        "npc_line": "State your business.",
    }
    ex = triple_to_example(t)
    assert "Recent conversation:" not in ex["system"]
