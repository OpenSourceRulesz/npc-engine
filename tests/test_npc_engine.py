"""CPU-only tests: converter + engine mechanics, no model weights required."""

import json

import torch

from npc_engine.data.convert_agentlans import convert_file, convert_row
from npc_engine.engine import NPCEngine
from npc_engine.personas import Persona

# ---------------------------------------------------------------- converter

ROW = {
    "setting": "A cozy hot spring.",
    "characters": {
        "Sarisha": "A heartbroken spiritual seeker in her 20s.",
        "Renn": "A weathered traveler with a scar.",
    },
    "conversation": [
        {"from": "Sarisha", "message": "I don't see the point anymore."},
        {"from": "Renn", "message": "You can't let events define your purpose."},
        {"from": "Sarisha", "message": "But what if my heart is too broken?"},
        {"from": "Ghost", "message": "Boo."},  # unknown speaker -> skipped
        {"from": "Renn", "message": "x"},       # too short -> skipped
    ],
}


def test_convert_row_basic():
    triples, skipped = convert_row(ROW)    assert len(triples) == 2
    t0 = triples[0]
    assert t0["persona"]["name"] == "Renn"
    assert t0["persona"]["description"] == "A weathered traveler with a scar."
    assert t0["world_state"] == "A cozy hot spring."
    assert t0["player_line"] == "I don't see the point anymore."
    assert t0["npc_line"] == "You can't let events define your purpose."
    assert "Sarisha: I don't see the point anymore." in t0["context"]
    assert skipped["unknown_speaker"] == 1
    assert skipped["bad_length"] == 1


def test_convert_row_context_window():
    triples, _ = convert_row(ROW, context_turns=1)
    t1 = triples[1]
    # only the immediately prior turn in context
    assert t1["context"] == "Renn: You can't let events define your purpose."


def test_convert_row_empty_conversation():
    assert convert_row({"conversation": []}) == ([], {"unknown_speaker": 0, "bad_length": 0})
    assert convert_row({})[0] == []


# ------------------------------------------------------------------- engine

PROMPT_LEN = 4
VOCAB = 16
MASK = 999
EOS = 2


class FakeTokenizer:
    def convert_tokens_to_ids(self, tok):
        assert tok == "<|mask|>"
        return MASK

    eos_token_id = EOS

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 10, 11, 12]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)


class FakeModel:
    """diffusion_generate returns fixed candidate token rows; forward returns
    logits with controllable peakedness."""

    def __init__(self, gen_rows, peak=30.0):
        self.gen_rows = gen_rows
        self.peak = peak

    def eval(self):
        return self

    def diffusion_generate(self, input_ids, attention_mask=None, max_new_tokens=64,
                           steps=32, temperature=0.8, top_p=0.95,
                           return_dict_in_generate=True):
        prompt = input_ids[0, :PROMPT_LEN]
        seqs = []
        for row in self.gen_rows:
            gen = torch.tensor(row + [EOS] * (max_new_tokens - len(row)), dtype=torch.long)
            seqs.append(torch.cat([prompt, gen]))

        class Out:
            pass
        out = Out()
        out.sequences = torch.stack(seqs)
        return out

    def __call__(self, seqs):
        B, L = seqs.shape
        logits = torch.zeros(B, L, VOCAB)
        # peaked logits -> confidence near 1.0; peak=0 -> uniform 1/VOCAB
        logits[..., 5] = self.peak

        class Out:
            pass
        out = Out()
        out.logits = logits
        return out


def make_engine(gen_rows, peak=30.0):
    eng = NPCEngine.__new__(NPCEngine)
    eng.tokenizer = FakeTokenizer()
    eng.model = FakeModel(gen_rows, peak=peak)
    eng.mask_token_id = MASK
    eng.eos_ids = {EOS}
    eng.device = "cpu"
    return eng


PERSONA = Persona(
    name="Test", role="a tester", personality="calm", speech_style="plain",
    knowledge="nothing", fallback_lines=["Safe line."],
)


def test_decode_truncates_at_eos():
    eng = make_engine([[7, 8]])
    seq = torch.tensor([1, 10, 11, 12, 7, 8, EOS, 9, 9])
    assert eng._decode_line(seq, PROMPT_LEN) == "w7 w8"


def test_decode_truncates_at_mask():
    eng = make_engine([[7]])
    seq = torch.tensor([1, 10, 11, 12, 7, MASK, 9])
    assert eng._decode_line(seq, PROMPT_LEN) == "w7"


def test_score_empty_span_is_zero():
    eng = make_engine([[EOS]])  # immediate eos -> no generated tokens
    seq = torch.tensor([[1, 10, 11, 12, EOS, EOS]])
    scores = eng._score_candidates(seq, PROMPT_LEN)
    assert scores[0] == (0.0, 0.0)


def test_respond_ranks_and_ships_best():
    rows = [[7, 8], [7, 8], [3, 4]]
    eng = make_engine(rows)
    r = eng.respond(PERSONA, "Somewhere.", "Hello?", n_candidates=3,
                    confidence_threshold=0.45)
    assert not r.used_fallback
    # identical candidates deduped -> 2 unique
    assert len(r.candidates) == 2
    assert r.text == r.candidates[0][0]
    assert r.confidence > 0.9


def test_respond_falls_back_when_unconfident():
    eng = make_engine([[7, 8]], peak=0.0)  # uniform logits -> conf 1/16
    r = eng.respond(PERSONA, "Somewhere.", "Hello?", n_candidates=1,
                    confidence_threshold=0.45)
    assert r.used_fallback
    assert r.text == "Safe line."


def test_respond_prompt_mentions_persona():
    eng = make_engine([[7, 8]])
    ids = eng._prompt_ids(PERSONA, "Somewhere.", "Hi")
    assert ids.shape == (1, PROMPT_LEN)


# ------------------------------------------------------------- loader kwargs

def test_loader_uses_quantization_config_for_8bit(monkeypatch):
    """Regression: load_in_8bit=True as a raw kwarg crashes Dream's custom
    __init__ on modern transformers; it must go via BitsAndBytesConfig."""
    import transformers

    captured = {}

    class StubModel(FakeModel):
        def to(self, device):
            return self

    def fake_from_pretrained(model_id, **kwargs):
        captured.update(kwargs)
        return StubModel([[7, 8]])

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        staticmethod(fake_from_pretrained))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: FakeTokenizer()))

    eng = NPCEngine(load_in_8bit=True, device="cpu")
    assert "load_in_8bit" not in captured
    assert "quantization_config" in captured
    assert captured.get("device_map") == "auto"
    assert eng.mask_token_id == MASK


def test_convert_file_survives_lone_surrogate(tmp_path):
    """Regression (2026-10-20 Colab run): the agentlans source contains lone
    UTF-16 surrogates (broken emoji); writing them strict utf-8 crashed the
    converter with UnicodeEncodeError. They must be replaced, not fatal."""
    row = {
        "setting": "A gate.",
        "characters": {"A": "Guard A.", "B": "Traveler B."},
        "conversation": [
            {"from": "A", "message": "Halt, who goes there?"},
            {"from": "B", "message": "Just a traveler \ud821 passing through."},
        ],
    }
    src = tmp_path / "src.jsonl"
    src.write_text(json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    stats = convert_file(src, out)
    assert stats["triples"] == 1
    written = out.read_text(encoding="utf-8")
    json.loads(written)  # must be valid, strictly-decodable utf-8 JSON
    assert "\ud821" not in written
