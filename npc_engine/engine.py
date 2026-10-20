"""NPCEngine — parallel candidate generation + confidence ranking for NPC lines.

How it works (diffusion-wise):
1. The persona card + world state + player line form the prompt. The prompt is
   kept fully clean (never masked).
2. We append a canvas of [MASK] tokens and expand it to a batch of N canvases.
   All N candidates are denoised in ONE batched diffusion pass — AR would need
   N sequential generations. Diversity comes from temperature/top-p sampling.
3. A single scoring forward pass computes, for each candidate, the mean
   max-probability over its generated tokens (confidence) plus the weakest
   token's probability. Candidates are ranked by confidence.
4. If the best candidate's confidence is below the threshold, the engine
   returns one of the persona's canned fallback lines instead (flagged), so a
   low-confidence generation never reaches the player.
"""

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .personas import Persona

DEFAULT_MODEL = "Dream-org/Dream-v0-Instruct-7B"
MASK_TOKEN = "<|mask|>"


@dataclass
class NPCResponse:
    text: str                 # the line the game should ship
    confidence: float         # confidence of the shipped line (1.0 for fallback)
    used_fallback: bool       # True if the model was too unsure and we canned it
    candidates: list = field(default_factory=list)  # [(text, confidence), ...] ranked
    latency_ms: float = 0.0   # wall time for the whole respond() call


class NPCEngine:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        load_in_8bit: bool = False,
        torch_dtype: torch.dtype | None = None,
        adapter_path: str | None = None,
    ):
        """Load the diffusion LM. On a 16GB GPU use load_in_8bit=True; on 24GB+
        fp16/bf16 is fine. adapter_path loads a LoRA adapter on top."""
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"trust_remote_code": True}
        if load_in_8bit:
            # Modern transformers: 8-bit loading goes through a quantization
            # config + device_map. Passing load_in_8bit=True directly leaks the
            # kwarg into the custom model's __init__ and crashes it.
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch_dtype or torch.float16
        self.model = AutoModel.from_pretrained(model_id, **kwargs)
        if not load_in_8bit:
            self.model = self.model.to(self.device)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        eos = self.tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

    # ---- prompt building -------------------------------------------------
    def _prompt_ids(self, persona: Persona, world_state: str, player_line: str,
                    history: list | None = None) -> torch.Tensor:
        messages = [
            {"role": "system", "content": persona.system_prompt(world_state, history)},
            {"role": "user", "content": player_line},
        ]
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    # ---- confidence scoring ----------------------------------------------
    @torch.no_grad()
    def _score_candidates(self, seqs: torch.Tensor, prompt_len: int):
        """One forward pass; per candidate, (mean max-prob, min max-prob) over
        the generated span (up to its first EOS)."""
        logits = self.model(seqs).logits.float()
        probs = F.softmax(logits, dim=-1)
        maxp, _ = probs.max(dim=-1)  # (B, L)
        out = []
        for b in range(seqs.shape[0]):
            span = []
            for t in range(prompt_len, seqs.shape[1]):
                tok = int(seqs[b, t])
                if tok in self.eos_ids or tok == self.mask_token_id:
                    break
                span.append(float(maxp[b, t]))
            if span:
                out.append((sum(span) / len(span), min(span)))
            else:
                out.append((0.0, 0.0))
        return out

    # ---- main entry point --------------------------------------------------
    @torch.no_grad()
    def respond(
        self,
        persona: Persona,
        world_state: str,
        player_line: str,
        history: list | None = None,
        n_candidates: int = 8,
        steps: int = 32,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        confidence_threshold: float = 0.45,
    ) -> NPCResponse:
        t0 = time.perf_counter()
        prompt_ids = self._prompt_ids(persona, world_state, player_line, history)
        prompt_len = prompt_ids.shape[1]

        canvas = torch.full(
            (1, max_new_tokens), self.mask_token_id, dtype=torch.long, device=self.device
        )
        input_ids = torch.cat([prompt_ids, canvas], dim=1)
        # parallel candidates: N canvases, ONE batched denoising run
        input_ids = input_ids.repeat(n_candidates, 1)
        attn = torch.ones_like(input_ids)

        gen = self.model.diffusion_generate if hasattr(self.model, "diffusion_generate") else self.model.generate
        out = gen(
            input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
        )
        seqs = out.sequences if hasattr(out, "sequences") else out

        scores = self._score_candidates(seqs, prompt_len)
        ranked = []
        for b in range(seqs.shape[0]):
            text = self._decode_line(seqs[b], prompt_len)
            if text:
                ranked.append((text, scores[b][0], scores[b][1]))
        # dedupe identical lines, rank by mean confidence then weakest link
        seen, uniq = set(), []
        for text, mean_c, min_c in sorted(ranked, key=lambda r: (r[1], r[2]), reverse=True):
            if text not in seen:
                seen.add(text)
                uniq.append((text, mean_c))
        latency_ms = (time.perf_counter() - t0) * 1000

        if not uniq or uniq[0][1] < confidence_threshold:
            import random
            fallback = random.choice(persona.fallback_lines) if persona.fallback_lines else "..."
            return NPCResponse(
                text=fallback, confidence=1.0, used_fallback=True,
                candidates=uniq, latency_ms=latency_ms,
            )
        return NPCResponse(
            text=uniq[0][0], confidence=uniq[0][1], used_fallback=False,
            candidates=uniq, latency_ms=latency_ms,
        )

    def _decode_line(self, seq: torch.Tensor, prompt_len: int) -> str:
        ids = []
        for t in range(prompt_len, seq.shape[0]):
            tok = int(seq[t])
            if tok in self.eos_ids or tok == self.mask_token_id:
                break
            ids.append(tok)
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()
