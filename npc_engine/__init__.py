"""NPC Engine Phase 0 — a tiny text-diffusion NPC dialogue harness.

Built on Dream 7B Instruct (Dream-org/Dream-v0-Instruct-7B), a masked diffusion LM.
The diffusion-native features used here:

- parallel candidate generation: N masked canvases denoised in ONE batched pass
  (AR would need N sequential decodes)
- confidence ranking: per-token max-prob over the generated span, from a single
  scoring forward pass — no second model
- confidence fallback: if the best candidate scores below threshold, the engine
  ships a safe canned line instead of a broken one (the anti-gameslop guard)
"""

from .personas import Persona, STARTER_PERSONAS
from .engine import NPCEngine, NPCResponse

__version__ = "0.1.1"

__all__ = ["Persona", "STARTER_PERSONAS", "NPCEngine", "NPCResponse"]
