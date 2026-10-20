"""Base of Three — a tiny playable survivor-camp game on the NPC engine.

Three survivors, three days. Each morning you assign tasks (scavenge / guard
/ rest); events land; each evening every survivor says ONE line generated
from their live state card (traits, morale, injuries, infection). The GAME
decides every fact — food, wounds, who lives — the engine only words it.

Win: reach day 4 with everyone alive and still in camp.
Lose: a survivor dies of an untreated infection, or someone's morale hits
zero and they walk out in the night.

The game logic is pure Python (injectable speaker + RNG) so it is fully
testable without a model. In Colab, the speaker wraps NPCEngine.respond().
"""

import random

from ..personas import Persona

BARK_PROMPT = (
    "The group is gathered at camp for the evening. Say one or two short "
    "lines, in character, about how you are doing tonight."
)

TASKS = ("scavenge", "guard", "rest")


class Survivor:
    def __init__(self, persona: Persona):
        self.persona = persona
        self.morale = 6          # 0 = walks out, 10 = thriving
        self.health = "ok"       # ok | hurt | infected
        self.today = "camp duty" # what happened to them today (for the card)
        self.task = None
        self.alive = True
        self.present = True

    @property
    def name(self):
        return self.persona.name

    def condition_text(self):
        return {
            "ok": "uninjured",
            "hurt": "hurt (scratched up and sore)",
            "infected": "infected — feverish, bitten two days ago, getting worse",
        }[self.health]

    def morale_word(self):
        if self.morale >= 8:
            return "hopeful"
        if self.morale >= 5:
            return "steady"
        if self.morale >= 3:
            return "shaken"
        return "at breaking point"


def starter_survivors():
    return [
        Survivor(Persona(
            name="Joss Marek", role="the camp's medic, ex-army",
            personality="calm, dry-humored, puts everyone else first",
            speech_style="short, dry, understated",
            knowledge="first aid, and exactly who is hurt and how badly",
            fallback_lines=["We'll manage. We always do.", "One thing at a time, people."],
        )),
        Survivor(Persona(
            name="Ruth Callahan", role="the camp's hunter and scavenger",
            personality="blunt, brave, impatient with panic",
            speech_style="terse and plainspoken",
            knowledge="the woods, the roads, and where supplies still are",
            fallback_lines=["Less talking. More doing.", "I've seen worse. Barely."],
        )),
        Survivor(Persona(
            name="Theo Lindqvist", role="a young runner, former grocery clerk",
            personality="nervous, kind, jokes when he is scared",
            speech_style="quick, apologetic, self-deprecating",
            knowledge="the camp's stores and who ate what",
            fallback_lines=["Okay. Okay okay okay.", "Please tell me that was the wind."],
        )),
    ]


class Camp:
    def __init__(self, rng=None):
        self.survivors = starter_survivors()
        self.food = 3
        self.meds = 0
        self.day = 1
        self.rng = rng or random.Random()
        self.over = False
        self.won = False
        self.log = []

    # ---- state -> text ----------------------------------------------------
    def world_state(self, s: Survivor, night_note: str) -> str:
        return (
            f"Evening at a small survivor camp in the woods, day {self.day} of "
            f"the second week of the outbreak. Food stored: {self.food} meals. "
            f"Medicine kits: {self.meds}. {night_note} "
            f"{s.name} feels {s.morale_word()} and is {s.condition_text()}. "
            f"Today {s.name} spent the day on {s.today}."
        )

    # ---- day resolution ---------------------------------------------------
    def assign(self, s: Survivor, task: str):
        assert task in TASKS
        s.task = task

    def resolve_day(self, night_note_holder: list):
        guarded = any(s.task == "guard" and s.present and s.alive
                      for s in self.survivors)
        for s in self.survivors:
            if not (s.alive and s.present):
                continue
            if s.task == "scavenge":
                s.today = "scavenging outside the fence"
                self.food += 1 if s.health != "ok" else 2
                if self.rng.random() < 0.25:
                    self.meds += 1
                    self.log.append(f"{s.name} found a medicine kit.")
                r = self.rng.random()
                if s.health == "ok" and r < 0.30:
                    s.health = "hurt"
                    s.morale -= 1
                    self.log.append(f"{s.name} got scratched up out there.")
                elif s.health == "hurt" and r < 0.50:
                    s.health = "infected"
                    s.morale -= 2
                    self.log.append(f"{s.name}'s scratches are infected — fever starting.")
            elif s.task == "guard":
                s.today = "standing guard at the fence"
            elif s.task == "rest":
                s.today = "resting at camp"
                if s.health == "hurt":
                    s.health = "ok"
                    self.log.append(f"{s.name} rested and is back on their feet.")
                s.morale = min(10, s.morale + 2)
        # evening meal
        if self.food >= 3:
            self.food -= 3
        else:
            self.food = 0
            for s in self.survivors:
                s.morale -= 2
            self.log.append("Not enough food — everyone went hungry.")
        if self.food >= 3:
            for s in self.survivors:
                s.morale = min(10, s.morale + 1)
        # night
        night_note = "The night is quiet."
        if self.day == 2 and not guarded:
            stolen = min(2, self.food)
            self.food -= stolen
            for s in self.survivors:
                s.morale -= 1
            night_note = "In the night, a stranger slipped in and stole supplies."
            self.log.append("A stranger raided the stores in the night — no one was on guard.")
        elif self.day == 2:
            night_note = "In the night, a stranger crept up to the fence but was driven off by the guard."
        night_note_holder.append(night_note)
        # infection progression: untreated + unrested = death by morning
        for s in self.survivors:
            if s.health == "infected" and s.task != "rest":
                s.alive = False
                self.log.append(f"{s.name} didn't make it through the night. The fever took them.")
        # morale collapse: walks out
        for s in self.survivors:
            if s.alive and s.morale <= 0:
                s.present = False
                self.log.append(f"{s.name} walked out into the dark and didn't come back.")
        if any(not s.alive or not s.present for s in self.survivors):
            self.over = True
            self.won = False
        elif self.day >= 3:
            self.over = True
            self.won = True

    def use_meds(self, s: Survivor):
        if self.meds > 0 and s.health == "infected":
            self.meds -= 1
            s.health = "hurt"
            self.log.append(f"You gave {s.name} the medicine. The fever breaks.")
            return True
        return False


# ---- interactive loop ------------------------------------------------------

def play(speak, input_fn=input, print_fn=print, seed=None):
    """speak(persona, world_state) -> str (one generated line)."""
    camp = Camp(rng=random.Random(seed))
    print_fn("BASE OF THREE — three survivors, three days. Keep them alive, "
             "fed, and from giving up.\n")
    while not camp.over:
        print_fn(f"--- DAY {camp.day} — food: {camp.food}, meds: {camp.meds} ---")
        for s in camp.survivors:
            print_fn(f"{s.name}: morale {s.morale}/10, {s.health}")
            while True:
                t = input_fn(f"  task for {s.name.split()[0]} "
                             f"[scavenge/guard/rest]: ").strip().lower()
                if t in TASKS:
                    camp.assign(s, t)
                    break
                print_fn("  please type scavenge, guard, or rest")
        # meds decision for anyone infected
        for s in camp.survivors:
            if s.health == "infected" and camp.meds > 0:
                a = input_fn(f"{s.name} is infected. Use a medicine kit? [y/n]: ")
                if a.strip().lower().startswith("y"):
                    camp.use_meds(s)
        note = []
        camp.resolve_day(note)
        for msg in camp.log:
            print_fn(" * " + msg)
        camp.log.clear()
        print_fn("")
        for s in camp.survivors:
            if s.alive and s.present:
                line = speak(s.persona, camp.world_state(s, note[0]))
                print_fn(f'{s.name}: "{line}"')
        print_fn("")
        camp.day += 1
    if camp.won:
        print_fn("DAY 4 — dawn. Everyone is still here. The camp holds. YOU WIN.")
    else:
        print_fn("The camp is broken. GAME OVER.")
    return camp.won
