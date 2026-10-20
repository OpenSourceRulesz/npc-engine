"""Tests for Base of Three game logic (no model — fake speaker + stub RNG)."""

from npc_engine.games.base_of_three import Camp, play


class StubRng:
    def __init__(self, values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0) if self.values else 0.99


def camp_with(rng_values):
    return Camp(rng=StubRng(rng_values))


def test_scavenge_brings_food_no_events():
    c = camp_with([0.99, 0.99])
    s = c.survivors[1]
    c.assign(s, "scavenge")
    c.assign(c.survivors[0], "guard")
    c.assign(c.survivors[2], "rest")
    note = []
    c.resolve_day(note)
    assert c.food == 3 + 2 - 3  # +2 scavenged, -3 eaten
    assert s.health == "ok"


def test_scavenge_scratch_then_infection():
    c = camp_with([0.99, 0.10])  # no meds; scratch roll < 0.30
    s = c.survivors[0]
    c.assign(s, "scavenge")
    c.assign(c.survivors[1], "rest")
    c.assign(c.survivors[2], "rest")
    c.resolve_day([])
    assert s.health == "hurt"
    c.day = 2
    c.rng = StubRng([0.99, 0.40])  # hurt scavenger: roll < 0.50 -> infected
    c.assign(s, "scavenge")
    c.assign(c.survivors[1], "rest")
    c.assign(c.survivors[2], "rest")
    c.resolve_day([])
    assert s.health == "infected"


def test_rest_heals_hurt_not_infected():
    c = camp_with([])
    s = c.survivors[0]
    s.health = "hurt"
    c.assign(s, "rest")
    c.assign(c.survivors[1], "guard")
    c.assign(c.survivors[2], "guard")
    c.resolve_day([])
    assert s.health == "ok"
    s2 = c.survivors[1]
    s2.health = "infected"
    c.day = 2
    c.assign(s2, "rest")
    c.assign(c.survivors[0], "guard")
    c.assign(c.survivors[2], "guard")
    c.resolve_day([])
    assert s2.health == "infected" and s2.alive  # rested -> survives the night


def test_infected_unrested_dies():
    c = camp_with([])
    s = c.survivors[0]
    s.health = "infected"
    c.assign(s, "guard")
    c.assign(c.survivors[1], "rest")
    c.assign(c.survivors[2], "rest")
    c.resolve_day([])
    assert not s.alive and c.over and not c.won


def test_meds_cure_infection():
    c = camp_with([])
    s = c.survivors[0]
    s.health = "infected"
    c.meds = 1
    assert c.use_meds(s) is True
    assert s.health == "hurt" and c.meds == 0
    assert c.use_meds(s) is False


def test_guard_prevents_day2_theft():
    c = camp_with([0.99, 0.99])
    c.day = 2
    c.food = 5
    c.assign(c.survivors[0], "guard")
    c.assign(c.survivors[1], "scavenge")
    c.assign(c.survivors[2], "rest")
    c.resolve_day([])
    assert c.food == 5 + 2 - 3  # no theft

    c2 = camp_with([0.99, 0.99])
    c2.day = 2
    c2.food = 5
    c2.assign(c2.survivors[0], "rest")
    c2.assign(c2.survivors[1], "scavenge")
    c2.assign(c2.survivors[2], "rest")
    c2.resolve_day([])
    assert c2.food == 5 + 2 - 3 - 2  # stranger stole 2


def test_morale_collapse_walks_out():
    c = camp_with([])
    c.food = 0
    c.survivors[0].morale = 1
    for s in c.survivors:
        c.assign(s, "guard")
    c.resolve_day([])
    assert not c.survivors[0].present and c.over and not c.won


def test_world_state_reflects_condition():
    c = camp_with([])
    s = c.survivors[0]
    s.health = "infected"
    s.morale = 2
    w = c.world_state(s, "The night is quiet.")
    assert "infected" in w and s.name in w and "breaking point" in w


def test_full_game_win_with_fake_speaker():
    calls = []

    def speak(persona, world):
        calls.append(world)
        return f"{persona.name} says something."

    answers = iter([
        "rest", "scavenge", "rest",   # day 1
        "rest", "rest", "guard",      # day 2
        "guard", "rest", "rest",      # day 3
    ])
    out = []
    won = play(speak, input_fn=lambda prompt: next(answers),
               print_fn=out.append, seed=0)
    assert won is True
    assert len(calls) == 9  # 3 survivors x 3 evenings
    assert any("day 1" in w for w in calls)
    assert any("YOU WIN" in line for line in out)
