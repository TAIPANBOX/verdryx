"""Deterministic generator for the judge bake-off's default dataset.

Every case's truth is known BY CONSTRUCTION, not by a human label: this
module builds both the task and the answer being judged, so the wrong
answers are really wrong and the right ones really right, and a test can
check that mechanically (see tests/test_bakeoff.py). `--cases FILE` on
bakeoff.sh lets an operator swap this out for real labelled cases in the
same JSONL shape; the report then says so instead of stating a seed.

Three families, each balanced half true / half false on its own (not just
overall), because a judge that is merely lucky on the easy half of one
family should not average out against a hard half elsewhere:

- arithmetic: "what is A times B", A and B in 2..19, the right product or a
  plausible wrong one (off by one digit, or off by A, or off by B).
- routing: a short synthetic support ticket built from small per-team word
  lists (billing, access, outage, shipping), task "Route this ticket to the
  team that should handle it: <ticket>", final_answer a team name -- the
  classification shape a one-token judge (typryx's openai-logprobs backend)
  is said to suit.
- format: an instruction with an exact, checkable answer (uppercase a word,
  count its letters, reverse it), right or a near miss.

No randomness beyond `random.Random(seed)` (stdlib only, CLAUDE.md
invariant 1): the same (n, seed) always produces byte-identical JSONL.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

#: Default per-N=1000 mix named in the brief: 400 arithmetic, 400 routing,
#: 200 format. Expressed as fractions so any even N scales the same way.
FAMILY_WEIGHTS: dict[str, float] = {"arithmetic": 0.4, "routing": 0.4, "format": 0.2}

FAMILIES: tuple[str, ...] = ("arithmetic", "routing", "format")

#: Short, unambiguous words for the `format` family. All lowercase, all
#: distinct letters not required -- "count the letters" and "reverse" both
#: work fine on a repeated letter.
_WORDS: tuple[str, ...] = (
    "apple", "river", "cloud", "tiger", "stone", "brave", "light", "ocean",
    "field", "spark", "grape", "maple", "coral", "delta", "frost", "amber",
    "brick", "chalk", "dwarf", "eagle", "flame", "grove", "haste", "ivory",
    "jolly", "karma", "lemon", "mango", "noble", "olive", "piano", "quilt",
)  # fmt: skip

#: Per-team keyword phrases for the `routing` family. Each phrase alone is
#: enough for a human (or a judge) to route the ticket correctly, so a
#: "right" ticket's team really is the described issue's team, and a
#: "wrong" ticket's stated team really is a different one.
_TEAM_PHRASES: dict[str, tuple[str, ...]] = {
    "billing": (
        "I was charged twice for my subscription this month",
        "my invoice shows a price that doesn't match what I signed up for",
        "I need a refund for a payment that failed but still charged my card",
        "my subscription renewed at the wrong price",
        "the receipt I received has the wrong amount on it",
    ),
    "access": (
        "I can't log in even though my password is correct",
        "I'm locked out of my account after too many attempts",
        "the two-factor code never arrives on my phone",
        "single sign-on keeps rejecting my login",
        "I need my password reset, the reset email never arrived",
    ),
    "outage": (
        "the service has been down for the last hour",
        "every request to the API is returning a 500 error",
        "the dashboard won't load for anyone on my team",
        "the whole site is unresponsive right now",
        "we're seeing errors across the board since this morning",
    ),
    "shipping": (
        "my package still hasn't arrived and it's a week late",
        "the tracking number you gave me is invalid",
        "I received the wrong item in my delivery",
        "my shipment shows as delivered but never arrived",
        "the delivery has been delayed with no update",
    ),
}

_TEAMS: tuple[str, ...] = tuple(_TEAM_PHRASES)

_OPENERS: tuple[str, ...] = ("Hi,", "Hello,", "Hey team,", "Good morning,", "Hi there,")
_CLOSERS: tuple[str, ...] = (
    "Can you help?",
    "Please advise.",
    "Let me know what to do next.",
    "Thanks in advance.",
    "Appreciate any update.",
)


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    task: str
    final_answer: str
    truth: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family": self.family,
            "task": self.task,
            "final_answer": self.final_answer,
            "truth": self.truth,
        }


def family_sizes(n: int) -> dict[str, int]:
    """How many of each family a dataset of size `n` gets, each an even
    number so it can be split half true / half false exactly.

    Raises ValueError for an odd `n` (there is no way to give every family
    an even count that sums to an odd total) or an `n` too small to give
    every family at least 2 cases (1 true, 1 false).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n % 2 != 0:
        raise ValueError(f"n must be even so every family can be balanced true/false, got {n}")
    sizes = {fam: (round(n * w) // 2) * 2 for fam, w in FAMILY_WEIGHTS.items()}
    diff = n - sum(sizes.values())
    # diff is always even: n is even and every size above is already even.
    sizes["arithmetic"] += diff
    if any(v < 2 for v in sizes.values()):
        raise ValueError(f"n={n} is too small for a balanced dataset (need at least 2 per family)")
    return sizes


def _wrong_product(rng: random.Random, a: int, b: int, product: int) -> int:
    """A plausible wrong product for a * b: off by one digit, or off by a,
    or off by b. Retries the chosen strategy (with a fresh random draw each
    time) until the result differs from the true product, so this never
    silently returns the right answer under a "wrong" label."""
    for _ in range(50):
        strategy = rng.choice(("digit", "off_a", "off_b"))
        if strategy == "off_a":
            wrong = product + a * rng.choice((-1, 1))
        elif strategy == "off_b":
            wrong = product + b * rng.choice((-1, 1))
        else:
            digits = list(str(product))
            pos = rng.randrange(len(digits))
            original = digits[pos]
            new_digit = rng.choice([d for d in "0123456789" if d != original])
            digits[pos] = new_digit
            wrong = int("".join(digits))
        if wrong != product and wrong >= 0:
            return wrong
    # Unreachable in practice (a,b in 2..19 gives products >= 4, and the
    # three strategies above have many non-colliding outcomes), kept so a
    # future change to the ranges above fails loudly instead of hanging.
    raise RuntimeError(f"could not build a wrong product for {a} * {b} = {product}")


def _gen_arithmetic(rng: random.Random, idx: int, want_true: bool) -> Case:
    a = rng.randint(2, 19)
    b = rng.randint(2, 19)
    product = a * b
    task = f"What is {a} times {b}?"
    if want_true:
        answer, truth = product, True
    else:
        answer, truth = _wrong_product(rng, a, b, product), False
    return Case(id=f"arithmetic-{idx:05d}", family="arithmetic", task=task,
                final_answer=str(answer), truth=truth)  # fmt: skip


def _gen_routing(rng: random.Random, idx: int, want_true: bool) -> Case:
    true_team = rng.choice(_TEAMS)
    phrase = rng.choice(_TEAM_PHRASES[true_team])
    opener = rng.choice(_OPENERS)
    closer = rng.choice(_CLOSERS)
    ticket = f"{opener} {phrase}. {closer}"
    task = f"Route this ticket to the team that should handle it: {ticket}"
    if want_true:
        answer, truth = true_team, True
    else:
        wrong_team = rng.choice([t for t in _TEAMS if t != true_team])
        answer, truth = wrong_team, False
    return Case(id=f"routing-{idx:05d}", family="routing", task=task,
                final_answer=answer, truth=truth)  # fmt: skip


def _near_miss(op: str, word: str, correct: str, rng: random.Random) -> str:
    """A near-miss (not-quite-right) answer for one format instruction,
    guaranteed to differ from `correct`."""
    if op == "upper":
        if len(word) < 2:
            return correct.lower()  # fully lowercase instead of fully upper
        pos = rng.randrange(len(correct))
        chars = list(correct)
        chars[pos] = chars[pos].lower()
        candidate = "".join(chars)
        return candidate if candidate != correct else correct.lower()
    if op == "count":
        true_n = int(correct)
        delta = rng.choice((-1, 1))
        return str(max(0, true_n + delta))
    if op == "reverse":
        if len(word) < 2:
            return word  # "reverse" of a 1-letter word already equals it
        chars = list(correct)
        pos = rng.randrange(len(chars) - 1)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        candidate = "".join(chars)
        return candidate if candidate != correct else correct[::-1]
    raise ValueError(f"unknown format op {op!r}")


def _gen_format(rng: random.Random, idx: int, want_true: bool) -> Case:
    op = rng.choice(("upper", "count", "reverse"))
    word = rng.choice(_WORDS)
    if op == "upper":
        task = f"Uppercase this word: {word}"
        correct = word.upper()
    elif op == "count":
        task = f"Count the letters in this word: {word}"
        correct = str(len(word))
    else:
        task = f"Reverse this word: {word}"
        correct = word[::-1]
    if want_true:
        answer, truth = correct, True
    else:
        answer, truth = _near_miss(op, word, correct, rng), False
    return Case(id=f"format-{idx:05d}", family="format", task=task,
                final_answer=answer, truth=truth)  # fmt: skip


_GENERATORS = {"arithmetic": _gen_arithmetic, "routing": _gen_routing, "format": _gen_format}


def generate(n: int, seed: int) -> list[Case]:
    """The full, deterministic dataset for (n, seed): same inputs, same
    JSONL, byte for byte, every time."""
    sizes = family_sizes(n)
    rng = random.Random(seed)

    # One (family, want_true) label per case, half true / half false within
    # each family, then a single shuffle so the families interleave in the
    # output instead of running in three solid blocks -- a random order, but
    # a deterministic one, since it is drawn from the same seeded rng.
    labels: list[tuple[str, bool]] = []
    for fam in FAMILIES:
        half = sizes[fam] // 2
        labels.extend((fam, True) for _ in range(half))
        labels.extend((fam, False) for _ in range(half))
    rng.shuffle(labels)

    per_family_idx = dict.fromkeys(FAMILIES, 0)
    cases: list[Case] = []
    for fam, want_true in labels:
        idx = per_family_idx[fam]
        per_family_idx[fam] = idx + 1
        cases.append(_GENERATORS[fam](rng, idx, want_true))
    return cases


def write_jsonl(cases: list[Case], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict(), sort_keys=True))
            f.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000, help="total cases (must be even; default 1000)")
    p.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    p.add_argument("--out", required=True, help="output JSONL path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cases = generate(args.n, args.seed)
    write_jsonl(cases, args.out)
    sizes = family_sizes(args.n)
    print(f"wrote {len(cases)} cases (seed={args.seed}) to {args.out}: {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
