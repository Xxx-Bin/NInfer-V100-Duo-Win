"""Generate a long-context messages JSON for the TP2 context ladder.

The prompt puts a six-digit code in the first sentence, pads with filler to the
requested repetition count, and asks for the code at the end, so a correct answer
proves the attention path still reaches the start of a long context.
"""

import argparse
import json
from pathlib import Path

FILLER = (
    "The archive room holds shelf after shelf of routine maintenance notes, "
    "and none of them mentions anything unusual. "
)

DEFAULT_NEEDLE = "918273"


def build(reps: int, needle: str = DEFAULT_NEEDLE) -> str:
    body = FILLER * reps
    return (
        f"Reference note. The access code for the vault is {needle}.\n\n"
        f"{body}\n"
        "Question: what is the access code for the vault? "
        "Reply with the six digits only."
    )


def write_messages(path: Path, text: str) -> None:
    path.write_text(
        json.dumps([{"role": "user", "content": text}], ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, required=True)
    parser.add_argument("--needle", default=DEFAULT_NEEDLE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    text = build(args.reps, args.needle)
    write_messages(args.out, text)
    print(f"{args.out}: reps={args.reps} chars={len(text)}")


if __name__ == "__main__":
    main()
