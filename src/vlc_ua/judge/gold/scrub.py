"""Scrub personal names out of gold rows before the data leaves the server.

The registry (ЄДРСР) already pseudonymises the parties: natural persons appear
as ОСОБА_1, addresses as АДРЕСА_1, identifiers as НОМЕР_1. What it leaves in
the open is everyone whose name is part of the court's own record — judges,
secretaries, experts, representatives, and sole traders, whose business name
IS their full name. Measured on the 9 655-row attribution gold (2026-09-20):

    full ПІБ (Прізвище Ім'я По-батькові)   206 occurrences, 183 fragments
    initials + surname ("В.М. Соколов")   2 411 occurrences, 767 fragments
    surname + initials ("Юрченко І.Я.")      12 occurrences,  10 fragments
    ЄДРПОУ / РНОКПП / ІПН numbers           104 occurrences,  59 fragments

So a tenth of the corpus carries a real name. This module removes them.

What it deliberately does NOT touch: ОСОБА_N, АДРЕСА_N, НОМЕР_N and the like
(already pseudonyms), case numbers, dates, sums, article numbers and the names
of courts and institutions — the label of a fragment depends on them.

Replacements are numbered per row, not globally: within one fragment the same
placeholder can be reused for readability, but nothing links a person across
fragments, which is the property that matters once the file is uploaded.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

U = "А-ЯІЇЄҐ"
L = "а-яіїєґ'’ʼ`"

# Patronymics are the only reliable marker of a Ukrainian personal name in
# running legal text, so the full-name rule is anchored on them.
_PATRONYMIC = r"(?:ович|овича|овичу|овичем|йович|йовича|йовичу|йовичем|" \
              r"івна|івни|івні|івну|івною|ївна|ївни|ївні|ївну|ївною)"

RULES: list[tuple[str, re.Pattern[str]]] = [
    # Прізвище Ім'я По-батькові (у будь-якому відмінку)
    ("ПІБ", re.compile(rf"\b[{U}][{L}]+\s+[{U}][{L}]+\s+[{U}][{L}]+{_PATRONYMIC}\b")),
    # Ім'я По-батькові Прізвище
    ("ПІБ", re.compile(rf"\b[{U}][{L}]+\s+[{U}][{L}]+{_PATRONYMIC}\s+[{U}][{L}]+\b")),
    # В.М. Соколов / В. М. Соколов / С. Бакуліна
    ("ПІБ", re.compile(rf"\b[{U}]\.\s?(?:[{U}]\.\s?)?[{U}][{L}]{{2,}}\b")),
    # Юрченко І.Я. / Юрченко І. Я.
    ("ПІБ", re.compile(rf"\b[{U}][{L}]{{2,}}\s+[{U}]\.\s?[{U}]?\.?(?=\s|,|;|\)|$)")),
    # коди, за якими особу знаходять у реєстрах
    ("КОД", re.compile(r"(?<=ЄДРПОУ)([^\d\n]{0,12})\d{6,10}")),
    ("КОД", re.compile(r"(?<=РНОКПП)([^\d\n]{0,12})\d{6,10}")),
    ("КОД", re.compile(r"(?<=ІПН)([^\d\n]{0,12})\d{6,10}")),
]

# Words that look like a name to rule 4 ("Прізвище І.") but are not one.
_KEEP = re.compile(rf"^(?:Стаття|Статті|Пункт|Частина|Абзац|Справа|Постанова|Ухвала|"
                   rf"Рішення|Закон|Кодекс|Суд|Суду|Судом|Велика|Палата|Верховний|"
                   rf"Касаційний|Господарський|Цивільний|Адміністративний|Кримінальний)\b")


def scrub_text(text: str) -> tuple[str, int]:
    """→ (text with names replaced, how many replacements)."""
    n = 0
    out = text
    for tag, rx in RULES:
        def repl(m: re.Match[str]) -> str:
            nonlocal n
            hit = m.group(0)
            if tag == "ПІБ" and _KEEP.match(hit):
                return hit
            n += 1
            if tag == "КОД":
                # keep the label, drop the number: "ЄДРПОУ 19364710" -> "ЄДРПОУ КОД"
                return f"{m.group(1)}КОД" if m.groups() else "КОД"
            return "ПІБ"
        out = rx.sub(repl, out)
    return out, n


def scrub_row(row: dict) -> tuple[dict, int]:
    r = dict(row)
    state = r.get("state")
    total = 0
    if isinstance(state, str):
        r["state"], total = scrub_text(state)
    elif isinstance(state, dict):
        new = {}
        for k, v in state.items():
            if isinstance(v, str):
                new[k], c = scrub_text(v)
                total += c
            else:
                new[k] = v
        r["state"] = new
    return r, total


def scrub_rows(rows: Iterable[dict]) -> Iterator[tuple[dict, int]]:
    for row in rows:
        yield scrub_row(row)


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="vlc-judge scrub-gold")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows, changed, total = 0, 0, 0
    with open(a.out, "w", encoding="utf-8") as f:
        for line in Path(a.src).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row, n = scrub_row(json.loads(line))
            rows += 1
            changed += 1 if n else 0
            total += n
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": rows, "rows_changed": changed, "replacements": total},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
