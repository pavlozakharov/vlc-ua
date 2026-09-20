"""Gold builder for quote attribution from section headers of court decisions.

A Supreme Court ruling is written in named sections. The header tells whose
voice a paragraph is in: the parties' arguments, the lower courts'
findings, the facts, or the court's own reasoning and conclusion. That is a
free, human-authored label for every paragraph, which is why this is the
first head to build: thousands of labelled fragments with no manual work.

Labels (aligned with the ``pos_section.kind`` vocabulary of the production
layer so the two can be compared directly):

* ``court``  : the court's own reasoning / conclusion ("Позиція Верховного Суду")
* ``party``  : arguments of participants, cassation complaint, response
* ``lower``  : restatement of first-instance / appellate rulings
* ``facts``  : factual circumstances established
* ``procedural`` : procedural history, referral, admissibility, resolutive part

Two honesty rules baked in:

1. A fragment is emitted only if its section is unambiguous (one header
   governs it). Text before the first header is skipped.
2. Every gold row records ``source`` = the exact header string it was
   derived from, so a bad label can be traced to a bad header rule instead
   of being argued about.

The header lexicon is a starting point taken from the standard structure of
Supreme Court rulings; the production layer's own detector can be plugged
in via ``header_kind`` to keep both in lockstep.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

KINDS = ("court", "party", "lower", "facts", "procedural")

# Order matters: first match wins. Patterns are matched on a normalized
# header line (lowercase, single spaces, no trailing punctuation).
HEADER_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("court", re.compile(r"^(позиці[яї] верховного суду|мотиви, з яких виходить верховний суд|"
                         r"оцінка аргументів учасників справи|мотивувальна частина|"
                         r"висновки за результатами розгляду|висновки верховного суду|"
                         r"щодо суті касаційн|мотиви суду|позиція суду|"
                         r"джерела права й акти їх застосування|нормативно-правове обґрунтування)")),
    ("party", re.compile(r"^(аргументи учасників справи|доводи (особи, яка подала )?касаційн|"
                         r"доводи (інших учасників|відзив|заперечен)|узагальнені доводи|"
                         r"короткий зміст (вимог )?касаційної скарги|"
                         r"короткий зміст (позовних вимог|позову|заяви|скарги)|"
                         r"позиція (позивача|відповідача|скаржника|інших учасників))")),
    ("lower", re.compile(r"^(короткий зміст (рішень|судових рішень|оскаржуван|ухвал|постанов)|"
                         r"рішення судів (першої|попередніх)|"
                         r"позиція суд(у|ів) (першої|апеляційної|попередніх)|"
                         r"короткий зміст судових рішень судів)")),
    ("facts", re.compile(r"^(фактичні обставини справи|обставини справи|"
                         r"встановлені судами обставини|фактичні обставини)")),
    ("procedural", re.compile(r"^(рух справи|процесуальні дії|історія справи|"
                              r"надходження касаційної скарги|підстави (для )?передачі|"
                              r"межі розгляду|провадження у суді касаційної інстанції|"
                              r"розподіл судових витрат|керуючись|постановив|ухвалив|"
                              r"резолютивна частина|щодо розподілу)")),
]

_HEADER_LINE = re.compile(r"^\s*(?:[ivx]+\.?\s+|\d+(?:\.\d+)*\.?\s+)?([^\n]{3,90})\s*$", re.I)


def normalize_header(line: str) -> str:
    s = re.sub(r"\s+", " ", line.strip().lower())
    s = re.sub(r"^[ivx]+\.?\s+|^\d+(?:\.\d+)*\.?\s+", "", s)
    return s.strip(" .:;—-«»\"'")


def header_kind(line: str) -> str | None:
    """Kind for a candidate header line, or None if it is not a header."""
    h = normalize_header(line)
    if not h or len(h) > 90:
        return None
    for kind, rx in HEADER_RULES:
        if rx.search(h):
            return kind
    return None


@dataclass(frozen=True)
class Section:
    kind: str
    header: str
    start: int   # char offset in the document (first char after header)
    end: int


def sections(text: str, kind_fn: Callable[[str], str | None] = header_kind) -> list[Section]:
    """Split a decision into governed sections by recognised header lines."""
    found: list[tuple[str, str, int, int]] = []
    for m in re.finditer(r"^[^\n]*$", text, flags=re.M):
        line = m.group(0)
        if not line.strip():
            continue
        k = kind_fn(line)
        if k:
            found.append((k, line.strip(), m.start(), m.end()))
    out: list[Section] = []
    for i, (k, hdr, s, e) in enumerate(found):
        nxt = found[i + 1][2] if i + 1 < len(found) else len(text)
        if nxt - e > 40:
            out.append(Section(kind=k, header=hdr, start=e, end=nxt))
    return out


def fragments(text: str, min_chars: int = 200, max_chars: int = 1200,
              kind_fn: Callable[[str], str | None] = header_kind) -> Iterator[tuple[Section, str]]:
    """Paragraph-sized fragments with their governing section."""
    for sec in sections(text, kind_fn):
        body = text[sec.start:sec.end]
        buf = ""
        for para in re.split(r"\n\s*\n|\n(?=\s*\d+\.\s)", body):
            p = re.sub(r"\s+", " ", para).strip()
            if not p:
                continue
            if len(buf) + len(p) + 1 <= max_chars:
                buf = (buf + " " + p).strip()
            else:
                if len(buf) >= min_chars:
                    yield sec, buf
                buf = p[:max_chars]
        if len(buf) >= min_chars:
            yield sec, buf


def build(docs: Iterable[tuple[str, str]], out_path: str | Path, question: str = "attribution",
          seed: int = 20260920, per_doc: int = 6, enriched_share: float = 0.0,
          kind_fn: Callable[[str], str | None] = header_kind) -> dict[str, int]:
    """Write gold JSONL from ``(doc_id, text)`` pairs.

    ``per_doc`` caps fragments per document so a few long rulings do not
    dominate. By default every row is marked ``sample="random"``: the mix of
    kinds is whatever the corpus has. ``enriched_share`` > 0 additionally
    marks that share of rows as ``enriched`` after balancing kinds, for
    sensitivity checks only.
    """
    rnd = random.Random(seed)
    rows: list[dict] = []
    counts = {k: 0 for k in KINDS}
    for doc_id, text in docs:
        frs = list(fragments(text, kind_fn=kind_fn))
        rnd.shuffle(frs)
        for sec, frag in frs[:per_doc]:
            rows.append({"id": f"{doc_id}:{sec.start}:{len(frag)}", "state": {"fragment": frag},
                         "question": question, "gold": sec.kind, "sample": "random",
                         "source": f"header:{sec.header}", "doc_id": str(doc_id)})
            counts[sec.kind] += 1
    if enriched_share > 0 and rows:
        by_kind: dict[str, list[dict]] = {}
        for r in rows:
            by_kind.setdefault(r["gold"], []).append(r)
        n_each = int(len(rows) * enriched_share / max(len(by_kind), 1))
        for k, rs in by_kind.items():
            for r in rs[:n_each]:
                r["sample"] = "enriched"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return counts


TASK = {
    "attribution": {
        "type": "choice",
        "instructions": ("Фрагмент постанови Верховного Суду. Чиїм голосом він написаний? "
                         "Обери один розділ, з якого походить цей текст."),
        "criteria": {
            "court": "власна позиція, мотиви або висновок Верховного Суду",
            "party": "доводи учасників справи: касаційна скарга, відзив, позовні вимоги",
            "lower": "переказ рішень судів першої або апеляційної інстанції",
            "facts": "фактичні обставини справи, встановлені судами",
            "procedural": "рух справи, процесуальні дії, судові витрати, резолютивна частина",
        },
    }
}


def read_sqlite_texts(db_path: str, limit: int | None = None,
                      judgment: str = "Постанова") -> Iterator[tuple[str, str]]:
    """Server-side source: full texts of rulings from edrsr.db.

    The table/column names follow the production schema documented in
    laws_src; adjust the SQL if the schema differs. Read-only URI.
    """
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql = ("SELECT doc_id, text FROM documents WHERE text IS NOT NULL AND length(text) > 3000 "
           "AND judgment = ? ORDER BY date DESC")
    if limit:
        sql += f" LIMIT {int(limit)}"
    for doc_id, text in con.execute(sql, (judgment,)):
        yield str(doc_id), text
