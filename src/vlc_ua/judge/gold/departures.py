"""Gold builders for the departure ("відступ") heads.

Two heads, two sources:

1. ``departure_pair``: for a sentence mentioning a departure and ONE target
   case, does the sentence report an *actually performed* departure by the
   named court from the conclusion in that target case? Options:
   ``departure`` / ``refusal`` (court declined to depart) / ``norm_quote``
   (quotes the procedural rule about departing) / ``generic`` (general
   remark) / ``other``.

   The question is asked per (sentence, target) pair on purpose: one
   sentence lawfully lists several targets, and a verifier that asks "is
   this pair right?" over the whole sentence anchors on the first target.

2. ``fragment_kind``: for a fragment that mentions a given case, which
   relation does it express? ``departure`` / ``referral`` (transfer to a
   Grand Chamber or joint chamber) / ``distinguishing`` / ``application`` /
   ``mention``. This is the per-fragment head for ``check_departure``.

Sources on the server:

* ``positions.db`` table ``departures`` (dep_case, from_case, quote, kind,
  source): positive pairs from the regex grammar. Their precision was
  measured at 87-88% on the reported branch, so they are *weak* positives
  until adjudicated; rows are marked ``source="grammar"``.
* the rejects dump of ``extract_departures3.py`` (``DUMP_REJECTS``): buckets
  such as ``заперечення`` are natural negatives for ``refusal``; ``генерика``
  for ``generic``. Marked ``source="grammar-reject:<bucket>"``.
* ``dep_gold.json`` positions with an ``evidence.quote``: human-curated
  (official LPD markup) positives, marked ``source="lpd"``.

Adjudicated rows (a human read the sentence) get ``source="human"`` and
``sample="random"``; everything else is ``sample="enriched"`` because the
grammar's output is not a random draw from the corpus.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

CASE_RX = re.compile(r"\b\d{1,5}/\d{1,6}/\d{2}(?:-[а-яa-zіїє]+)?\b|\b\d{1,3}-\d{2,5}[а-я]{1,3}\d{2}\b")

TASK = {
    "departure_pair": {
        "type": "choice",
        "instructions": ("Речення з постанови Верховного Суду і одна ЦІЛЬ — номер справи, "
                         "згаданий у ньому. Що саме речення повідомляє про висновок у цій цільовій "
                         "справі? Оцінюй лише названу ціль, навіть якщо в реченні перелічено кілька справ."),
        "criteria": {
            "departure": "названий суд (ВП ВС, об'єднана або судова палата) ФАКТИЧНО відступив від висновку, викладеного у цільовій справі",
            "refusal": "суд розглянув і НЕ знайшов підстав відступати від висновку у цільовій справі",
            "norm_quote": "цитата процесуальної норми про право або порядок відступу, без конкретного відступу",
            "generic": "загальне міркування про відступи або наслідки відступу, без відступу від цільової справи",
            "other": "цільова справа згадана з іншої причини (застосування висновку, фабула, посилання сторони)",
        },
    },
    "fragment_kind": {
        "type": "choice",
        "instructions": ("Уривок судового рішення, що згадує вказану справу. Який стосунок до "
                         "позиції з цієї справи він виражає?"),
        "criteria": {
            "departure": "здійснений відступ від висновку цієї справи",
            "referral": "передача справи на розгляд Великої Палати або об'єднаної палати для можливого відступу",
            "distinguishing": "розмежування обставин: висновок не застосовується, бо правовідносини не подібні",
            "application": "застосування висновку цієї справи як чинного",
            "mention": "нейтральна згадка, переказ доводу сторони або цитата без власної оцінки суду",
        },
    },
}


def _row(rid: str, state: dict, question: str, gold: str, sample: str, source: str) -> dict:
    return {"id": rid, "state": state, "question": question, "gold": gold,
            "sample": sample, "source": source}


def from_departures_table(db_path: str, limit: int | None = None) -> Iterator[dict]:
    """Weak positives for ``departure_pair`` from positions.db.departures."""
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql = "SELECT rowid, dep_case, from_case, quote, kind, source FROM departures WHERE quote IS NOT NULL"
    if limit:
        sql += f" LIMIT {int(limit)}"
    for rowid, dep_case, from_case, quote, kind, source in con.execute(sql):
        if not from_case or not quote:
            continue
        gold = "departure" if (kind or "departure") == "departure" else "other"
        yield _row(f"dep:{rowid}", {"sentence": quote, "target_case": from_case,
                                    "citing_case": dep_case},
                   "departure_pair", gold, "enriched", f"grammar:{source or ''}:{kind or ''}")


BUCKET_LABEL = {
    "заперечення": "refusal",
    "генерика": "generic",
    "без суб'єкта": None,        # unknown without reading: skip
    "без зони цілі": None,
    "ціль не валідна": None,
    "без «виклад»": None,
    "репорт без джерела": None,
    "інверсія": None,
}


def from_rejects_dump(path: str | Path, limit: int | None = None) -> Iterator[dict]:
    """Negatives for ``departure_pair`` from the extractor's rejects dump."""
    n = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        label = BUCKET_LABEL.get(d.get("bucket"))
        if not label:
            continue
        sent = d.get("sent") or ""
        targets = [c for c in CASE_RX.findall(sent) if c != d.get("cause_num")]
        if not targets:
            continue
        for t in targets[:3]:
            yield _row(f"rej:{d.get('doc_id')}:{abs(hash(sent)) % 10**8}:{t}",
                       {"sentence": sent, "target_case": t, "citing_case": d.get("cause_num")},
                       "departure_pair", label, "enriched", f"grammar-reject:{d.get('bucket')}")
            n += 1
            if limit and n >= limit:
                return


def from_dep_gold(path: str | Path) -> Iterator[dict]:
    """Human-curated positives from the LPD markup (dep_gold.json)."""
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    for p in g.get("positions", []):
        ev = p.get("evidence") or {}
        quote, cases = ev.get("quote"), p.get("cases") or []
        if not quote or not cases:
            continue
        for c in cases:
            if c in quote:
                yield _row(f"lpd:{p.get('lpd_id')}:{c}", {"sentence": quote, "target_case": c,
                                                           "citing_case": ev.get("cause_num")},
                           "departure_pair", "departure", "random", "lpd")


def merge_adjudications(rows: Iterable[dict], adjudication_path: str | Path | None) -> Iterator[dict]:
    """Overlay human verdicts: ``{"id": ..., "gold": ...}`` per line.

    Adjudicated rows become ``sample="random"`` and ``source="human"`` only if
    the adjudication file says they were drawn at random (``"draw": "random"``).
    """
    adj: dict[str, dict] = {}
    if adjudication_path and Path(adjudication_path).exists():
        for line in Path(adjudication_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                adj[d["id"]] = d
    for r in rows:
        a = adj.get(r["id"])
        if a:
            r = dict(r)
            r["gold"] = a["gold"]
            r["source"] = "human"
            r["sample"] = "random" if a.get("draw") == "random" else r["sample"]
        yield r


def write(rows: Iterable[dict], out_path: str | Path) -> int:
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
