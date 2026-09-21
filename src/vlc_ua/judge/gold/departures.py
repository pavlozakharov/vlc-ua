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

import hashlib
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


def _digest(text: str) -> str:
    """Stable short hash of a sentence.

    ``hash()`` is salted per process (PYTHONHASHSEED), so row ids built from
    it change on every run and an adjudication file keyed by id stops
    matching the gold it was written against.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


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


# Keys are the bucket strings the extractor actually writes
# (extract_departures3.py, dumps ~/.edrsr/dep_rejects.jsonl and
# dep_g_rejects.jsonl, checked 2026-09-20). Only two buckets carry a usable
# label; the rest cannot be labelled without reading the sentence.
BUCKET_LABEL = {
    "заперечення": "refusal",
    "генерика": "generic",
    "без суб'єкта ВП/ОП": None,   # unknown without reading: skip
    "без зони цілі": None,
    "ціль не валідна/нема": None,
    "без «виклад»": None,
    "репорт без джерела": None,
    "інверсія": None,
    "G-не постанова": None,
    "G-номер перед маркером": None,
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
            yield _row(f"rej:{d.get('doc_id')}:{_digest(sent)}:{t}",
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
    """Write gold JSONL, dropping repeated ids.

    The same sentence is rejected once per window, so one (doc, sentence,
    target) triple can reach the writer several times. The harness keys
    answers by ``id``, so a duplicate id silently overwrites its twin and
    the row count stops matching the number of questions asked.
    """
    n = 0
    seen: set[str] = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# --- Evidence relabelling -------------------------------------------------
#
# Measured on 2026-09-20 against 48 rows read one by one (the document form,
# "Постанова" or "Ухвала", settled the ambiguous ones):
#
#   * rows built from ``dep_gold.json``: in 71 of 82 pairs the target case is
#     the case that PERFORMED the departure, not the case departed FROM, yet
#     every such pair was labelled ``departure``. Read sample: 0 of 5 correct.
#   * rows built from the rejects bucket "заперечення": the extractor's guard
#     looks for a refusal within ±60 characters of the marker, which reaches
#     into neighbouring sentences — the refusal is visible inside the dumped
#     sentence in only 509 of 5304 rows (9.6%). Read sample: 0 of 15 correct.
#   * rows from ``departures.kind`` other than "departure" were all labelled
#     ``other``, but 176 of them report a performed departure from the target.
#     Read sample: 4 of 7 wrong.
#   * 234 of 4195 rows do not contain their own target case number: the 400
#     character quote window cut it off, so the question cannot be answered
#     from the row at all.
#
# The relabeller therefore keeps only labels that can be read off the sentence
# itself, and drops the rest. Dropped above all: "вважає за необхідне
# відступити" — in a ruling of the Grand Chamber or a joint chamber that IS
# the departure, in a referral order it is only an intention, and Grand
# Chamber rulings quote the referring panel's intention verbatim, so the
# sentence alone does not decide. Those rows need a human.

PERFORMED = re.compile(r"відступ(?:ила|ив|ило|или|ає|ають|аючи|ивши)", re.I | re.U)
INTENT = re.compile(r"вважа\w*\s+(?:за\s+)?необхідн\w*\s+відступити|"
                    r"дійш\w+\s+висновку\s+про\s+необхідність\s+відступ", re.I | re.U)
REFUSED = re.compile(r"(?<![а-яіїєґ'])не\s+(?:вважа|вбача|знаход)\w*[^.]{0,50}?відступ|"
                     r"(?<![а-яіїєґ'])не\s+відступ|відсутні\s+підстави\s+для\s+відступ|"
                     r"нема[єе]?\s+підстав\s+для\s+відступ", re.I | re.U)
# The referral is also written as a noun ("дійшов висновку про необхідність
# передачі справи ... на розгляд об'єднаної палати"), and that form carries the
# procedural article with it, so without the noun the row falls through to the
# norm-quote rule and is labelled as a quotation of the rule instead.
REFERRAL = re.compile(r"(?:перед(?:ає|ати|ано|аючи|ала|ав)|передач[іїу])\s+справ[уи]\s+"
                      r"[^.]{0,100}?на\s+розгляд\s+(?:Велик|Об[’'`ʼ]?єднан|судов)", re.I | re.U)
NORM_CITE = re.compile(r"стат(?:ті|тею|тях|тей)\s*(?:302|303|346|347|403)\b", re.I | re.U)
GENERIC = re.compile(r"у разі,?\s+коли\s+(?:вона|Велика Палата)[^,]{0,40}відступила|"
                     r"незалежно від того,?\s+чи перераховані", re.I | re.U)


def evidence_label(sentence: str, target: str) -> tuple[str | None, str]:
    """(label, why) read off the sentence, or (None, why) when it cannot be.

    Order matters: an explicit refusal or a quoted procedural rule decides
    before the direction test, because both contain the departure verb too.
    """
    s = sentence or ""
    at = s.find(target)
    if at < 0:
        return None, "цілі немає в тексті речення"
    if REFUSED.search(s):
        return "refusal", "у реченні є пряма відмова відступати"
    if GENERIC.search(s):
        return "generic", "загальне міркування про відступ"
    if REFERRAL.search(s):
        return "other", "передача справи на розгляд палати"
    perf = PERFORMED.search(s)
    if perf:
        if at > perf.start():
            return "departure", "ціль стоїть після перформатива «відступив/відступила»"
        return "other", "ціль стоїть перед перформативом: це суд, ЯКИЙ відступив"
    if NORM_CITE.search(s):
        return "norm_quote", "цитата процесуальної норми про порядок відступу"
    if INTENT.search(s):
        return None, "намір «вважає за необхідне відступити»: речення не вирішує, хто суб'єкт"
    return None, "у реченні немає ознаки відступу"


def relabel_by_evidence(rows: Iterable[dict]) -> Iterator[dict]:
    """Keep only rows whose label is visible in the sentence; mark the source."""
    for r in rows:
        st = r.get("state") or {}
        label, why = evidence_label(st.get("sentence", ""), st.get("target_case", ""))
        if not label:
            continue
        out = dict(r)
        out["gold"] = label
        out["source"] = f"evidence({label}) <- {r.get('source', '')}"
        out["evidence"] = why
        yield out
