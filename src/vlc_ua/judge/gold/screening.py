"""Gold builder for rubric screening (the DAG "does this ruling fall under the
rubric" head) and for the practice-sweep stance head.

Sources on the server:

* the SKILL.state pilot slice (``slice.jsonl``: doc_id, base label) with
  its adjudication sheet (``adjudication.md`` or a JSONL of human verdicts);
* ``sweep_classify`` outputs (``*-classified.jsonl``) for the stance head,
  whose labels come from Haiku and are therefore *enriched/weak* until a
  human reads them.

The rubric text is part of the state, not of the question, so one task file
serves every rubric and the instruction stays a versioned artifact.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

TASK = {
    "rubric_screen": {
        "type": "choice",
        "instructions": ("Дано РУБРИКУ правового питання і повний текст судового рішення. "
                         "Чи підпадає рішення під рубрику по суті, а не за збігом слів?"),
        "criteria": {
            "yes": "суд по суті вирішував питання рубрики або сформулював щодо нього висновок",
            "maybe": "питання рубрики згадане або застосоване побічно; потрібен повний прочит людиною",
            "no": "рішення про інше; збіг лише термінологічний або процесуальний",
        },
    },
    "stance": {
        "type": "choice",
        "instructions": ("Дано ТЕЗУ і витяг з рішення Верховного Суду (конспект або позиція). "
                         "Як рішення співвідноситься з тезою? «Не підтримує» не дорівнює «заперечує»."),
        "criteria": {
            "for": "суд по суті підтримав підхід тези",
            "against": "суд по суті відкинув підхід тези у тих самих правовідносинах",
            "neutral": "та сама тема, але вердикту щодо тези немає: процесуальне, доказове, окремий аспект",
            "unclear": "інша тема або витягу замало",
        },
    },
    "needs_fulltext": {
        "type": "noul",
        "instructions": "Витягу замало для впевненого вердикту щодо тези; потрібен повний текст рішення.",
    },
}

BASE_MAP = {"так": "yes", "можливо": "maybe", "ні": "no", "yes": "yes", "maybe": "maybe", "no": "no"}
STANCE_MAP = {"за": "for", "проти": "against", "нейтральне": "neutral", "незрозуміло": "unclear"}


def from_dag_slice(slice_path: str | Path, rubric: str, fetch_text: Callable[[str], str],
                   adjudication_path: str | Path | None = None, max_chars: int = 28000) -> Iterator[dict]:
    """Rows for ``rubric_screen``. ``fetch_text(doc_id)`` returns the ruling text.

    Base labels come from the two-vendor S2 screening and are ``enriched``;
    adjudicated rows (``{"doc_id": ..., "gold": ..., "draw": "random"}``)
    become human/random.
    """
    adj: dict[str, dict] = {}
    if adjudication_path and Path(adjudication_path).exists():
        for line in Path(adjudication_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                adj[str(d["doc_id"])] = d
    for line in Path(slice_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        doc_id = str(d.get("doc_id"))
        base = BASE_MAP.get(str(d.get("base") or d.get("label") or "").lower())
        a = adj.get(doc_id)
        gold = BASE_MAP.get(str(a["gold"]).lower(), a["gold"]) if a else base
        if not gold:
            continue
        text = fetch_text(doc_id)[:max_chars]
        yield {"id": f"screen:{doc_id}", "state": {"rubric": rubric, "decision_text": text},
               "question": "rubric_screen", "gold": gold,
               "sample": "random" if (a and a.get("draw") == "random") else "enriched",
               "source": "human" if a else "s2-two-vendor", "doc_id": doc_id}


def from_sweep_classified(path: str | Path, adjudication_path: str | Path | None = None) -> Iterator[dict]:
    """Rows for ``stance`` and ``needs_fulltext`` from sweep_classify output."""
    adj: dict[str, dict] = {}
    if adjudication_path and Path(adjudication_path).exists():
        for line in Path(adjudication_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                adj[str(d["doc_id"])] = d
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        doc_id = str(d.get("doc_id"))
        body = "\n".join(x for x in (d.get("summary"), d.get("position")) if x)
        if not body:
            continue
        state = {"thesis": d.get("cls_thesis"), "case": d.get("case"), "date": d.get("date"),
                 "excerpt": body[:3000]}
        a = adj.get(doc_id)
        stance = STANCE_MAP.get(str(a["gold"]).lower(), a["gold"]) if a else STANCE_MAP.get(d.get("cls_stance"))
        if stance:
            yield {"id": f"stance:{doc_id}", "state": state, "question": "stance", "gold": stance,
                   "sample": "random" if (a and a.get("draw") == "random") else "enriched",
                   "source": "human" if a else "haiku", "doc_id": doc_id}
        if a is None and d.get("cls_needs_fulltext") is not None:
            yield {"id": f"nft:{doc_id}", "state": state, "question": "needs_fulltext",
                   "gold": "yes" if d.get("cls_needs_fulltext") else "no",
                   "sample": "enriched", "source": "haiku", "doc_id": doc_id}
