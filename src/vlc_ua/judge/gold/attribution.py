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
#
# Added 2026-09-20 after a sweep of real rulings (inspect over 14 decisions)
# and a reading of 30 gold rows, where 5 of 8 mismatches were caused by an
# unrecognised header: the section above it kept running and swallowed the
# next section's text. Each added pattern was seen in the corpus:
#   party      "короткий зміст вимог і доводів касаційної скарги"
#   lower      "короткий зміст рішення суду першої інстанції" (singular)
#   facts      "СТИСЛИЙ ВИКЛАД ОБСТАВИН СПРАВИ, ВСТАНОВЛЕНИХ СУДАМИ …",
#              "Обставини, встановлені судами"
#   procedural "описова частина", "учасники справи", "щодо судових витрат"
HEADER_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("court", re.compile(r"^(позиці[яї] (верховного суду|великої палати|об.єднаної палати)|"
                         r"мотиви, з яких виход(ить|ив|ила|или)|"
                         r"мотиви, яким[иі] керується (верховний суд|велика палата)|"
                         r"мотиви і доводи верховного суду|"
                         r"оцінка аргументів учасників справи|оцінка верховного суду|мотивувальна частина|"
                         r"висновки за результатами розгляду|висновки верховного суду|"
                         r"щодо суті касаційн|мотиви суду|"
                         # «позиція суду першої інстанції» — це lower, а не court:
                         # правило court перевіряється першим і без застереження
                         # з'їдало заголовок нижчої інстанції.
                         r"позиція суду(?! (першої|апеляційної|попередніх))|"
                         r"джерела права й акти їх застосування|нормативно-правове обґрунтування|"
                         r"нормативне [ву]?регулювання)")),
    ("party", re.compile(r"^(аргументи (інших )?учасників справи|"
                         r"доводи (особи, яка подала )?(касаційн|відзив|заперечен)|"
                         r"доводи (інших учасників|відзив|заперечен)|узагальнені доводи|"
                         r"короткий зміст (?!рішен|судових|оскаржуван|ухвал|постанов)"
                         r"[а-яіїєґ' ]{0,45}касаційної скарги|"
                         r"короткий зміст (позовних вимог|позову|заяви|скарги)|"
                         r"позиці[їя] учасників судового провадження|"
                         r"позиція (позивача|відповідача|скаржника|"
                         r"інших учасників|учасників справи|іншого учасника))")),
    ("lower", re.compile(r"^(короткий зміст (рішень|рішення|судових рішень|оскаржуван|ухвал|постанов)|"
                         r"рішення судів (першої|попередніх)|"
                         r"позиція суд(у|ів) (першої|апеляційної|попередніх)|"
                        r"оцінка суд(у|ів) (першої|апеляційної|попередніх)|"
                         r"короткий зміст судових рішень судів)")),
    ("facts", re.compile(r"^(фактичні обставини справи|обставини справи|"
                         r"встановлені судами обставини|фактичні обставини|"
                         r"стислий виклад обставин справи|виклад обставин справи|"
                         r"обставини, встановлені суд|установлені судами обставини|"
                         r"[ву]становлені суд(ами|ом)[^\n]{0,45}обставини)")),
    ("procedural", re.compile(r"^(рух (справи|касаційної скарги)|процесуальні дії|історія справи|"
                              r"надходження касаційної скарги|підстави (для )?передачі|"
                              r"мотиви перед(ачі|ання) справи|"
                              r"межі розгляду|провадження [ву] суді касаційної інстанції|"
                              r"розподіл судових витрат|керуючись|постановив|ухвалив|"
                              r"резолютивна частина|щодо розподілу|щодо судових витрат|судові витрати$|"
                              r"описова частина|учасники справи|вступна частина)")),
]

# Нумерація розділу з'їдається перед звіркою з лексиконом. Римські цифри в
# постановах ВС набрані ЗМІШАНО: «ІV» — це кирилична І (U+0406) плюс латинська V,
# «VІІ» — латинська V плюс дві кириличні І. Латинський клас [ivx] таку нумерацію
# не знімає, заголовок не впізнається, і розділ тягнеться до наступного впізнаного
# заголовка. Замір 21.09.2026 по 1 909 постановах еталона: 564 втрачені заголовки,
# серед них найчастіший у корпусі «ПОЗИЦІЯ ВЕРХОВНОГО СУДУ».
_NUM_PREFIX = re.compile(r"^(?:[ivxlcdmіхсм]{1,7}\.\s*|[ivxlcdmіхсм]{2,7}\s+|\d+(?:\.\d+)*\.?\s+)")


# Резолютивне дієслово в постановах часто набране врозрядку:
# «П О С Т А Н О В И В :». Замір 21.09.2026 по 1 909 постановах — 281 такий
# рядок; кожен із них означав, що резолютивна частина не відокремилась і
# поїхала в попередній розділ, найчастіше в мотиви суду.
_SPACED_OUT = re.compile(r"^(?:[^\W\d_]\s+){3,}[^\W\d_]\s*:?\s*$")


def normalize_header(line: str) -> str:
    s = re.sub(r"\s+", " ", line.strip().lower())
    if _SPACED_OUT.match(s):
        s = re.sub(r"(?<=\w) (?=\w)", "", s)
    s = _NUM_PREFIX.sub("", s)
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
        if nxt - e <= 40:
            continue
        # Рядок може збігтися з лексиконом, лишаючись початком речення, яке
        # переноситься далі: «Доводи касаційної скарги про те, що належним
        # способом захисту…» — і тоді міркування суду дістає ярлик party.
        # Ознака — продовження з малої літери; двокрапка в кінці рядка знімає
        # підозру, бо «Учасники справи:» — справжній заголовок, після якого
        # теж іде мала літера (570 таких у корпусі проти ~47 обірваних речень).
        # Довжина — друга умова: справжній заголовок короткий, обірване
        # речення довге; без неї відсіювались би й «Рух справи» з абзацом,
        # що починається з малої літери.
        if (len(hdr) > 40 and not hdr.rstrip().endswith(":")
                and text[e:nxt].lstrip()[:1].islower()):
            continue
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
        # ``sec.start`` is the section's offset, shared by every fragment of that
        # section, and the length repeats too (max_chars truncates many to the
        # same size), so an id built from the pair collides. Measured on the
        # 2026-09-20 build: 155 ids repeated, 188 of 9 655 rows never reached
        # the harness, which keys answers by id. The ordinal makes it unique.
        for n, (sec, frag) in enumerate(frs[:per_doc]):
            rows.append({"id": f"{doc_id}:{sec.start}:{n}:{len(frag)}", "state": {"fragment": frag},
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


SUPREME_COURT_CODES = ("9901", "9911", "9921", "9931", "9941", "9951")


def read_sqlite_texts(db_path: str, limit: int | None = None,
                      judgment: str = "Постанова",
                      courts: tuple[str, ...] = SUPREME_COURT_CODES,
                      min_chars: int = 3000, chunk: int = 500,
                      max_candidates: int = 100_000, skip: int = 0,
                      progress=None) -> Iterator[tuple[str, str]]:
    """Server-side source: full texts of rulings from edrsr.db.

    Production schema (``~/.edrsr/edrsr.db``, verified 2026-09-20)::

        documents(doc_id TEXT PRIMARY KEY, court_code, judgment_code,
                  justice_kind, category_code, cause_num, adjudication_date,
                  receipt_date, judge, doc_url, status, date_publ, full_text)
        judgment_forms(judgment_code TEXT PRIMARY KEY, name TEXT)

    Three differences from the original SQL, all forced by the real schema
    and the size of the table (~25 M rows, 39 GB):

    1. columns: ``full_text`` (not ``text``), ``judgment_code`` resolved
       through ``judgment_forms`` (not a ``judgment`` name column),
       ``adjudication_date`` (not ``date``);
    2. recency: there is no index on the date, so ``ORDER BY date DESC``
       scans the table for minutes. Rows are inserted in publication order,
       so ``rowid`` is a monotone proxy for the date (checked: the highest
       rowids of Supreme Court rulings carry 2026-09 dates). Candidates are
       taken index-only by ``rowid DESC`` over ``idx_doc_filter``;
    3. missing texts: the freshest rows are cards without ``full_text`` (the
       daily refresh writes the card first and the text later), so the
       candidate list is walked in chunks until ``limit`` texted rulings are
       collected, instead of taking the first ``limit`` candidates.

    ``courts`` defaults to the Supreme Court (four cassation courts, the
    Grand Chamber, and the court's own code): the header lexicon describes
    their structure and the task asks about "постанова Верховного Суду".

    ``skip`` drops that many candidates from the front before collecting, so a
    holdout can be built from rulings the training set never saw: the pair
    (limit, skip) carves disjoint slices out of one rowid-ordered candidate
    list, and rowid order is publication order, so a skip is also a step back
    in time.
    """
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    row = con.execute("SELECT judgment_code FROM judgment_forms WHERE name = ?",
                      (judgment,)).fetchone()
    if not row:
        raise ValueError(f"judgment form {judgment!r} not found in judgment_forms")
    code = row[0]
    marks = ",".join("?" for _ in courts)
    want = int(limit) if limit else None
    cap = max_candidates if want is None else min(max_candidates, want * 40)
    cap = cap + int(skip)
    cands = [r[0] for r in con.execute(
        f"SELECT rowid FROM documents WHERE court_code IN ({marks}) AND judgment_code = ? "
        f"ORDER BY rowid DESC LIMIT {int(cap)}", (*courts, code))][int(skip):]
    seen = kept = 0
    for i in range(0, len(cands), chunk):
        batch = cands[i:i + chunk]
        q = ("SELECT rowid, doc_id, full_text FROM documents WHERE rowid IN (%s) "
             "ORDER BY rowid DESC" % ",".join(str(r) for r in batch))
        for _rid, doc_id, text in con.execute(q):
            seen += 1
            if not text or len(text) <= min_chars:
                continue
            yield str(doc_id), text
            kept += 1
            if want and kept >= want:
                if progress:
                    progress(seen, kept)
                return
        if progress:
            progress(seen, kept)
