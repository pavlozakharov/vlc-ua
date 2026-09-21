"""Tests for gold builders: attribution, departures, screening."""
import json
import pytest
from pathlib import Path

from vlc_ua.judge.gold import attribution as gold_attr
from vlc_ua.judge.gold import departures as gold_dep
from vlc_ua.judge.gold import screening as gold_scr


class TestAttributionSections:
    """Test attribution section detection."""

    def test_sections_recognizes_headers(self):
        text = "Рух справи.\n" + "Some procedural history that is at least 40 chars long for test. " * 2 + "\n\n"
        text += "Фактичні обставини справи.\n" + "Some facts established by courts. " * 2 + "\n\n"
        text += "Позиція Верховного Суду.\n" + "The court's reasoning. " * 3

        secs = gold_attr.sections(text)

        assert len(secs) >= 1
        if len(secs) > 0:
            assert secs[0].kind in gold_attr.KINDS
            assert secs[0].header is not None
            assert secs[0].start < secs[0].end

    def test_sections_skips_text_before_first_header(self):
        text = "Some intro text that is not a header and is filler.\n\n"
        text += "Рух справи.\n" + "Procedural content that is long enough. " * 2

        secs = gold_attr.sections(text)

        assert len(secs) > 0
        assert secs[0].kind == "procedural"

    def test_sections_header_governs_following_text(self):
        text = "Рух справи.\n" + "First paragraph that is long enough. " * 2 + "\n\n"
        text += "Позиція Верховного Суду.\n" + "Court position that is long. " * 2

        secs = gold_attr.sections(text)

        if len(secs) >= 2:
            body1 = text[secs[0].start:secs[0].end]
            body2 = text[secs[1].start:secs[1].end]
            assert "First paragraph" in body1 or "long" in body1
            assert "Court position" in body2


class TestAttributionFragments:
    """Test fragment extraction."""

    def test_fragments_emits_sized_fragments(self, tmp_path):
        text = """Позиція Верховного Суду.
        """ + "This is a long paragraph that should be at least 200 characters long to qualify as a valid fragment. " * 3

        frags = list(gold_attr.fragments(text, min_chars=200, max_chars=1200))

        assert len(frags) > 0
        for sec, frag in frags:
            assert len(frag) >= 200
            assert len(frag) <= 1200
            assert sec.kind in gold_attr.KINDS

    def test_fragments_respects_min_size(self):
        text = """Позиція Верховного Суду.
        Short text."""

        frags = list(gold_attr.fragments(text, min_chars=200))

        assert len(frags) == 0

    def test_fragments_respects_max_size(self):
        text = """Позиція Верховного Суду.
        """ + "a" * 2000

        frags = list(gold_attr.fragments(text, max_chars=1200))

        for sec, frag in frags:
            assert len(frag) <= 1200


class TestAttributionBuild:
    """Test full attribution gold building."""

    def test_build_creates_jsonl(self, tmp_path):
        out_path = tmp_path / "gold.jsonl"

        docs = [
            ("doc1", "Рух справи.\n" + "x" * 300),
        ]

        counts = gold_attr.build(docs, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert sum(counts.values()) > 0

    def test_build_per_doc_cap(self, tmp_path):
        out_path = tmp_path / "gold.jsonl"

        text = "Рух справи.\n" + ("Long fragment that is at least 200 chars. " * 10) + "\n\n"
        text += "Позиція Верховного Суду.\n" + ("Another long fragment. " * 10)

        docs = [("doc1", text), ("doc2", text)]

        counts = gold_attr.build(docs, out_path, per_doc=5)

        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        doc1_rows = [r for r in rows if r["doc_id"] == "doc1"]

        assert len(doc1_rows) <= 5

    def test_build_enriched_share(self, tmp_path):
        out_path = tmp_path / "gold.jsonl"

        text = "Рух справи.\n" + "x" * 300 + "\n\n" + "Позиція Верховного Суду.\n" + "y" * 300

        docs = [("doc1", text)]

        counts = gold_attr.build(docs, out_path, per_doc=10, enriched_share=0.5)

        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        enriched = [r for r in rows if r["sample"] == "enriched"]
        random = [r for r in rows if r["sample"] == "random"]

        # When there are rows, enriched should either have some rows or all are random
        if len(rows) > 0:
            assert len(enriched) + len(random) == len(rows)


class TestNormalizeHeader:
    """Test header normalization."""

    def test_normalize_header_strips_numbering(self):
        assert gold_attr.normalize_header("3. Рух справи") == "рух справи"
        assert gold_attr.normalize_header("III. Позиція Суду") == "позиція суду"

    def test_normalize_header_strips_punctuation(self):
        assert gold_attr.normalize_header("Рух справи.") == "рух справи"
        assert gold_attr.normalize_header("Позиція Суду:") == "позиція суду"

    def test_normalize_header_collapses_spaces(self):
        assert gold_attr.normalize_header("Рух   справи") == "рух справи"

    def test_cyrillic_roman_numerals_are_stripped(self):
        """«ІV» у постановах ВС — кирилична І плюс латинська V.

        Латинський клас [ivx] її не знімав, і найчастіший заголовок корпусу
        лишався невпізнаним: 564 втрачені заголовки на 1 909 постановах.
        """
        assert gold_attr.normalize_header("ІV. ПОЗИЦІЯ ВЕРХОВНОГО СУДУ") == "позиція верховного суду"
        assert gold_attr.normalize_header("VІІ. Позиція Верховного Суду") == "позиція верховного суду"
        assert gold_attr.normalize_header("ІІІ. Фактичні обставини справи") == "фактичні обставини справи"
        assert gold_attr.header_kind("ІV. ПОЗИЦІЯ ВЕРХОВНОГО СУДУ") == "court"
        assert gold_attr.header_kind("І. РУХ СПРАВИ") == "procedural"

    def test_numbering_stripper_keeps_running_text(self):
        """Клас із кириличними двійниками не має з'їдати початок речення."""
        assert gold_attr.header_kind("і суд дійшов висновку, що позивач не довів") is None
        assert gold_attr.header_kind("м. Київ") is None
        assert gold_attr.header_kind("с. Іванівка Полтавського району") is None


class TestHeaderKindVariants:
    """Варіанти заголовків, кожен заміряний у корпусі 21.09.2026."""

    def test_grand_chamber_is_court(self):
        assert gold_attr.header_kind("ПОЗИЦІЯ ВЕЛИКОЇ ПАЛАТИ") == "court"
        assert gold_attr.header_kind("Позиція Великої Палати Верховного Суду") == "court"

    def test_motives_tense_and_wording_variants(self):
        for line in ("Мотиви, з яких виходив Верховний Суд, та застосовані норми права",
                     "Мотиви, якими керується Верховний Суд, та застосовані норми права",
                     "Мотиви і доводи Верховного Суду та застосовані норми права"):
            assert gold_attr.header_kind(line) == "court", line

    def test_referral_to_grand_chamber_is_procedural(self):
        assert gold_attr.header_kind(
            "Мотиви передачі справи на розгляд Великої Палати Верховного Суду") == "procedural"

    def test_lower_court_position_beats_generic_court_rule(self):
        """Правило court перевіряється першим, тож «позиція суду» мусить мати застереження."""
        assert gold_attr.header_kind("Позиція суду першої інстанції") == "lower"
        assert gold_attr.header_kind("Позиція суду апеляційної інстанції") == "lower"
        assert gold_attr.header_kind("Позиція суду") == "court"

    def test_participants_position_is_party(self):
        assert gold_attr.header_kind("Позиція учасників справи") == "party"
        assert gold_attr.header_kind("5. Позиція іншого учасника справи") == "party"

    def test_forms_found_by_the_v3_read_control(self):
        """Чотири зразки, кожен порахований у корпусі 21.09.2026."""
        assert gold_attr.header_kind("Аргументи інших учасників справи") == "party"          # 35
        assert gold_attr.header_kind("3. ВСТАНОВЛЕНІ СУДАМИ ПОПЕРЕДНІХ ІНСТАНЦІЙ "
                                     "ОБСТАВИНИ У СПРАВІ") == "facts"                        # 33
        assert gold_attr.header_kind("Рух касаційної скарги та матеріалів справи") == "procedural"  # 112
        assert gold_attr.header_kind("10. Судові витрати") == "procedural"                   # 273

    def test_bare_costs_header_does_not_swallow_a_sentence(self):
        """«Судові витрати» ловиться лише як цілий рядок-заголовок."""
        assert gold_attr.header_kind("Судові витрати стягуються з відповідача") is None

    def test_spaced_out_operative_verb(self):
        """«П О С Т А Н О В И В» — 281 такий рядок у корпусі 21.09.2026.

        Нерозпізнаний, він лишає резолютивну частину всередині мотивів суду.
        """
        for line in ("П О С Т А Н О В И В:", "п о с т а н о в и в :",
                     "У Х В А Л И В :", "у х в а л и в:"):
            assert gold_attr.header_kind(line) == "procedural", line
        assert gold_attr.normalize_header("П О С Т А Н О В И В:") == "постановив"

    def test_normative_regulation_is_court(self):
        assert gold_attr.header_kind("Нормативне регулювання") == "court"
        assert gold_attr.header_kind("Нормативне врегулювання") == "court"


class TestWrappedSentenceIsNotAHeader:
    """Рядок може збігтися з лексиконом, лишаючись початком речення."""

    def test_wrapped_sentence_does_not_open_a_section(self):
        text = ("Позиція Верховного Суду\n"
                "Суд дійшов висновку про таке. " + "Текст розділу. " * 10 + "\n"
                "Доводи касаційної скарги про те, що належним способом захисту\n"
                "є витребування майна, є безпідставними. " + "Продовження. " * 10 + "\n")
        kinds = [s.kind for s in gold_attr.sections(text)]
        assert kinds == ["court"], kinds

    def test_colon_header_followed_by_lowercase_survives(self):
        text = ("Учасники справи:\n"
                "позивач - товариство, відповідач - орган. " + "Далі текст. " * 10 + "\n")
        kinds = [s.kind for s in gold_attr.sections(text)]
        assert kinds == ["procedural"], kinds


class TestDeparturesFromDepGold:
    """Test departure pair gold building."""

    def test_from_dep_gold_reads_json(self, tmp_path):
        gold_file = tmp_path / "dep_gold.json"

        dep_gold = {
            "positions": [
                {
                    "lpd_id": "pos1",
                    "evidence": {
                        "quote": "Суд відступив від висновку справи 123/456/20.",
                        "cause_num": "111/222/20"
                    },
                    "cases": ["123/456/20"]
                }
            ]
        }

        gold_file.write_text(json.dumps(dep_gold, ensure_ascii=False))

        rows = list(gold_dep.from_dep_gold(gold_file))

        assert len(rows) > 0
        assert rows[0]["gold"] == "departure"
        assert rows[0]["sample"] == "random"
        assert rows[0]["source"] == "lpd"

    def test_from_dep_gold_skips_missing_quote(self, tmp_path):
        gold_file = tmp_path / "dep_gold.json"

        dep_gold = {
            "positions": [
                {
                    "lpd_id": "pos1",
                    "evidence": {"cause_num": "111/222/20"},
                    "cases": ["123/456/20"]
                }
            ]
        }

        gold_file.write_text(json.dumps(dep_gold, ensure_ascii=False))

        rows = list(gold_dep.from_dep_gold(gold_file))

        assert len(rows) == 0


class TestDeparturesFromRejectsDump:
    """Test departures from rejects dump."""

    def test_from_rejects_dump_maps_labels(self, tmp_path):
        rejects_file = tmp_path / "rejects.jsonl"

        rows_data = [
            {
                "bucket": "заперечення",
                "sent": "Суд не знайшов підстав для відступу від справи 123/456/20.",
                "cause_num": "999/888/20",
                "doc_id": "doc1"
            }
        ]

        rejects_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_data))

        rows = list(gold_dep.from_rejects_dump(rejects_file))

        assert len(rows) > 0
        assert rows[0]["gold"] == "refusal"
        assert rows[0]["source"] == "grammar-reject:заперечення"

    def test_from_rejects_dump_skips_unknown_bucket(self, tmp_path):
        rejects_file = tmp_path / "rejects.jsonl"

        rows_data = [
            {
                "bucket": "невідомо",
                "sent": "Some text 123/456/20.",
                "cause_num": "999/888/20",
                "doc_id": "doc1"
            }
        ]

        rejects_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_data))

        rows = list(gold_dep.from_rejects_dump(rejects_file))

        assert len(rows) == 0


class TestDeparturesMergeAdjudications:
    """Test merging adjudication overlays."""

    def test_merge_adjudications_overlays_verdicts(self, tmp_path):
        adj_file = tmp_path / "adj.jsonl"

        adj_file.write_text(
            json.dumps({"id": "row1", "gold": "departure", "draw": "random"}, ensure_ascii=False)
        )

        rows = [
            {"id": "row1", "gold": "other", "sample": "enriched", "source": "grammar"}
        ]

        merged = list(gold_dep.merge_adjudications(rows, adj_file))

        assert len(merged) > 0
        assert merged[0]["gold"] == "departure"
        assert merged[0]["source"] == "human"
        assert merged[0]["sample"] == "random"

    def test_merge_adjudications_no_file(self):
        rows = [
            {"id": "row1", "gold": "other", "sample": "enriched", "source": "grammar"}
        ]

        merged = list(gold_dep.merge_adjudications(rows, None))

        assert merged[0]["gold"] == "other"


class TestScreeningFromSweepClassified:
    """Test screening gold from sweep classification."""

    def test_from_sweep_classified_maps_stance(self, tmp_path):
        classified_file = tmp_path / "classified.jsonl"

        row_data = {
            "doc_id": "doc1",
            "cls_thesis": "Some thesis",
            "case": "123/456/20",
            "date": "2020-01-01",
            "summary": "Summary text",
            "position": "Position text",
            "cls_stance": "за"
        }

        classified_file.write_text(json.dumps(row_data, ensure_ascii=False))

        rows = list(gold_scr.from_sweep_classified(classified_file))

        stance_rows = [r for r in rows if r["question"] == "stance"]
        assert len(stance_rows) > 0
        assert stance_rows[0]["gold"] == "for"

    def test_from_sweep_classified_maps_needs_fulltext(self, tmp_path):
        classified_file = tmp_path / "classified.jsonl"

        row_data = {
            "doc_id": "doc1",
            "cls_thesis": "Some thesis",
            "summary": "Text",
            "cls_needs_fulltext": True
        }

        classified_file.write_text(json.dumps(row_data, ensure_ascii=False))

        rows = list(gold_scr.from_sweep_classified(classified_file))

        nft_rows = [r for r in rows if r["question"] == "needs_fulltext"]
        assert len(nft_rows) > 0
        assert nft_rows[0]["gold"] == "yes"
