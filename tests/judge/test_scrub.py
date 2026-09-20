"""Tests for the pre-upload scrubber: it must take names and nothing else."""
from vlc_ua.judge.gold.scrub import scrub_text, scrub_row


class TestTakesNames:
    def test_full_name_with_patronymic(self):
        out, n = scrub_text("залишити позов ОСОБА_1 до Федорова Олега Анатолійовича без руху")
        assert "Федоров" not in out and "ПІБ" in out and n == 1
        assert "ОСОБА_1" in out

    def test_judges_signature_block(self):
        out, n = scrub_text("Головуючий О. О. Шишов Судді І. А. Васильєва М. М. Яковенко")
        assert "Шишов" not in out and "Васильєва" not in out and "Яковенко" not in out
        assert n >= 3

    def test_surname_then_initials(self):
        out, _ = scrub_text("у складі судді Федоріщева С. С. заяву задоволено")
        assert "Федоріщева" not in out

    def test_registry_codes(self):
        out, _ = scrub_text("ТОВ «Ромашка» (код ЄДРПОУ 19364710) звернулося")
        assert "19364710" not in out and "ЄДРПОУ" in out


class TestKeepsEverythingElse:
    def test_case_numbers_dates_articles_survive(self):
        src = ("Постановою від 16.03.2021 у справі № 906/1174/18 Велика Палата Верховного "
               "Суду застосувала статтю 625 ЦК України і стягнула 78 637 грн")
        out, n = scrub_text(src)
        assert out == src and n == 0

    def test_registry_pseudonyms_survive(self):
        src = "ОСОБА_1 за АДРЕСА_2 отримав НОМЕР_3"
        out, n = scrub_text(src)
        assert out == src and n == 0

    def test_court_names_survive(self):
        src = "Касаційний господарський суд у складі Верховного Суду"
        out, n = scrub_text(src)
        assert out == src and n == 0

    def test_numbered_heading_is_not_a_name(self):
        src = "V. ВИСНОВКИ ВЕРХОВНОГО СУДУ"
        out, n = scrub_text(src)
        assert out == src and n == 0


class TestRow:
    def test_row_keeps_label_and_id(self):
        row = {"id": "x", "gold": "court", "question": "attribution",
               "state": {"fragment": "Головуючий О. О. Шишов"}}
        out, n = scrub_row(row)
        assert out["id"] == "x" and out["gold"] == "court" and n >= 1
        assert "Шишов" not in out["state"]["fragment"]
