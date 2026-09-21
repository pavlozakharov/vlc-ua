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


class TestGluedPatronymic:
    """Витяг з RTF склеює слова, і по-батькові лишається без межі слова.

    Заміряно 21.09.2026 на збірці v3: після першої редакції скрубера в
    10 046 рядках лишалось 2 повні ПІБ, обидва склеєні з наступним словом
    ("Васильовичазалишити"). Вивантаження таких рядків назовні і є те, чого
    цей модуль має не допускати.
    """

    def test_glued_full_name_is_removed(self):
        out, n = scrub_text("адвоката Моспана Віталія Васильовичазалишити без задоволення")
        assert "Моспана" not in out and "Віталія" not in out and "Васильович" not in out
        assert n == 1
        assert "залишити" in out   # склеєне слово не з'їдається цілком

    def test_glued_surname_name_patronymic(self):
        out, _ = scrub_text("Кобилецького Вячеслава Вікторовичапро ухвалення рішення")
        assert "Кобилецького" not in out and "Вікторович" not in out
        assert "про ухвалення рішення" in out

    def test_name_patronymic_without_surname(self):
        out, _ = scrub_text("за участю Віталія Васильовича")
        assert "Віталія" not in out and "Васильовича" not in out

    def test_pseudonyms_and_institutions_survive(self):
        for keep in ("ОСОБА_1 звернувся до суду",
                     "Верховний Суд у складі колегії суддів",
                     "Касаційний господарський суд"):
            out, n = scrub_text(keep)
            assert out == keep and n == 0, out


class TestInitialsWithoutSecondDot:
    """«О.В Білоус» — друга ініціала без крапки; знайдено перевіркою перед
    вивантаженням збірки v5: один такий підпис на 10 156 рядків."""

    def test_second_initial_without_dot(self):
        out, n = scrub_text("Судді: О.В Білоус")
        assert "Білоус" not in out and n == 1

    def test_article_reference_is_not_a_name(self):
        out, n = scrub_text("стаття 5 ЦК України")
        assert n == 0 and out == "стаття 5 ЦК України"
