"""Keyword baselines. Every head is measured against one before any model.

An independent replication found a hand-written keyword rule beating the
vendor model on an easy set (97.2% vs 91.7%), and this project measured the
same thing on departures ("граматика > модель"). A head that cannot beat
its keyword baseline on the random slice has no business in production.

Rules return per-option scores in [0, 1]; the ScoringJudge softmax turns
them into a distribution. Scores are deliberately coarse so that the
baseline's "confidence" is honest about being a rule.
"""
from __future__ import annotations

import re
from typing import Mapping

from ..backend import ScoringJudge
from ..types import Question, State

_ATTR = {
    "court": [r"верховн\w+ суд\w* (виходить|вважає|зазначає|дійш\w+ висновк|погоджу|не погоджу|відхиляє|бере до уваги)",
              r"колегія суддів (вважає|зазначає|дійш\w+|погоджу|виходить)",
              r"велика палата (вважає|зазначає|дійш\w+|відступ|виходить)",
              r"об['’]єднана палата", r"з огляду на (наведене|викладене)", r"отже,", r"таким чином,"],
    "party": [r"скаржник (зазначає|вказує|посилається|вважає|наголошує)",
              r"(позивач|відповідач|заявник) (зазначає|вказує|посилається|вважає|просить)",
              r"у касаційній скарзі", r"у відзиві", r"доводи касаційної скарги", r"просить (скасувати|залишити)"],
    "lower": [r"суд (першої|апеляційної) інстанції (виходив|дійшов|встановив|зазначив|вважав|відмовив|задовольнив)",
              r"(рішенням|ухвалою|постановою) .{0,60}(суду|суд) .{0,40}(відмовлено|задоволено|скасовано)",
              r"апеляційний суд (погодився|скасував|змінив|залишив)"],
    "facts": [r"суди встановили", r"судами встановлено", r"установлено, що", r"згідно з (договором|актом|випискою)",
              r"\b\d{2}\.\d{2}\.\d{4}\b.{0,80}(укладено|видано|зареєстровано|подано)"],
    "procedural": [r"керуючись стат", r"постановив:", r"ухвалив:", r"судовий збір", r"справу передано",
                   r"відкрито касаційне провадження", r"надійшла касаційна скарга", r"розподіл судових витрат"],
}

_DEP = {
    "departure": [r"відступ(ила|ив|лено|ає|ають|аючи|ивши) від (висновк|правов\w+ позиці|правов\w+ висновк)"],
    "refusal": [r"не (вбачає|знаходить|вбачають|знайшла) підстав для відступ", r"відсутні підстави для відступ",
                r"немає підстав (для )?відступ"],
    "norm_quote": [r"вважає за необхідне відступити", r"якщо (ця )?колегія вважає за необхідне відступити",
                   r"передає справу на розгляд .{0,40}палати"],
    "generic": [r"у разі, коли .{0,60}відступ", r"незалежно від того,? чи", r"неодноразово наголошувала"],
    "other": [],
}

_FRAG = {
    "departure": _DEP["departure"],
    "referral": [r"передати справу на розгляд (великої палати|об['’]єднаної палати)", r"підстав\w* для передачі",
                 r"виключна правова проблема"],
    "distinguishing": [r"не є подібними", r"правовідносини .{0,40}не (є )?подібн", r"відмінн\w+ (фактичн\w+ )?обставин",
                       r"за інших фактичних обставин", r"не застосов\w+ до спірних правовідносин"],
    "application": [r"(застосував|застосовує|застосовуючи|з урахуванням) (висновк|правов\w+ позиці)",
                    r"аналогічн\w+ (правов\w+ )?(висновк|позиці)", r"викладен\w+ у постанов\w+ .{0,80}(враховує|застосов)"],
    "mention": [],
}

RULES: dict[str, dict[str, list[str]]] = {
    "attribution": _ATTR, "departure_pair": _DEP, "fragment_kind": _FRAG,
}


def _text_of(state: State) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        return " ".join(str(v) for v in state.values() if isinstance(v, str))
    return " ".join(str(x) for x in state)


def make_keyword_judge(question_name: str, base: float = 0.1, hit: float = 1.5) -> ScoringJudge:
    """A ScoringJudge whose scores count regex hits per option.

    Options with no rules (``other``, ``mention``) get ``base``: they win only
    when nothing else fires, which is exactly the semantics of a fallback.
    """
    rules = RULES[question_name]
    compiled = {opt: [re.compile(p, re.I) for p in pats] for opt, pats in rules.items()}

    def score(state: State, question: Question, option: str) -> float:
        text = _text_of(state).lower()
        pats = compiled.get(option)
        if not pats:
            return base if option in rules else 0.0
        hits = sum(1 for rx in pats if rx.search(text))
        return base + hit * hits

    return ScoringJudge(score, name=f"keyword:{question_name}")
