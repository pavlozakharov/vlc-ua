# VLC-UA — Verifiable Legal Citations for Ukrainian Law

**Status: pre-development skeleton.** This repository hosts an upcoming
open-source Python library that makes legal citations in Ukrainian law
*verifiable* — built from the working verification discipline of a
practicing attorney, extracted as a clean-room library.

## Why

Legal AI systems fail dangerously in a specific way: they cite laws and
court decisions that do not say what the citation claims. The failure
modes are legally specific and invisible to generic RAG "grounding"
checks:

- a quote is **verbatim present** in the decision — but it is a
  restatement of the *losing party's argument*, not the court's
  conclusion;
- a legal position is **accurately quoted** — but later Supreme Court
  panels have *departed* from it;
- a citation's requisites (date, document type, case number) are subtly
  wrong — OCR noise, sibling documents of the same case, altered
  punctuation inside "verified" quotes.

Each of these is documented from real practice. For Ukrainian law — 40M
citizens plus ~1M refugees in the EU — no open tooling addresses them.

## Planned components

| Layer | What it does |
|---|---|
| `verbatim` | Quote verification against decision texts: normalization for Ukrainian legal text (homoglyphs, quotation marks, OCR traps), honest divergence reporting |
| `attribution` | Judgment-section attribution: court's own reasoning vs party's position vs lower-court restatement — with conflict-aware output when signals disagree |
| `vitality` | Departure-signal detection for cited legal positions ("signal, not verdict" semantics) |
| `references` | Parser for Ukrainian statute citations: superscript numbers, renamed acts, indirect references — to canonical identifiers |

Plus: open test corpora, EN/UK documentation, and a reference MCP-server
integration so any LLM-agent framework can adopt citation verification.

## Roadmap

Milestones M1–M6 as described in our NGI Zero Commons Fund application
(2026). Development starts upon funding decision; the discipline the
library encodes already runs daily in practice.

## License

Apache-2.0 (see LICENSE).

## Author

Pavlo Zakharov — attorney at the Ukrainian bar, based in Germany;
independent legal-AI researcher.
