# Verification and current limitations

Date: 2026-10-28. This package contains a research specification, not a completed implementation of the proposed method.

Checks performed:
- Original article v13 read and used as task basis. Its result table is empty and named parameters/author fields remain unfilled. The article occupies 12 rendered pages in the current layout.
- GitHub `janfik200-prog/BZ` default `main` inspected: functions in georag/graph/facts.py, confirm.py, kinds.py, synonyms.py, store.py, and chat/checks.py confirmed; the proposed MinHash and context-applicability end-to-end implementation not confirmed in the reviewed public code. Local development may be newer.
- 5 JSON-schema contracts checked using Draft202012Validator, and one SYNTHETIC example validated against all schemas. Example is a fictional sample, not a geological result.
- 40 explicitly synthetic regression scenarios prepared; they cannot be used for scientific performance estimates.
- `qa/evaluate_applicability.py` is a read-only standalone scorer for gold/pred CSVs; its 4 unit tests passed. No real gold/predictions were provided, so no real performance metric has been calculated.
- The 14-page full DOCX and 1-page executor assignment DOCX were rendered by `render_docx.py`; visual contact sheets inspected for overflow/clipping. These are engineering handoff documents and are not intended to fit the 6–10-page conference article limit.
- The official EEKM/IPUZ paper guidelines specify 6–10 pages (28,000 characters), DOCX and no page numbers for the paper (https://conf-eekm.ru/requirements/).
- All strong novelty claims are contingent on a new independently annotated experiment, with FTR and DIRECT coverage evaluated jointly.
