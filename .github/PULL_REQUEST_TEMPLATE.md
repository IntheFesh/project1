# Summary

<!-- What does this change and why? -->

## Checklist
- [ ] `make check` is green (ruff + 128 tests + `[eval-gate] PASS`).
- [ ] Tests added/updated for any behaviour change.
- [ ] Data-leakage guards still pass (`tests/test_leakage.py`).
- [ ] **No fabricated metrics.** If a reported number could change, I re-ran the affected
      evaluation and updated `results/*.json` (no table was hand-edited).
- [ ] Anything not actually run (e.g. τ²-bench, BFCL-V4) is labelled future work.

## Notes
<!-- Numbers that changed, follow-ups, or anything a reviewer should know. -->
