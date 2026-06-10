## Summary

<!-- What does this PR change, and why? -->

## Checklist

<!-- See CONTRIBUTING.md for details. There is no CI gate yet, so please run these locally. -->

- [ ] `ruff check .` and `ruff format .` pass
- [ ] `mypy pykmc/` passes
- [ ] `pytest` passes
- [ ] New public functions/classes/modules have **NumPy-style docstrings**
- [ ] New public module → added `docs/api/<module>.md` **and** a nav entry in `mkdocs.yml`
- [ ] New/changed `Config` fields verified via `python scripts/generate_parameters_doc.py`
- [ ] `mkdocs build` succeeds, and `mkdocs build --strict` raises no new warnings for files you touched
- [ ] User-facing behavior changes are reflected in the relevant `docs/` page
