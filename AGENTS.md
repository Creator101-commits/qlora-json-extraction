# QLoRA JSON Extraction

A reproducible Python study comparing prompted baselines with 4-bit QLoRA for extracting schema-valid receipt JSON from noisy OCR text.

## Build commands

```bash
# Verification commands.
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

## Platform support

- Linux with CUDA is the target for training; Python 3.12+.
- CPU-safe tests are required; practical training targets a 16 GB-class GPU and compatible bitsandbytes setup.

## Key conventions

- Group exact and near-duplicate documents before splitting; test documents cannot drive prompts, demonstrations, token limits, or checkpoint selection.
- Validate every output with the versioned Pydantic/JSON Schema and score canonical parsed values.
- v1 scope is real CORD data only: the synthetic-data ablation and DPO phases are out of scope per the settled brief.

## Rules

- Default branch is `main`; never force-push it or rewrite shared history.
- Ask before adding any dependency or package manager.
- Never fabricate data, benchmarks, logs or results.
- Done means the relevant tests actually ran and passed; never skip or mock them to finish.
- No emojis anywhere: code, comments, commits, READMEs, docs.
- No em dashes (the long dash, U+2014) anywhere: code, comments, commits, READMEs, docs. Use a comma, colon, parentheses, or a new sentence instead.
- Keep README in sync with code changes, following its existing template.
- Comments are brief and explain why, only when the code doesn't already say it.
- Keep dataset splits and eval leakage controls intact.

## Commits

- Conventional Commits: `type(scope): summary`. Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`; scope is optional.
- Summary in imperative mood, lowercase, no trailing period, at most 72 characters.
- One logical change per commit; keep formatting-only changes separate.
- Add a body only when the reason is not obvious: explain why, not what, wrapped at 72.
- Breaking changes: `type!:` in the summary plus a `BREAKING CHANGE:` footer.
- Commit only when asked. Never commit secrets, build output, or failing code.
