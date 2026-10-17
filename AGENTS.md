# RepoEngram contributor instructions

RepoEngram provides evidence-backed repository memory and project Skills. Public package metadata and CLI branding use repoengram; internal Python modules and compatible commands retain skillforge.

## Current checkpoint

The 0.1.1 snapshot dated 2026-10-17 is documented in docs/checkpoint-20261017.md. Production source provenance is in docs/release-provenance.json. Preserve known failures, evaluation limitations and cost uncertainty; do not claim a stable overall improvement percentage. Active optimization is paused unless the user requests a new plan.

## Changes and validation

- Keep source selection, evidence reading, presentation and transport responsibilities separate.
- Preserve existing input, reference, read and output budgets. Do not add a model stage without an evidence-backed need and explicit authorization.
- Run the selected offline release regression with python -m pytest. The explicit testpaths in pyproject.toml define the published validation scope; do not portray it as the entire private research suite.
- Use a repository-local pytest --basetemp when the system temporary directory is inaccessible.
- Preserve tests and public CLI aliases. Do not weaken existing assertions to make a release pass.
- Models, paid evaluations and external-service calls require explicit user authorization. If authorized, use the model and budget requested by the user and retain failures.
- Keep credentials, model raw streams, local runtime artifacts and third-party checkouts out of public commits.
- Do not publish task-specific answers as project knowledge. Knowledge must come from actual repository evidence.
