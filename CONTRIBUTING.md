# Contributing

This repository is primarily a research artifact accompanying an academic project.

## Development Principles

Contributions should preserve:

- reproducibility;
- experiment provenance;
- claim boundaries;
- separation between reusable algorithm code and experiment artifacts.

## Before Opening a Pull Request

Run:

```bash
pytest -q
git diff --check
```

Do not commit:

- credentials;
- API tokens;
- Neo4j passwords;
- private datasets;
- local virtual environments;
- model checkpoints unless explicitly approved.

## Experimental Changes

Any change affecting a frozen formal result should create a new experiment version.

Do not silently overwrite frozen results or move historical tags.

## Research Integrity

Clearly distinguish:

```text
development
pilot
formal
diagnostic
excluded
```

Do not report development results as independent held-out evidence.
