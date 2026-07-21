# Experiment Plan

1. Import Dataset A into Neo4j.
2. Run `python -m scripts.inspect_neo4j`.
3. Run `python -m src.run_pilot --limit 10`.
4. Check `outputs/pilot/summary.csv`.
5. Run `python -m src.run_pilot --limit 12 --balanced --output-dir outputs/pilot_balanced`.
6. Run full v0.1.
7. Run full v0.2.
8. Use `summary.csv` in the dissertation results table.

Expected C3-Lite strengths:

- route selection;
- conflict/update detection;
- procedural rule usage;
- explainability.
