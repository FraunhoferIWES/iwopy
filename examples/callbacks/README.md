# IWOPY Callback Examples

These examples combine a custom callback for live terminal output with
`iwopy.OptimizationHistory` for recording and plotting intermediate results.

Run the SLSQP example:

```bash
uv run python examples/callbacks/run_slsqp.py
```

Run the pymoo example:

```bash
uv run --extra pymoo python examples/callbacks/run_pymoo.py
```

Both scripts save an objective-history plot under `examples/callbacks/output/`.
The output directory is ignored by Git.