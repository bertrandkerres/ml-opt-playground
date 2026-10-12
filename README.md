# ML and Optimization Playground
This repo contains an unordered collection of different scripts to solve puzzles, example problems, etc.
It contains code in both python and Julia; environments are defined under `envs/`.

## Setup

```bash
# Python env
uv sync --project envs/python

# Julia env
julia --project=envs/julia -e 'using Pkg; Pkg.instantiate()'
```

Jupyter notebooks in python are stored as `.py`-file and synced using `jupytext`:
```bash
uv run --with jupytext --no-project jupytext --sync <notebook>
```
For Julia, direct scripts or `Pluto.jl` is used, so no sync is necessary.