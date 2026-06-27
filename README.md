# Physics of llms repro

## Before first use

### UV
In case you do not have uv, pls make yourself a favor and [install](https://docs.astral.sh/uv/getting-started/installation).

### Env
```
uv sync --all-extras # so that we have testing, precommit and fa
make pre-commit
git submodule update --init --recursive # clones iGSM
```
