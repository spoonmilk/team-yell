# Team Yell: Adversarial Learning on Speech-to-text Models

## Dependencies & Development

### REQUIRED: Python 3.11

Currently, pytorch does not support python 3.13. Downgrade to 3.11 with `pyenv` or related tools

### Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

### Install dependencies with poetry

```bash
poetry env use 3.11
poetry install
```

### Use poetry to activate virtual environment

```bash
# cd team-yell
poetry env activate
```
