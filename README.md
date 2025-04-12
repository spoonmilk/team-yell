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

### Download LibriSpeech Dataset

```bash
# cd team-yell
wget https://us.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz
rm dev-clean.tar.gz
```
