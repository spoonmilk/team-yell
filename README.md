# Team Yell: Adversarial Learning on Speech-to-text Models

## File Structure

- ``closed-src/`` for black-box attack architecture
- ``open-src/`` for white-box attack architecture 
- ``utilities/`` for useful functions used in multiple project areas
- ``sketchpad/`` for experimentation/fiddling about with whisper

## Dependencies & Development

### REQUIRED: Python 3.12 OR LOWER

Currently, pytorch does not support python 3.13. Oscar uses 3.9

### Generate .venv and activate

```bash
python3.9 -m venv .venv
source .venv/bin/activate
```

### Install dependencies from requirements.txt

```bash
pip install -r requirements.txt
```

### Deactivate environment

```bash
deactivate
```

## Data Acquisition

### Download LibriSpeech Dataset

```bash
# cd team-yell
wget https://us.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz
rm dev-clean.tar.gz
```
