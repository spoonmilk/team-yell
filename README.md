# Team Yell: Adversarial Learning on Speech-to-text Models

## Read the papers!

See `./team_yell_final_paper.pdf` and `./team_yell_poster.pdf`

## File Structure

- `./docs/` for internal project documentation
- `./src/` for project code
- `./src/models` for perturbation models (closed and open)
- `./src/attacks/` for attack optimization functions
- `./src/utilities/` for useful functions used in multiple project areas
- `./src/sketchpad/` for experimentation/fiddling about with whisper
- `./src/testing/` for model testing functions

**_All papers are in the top-level directory_**

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

### Preprocess and Save Waveforms and Transcripts

```bash
python3 -m src.utilities.preprocess_wav
```

Check to make sure that a "data" directory was made in src with two files in it.

## Testing

### API Key Set Up

Create API keys with Assembly AI, Gladia, and Speechmatics and put them into a `.env` file in the testing directory.

```bash
# cd team-yell/src/testing
touch .env
```

Format for `.env` should be:

```txt
AAI_API_KEY = "<api_key>"
GLADIA_API_KEY = "<api_key>"
SPEECHMATICS_API_KEY = "<api_key>"
```

### Training

Adjust hyperparameters in `./src/attacks/es_optim.py` as wanted and run

```bash
python3 -m src.attacks.es_optim
```

### Running Tests

Uncomment the desired tests the bottom of `test.py` and run.

```bash
python3 -m src.testing.test
```
