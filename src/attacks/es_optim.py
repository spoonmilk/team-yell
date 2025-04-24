import torch as pt
from torch import nn
import whisper
from ..models.perturbation_model import WavPerturbationModel
from ..utilities.wer import wer
from ..utilities.data_access import grab_batch

# TRAINING HYPERPARAMETERS
POP_SIZE = 50
BATCH_SIZE = 10
LEARNING_RATE = 0.01
NOISE_MEAN = 0
NOISE_STD_DEV_RNG_PORTION = 0.05
MODEL_TYPE = "tiny"
NUM_WORKERS = 5
NUM_EPOCHS = 100

# MODEL PARAMETERS
NUM_LAYERS = 3
NUM_CHANNELS = 32
KERNEL_SIZE = 3
MAX_DELTA = 0.01

# HOME DIRECTORY FOR YOUR PROJECT - CHANGE THIS FOR YOUR SYSTEM!!!
HOME_DIR = "/home/spoonmilk/university/csci1470/team-yell"

# Saving and loading model
CHECKPOINT_PATH = f"{HOME_DIR}/src/attacks/checkpoints/wavperturbation_model.pt"

# Load Whisper
device = "cuda" if pt.cuda.is_available() else "cpu"
whisper_model = whisper.load_model(MODEL_TYPE)


def whisper_transcribe(audio_data: pt.Tensor) -> list[str]:
    """Transcribes all audio sequences encapsulated within an input tensor and returns whisper's transcriptions of them"""
    transcripts = []
    for audio in audio_data:
        # Normalize for whisper
        audio = audio.squeeze().cpu()
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
        if mel.ndim == 3:
            mel = mel[0]
        opts = whisper.DecodingOptions()
        # Decode the audio
        # Transfers back to gpu for decoding
        mel = mel.to(device)
        result = whisper.decode(whisper_model, mel, opts)
        # decode returns a single DecodingResult
        transcripts.append(result.text)
    return transcripts


def noise_params(model: nn.Module):
    device = next(model.parameters()).device
    with pt.no_grad():
        for param in model.parameters():
            data = param.data
            span = (data.max() - data.min()).clamp_min(0.0)
            std_dev = span * NOISE_STD_DEV_RNG_PORTION
            noise = pt.randn_like(data, device=device) * std_dev
            data.add_(noise)


def compute_reward(clean_transcription: list[str], perturbed_transcription: list[str]):
    """
    Compute the reward for the perturbed transcription.

    Returns the 'fitness' of the perturbed transcription.
    The fitness is the average WER between the perturbed and clean transcriptions.
    We want WER to be as large as possible
    """
    wers = []
    for gt, pr in zip(clean_transcription, perturbed_transcription):
        w = wer(gt, pr)
        w_clipped = min(w, 1.0)
        wers.append(w_clipped)
    return sum(wers) / len(wers)


def epoch(
    model: WavPerturbationModel,
    pop_sz: int = POP_SIZE,
    batch_sz: int = BATCH_SIZE,
):
    device = next(model.parameters()).device
    # Grab batch of audio + transcriptions - NOT COMPUTE INTENSIVE
    clean_audio_batch, transcriptions = grab_batch(batch_sz)
    clean_audio_batch = pt.stack([a.to(device) for a in clean_audio_batch], dim=0)

    # Create population of cloned + noised models - NOT COMPUTE INTENSIVE
    population = []
    for _ in range(pop_sz):
        copy = WavPerturbationModel(*model.options)
        copy.load_state_dict(model.state_dict())
        noise_params(copy)
        population.append(copy)

    # Run perturbed audio through whisper - COMPUTE INTENSIVE
    all_preds = []
    with pt.inference_mode():
        for child in population:
            perturbed = child(clean_audio_batch)
            preds = whisper_transcribe(perturbed)  # this calls whisper.decode()
            all_preds.append(preds)

    scores = pt.tensor(
        [compute_reward(transcriptions, preds) for preds in all_preds], device=device
    )
    fitness = pt.exp(scores)
    weights = fitness / fitness.sum()

    # Update model weights
    with pt.no_grad():
        params = list(model.parameters())
        for idx, p in enumerate(params):
            child_params = pt.stack(
                [list(pop.parameters())[idx].data for pop in population], dim=0
            )
            diffs = child_params - p.data.unsqueeze(0)
            # broadcast weights
            w = weights.view(-1, *([1] * (diffs.dim() - 1)))
            step = (w * diffs).sum(0)
            p.data.add_(LEARNING_RATE * step)

    return float(scores.mean().cpu())


def train_es(
    model: WavPerturbationModel,
    epochs: int = NUM_EPOCHS,
):
    print(f"Starting ES training on device={device}")
    for i in range(1, epochs + 1):
        avg_wer = epoch(model, POP_SIZE, BATCH_SIZE)
        print(f"Epoch {i}/{epochs} — avg WER: {avg_wer:.4f}")
    # Save model
    pt.save(
        model.state_dict(),
        CHECKPOINT_PATH,
    )
    print("Model saved!")


if __name__ == "__main__":
    # build fresh attack model
    attack_model = WavPerturbationModel(
        kernel_size=KERNEL_SIZE,
        num_channels=NUM_CHANNELS,
        num_layers=NUM_LAYERS,
        max_delta=MAX_DELTA,
    )
    train_es(attack_model, 4)
