import os
from pathlib import Path

import torch as pt
import whisper
from torch import nn
import numpy as np

from ..models.perturbation_model import WavPerturbationModel
from ..utilities.data_access import grab_batch
from ..utilities.wer import wer

# POTENTIAL REASONS FOR APPARENT NON-LEARNING
# - Not enough epochs/training time
# - Model too complex (too many weights could lead to slow convergence for ES if you think about it)
# - Learning too slow - might need to increase learning rate
# - Max delta too big - might need to loosen max delta restriction
# - Every population member takes a step in the wrong direction so model as a whole does - might need to increase population size
# - Loss function not fine-grained enough for learning to take place

# TASKS
# - Restrict learning to the top 5 or 10 performers - DONE
# - Separate training and testing data - DONE
# - Create testing utility with access to different speech to text models
# - Finish batcher
# - Create other perturbation models

# TRAINING HYPERPARAMETERS
POP_SIZE = 50
BATCH_SIZE = 10
LEARNING_RATE = 1
NOISE_MEAN = 0
NOISE_STD_DEV_RNG_PORTION = 0.05
MODEL_TYPE = "tiny"
NUM_WORKERS = 5
NUM_EPOCHS = 100
PERFORMANCE_CUTOFF = 0.2

# MODEL PARAMETERS
NUM_LAYERS = 3
NUM_CHANNELS = 32
KERNEL_SIZE = 3
MAX_DELTA = 0.1

try:
    BASE_DIR = Path(__file__).resolve().parent
    CHECKPOINT_PATH = str(BASE_DIR / "checkpoints" / "wavperturbation_model.pt")
except NameError:
    AUDIO_DIR = os.path.abspath("./checkpoints/wavperturbation_model.pt")
# # HOME DIRECTORY FOR YOUR PROJECT - CHANGE THIS FOR YOUR SYSTEM!!!
# HOME_DIR = "/home/spoonmilk/university/csci1470/team-yell"

# # Saving and loading model
# CHECKPOINT_PATH = f"{HOME_DIR}/src/attacks/checkpoints/wavperturbation_model.pt"

# Load Whisper
device = "cuda" if pt.cuda.is_available() else "cpu"
whisper_model = whisper.load_model(MODEL_TYPE)

#NOT CHECKED
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

#CHECKED
def noise_params(model: nn.Module):
    device = next(model.parameters()).device
    with pt.no_grad():
        for param in model.parameters():
            data = param.data
            span = (data.max() - data.min()).clamp_min(0.0) #NOTE: this means that some parameters will never have noise added (1x1 bias, eg.)
            std_dev = span * NOISE_STD_DEV_RNG_PORTION
            noise = pt.randn_like(data, device=device) * std_dev
            data.add_(noise)


#CHECKED - NOTE: very heavily penalizes missed words (all words afterwards are considered incorrect)
def compute_reward(clean_transcription: list[str], perturbed_transcription: list[str]) -> list[float]:
    """
    Compute the reward for the perturbed transcription.

    Returns the 'dfitness' of the perturbed transcription.
    The fitness is the average WER between the perturbed and clean transcriptions.
    We want WER to be as large as possible
    """
    wers = []
    for gt, pr in zip(clean_transcription, perturbed_transcription):
        w = wer(gt, pr)
        w_clipped = min(w, 1.0)
        wers.append(w_clipped)
    return sum(wers) / len(wers)

#CHECKED
def create_population(model: WavPerturbationModel, pop_sz: int) -> list[WavPerturbationModel]:
    population = [] 
    for _ in range(pop_sz):
        copy = WavPerturbationModel(*model.options)
        copy.load_state_dict(model.state_dict())
        noise_params(copy)
        population.append(copy)
    return population


#CHECKED
def update_model_weights(model: WavPerturbationModel, population: list[WavPerturbationModel], weights: pt.Tensor):
    with pt.no_grad():
        params = list(model.parameters())
        for idx, parent_p in enumerate(params):
            child_params = pt.stack(
                [list(pop.parameters())[idx].data for pop in population], dim=0
            )
            delta_p = child_params - parent_p.data
            weights_bc = weights.view(-1, *([1] * (delta_p.dim() - 1)))
            # Weighted sum of weights
            step = (weights_bc * delta_p).sum(0)
            parent_p.data.add_(LEARNING_RATE*step)

#CHECKED
def scores_to_weights(scores: pt.Tensor) -> pt.Tensor:
    sorted_scores = pt.sort(scores, descending=True)[0]
    top_scores = sorted_scores[:int(PERFORMANCE_CUTOFF*scores.shape[0])].numpy()
    filter_scores = np.vectorize(lambda score: score if (score in top_scores) else 0)
    scores = pt.tensor(filter_scores(scores.numpy()))
    fitness = pt.exp(scores) #Can result in NAN if one member of scores is particularly bigger than all the others... we prob don't have to worry about that though
    weights = fitness / fitness.sum()
    return weights

def epoch(
    model: WavPerturbationModel,
    pop_sz: int = POP_SIZE,
    batch_sz: int = BATCH_SIZE,
):
    device = next(model.parameters()).device
    # Grab batch of audio + transcriptions - NOT COMPUTE INTENSIVE
    clean_audio_batch, transcriptions = grab_batch(batch_sz)
    clean_audio_batch = clean_audio_batch.to(device)

    # Create population of cloned + noised models - NOT COMPUTE INTENSIVE
    population = create_population(model, pop_sz)

    # Run perturbed audio through whisper - COMPUTE INTENSIVE
    all_preds = []
    with pt.inference_mode():
        for child in population:
            perturbed = child(
                clean_audio_batch
            )  # this calls model.forward() which produces a residual
            # Add the residual to the clean audio
            perturbed = clean_audio_batch + perturbed
            preds = whisper_transcribe(perturbed)  # this calls whisper.decode()
            all_preds.append(preds)

    scores = pt.tensor(
        [compute_reward(transcriptions, preds) for preds in all_preds], device=device
    )
    weights = scores_to_weights(scores)
    # Update model weights
    update_model_weights(model, population, weights)

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
        model,
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
