import os
from pathlib import Path

import numpy as np
import torch as pt
import whisper
from torch import nn
from pystoi import stoi
import torchaudio

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
# - Clean functions

# TRAINING HYPERPARAMETERS
POP_SIZE = 50
BATCH_SIZE = 10
LEARNING_RATE = 1
NOISE_MEAN = 0
NOISE_STD_DEV_RNG_PORTION = 0.05
MODEL_TYPE = "tiny"
NUM_WORKERS = 5
NUM_EPOCHS = 10
PERFORMANCE_CUTOFF = 0.1
SCALE_FACTOR = 0.5

# MODEL PARAMETERS
NUM_LAYERS = 3
NUM_CHANNELS = 32
KERNEL_SIZE = 3
MAX_DELTA = 0.05

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

mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=512, hop_length=128, n_mels=64
)

mel_weights = pt.tensor(
    [1.0 if 300 <= (i / 64) * 8000 <= 3000 else 0.5 for i in range(64)]
).view(1, 64, 1)  # shape (1, F, 1)


def pad_to_length(sig: np.ndarray, min_len: int) -> np.ndarray:
    if sig.ndim != 1:
        sig = sig.flatten()
    if sig.shape[0] < min_len:
        pad_amt = min_len - sig.shape[0]
        return np.pad(sig, (0, pad_amt), mode="constant")
    return sig


# NOT CHECKED
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


# CHECKED
def noise_params(model: nn.Module, epoch: int = 0):
    """
    Adds noise to the model parameters based on the current epoch.
    Noise is scaled by the mutation strength, which decreases over time.
    """
    device = next(model.parameters()).device
    with pt.no_grad():
        strength = mutation_strength(epoch, NUM_EPOCHS)
        for param in model.parameters():
            param.data.add_(pt.randn_like(param, device=device) * strength)


# LOGIT LOSS FUNCTIONS
def extract_logits(perturbed_audio: pt.Tensor):
    """Takes in a perturbed audio tensor and returns logits from Whisper"""
    tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=True, task="transcribe", language="en"
    )
    tokens = pt.tensor([[tokenizer.sot]], device=whisper_model.device)

    mels = []
    for x in perturbed_audio:
        audio = x.squeeze().cpu()
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels)
        mels.append(mel)
    mel_batch = pt.stack(mels, dim=0).to(device)
    tokens = pt.full(
        (mel_batch.size(0), 1), tokenizer.sot, dtype=pt.long, device=device
    )
    with pt.inference_mode():
        return whisper_model(mel_batch, tokens)


def logit_entropy(logits: pt.Tensor):
    """Returns the entropy of produced Whisper logits"""
    # Get probabilities from logits
    probs = pt.nn.functional.softmax(logits, dim=-1)
    log_probs = pt.nn.functional.log_softmax(logits, dim=-1)
    # Take entropy across
    entropy = (probs * log_probs).sum(dim=-1)
    entropy = -entropy.mean()
    return entropy


def scheduled_rewards(epoch: int, total_epochs: int) -> tuple[float, float, float]:
    """
    Returns the scheduled rewards for the current epoch
    The rewards are scaled by the mutation strength, which decreases over time.
    """
    t = epoch / total_epochs
    adversarial_bonus = max((2.0 * (1 - t) + 0.1 * t), 1.0)
    distortion_penalty = min((0.1 * (1 - t) + 1.2 * t), 1.0)
    interpretability_incentive = 0.1 * (1 - t) + 1.0 * t
    return adversarial_bonus, distortion_penalty, interpretability_incentive


def phase_deviation(
    clean: pt.Tensor, pert: pt.Tensor, n_fft=512, hop_length=128
) -> pt.Tensor:
    """
    Returns a scalar: the mean |sin((delta)phase)| across time–frequency bins.
    """
    # shape: (batch, time) → (batch, freq, frames)
    clean_spec = pt.stft(clean, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    pert_spec = pt.stft(pert, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    # extract angles
    phase_diff = pt.angle(pert_spec) - pt.angle(clean_spec)
    return pt.abs(pt.sin(phase_diff)).mean()


def compute_logit_reward(
    clean_audio: pt.Tensor,
    perturbed_audio: pt.Tensor,
    logits: pt.Tensor,
    sched_rewards: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """
    Computes a multi-factor reward for the perturbed audio.
    Logit entropy is our base reward, and we penalize the distortion of the audio, which is mostly in low frequencies.
    High frequencies are incentivized as they help with intelligibility.

    Args:
        clean_audio (pt.Tensor): The clean audio tensor.
        perturbed_audio (pt.Tensor): The perturbed audio tensor.
        logits (pt.Tensor): The logits from the Whisper model.
        adversarial_bonus (float): The weight for the logit entropy reward.
        distortion_penalty (float): The weight for the distortion penalty.
        interpretability_incentive (float): The weight for the high frequency incentive.
    """
    entropy = logit_entropy(logits)
    # Take mean squared change between clean and perturbed audio
    clean_mel = mel_spec(clean_audio)  # (1, F, N)
    pert_mel = mel_spec(perturbed_audio)  # (1, F, N)
    mel_mse = ((pert_mel - clean_mel) ** 2).mean()

    # Compute phase deviation
    phase_dev = phase_deviation(clean_audio, perturbed_audio)

    clean_np = clean_audio.squeeze().cpu().numpy()
    pert_np = perturbed_audio.squeeze().cpu().numpy()

    # Pad to 16000 samples for STOI
    clean_np = pad_to_length(clean_np, 16000)
    pert_np = pad_to_length(pert_np, 16000)

    s = stoi(clean_np, pert_np, fs_sig=16000, extended=False)

    # Normalize s to within similar bounds as the other rewards
    s = (s - 0.3) / (1.0 - 0.3)  # Normalize to [0, 1]

    # Normalize mse to within [0, 1]
    mel_mse = min((mel_mse) / (50.0), 1.0)  # Normalize to [0, 1]
    adversarial_bonus, distortion_penalty, interpretability_incentive = sched_rewards

    print(
        f"Entropy: {entropy:.4f}, "
        f"Mel MSE: {mel_mse:.4f}, "
        f"STOI: {s:.4f}, "
        f"Phase deviation: {phase_dev:.4f}, "
        f"Adversarial bonus: {adversarial_bonus:.4f}, "
        f"Distortion penalty: {distortion_penalty:.4f}, "
        f"Interpretability incentive: {interpretability_incentive:.4f}\n"
    )

    return (
        adversarial_bonus * entropy
        - distortion_penalty * mel_mse
        + phase_dev
        + interpretability_incentive * s
    )


# CHECKED - NOTE: very heavily penalizes missed words (all words afterwards are considered incorrect)
def compute_wer_reward(
    clean_transcription: list[str], perturbed_transcription: list[str]
) -> list[float]:
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


# CHECKED
def create_population(
    model: WavPerturbationModel, pop_sz: int, epoch: int = 0
) -> list[WavPerturbationModel]:
    """Creates a population of perturbed models"""
    population = []
    for _ in range(pop_sz):
        copy = WavPerturbationModel(*model.options)
        copy.load_state_dict(model.state_dict())
        noise_params(copy, epoch=epoch)
        population.append(copy)
    return population


# CHECKED
def update_model_weights(
    model: WavPerturbationModel,
    population: list[WavPerturbationModel],
    weights: pt.Tensor,
):
    """Update the model weights based on the population and their scores"""
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
            parent_p.data.add_(LEARNING_RATE * step)


def scores_to_weights(
    scores: pt.Tensor, cutoff: float = PERFORMANCE_CUTOFF, scale_factor: float = 0.1
) -> pt.Tensor:
    """
    Returns the weights for the population based on their scores
    Scores below the cutoff are set to -inf to filter them out
    Scale factor helps with softmax stability, tunable

    Args:
        scores (pt.Tensor): The scores for the population
        cutoff (float): The cutoff for the scores
        scale_factor (float): The scale factor for the softmax
    """
    sorted_scores = pt.sort(scores, descending=True)[0]
    k = max(1, int(cutoff * scores.shape[0]))
    top_scores = sorted_scores[:k].cpu().numpy()

    # Filter out low performers
    scores_np = scores.cpu().numpy()
    filter_fn = np.vectorize(lambda s: s if (s in top_scores) else float("-inf"))
    filtered_np = filter_fn(scores_np)

    # Convert back to tensor
    filtered = pt.tensor(filtered_np, device=scores.device)

    # Softmax to get weights
    return pt.softmax(filtered / scale_factor, dim=0)


def mutation_strength(epoch, total_epochs, sig_o=0.5, sig_t=0.01):
    """
    Returns the mutation strength for the current epoch

    See simulated annealing. Basic idea is to start with high mutations
    to get the search moving, then reduce as the model trains to converge.

    Args:
        epoch (int): The current epoch
        total_epochs (int): The total number of epochs
        sig_o (float): The initial mutation strength
        sig_t (float): The final mutation strength
    """
    t = epoch / total_epochs
    return sig_o * (1 - t) + sig_t * t


def epoch(
    model: WavPerturbationModel,
    pop_sz: int = POP_SIZE,
    batch_sz: int = BATCH_SIZE,
    train_type: str = "transcript",
    epoch: int = 0,
):
    """
    Run a single epoch of training

    Args:
        model (WavPerturbationModel): The model to train
        pop_sz (int): The size of the population
        batch_sz (int): The size of the batch
        train_type (str): The type of training to perform ; logit or transcript
                          transcript = train using WER as reward
                          logit = train using logit entropy as reward
                          generally, logit is better
        epoch (int): The current epoch
    """
    device = next(model.parameters()).device
    # Grab batch of audio + transcriptions - NOT COMPUTE INTENSIVE
    clean_audio_batch, transcriptions = grab_batch(batch_sz)
    clean_audio_batch = clean_audio_batch.to(device)

    # Create population of cloned + noised models - NOT COMPUTE INTENSIVE
    population = create_population(model, pop_sz, epoch=epoch)

    perturbed_list = []
    with pt.inference_mode():
        for child in population:
            delta = child(clean_audio_batch)
            perturbed_list.append(clean_audio_batch + delta)

    # 2) compute scores via the chosen reward
    if train_type == "transcript":
        preds = [whisper_transcribe(x) for x in perturbed_list]
        scores = pt.tensor(
            [compute_wer_reward(transcriptions, p) for p in preds], device=device
        )
    elif train_type == "logit":
        logits = [extract_logits(x) for x in perturbed_list]
        sched_rewards = scheduled_rewards(epoch, NUM_EPOCHS)
        scores = pt.tensor(
            [
                compute_logit_reward(
                    clean_audio_batch,
                    perturbed_list[i],
                    logits[i],
                    sched_rewards=sched_rewards,
                )
                for i in range(len(perturbed_list))
            ],
            device=device,
        )
        print(
            f"Scores  →  min={scores.min().item():.4f}, "
            f"max={scores.max().item():.4f}, "
            f"mean={scores.mean().item():.4f}, "
            f"max-min={scores.max().item() - scores.min().item():.4f}\n"
        )

    else:
        raise ValueError("Invalid training type. Choose transcript or logit.")

    weights = scores_to_weights(scores)
    # Update model weights
    update_model_weights(model, population, weights)

    if train_type == "logit":
        print(f"Mean score: {scores.mean().cpu()}")
    else:
        print(f"Avg WER: {scores.mean().cpu()}")

    return float(scores.mean().cpu())


def train_es(
    model: WavPerturbationModel,
    epochs: int = NUM_EPOCHS,
    type: str = "transcript",
):
    """Train the model using evolutionary strategies"""

    print(f"Starting ES training on device={device}")
    for i in range(1, epochs + 1):
        epoch(model, POP_SIZE, BATCH_SIZE, type, epoch=i)
        # print(f"Epoch {i}/{epochs} — avg WER: {avg_wer:.4f}")
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
    train_es(attack_model, NUM_EPOCHS, type="logit")
