import os
from pathlib import Path

import numpy as np
import torch as pt
import whisper
from torch import nn

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
NUM_EPOCHS = 50
PERFORMANCE_CUTOFF = 0.1
SCALE_FACTOR = 0.5

# MODEL PARAMETERS
NUM_LAYERS = 3
NUM_CHANNELS = 32
KERNEL_SIZE = 3
MAX_DELTA = 0.04

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


def compute_logit_reward(
    clean_audio: pt.Tensor,
    perturbed_audio: pt.Tensor,
    logits: pt.Tensor,
    adversarial_bonus: float = 1.0,
    distortion_penalty: float = 0.8,
    hf_incentive: float = 1.0,
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
        hf_incentive (float): The weight for the high frequency incentive.

    Returns:
        tuple: (reward_value, metrics_dict) where metrics_dict contains all computed metrics
    """

    # COMPONENT 1: Logit entropy
    entropy = logit_entropy(logits)

    # COMPONENT 2: Distortion penalty
    # Calculate the mean squared error between the clean and perturbed audio
    delta = pt.mean(pt.square(perturbed_audio - clean_audio))

    # COMPONENT 3: High frequency incentive
    # Calculate the STFT of the perturbed audio
    # We use the STFT to get the magnitude of the audio signal
    spec = pt.stft(
        perturbed_audio.squeeze(1), n_fft=256, hop_length=128, return_complex=True
    )
    mag = spec.abs()

    # High frequency cutoff
    # Tune for intelligibility
    freq_high = 0.95
    cutoff_idx = int(freq_high * mag.size(1))

    # We incentivize high frequency energy and penalize low frequency energy
    low_eng = mag[:, : (mag.size(1) // 2)].mean()
    high_eng = mag[:, cutoff_idx:].mean()
    hf_ratio = (high_eng + 1e-8) / (low_eng + 1e-8)

    # Calculate discounted delta and hf bonus
    discounted_delta = delta * distortion_penalty
    hf_bonus = hf_ratio * hf_incentive

    # Calculate final reward
    reward = adversarial_bonus * entropy - discounted_delta + hf_bonus

    # Return both the reward and a dictionary of metrics
    metrics = {
        "entropy": entropy.item(),
        "delta": delta.item(),
        "discounted_delta": discounted_delta.item(),
        "hf_ratio": hf_ratio.item(),
        "hf_bonus": hf_bonus.item(),
    }

    return reward, metrics


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

        # Updated to collect rewards and metrics
        rewards_and_metrics = [
            compute_logit_reward(clean_audio_batch, perturbed_list[i], logits[i])
            for i in range(len(perturbed_list))
        ]

        # Unpack rewards and metrics
        scores = pt.tensor([r[0] for r in rewards_and_metrics], device=device)
        all_metrics = [m[1] for m in rewards_and_metrics]

        # Calculate mean metrics
        mean_metrics = {
            k: sum(metric[k] for metric in all_metrics) / len(all_metrics)
            for k in all_metrics[0]
        }

        # Print mean metrics for the epoch
        print(f"\n--- Epoch {epoch} Mean Metrics ---")
        print(f"Mean Entropy: {mean_metrics['entropy']:.4f}")
        print(f"Mean Delta: {mean_metrics['delta']:.4f}")
        print(f"Mean Discounted Delta: {mean_metrics['discounted_delta']:.4f}")
        print(f"Mean HF Ratio: {mean_metrics['hf_ratio']:.4f}")
        print(f"Mean HF Bonus: {mean_metrics['hf_bonus']:.4f}")
        print("---------------------------\n")

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
