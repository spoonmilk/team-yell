import torch as pt
from torch import nn
import whisper
from ..models.perturbation_model import WavPerturbationModel
from ..utilities.wer import wer
from ..utilities.preprocess_wav import load_data
from concurrent.futures import ProcessPoolExecutor
import random

POP_SIZE = 50
BATCH_SIZE = 10
LEARNING_RATE = 0.01
NOISE_MEAN = 0
NOISE_STD_DEV_RNG_PORTION = 0.05
MODEL_TYPE = "tiny"
NUM_WORKERS = 5

whisper_model = whisper.load_model(MODEL_TYPE)
waves, transcripts = load_data()
assert len(waves) == len(transcripts)


def whisper_transcribe(audio_data: pt.Tensor) -> list[str]:
    """Transcribes all audio sequences encapsulated within an input tensor and returns whisper's transcriptions of them"""
    sized_data = whisper.pad_or_trim(audio_data)
    log_mel_data = whisper.log_mel_spectrogram(
        sized_data, n_mels=whisper_model.dims.n_mels
    ).to(whisper_model.device)
    results = whisper.decode(whisper_model, log_mel_data, whisper.DecodingOptions())
    return results


def grab_batch(batch_sz: int) -> tuple[pt.Tensor, list[str]]:
    """Size of batch -> list of audio tensors correlated with list of transcriptions"""
    indices = random.sample(range(len(transcripts)), batch_sz)
    batch_waves = pt.gather(waves, 0, indices)
    batch_trans = [transcripts[idx] for idx in indices]
    return batch_waves, batch_trans


def noise_params(model: nn.Module):
    device = next(model.parameters()).device
    with pt.no_grad():
        for param in model.parameters():
            data = param.data
            span = (data.max() - data.min()).clamp_min(0.0)
            std_dev = NOISE_STD_DEV_RNG_PORTION * span
            noise = pt.randn_like(data, device=device) * std_dev
            data.add_(noise)


def epoch(
    model: WavPerturbationModel,
    pop_sz: int = POP_SIZE,
    batch_sz: int = BATCH_SIZE,
    # learning_rate: float = LEARNING_RATE,
):
    device = next(model.parameters()).device

    # Grab batch of audio + transcriptions - NOT COMPUTE INTENSIVE
    clean_audio_batch, transcriptions = grab_batch(batch_sz)
    clean_audio_batch = clean_audio_batch.to(device)

    # Run clean audio through whisper - COMPUTE INTENSIVE
    clean_whisper_preds = []
    for audio in clean_audio_batch:
        whisper_transcription = whisper_transcribe(audio)
        clean_whisper_preds.append(whisper_transcription)

    # Create population of cloned + noised models - NOT COMPUTE INTENSIVE
    population = []
    for _ in range(pop_sz):
        copy = WavPerturbationModel(*model.options)
        copy.load_state_dict(model.state_dict())
        noise_params(copy)
        population.append(copy)

    # Run perturbed audio through whisper - COMPUTE INTENSIVE
    def worker(pop_model: WavPerturbationModel):
        pert = pop_model(clean_audio_batch)
        return whisper_transcribe(pert)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as exe:
        all_perturb_preds = list(exe.map(worker, population))

    # Use WER to see how well each model did - NOT COMPUTE INTENSIVE
    scores = pt.tensor(
        [
            compute_reward(clean_whisper_preds, perturbed_preds)
            for perturbed_preds in all_perturb_preds
        ],
        device=device,
    )
    # Compute fitness of each model - NOT COMPUTE INTENSIVE
    fitness = pt.exp(scores).clone()
    model_weightings = fitness / pt.sum(fitness)

    # Update model weights
    with pt.no_grad():
        params = list(model.parameters())
        for idx, p in enumerate(params):
            child_params = pt.stack(
                [list(pop.parameters())[idx].data for pop in population], dim=0
            )
            diffs = child_params - p.data.unsqueeze(0)
            # broadcast weights
            w = model_weightings.view(-1, *([1] * (diffs.dim() - 1)))
            step = (w * diffs).sum(0)
            p.data.add_(LEARNING_RATE * step)


def compute_reward(clean_transcription, perturbed_transcription):
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
