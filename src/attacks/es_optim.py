import torch as pt
import numpy as np
from torch import nn
import whisper
from ..models.perturbation_model import WavPerturbationModel
from ..utilities.wer import wer
from ..utilities.preprocess_wav import load_data
import re
from concurrent.futures import ThreadPoolExecutor
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
    # Create population of cloned + noised models - NOT COMPUTE INTENSIVE
    population = []
    for _ in range(pop_sz):
        copy = WavPerturbationModel(*model.options)
        copy.load_state_dict(model.state_dict())
        noise_params(copy)
        population.append(copy)
    # Grab batch of audio + transcriptions - NOT COMPUTE INTENSIVE
    audio, transcriptions = grab_batch(batch_sz)
    # Get noised output from models of batch - MILDLY COMPUTE INTENSIVE
    perturbed_audio = map(lambda pop_mem: pop_mem(audio), population)
    # Run perturbed audio through whisper - COMPUTE INTENSIVE
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        perturbed_transes = list(executor.map(whisper_transcribe, perturbed_audio))
        executor.shutdown(wait=True)
    # Use WER to see how well each model did - NOT COMPUTE INTENSIVE
    assert len(transcriptions) == len(perturbed_transes)
    induced_wers = [
        [wer(actual_trans, perturbed_trans) for perturbed_trans in perturbed_transes]
        for actual_trans in transcriptions
    ]
    induced_wers = pt.tensor(induced_wers)
    induced_wer_per_model = pt.sum(induced_wers, 1)
    # Use WER results to weight each model - NOT COMPUTE INTENSIVE
    model_weightings = induced_wer_per_model / pt.sum(induced_wer_per_model)
    # Construct and new model from weighted sum of each of the created models - NOT COMPUTE EXPENSIVE
    with pt.no_grad():
        for param_idx in range(len(model.parameters)):
            new_param = pt.sum(
                model_weightings
                * pt.tensor([pop_mem.parameters[param_idx] for pop_mem in population])
            )
            model.parameters[param_idx].data = new_param


def reward_fn(perturbed, clean, transcripts):
    """
    Reward function for evolutionary strategies adversarial training.
    """
    # Let whisper transcribe the perturbed audio


def compute_reward(clean_transcription, perturbed_transcription):
    """
    Compute the reward for the perturbed transcription.

    Returns the 'fitness' of the perturbed transcription.
    The fitness is the average WER between the perturbed and clean transcriptions.
    We want WER to be as large as possible
    """
    wers = []
    for gt, pr in zip(perturbed_transcription, clean_transcription):
        w = wer(gt, pr)
        w_clipped = min(w, 1.0)
        wers.append(w_clipped)
    return sum(wers) / len(wers)
