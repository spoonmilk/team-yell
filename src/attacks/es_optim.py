import torch as pt
from torch import nn
import whisper


def reward_fn(perturbed, clean, transcripts):
    """
    Reward function for evolutionary strategies adversarial training.
    """

    # Let whisper transcribe the perturbed audio
