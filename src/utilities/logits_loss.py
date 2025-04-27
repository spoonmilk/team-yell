import whisper
import torch
from torch import functional as F


def extract_logits(perturbed_audio: torch.Tensorim, model_type: str, model_device: str):
    """Takes in a perturbed audio and ground truth transcription, outputting the logits of Whisper's forward pass"""
    # Whisper and whisper tokenizer initialization
    w_model = whisper.load_model(model_type)
    w_model.to(model_device)

    tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=True, task="transcribe", language="en"
    )
    tokens = torch.tensor([[tokenizer.sot]], device=w_model.device)

    # Reset whisper model gradients
    w_model.zero_grad()
    # Get whisper transcription logits
    sized_audio = whisper.pad_or_trim(perturbed_audio)
    mel = whisper.log_mel_spectrogram(sized_audio, n_mels=w_model.dims.n_mels).to(
        w_model.device
    )
    whisper_logits = w_model.forward(mel, tokens)
    return whisper_logits

def logit_entropy(logits: torch.Tensor):
    """Returns the entropy of produced Whisper logits"""
    # Get probabilities from logits
    log_probs = F.softmax(logits)
    # Take entropy across
    entropy = (-(log_probs * torch.exp(log_probs)).sum(dim=-1)).mean()
    return entropy