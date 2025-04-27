import whisper
import torch


def extract_logits(perturbed_audio: torch.Tensorim, model_type: str, model_device: str):
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
