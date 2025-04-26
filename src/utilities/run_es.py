import torch
import torchaudio
import whisper
from .wer import wer
from ..attacks.es_optim import train_es
from ..models.perturbation_model import WavPerturbationModel

# PATHS/UTILS

# HOME PATH - CHANGE THIS FOR YOUR SYSTEM!!!
HOME_PATH = "/home/spoonmilk/university/csci1470/team-yell"

# MODEL PATH
MODEL_PATH = f"{HOME_PATH}/src/attacks/checkpoints/wavperturbation_model.pt"

# MODEL PARAMETERS
NUM_LAYERS = 3
NUM_CHANNELS = 32
KERNEL_SIZE = 3
MAX_DELTA = 0.01

# TRAINING PARAMETERS
NUM_EPOCHS = 1

# TORCHECK CODE

attack_model = WavPerturbationModel(
        kernel_size=KERNEL_SIZE,
        num_channels=NUM_CHANNELS,
        num_layers=NUM_LAYERS,
        max_delta=MAX_DELTA,
    )

for param in attack_model.parameters():
    print(param)

train_es(attack_model, NUM_EPOCHS)

model = torch.load(MODEL_PATH, weights_only=False)
model.eval()

for param in model.parameters():
    print(param)

waveform, sample_rate = torchaudio.load(
    f"{HOME_PATH}/src/LibriSpeech/dev-clean/1988/24833/1988-24833-0000.flac"
)

with torch.no_grad():
    waveform = waveform.unsqueeze(0)

    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)
    residual = model(waveform)
    perturbed = waveform.squeeze(1) + residual

torchaudio.save(
    "/home/spoonmilk/university/csci1470/team-yell/src/data/results/output.wav",
    perturbed.cpu(),
    sample_rate,
)

whisper_model = whisper.load_model("tiny")
audio = perturbed.squeeze().cpu()
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
if mel.ndim == 3:
    mel = mel[0]
opts = whisper.DecodingOptions()
result = whisper.decode(whisper_model, mel, opts)
print(result.text)

wer = wer("The two stray kittens gradually make themselves at home", result.text)
print("WER:", wer)
