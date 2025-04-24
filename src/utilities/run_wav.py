import torch
import torchaudio
from ..models.perturbation_model import WavPerturbationModel

# HOME PATH - CHANGE THIS FOR YOUR SYSTEM!!!
HOME_PATH = "/home/spoonmilk/university/csci1470/team-yell"

# MODEL PATH
MODEL_PATH = f"{HOME_PATH}/src/attacks/checkpoints/wavperturbation_model.pt"

model = WavPerturbationModel(
    num_layers=3,
    num_channels=32,
    kernel_size=3,
    max_delta=0.01,
)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

waveform, sample_rate = torchaudio.load(
    "/home/spoonmilk/university/csci1470/team-yell/src/LibriSpeech/dev-clean/1988/24833/1988-24833-0000.flac",
    normalize=True,
)

with torch.no_grad():
    # Pass through model
    output = model(waveform)

torchaudio.save(
    "/home/spoonmilk/university/csci1470/team-yell/src/data/results/output.wav",
    output.cpu(),
    sample_rate,
)
