import numpy as np
import torch as pt
import torchaudio
import io
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections.abc import Callable
from audiomentations import AddGaussianNoise
from pathlib import Path
import os

import whisper
import assemblyai as aai
# deepspeech - no longer available
# deepseek - API is doodoo

from ..utilities.data_access import load_data
from ..utilities.wer import wer

NUM_WORKERS = 5
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
    CHECKPOINTS_DIR = str(BASE_DIR / "attacks" / "checkpoints") + "/"
except NameError:
    CHECKPOINTS_DIR = os.path.abspath("../attacks/checkpoints") + "/"

AAI_API_KEY = "<insert_API_key>"

test_waves, test_transcripts = load_data(test=True)
test_waves = test_waves[:10]
test_transcripts = test_transcripts[:10]

# GENERAL FUNCTIONS


def grab_perturbation_model(rel_path: str):
    perturbation_model = pt.load(CHECKPOINTS_DIR + rel_path)
    return perturbation_model


def test_audio_set(audio: pt.Tensor, transcripts: list[str], test_func: Callable[[pt.Tensor, str], float]) -> float:
    assert len(audio) == len(transcripts)
    # Run through tests with progress bar
    progress_bar = tqdm(total=len(test_transcripts))
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        wer_futures = [executor.submit(test_func, audio[idx], transcripts[idx]) for idx in range(len(test_transcripts))]
        for fut in wer_futures:
            fut.add_done_callback(lambda _fut: progress_bar.update(1))
        executor.shutdown(wait=True)
    progress_bar.close()
    # Calculate mean wer
    wers = np.array([fut.result() for fut in wer_futures])
    mean_wer = np.mean(wers)
    return mean_wer

# TARGET MODEL SPECIFIC TESTING FUNCTIONS


def curried_test_one_aai(config: aai.TranscriptionConfig) -> Callable[[pt.Tensor, str], float]:
    def test_one_aai(wave: pt.Tensor, trans: str) -> float:
        dimmed_wave = wave.unsqueeze(0)
        bytes_obj = io.BytesIO()
        torchaudio.save(bytes_obj, dimmed_wave, 16000, format='flac')
        data = bytes_obj.getvalue()
        aai_transcript = aai.Transcriber(config=config).transcribe(data)
        if aai_transcript.status == "error":
            print(f"ERROR ENCOUNTERED WHILE TRANSCRIBING: {aai_transcript.error}")
            return 0
        test_wer = wer(trans, aai_transcript.text)
        return test_wer
    return test_one_aai


def test_aai(perturbation_model: pt.nn.Module, aai_level: str = "nano"):
    # Set API key
    aai.settings.api_key = AAI_API_KEY
    # Grab appropriate model config
    if aai_level == "best":
        config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)
    elif aai_level == "nano":
        config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.nano)
    else:
        raise Exception(f"Invalid aai model ({aai_level}) specified")
    # Get curried test function
    test_one_aai = curried_test_one_aai(config)
    # Test model on each unperturbed audio clip
    mean_wer_unperturbed = test_audio_set(test_waves, test_transcripts, test_one_aai)
    print(f"AAI {aai_level} MODEL UNPERTURBED MEAN_WER: {mean_wer_unperturbed}")
    # Test model on each perturbed audio clip
    with pt.no_grad():
        perturbed_waves = perturbation_model(test_waves)
    mean_wer_perturbed = test_audio_set(perturbed_waves, test_transcripts, test_one_aai)
    print(f"AAI {aai_level} MODEL PERTURBED MEAN_WER: {mean_wer_perturbed}")
    print(f"AAI {aai_level} MODEL WER DEPROVEMENT WITH PERTURBATION: {mean_wer_perturbed - mean_wer_unperturbed}")

# UNTESTED B/C I CAN'T RUN WHISPER


def whisper_transcribe(w_model: whisper.Whisper, audio: pt.Tensor) -> list[str]:
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(w_model.device)
    opts = whisper.DecodingOptions(fp16=False)
    results = whisper.decode(w_model, mel, opts)
    return [result.text for result in results]

# UNTESTED


def test_set_whisper(w_model: whisper.Whisper, audio: pt.Tensor, transcripts: list[str]) -> float:
    whisper_transcripts = whisper_transcribe(w_model, audio)
    wers = np.array(list(map(lambda w_trans, trans: wer(trans, w_trans), whisper_transcripts, transcripts)))
    return np.mean(wers)

# UNTESTED


def test_whisper(perturbation_model: pt.nn.Module, whisper_level: str = "tiny"):
    print("Loading model at level:", whisper_level)
    whisper_model = whisper.load_model(whisper_level)
    print("Model loaded, running test_set_whisper with the test data")
    unperturbed_wer = test_set_whisper(whisper_model, test_waves, test_transcripts)
    print(f"WHISPER {whisper_level} MODEL UNPERTURBED MEAN_WER: {unperturbed_wer}")
    print("Running perturbation model to generate perturbed audio of the test_waves")
    with pt.no_grad():
        perturbed_waves = perturbation_model(test_waves)
    print("Running test_set_whisper with perturbed audio")
    perturbed_wer = test_set_whisper(whisper_model, perturbed_waves, test_transcripts)
    torchaudio.save("perturbed wav 0.wav", pt.reshape(perturbed_waves[0], (1, 80000)), 16000)
    torchaudio.save("clean wav 0.wav", pt.reshape(test_waves[0], (1, 80000)), 16000)
    print(f"WHISPER {whisper_level} MODEL PERTURBED MEAN_WER: {perturbed_wer}")
    print(f"WHISPER {whisper_level} MODEL WER DEPROVEMENT WITH PERTURBATION: {perturbed_wer - unperturbed_wer}")


def test_noisy_whisper(whisper_level: str = "tiny"):
    print("test waves shape", test_waves.shape)
    print("Loading model at level:", whisper_level)
    whisper_model = whisper.load_model(whisper_level)
    print("Model loaded, running test_set_whisper with the test data")
    unperturbed_wer = test_set_whisper(whisper_model, test_waves, test_transcripts)
    print(f"WHISPER {whisper_level} MODEL UNPERTURBED MEAN_WER: {unperturbed_wer}")
    print("Generating noisy test_waves")
    transform = AddGaussianNoise(
        min_amplitude=0.03,
        max_amplitude=0.045,
        p=1.0
    )
    noisy_waves = transform(test_waves, 16000)
    torchaudio.save("noisy wav 0.wav", pt.reshape(noisy_waves[0], (1, 80000)), 16000)
    torchaudio.save("clean wav 0.wav", pt.reshape(test_waves[0], (1, 80000)), 16000)
    print("Running test_set_whisper with noisy audio")
    noisy_wer = test_set_whisper(whisper_model, noisy_waves, test_transcripts)
    print(f"WHISPER {whisper_level} MODEL NOISY MEAN_WER: {noisy_wer}")
    print(f"WHISPER {whisper_level} MODEL WER DEPROVEMENT WITH NOISE: {noisy_wer - unperturbed_wer}")


def test_one_gladia(wave: pt.Tensor, trans: str) -> float:
    pass


if __name__ == "__main__":
    model = grab_perturbation_model("wavperturbation_model.pt")
    test_whisper(model)
    # test_noisy_whisper()
