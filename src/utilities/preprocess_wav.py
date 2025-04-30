import torch
import torch.nn.functional as F
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Iterator, Any, Tuple, List
import random
import torchaudio
import pickle
from .data_access import save_data

# Path to LibriSpeech dev-clean folder (adjust as needed)
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
    AUDIO_DIR = str(BASE_DIR / "LibriSpeech" / "dev-clean") + "/"
except NameError:
    AUDIO_DIR = os.path.abspath("../LibriSpeech/dev-clean/") + "/"

# Number of threads for parallel I/O
NUM_WORKERS = 5

# Desired sample rate and max length (in samples)
TARGET_SR = 16000
MAX_LEN_S = 5  # seconds
MAX_LEN = TARGET_SR * MAX_LEN_S
PREEMPHASIS = 0.97

# Test to training ratio
TEST_PORTION = 0.1

class Locked:
    def __init__(self, inner: Any):
        self.inner = inner
        self.lock = threading.Lock()


def flacfiles(directory: str) -> Iterator[str]:
    """
    Recursively yield relative paths to .flac files under `directory`.

    Args:
        directory (str): Base directory to search.

    Yields:
        str: Relative path (to AUDIO_DIR) of each .flac file.
    """
    for entry in os.scandir(directory):
        if entry.is_dir():
            yield from flacfiles(entry.path)
        elif entry.is_file() and entry.path.endswith(".flac"):
            yield os.path.relpath(entry.path, AUDIO_DIR)


def find_transcription(flac_path: str) -> str:
    """
    Find and return the transcript for a given .flac file path.

    Args:
        flac_path (str): Relative path (to AUDIO_DIR) of the .flac file.

    Returns:
        str: Sentence transcript.
    """
    parts = flac_path.split(os.sep)
    parent = os.path.join(AUDIO_DIR, *parts[:-1])
    base = parts[-1].replace(".flac", "")
    trans_file = None
    for f in os.listdir(parent):
        if f.endswith(".trans.txt"):
            trans_file = os.path.join(parent, f)
            break
    if trans_file is None:
        raise FileNotFoundError(f"No transcription file in {parent}")
    with open(trans_file, "r") as fp:
        for line in fp:
            if line.startswith(base):
                return line.split(maxsplit=1)[1].strip()
    raise RuntimeError(f"No transcription for {base} in {trans_file}")

def clip_wave(wav: torch.Tensor) -> torch.Tensor:
    L = wav.size(1)
    if L < MAX_LEN:
        wav = F.pad(wav, (0, MAX_LEN - L))
    else:
        wav = wav[:, :MAX_LEN]
    return wav

def pad_to_max(wavs: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(map(lambda tens: tens.shape[-1], wavs))
    padded_wavs = list(map(lambda tens: torch.cat((tens.squeeze(0), torch.zeros((max_len - tens.shape[1],))), dim=0), wavs))
    return torch.stack(padded_wavs, dim=0)

def preprocess_wave(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """
    Resample, mono mix, pre-emphasize, pad/trim, and normalize raw audio.

    Args:
        wav (torch.Tensor): Input waveform of shape (channels, samples).
        sr (int): Original sample rate.

    Returns:
        torch.Tensor: Processed waveform of shape (1, MAX_LEN),
                      values in [-1,1].
    """
    # Resample
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
    # Mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Pre-emphasis
    wav = torch.cat([wav[:, :1], wav[:, 1:] - PREEMPHASIS * wav[:, :-1]], dim=1)
    # Pad/trim
    # if clip:
    #     L = wav.size(1)
    #     if L < MAX_LEN:
    #         wav = F.pad(wav, (0, MAX_LEN - L))
    #     else:
    #         wav = wav[:, :MAX_LEN]
    # Normalize to [-1,1]
    max_val = wav.abs().max().clamp(min=1e-4)
    wav = wav / max_val
    return wav


def extract_data(audio_paths: Locked, wave_list: Locked, trans_list: Locked) -> None:
    """
    Thread-safe extraction: pop a random path, load and preprocess audio,
    find its transcription, and append both to shared lists.

    Args:
        audio_paths (Locked): Locked list of relative .flac paths.
        wave_list (Locked): Locked list to append processed waveforms.
        trans_list (Locked): Locked list to append token lists.

    Returns:
        None
    """
    with audio_paths.lock:
        rel_path = audio_paths.inner.pop(random.randrange(len(audio_paths.inner)))
    full = AUDIO_DIR + rel_path
    wav, sr = torchaudio.load(full)  # (channels, samples)
    wav_proc = preprocess_wave(wav, sr)  # (1, MAX_LEN)
    transcript = find_transcription(rel_path)
    with wave_list.lock, trans_list.lock:
        wave_list.inner.append(wav_proc)
        trans_list.inner.append(transcript)


def grab_waveforms(num_files: int) -> Tuple[torch.Tensor, list[str]]:
    """
    Load, preprocess, and batch a set of random audio files for Whisper.

    Args:
        num_files (int): Number of random examples to load.

    Returns:
        Tuple:
            - torch.Tensor: Batch of waveforms, shape (B, 1, MAX_LEN)
            - List[List[str]]: Corresponding transcripts, length B
    """
    paths = list(flacfiles(AUDIO_DIR))
    assert num_files <= len(paths), "Request exceeds available files"
    audio_paths = Locked(paths.copy())
    wave_list = Locked([])
    trans_list = Locked([])

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for _ in range(num_files):
            executor.submit(extract_data, audio_paths, wave_list, trans_list)
        executor.shutdown(wait=True)

    waves = wave_list.inner  # list of (1, MAX_LEN)
    trans = trans_list.inner  # list of token lists
    waves = list(map(clip_wave, waves))
    batch = torch.stack(waves, dim=0)  # (B, 1, MAX_LEN)
    return batch, trans

def grab_all_waveforms() -> Tuple[torch.Tensor, list[str], torch.Tensor, list[str]]:
    paths = list(flacfiles(AUDIO_DIR))
    audio_paths = Locked(paths.copy())
    wave_list = Locked([])
    trans_list = Locked([])

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for _ in range(2703): #2703 is the number of files in the Librispeech
            executor.submit(extract_data, audio_paths, wave_list, trans_list)
        executor.shutdown(wait=True)

    waves = wave_list.inner # list of (1, MAX_LEN)
    trans = trans_list.inner  # list of token lists
    break_idx = int(len(waves)*(1-TEST_PORTION))
    train_waves = torch.squeeze(torch.stack(list(map(clip_wave, waves[:break_idx])), dim=0))
    test_waves = torch.squeeze(pad_to_max(waves[break_idx:]))
    train_trans = trans[:break_idx]
    test_trans = trans[break_idx:]
    return (train_waves, train_trans, test_waves, test_trans)

# Demo
if __name__ == "__main__":
    # batch, transcripts = grab_waveforms(8)
    # print("Batch shape:", batch.shape)  # e.g., (8, 1, 80000)
    # print("First transcript:", transcripts[0])
    train_waves, train_trans, test_waves, test_trans = grab_all_waveforms()
    save_data(train_waves, train_trans, test_waves, test_trans)



