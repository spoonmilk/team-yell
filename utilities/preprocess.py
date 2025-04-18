import torch as pt
#from multiprocessing import Pool, Lock, Process, Manager
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Iterator, Any
from collections.abc import Callable
import re
import random
import torchaudio
#from dataclasses import dataclass

try: #Path to LibriSpeech dev-clean folder
    AUDIO_DIR = str(Path(__file__).resolve().parent.parent) + "/LibriSpeech/dev-clean/" #Will work in normal use cases (running this file directly, importing)
except NameError:
    AUDIO_DIR = os.path.abspath("../LibriSpeech/dev-clean") + "/" #Works for when this file gets run interactively (import by interactive python interpreter)
NUM_WORKERS = 5 #Number of threads you're willing to spawn to parallelize audio file reading

# DATATYPES

class Locked():
    """Struct to make dealing with mutexes easier"""
    def __init__(self, inner: Any):
        self.inner = inner
        self.lock = threading.Lock()

# HELPER FUNCTIONS

def flacfiles(directory: str) -> Iterator[str]:
    """Yields a subpath of the input file path that leads to a flac file within the directory 
    (or a subdirectory of the directory) at the input file path until there are no more such subpaths to yield"""
    for entry in os.scandir(directory):
        if entry.is_dir():
            yield from flacfiles(entry.path)
        elif entry.is_file():
            if not re.search(r"\.flac$", entry.path): continue #Skip over non-flac files
            yield os.path.relpath(entry.path, AUDIO_DIR)
        else: continue #Something's weird with the file/directory

def find_transcription(flac_file: str) -> str:
    """Finds and returns the transcription the flac file found at the input file path"""
    split_path = flac_file.split("/")
    parent_dir = "/".join(split_path[:-1]) + "/"
    file_name = re.sub(r"\.flac$", "", split_path[-1])
    for file in os.listdir(parent_dir):
        if re.search(r"\.trans\.txt$", file): trans_file = parent_dir + file
    assert trans_file, f"Unable to find transcription file for {flac_file}"
    with open(trans_file) as fl:
        while (line := fl.readline()):
            if re.match(file_name, line): return re.sub(r"^[^\s]*\s", "", line).rstrip()
    raise Exception(f"No transcription for {file_name} found in {trans_file}")
        
def extract_data(audio_paths: Locked, read_files: Locked, file_transcriptions: Locked, audio_trans: Callable[[pt.Tensor],pt.Tensor] = None):
    """Pops a random path to an audio file from the input thread-safe list of them, reads (and transforms) its audio data, finds its
    transcription, and then appends these two pieces of data to their respective input thread-safe lists"""
    with audio_paths.lock:
        path = audio_paths.inner.pop(random.randrange(len(audio_paths.inner)))
    file = AUDIO_DIR + path
    audio_data, _ = torchaudio.load(file) #Loads in time series data of the file as a pytorch tensor
    if audio_trans: audio_data = audio_trans(audio_data)
    audio_data = audio_data[0] #Returned tensors from load are shape (1, max_amplitude (80), length (in time)), so just down-dimming them to shape (max_amplitude, length) here
    trans = find_transcription(file).split(" ")
    with read_files.lock, file_transcriptions.lock: #Both locks must be obtained at the same time to ensure proper ordering
        read_files.inner.append(audio_data)
        file_transcriptions.inner.append(trans)

# MAIN

def grab_audio(num_files: int) -> tuple[pt.Tensor, list[list[str]]]:
    """Grabs num_files many random audio files from LibriSpeech, reads (and pads) all of their data into a tensor of shape
     (num_files, max_amplitude (80), max_audio_length), creates an associated list of transcriptions, and returns the two"""
    #Create a list of filepaths to the audio files
    audio_paths = list(flacfiles(AUDIO_DIR))
    assert num_files <= len(audio_paths) #NOTE: there are 2703 files in LibriSpeech, I think
    #Grab num_files random audio files and extract their data as log mel spectrograms
    audio_paths = Locked(audio_paths)
    read_files = Locked([]); file_transcriptions = Locked([])
    melspectroify = torchaudio.transforms.MelSpectrogram(n_mels=80) #Default values are pretty good for speech (would want to change them for music or another kind of audio)
    logify = torchaudio.transforms.AmplitudeToDB(top_db=80) #80 is noted as a "reasonable" cutoff by torchaudio API *shrug*
    logmelspectroify = lambda time_series_audio: logify(melspectroify(time_series_audio))
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for _ in range(num_files): executor.submit(extract_data, audio_paths, read_files, file_transcriptions, logmelspectroify)
        executor.shutdown(wait=True)
    read_files = read_files.inner; file_transcriptions = file_transcriptions.inner
    #Add padding to audio data
    max_len_audio = max(map(lambda seq: seq.shape[1], read_files))
    padded_audio = list(map(lambda seq: pt.cat((seq, pt.zeros((seq.shape[0], max_len_audio - seq.shape[1]))), dim=1), read_files))
    #Tensorify the audio and return
    audio_tensor = pt.stack(padded_audio)
    return (audio_tensor, file_transcriptions)

# Quick demo
if __name__ == "__main__":
    audio_data, transcriptions  = grab_audio(100)
    print(f"AUDIO DATA SHAPE: {audio_data.shape}")
    print("FIRST ENTRY AUDIO:\n", audio_data[0])
    print("FIRST ENTRY TRANSCRIPTION:\n", transcriptions[0])