import whisper
import torch as pt
import torchaudio
import re
from functools import reduce
from concurrent.futures import ThreadPoolExecutor
#import ..utilities.preprocess

MODEL_TYPE = "small"
NUM_WORKERS = 5
model = whisper.load_model(MODEL_TYPE)

def whisper_transcribe(audio_data: pt.Tensor) -> list[list[str]]:
    """Transcribes all audio sequences encapsulated within an input tensor and returns whisper's transcriptions of them"""
    sized_data = whisper.pad_or_trim(audio_data)
    log_mel_data = whisper.log_mel_spectrogram(sized_data, n_mels=model.dims.n_mels).to(model.device)
    results = whisper.decode(model, log_mel_data, whisper.DecodingOptions())
    transcriptions = list(map(lambda res: re.sub(r"[^A-Za-z\s]", "", res.text).upper().split(" "), results))
    return transcriptions

def string_error(str_list1: list[str], str_list2: list[str]) -> int:
    """Sums the number of different words across the two input lists of strings"""
    if len(str_list1) != len(str_list2):
        padding = ["" for _ in range(abs(len(str_list1) - len(str_list2)))]
        if len(str_list1) < len(str_list2): str_list1 += padding
        else: str_list2 += padding
    return reduce(lambda cum, idx: cum + 1 if str_list1[idx] != str_list2[idx] else cum, range(len(str_list1)), 0)

def transcription_MSE(og_audio: pt.Tensor, noise: pt.Tensor, labels: list[str]) -> float:
    """UNTESTED Calculates the MSE of whisper when the input audio tensor has noise applied to it"""
    altered_audio = og_audio + noise
    transcriptions = whisper_transcribe(altered_audio)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        error = pt.tensor(executor.map(string_error, labels, transcriptions))
        executor.shutdown(wait=True)
    return error**2/error.shape[0]

# if __name__ == "__main__":
#     audio, _ = torchaudio.load("/Users/zarquon/CS1470/team-yell/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac")
#     print(whisper_transcribe(audio))