import numpy as np
import torch as pt
import torchaudio
import io
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections.abc import Callable
from typing import Any
from pathlib import Path
import os
import requests
import time

import whisper
import assemblyai as aai
from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError


from ..utilities.data_access import load_data
from ..utilities.wer import wer

NUM_WORKERS = 5
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
    CHECKPOINTS_DIR = str(BASE_DIR / "attacks" / "checkpoints") + "/"
    TEMP_DIR = str(Path(__file__).resolve().parent / "temp") + "/"
except NameError:
    CHECKPOINTS_DIR = os.path.abspath("../attacks/checkpoints") + "/"
    TEMP_DIR = os.path.abspath("../temp") + "/"

AAI_API_KEY = "82d2691ae7d340c895678ab5c284e616"
GLADIA_API_KEY = "44a5db2a-f937-45ab-9211-18f063870ef6"
SPEECHMATICS_API_KEY = "gF8yNtvnU3ovRr6XIXoaLhWibuwHrr70"

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

def try_until(task: Callable[[Any], Any], args: tuple, err_msg: str) -> Any: #Typing left somewhat ambiguous because task return type is so variable
    while True:
        try: task_result = task(*args)
        except Exception as e:
            print(err_msg + ":", e)
            time.sleep(1)
        else: break
    return task_result

# ASSEMBLY AI TEST FUNCTIONS

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

# WHISPER TEST FUNCTIONS

def whisper_transcribe(w_model: whisper.Whisper, audio: pt.Tensor) -> list[str]: 
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(w_model.device)
    opts = whisper.DecodingOptions(fp16=False)
    results = whisper.decode(w_model, mel, opts)
    return [result.text for result in results]

def test_set_whisper(w_model: whisper.Whisper, audio: pt.Tensor, transcripts: list[str]) -> float:
    whisper_transcripts = whisper_transcribe(w_model, audio)
    wers = np.array(list(map(lambda w_trans, trans: wer(trans, w_trans), whisper_transcripts, transcripts)))
    return np.mean(wers)

def test_whisper(perturbation_model: pt.nn.Module, whisper_level: str = "tiny"):
    print("Loading model at level:", whisper_level)
    whisper_model = whisper.load_model(whisper_level)
    print("Mode loaded, running test_set_whisper with the test data")
    unperturbed_wer = test_set_whisper(whisper_model, test_waves, test_transcripts)
    print(f"WHISPER {whisper_level} MODEL UNPERTURBED MEAN_WER: {unperturbed_wer}")
    print("Running perturbation model to generate perturbed audio of the test_waves")
    with pt.no_grad():
        perturbed_waves = perturbation_model(test_waves)
    print("Running test_set_whisper with perturbed audio")
    perturbed_wer = test_set_whisper(whisper_model, perturbed_waves, test_transcripts)
    print(f"WHISPER {whisper_level} MODEL PERTURBED MEAN_WER: {perturbed_wer}")
    print(f"WHISPER {whisper_level} MODEL WER DEPROVEMENT WITH PERTURBATION: {perturbed_wer - unperturbed_wer}")

#GLADIA TEST FUNCTIONS

def request_transcript(headers: dict, data: dict) -> str:
    trans_req_resp = requests.post("https://api.gladia.io/v2/pre-recorded/", headers=headers, json=data).json()
    result_url = trans_req_resp.get("result_url")
    if not result_url:
        raise Exception("Result url None - result status:", trans_req_resp)
    while True:
        poll_resp = requests.get(result_url, headers=headers).json()
        match poll_resp.get("status"):
            case "done":
                g_response = poll_resp.get("result")
                break
            case "error":
                raise Exception(f"POLLING ERROR: {poll_resp}")
            case _:
                time.sleep(0.1)
    return g_response['transcription']['full_transcript']

def upload_file(headers: dict, files: list) -> str:
    upload_resp = requests.post("https://api.gladia.io/v2/upload/", headers=headers, files=files).json()
    audio_url = upload_resp.get("audio_url")
    return audio_url

def test_one_gladia(wave: pt.Tensor, trans: str) -> float: #CREDIT: This function (and request_transcript + upload_file) takes from the Gladia sample python API call documentation
    #Get wave as bytes
    dimmed_wave = wave.unsqueeze(0)
    bytes_obj = io.BytesIO()
    torchaudio.save(bytes_obj, dimmed_wave, 16000, format='flac')
    file_content = bytes_obj.getvalue()
    #Upload wave (as a file) to gladia
    headers = {
        "x-gladia-key": GLADIA_API_KEY,
        "accept": "application/json",
    }
    file_path = "dummy/path/"; file_extension = ".flac"
    files = [ #I wonder if you could upload multiple files at once... would be awfully nice, but given that only one URL is returned, I think not
        ("audio", (file_path, file_content, "audio/" + file_extension[1:])),
    ]
    audio_url = try_until(upload_file, (headers, files), "Error while attempting to upload audio clip")
    #Request transcription
    headers["Content-Type"] = "application/json"
    data = { "audio_url": audio_url }
    g_trans = try_until(request_transcript, (headers, data), "Error while attempting to request transcript")
    #Run wer and return
    test_wer = wer(trans, g_trans)
    return test_wer

def test_gladia(perturbation_model: pt.nn.Module):
    unperturbed_wer = test_audio_set(test_waves, test_transcripts, test_one_gladia)
    print(f"GLADIA MODEL UNPERTURBED MEAN_WER: {unperturbed_wer}")
    with pt.no_grad():
        perturbed_waves = perturbation_model(test_waves)
    perturbed_wer = test_audio_set(perturbed_waves, test_transcripts, test_one_gladia)
    print(f"GLADIA MODEL PERTURBED MEAN_WER: {perturbed_wer}")
    print(f"GLADIA MODEL WER DEPROVEMENT WITH PERTURBATION: {perturbed_wer - unperturbed_wer}")

# SPEECHMATICS TEST FUNCTIONS

def speechmatics_transcribe(client: BatchClient, file_path: str, conf: dict) -> str:
    try: #CREDIT: Only slightly modified from the speechmatics API documentation
        job_id = client.submit_job(audio=file_path, transcription_config=conf)
        s_transcript = client.wait_for_completion(job_id, transcription_format='txt')
    except HTTPStatusError as e:
        if e.response.status_code == 401:
            print('Invalid API key - Check your API_KEY at the top of the code!')
        elif e.response.status_code == 400:
            print(e.response.json()['detail'])
        else:
            raise e
    return s_transcript

def curried_test_one_speechmatics(settings: ConnectionSettings, conf: dict) -> float:
    os.makedirs(TEMP_DIR, exist_ok=True)
    def test_one_speechmatics(wave: pt.Tensor, trans: str) -> float:
        #Turn audio sample into file (b/c that's the way the API expects it, as relatively inefficient as it is)
        dimmed_wave = wave.unsqueeze(0)
        split_trans = trans.split()
        word_num = min(len(split_trans), 4)
        file_name = "".join(split_trans[0:word_num])
        file_path = TEMP_DIR + file_name
        with open(file_path, "wb") as fl:
            torchaudio.save(fl, dimmed_wave, 16000, format='flac')
        #Send audio file over API and grab transcript
        with BatchClient(settings) as client:
            s_transcript = try_until(speechmatics_transcribe, (client, file_path, conf), "Error having speechmatics transcribe file")
        #Get wer of transcripts
        os.remove(file_path)
        test_wer = wer(trans, s_transcript)
        return test_wer
    return test_one_speechmatics

def test_speechmatics(perturbation_model: pt.nn.Module):
    settings = ConnectionSettings(
        url="https://asr.api.speechmatics.com/v2",
        auth_token=SPEECHMATICS_API_KEY,
    )
    conf = {
        "type": "transcription",
        "transcription_config": {
            "language": "en"
        }
    }
    test_one_speechmatics = curried_test_one_speechmatics(settings, conf)
    unperturbed_wer = test_audio_set(test_waves, test_transcripts, test_one_speechmatics)
    print(f"SPEECHMATICS MODEL UNPERTURBED MEAN_WER: {unperturbed_wer}")
    with pt.no_grad():
        perturbed_waves = perturbation_model(test_waves)
    perturbed_wer = test_audio_set(perturbed_waves, test_transcripts, test_one_speechmatics)
    print(f"SPEECHMATICS MODEL PERTURBED MEAN_WER: {perturbed_wer}")
    print(f"SPEECHMATICS MODEL WER DEPROVEMENT WITH PERTURBATION: {perturbed_wer - unperturbed_wer}")


if __name__ == "__main__":
    model = grab_perturbation_model("wavperturbation_model.pt")
    test_aai(model)

#LIST OF THINGS TO DO:
# - Modularize the test_* functions - TRIAGED OUT
# - Change the saved tensors to be full length - DONE
