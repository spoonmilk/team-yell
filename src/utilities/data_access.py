import random
import torch
import numpy as np
import pickle
from pathlib import Path
import os

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = str(BASE_DIR / "data") + "/"
except NameError:
    DATA_DIR = os.path.abspath("../data/") + "/"

os.makedirs(DATA_DIR, exist_ok=True)

def save_data(waves: torch.Tensor, transcripts: list[str]):
    torch.save(waves, DATA_DIR + 'waves.pt')
    with open(DATA_DIR + 'transcripts.pkl', 'wb') as fl:
        pickle.dump(transcripts, fl)

def load_data() -> tuple[torch.Tensor, list[str]]:
    waves = torch.load(DATA_DIR + 'waves.pt')
    with open(DATA_DIR + 'transcripts.pkl', 'rb') as fl:
        transcripts = pickle.load(fl)
    return waves, transcripts

if Path(DATA_DIR + 'transcripts.pkl').is_file() and Path(DATA_DIR + 'waves.pt').is_file():
    waves, transcripts = load_data()
    assert len(waves) == len(transcripts)

    def grab_batch(batch_sz: int) -> tuple[torch.Tensor, list[str]]:
        """Size of batch -> list of audio tensors correlated with list of transcriptions"""
        indices = random.sample(range(len(transcripts)), batch_sz)
        batch_waves = torch.stack([waves[idx] for idx in indices], 0)
        batch_trans = [transcripts[idx] for idx in indices]
        return batch_waves, batch_trans
    
    # def batches(batch_sz: int, num_batches: int = None):
    #     assert num_batches*batch_sz <= len(transcripts) #Must be enough possible batches in the first place
    #     indices = np.arange(len(transcripts))
    #     while (len(indices) >= batch_sz) or 