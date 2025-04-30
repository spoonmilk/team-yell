import random
import torch
import numpy as np
import pickle
from pathlib import Path
import os

TEST_PORTION = 0.1

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = str(BASE_DIR / "data") + "/"
except NameError:
    DATA_DIR = os.path.abspath("../data/") + "/"

os.makedirs(DATA_DIR, exist_ok=True)

def save_data(train_waves: torch.Tensor, train_trans: list[str], test_waves: torch.Tensor, test_trans: list[str]):
    # test_num = int(TEST_PORTION*waves.shape[0])
    # indices = random.sample(range(len(transcripts)), test_num)
    #Separate waves and save them
    # waves = np.squeeze(waves.numpy())
    # test_waves = torch.tensor(np.take(waves, indices, axis=0))
    # train_waves = torch.tensor(np.delete(waves, indices, axis=0))
    torch.save(test_waves, DATA_DIR + 'test_waves.pt')
    torch.save(train_waves, DATA_DIR + 'train_waves.pt')
    #Separate transcripts
    # transcripts = np.array(transcripts)
    # test_trans = list(np.take(transcripts, indices))
    # train_trans = list(np.delete(transcripts, indices))
    with open(DATA_DIR + 'test_transcripts.pkl', 'wb') as fl:
        pickle.dump(test_trans, fl)
    with open(DATA_DIR + 'train_transcripts.pkl', 'wb') as fl:
        pickle.dump(train_trans, fl)

def load_data(test: bool = False) -> tuple[torch.Tensor, list[str]]:
    if test: prefix = 'test'
    else: prefix = 'train'
    wave_path = prefix + '_waves.pt'; trans_path = prefix + '_transcripts.pkl'
    waves = torch.load(DATA_DIR + wave_path)
    with open(DATA_DIR + trans_path, 'rb') as fl:
        transcripts = pickle.load(fl)
    return waves, transcripts

if Path(DATA_DIR + 'test_transcripts.pkl').is_file() and Path(DATA_DIR + 'train_waves.pt').is_file():
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