import torch as pt
import torch.nn as nn
from torch import optim
from ..utilities.data_access import grab_batch
from ..models.perturbation_model import WavPerturbationModel
import whisper

MODEL_TYPE = "tiny"

ce_loss = nn.CrossEntropyLoss() #Expects logits (and labels) as inputs and applies softmax internally 

w_model = whisper.load_model(MODEL_TYPE)
#CREDIT (following two lines and help in tokenizer usage in loss function): ChatGPT (see appendix)
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, task="transcribe", language="en")
tokens = pt.tensor([[tokenizer.sot]], device=w_model.device)

def loss(perturbed_audio: pt.Tensor, transcripts: list[str]) -> float:
    #Reset whisper model gradients (just in case)
    w_model.zero_grad()
    #Get whisper transcription logits
    sized_audio = whisper.pad_or_trim(perturbed_audio)
    mel = whisper.log_mel_spectrogram(sized_audio, n_mels=w_model.dims.n_mels).to(w_model.device)
    whisper_logits = w_model.forward(mel, tokens)
    #Convert transcripts to tokenized transcripts
    sots = pt.fill((len(transcripts), 1), tokenizer.sot, device=w_model.device)
    eots = pt.fill((len(transcripts), 1), tokenizer.eot, device=w_model.device)
    transcripts_tokens = pt.tensor([tokenizer.encode(trans) for trans in transcripts], device=w_model.device) #CREDIT: OpenAI's ChatGPT
    token_transcripts = pt.concat((sots, transcripts_tokens, eots), dim=1)
    #Return Cross Entropy loss of (softmaxed) logits and tokenized transcripts
    return ce_loss(whisper_logits, token_transcripts)

def pga_epoch(model: WavPerturbationModel, batch_sz: int = 30, batch_num: int = 10):
    optimizer = optim.Adam(model.parameters(), lr=0.001, maximize=True) #maximize = True for gradient ascent
    for _ in range(len(batch_num)): 
        optimizer.zero_grad() #Reset relevant gradients
        waves, transcripts = grab_batch(batch_sz)
        perturbed_audio = model(waves)
        batch_loss = loss(perturbed_audio, transcripts)
        batch_loss.backwards()
        optimizer.step()

#CREDIT: https://medium.com/@zachariaharungeorge/a-deep-dive-into-the-fast-gradient-sign-method-611826e34865 for fgsm concept
def fgsm(audio_clip: pt.Tensor, trans: str, epsilon):
    #Find gradient of loss with respect to singular audio_clip
    audio_clip.requires_grad = True 
    loss = loss(pt.unsqueeze(audio_clip), [trans])
    loss.backward()
    grad = audio_clip.grad
    #Generate perturbed version of audio based off of step in direction of loss gradient
    perturbation = epsilon * pt.sign(grad)
    perturbed_audio = audio_clip + perturbation
    #Return
    return perturbed_audio.detach()

if __name__ == "__main__":
    #fgsm example to test loss function
    waves, transes = grab_batch(1)
    wave = waves[0];  trans = transes[0] 
    #model = WavPerturbationModel(10,32,2,0.01) #Picked these numbers kinda randomly... - model not actually needed
    fgsm(wave, trans, 0.01)
