import torch as pt
import torch.nn as nn
import whisper
from torch import optim

from ...models.perturbation_model import WavPerturbationModel
from ...utilities.data_access import grab_batch

"""
PGA (Projected Gradient Ascent) is a relatively simple white box attack involving feeding perturbed inputs through a
target model, calculating some loss of the model, and backpropogating through it to alter a perturbation model's weights.
We originally worked on implementing this approach thinking it would be a good point of comparison for our black box
attacks, but ran into difficulties with Whisper's tokenizer and pytorch's CE Loss function that were difficult to resolve,
and we figured it would be a better idea to focus all of our time and effort on getting ES alone to work. It's worth
noting, though, that we later developed a logit based loss function for our ES model that would work for this PGA model
as well (it's based around the entropy of the logits instead of the cross entropy of the logits and ground truth
transcription tokens), so with a little more work, we could probably get a version of this working as well. 
"""

MODEL_TYPE = "tiny"

ce_loss = (
    nn.CrossEntropyLoss()
)  # Expects logits (and labels) as inputs and applies softmax internally

w_model = whisper.load_model(MODEL_TYPE)
w_model.to(pt.device('cpu'))
# CREDIT (following two lines and help in tokenizer usage in loss function): ChatGPT (would include transcripts and more detailed reports if this was actually code we used for any of the deliverables, but this code is essentially unused for this project)
tokenizer = whisper.tokenizer.get_tokenizer(
    multilingual=True, task="transcribe", language="en"
)
tokens = pt.tensor([[tokenizer.sot]], device=w_model.device)

def loss(perturbed_audio: pt.Tensor, transcripts: list[str]) -> float:
    # Reset whisper model gradients (just in case)
    w_model.zero_grad()
    # Get whisper transcription logits
    sized_audio = whisper.pad_or_trim(perturbed_audio)
    mel = whisper.log_mel_spectrogram(sized_audio, n_mels=w_model.dims.n_mels).to(
        w_model.device
    )
    whisper_logits = w_model.forward(mel, tokens)
    # Convert transcripts to tokenized transcripts
    sots = pt.empty((len(transcripts), 1), device=w_model.device)
    eots = sots.detach().clone()
    sots.fill_(tokenizer.sot)
    eots.fill_(tokenizer.eot)
    transcripts_tokens = pt.tensor(
        [tokenizer.encode(trans) for trans in transcripts], device=w_model.device
    )  # CREDIT: OpenAI's ChatGPT
    padding = pt.zeros(
        (
            len(transcripts),
            whisper_logits.shape[-1] - (transcripts_tokens.shape[1] + 2),
        ),
        device=w_model.device,
    )
    token_transcripts = pt.concat((sots, transcripts_tokens, eots, padding), dim=1)
    token_transcripts = token_transcripts.to(pt.long) #CE loss needs this for whatever reason *shrug*
    # Return Cross Entropy loss of (softmaxed) logits and tokenized transcripts
    print(token_transcripts.shape, whisper_logits.shape)
    return ce_loss(whisper_logits, token_transcripts)

def pga_epoch(model: WavPerturbationModel, batch_sz: int = 30, batch_num: int = 10):
    optimizer = optim.Adam(
        model.parameters(), lr=0.001, maximize=True
    )  # maximize = True for gradient ascent
    for _ in range(len(batch_num)):
        optimizer.zero_grad()  # Reset relevant gradients
        waves, transcripts = grab_batch(batch_sz)
        perturbed_audio = model(waves)
        batch_loss = loss(perturbed_audio, transcripts)
        batch_loss.backwards()
        optimizer.step()

# CREDIT: https://medium.com/@zachariaharungeorge/a-deep-dive-into-the-fast-gradient-sign-method-611826e34865 for fgsm concept
def fgsm(audio_clip: pt.Tensor, trans: str, epsilon):
    # Find gradient of loss with respect to singular audio_clip
    audio_clip.requires_grad = True
    fgsm_loss = loss(audio_clip, [trans])
    fgsm_loss.backward()
    grad = audio_clip.grad
    # Generate perturbed version of audio based off of step in direction of loss gradient
    perturbation = epsilon * pt.sign(grad)
    perturbed_audio = audio_clip + perturbation
    # Return
    return perturbed_audio.detach()

if __name__ == "__main__":
    # fgsm example to test loss function
    waves, transes = grab_batch(1)
    wave = waves[0]
    trans = transes[0]
    # model = WavPerturbationModel(10,32,2,0.01) #Picked these numbers kinda randomly... - model not actually needed
    fgsm(wave, trans, 0.01)
