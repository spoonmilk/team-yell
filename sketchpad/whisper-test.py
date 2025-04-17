import whisper

# Load tiny model
model = whisper.load_model("tiny")

# Grab data from scratch
audio = whisper.load_audio("/users/akhosrow/scratch/LibriSpeech/dev-clean/174/50561/174-50561-0000.flac")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)


