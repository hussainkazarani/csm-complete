from generator import load_csm_1b
import torchaudio
import torch

if torch.cuda.is_available():
    device = "cuda"
    print("true for gpu")
else:
    device = "cpu"

generator = load_csm_1b(device=device)

audio = generator.generate(
    text="Hello, this is my voice speaking clearly and naturally.",
    speaker=0,
    context=[],
    max_audio_length_ms=5_000,
)

torchaudio.save("csm_voice.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)

