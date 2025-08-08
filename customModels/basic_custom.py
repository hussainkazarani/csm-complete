from generator import load_csm_1b, Segment
import torchaudio
import torch

if torch.cuda.is_available():
    device = "cuda"
    print("true for gpu")
else:
    device = "cpu"

generator = load_csm_1b(device=device)

base_audio = generator.generate(
    text="Hey there, I would really love to talk with you.",
    speaker=0,
    context=[],
    max_audio_length_ms=5_000,
)

torchaudio.save("base_voice.wav", base_audio.unsqueeze(0).cpu(), generator.sample_rate)

base_segment = Segment(
    text="Hey there, I would really love to talk with you.",
    speaker=0,
    audio=base_audio
)

audio = generator.generate(
    text="Hello, this is my voice speaking clearly and naturally.",
    speaker=0,
    context=[],
    max_audio_length_ms=5_000,
)

torchaudio.save("csm_custom_voice.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)

