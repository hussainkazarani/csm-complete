from generator import load_csm_1b
import torchaudio
import torch
from datetime import datetime
import warnings

# Skip Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

timestamp = datetime.now().strftime("%H-%M-%S")
output_path = f"voices/server/ubuntu_{timestamp}.wav"
torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)

print(f"Saved audio to {output_path}")
