from generator import load_csm_1b, Segment
import torchaudio
import torch
import time
from datetime import datetime
import warnings

# Skip Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
generator = load_csm_1b(device=device)

# Load existing base audio file
base_audio_path = "voices/test/base_voice.wav"
base_audio, sample_rate = torchaudio.load(base_audio_path)

# If stereo, convert to mono
if base_audio.shape[0] > 1:
    base_audio = base_audio.mean(dim=0, keepdim=True)

# If needed, resample to match generator.sample_rate
if sample_rate != generator.sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=generator.sample_rate)
    base_audio = resampler(base_audio)

# Remove batch dimension for Segment (just use 1D tensor)
base_audio = base_audio.squeeze(0).to(device)

# Create base segment
base_segment = Segment(
    text="Hey there, I would really love to talk with you.",
    speaker=0,
    audio=base_audio
)

# Generate new audio with context if needed
start = time.time()
audio = generator.generate(
    text="Hello, this is my voice speaking clearly and naturally.",
    speaker=0,
    context=[],  # optionally pass [base_segment] for voice continuity
    max_audio_length_ms=5_000,
)
end = time.time()
print(f"Generation time: {end - start:.2f} seconds",flush=True)

# Save the new audio
timestamp = datetime.now().strftime("%H-%M-%S")
output_path = f"voices/server/ubuntu_{timestamp}.wav"
torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)

print(f"Saved audio to {output_path}")
