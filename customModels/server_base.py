from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import torch
import torchaudio
from generator import load_csm_1b, Segment
import time
from datetime import datetime
import warnings

# Skip Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = load_csm_1b(device=device)  # Model loads ONCE

# ------------- 1. Load and prepare your reference ("cloned") voice once -------------
REF_AUDIO_PATH = "voices/test/base_voice.wav"  # <-- replace with your file
REF_TRANSCRIPT = "Hey there, I would really love to talk with you."  # Write the exact spoken words

# Load and resample reference audio to sample rate expected by model
ref_audio_tensor, sr = torchaudio.load(REF_AUDIO_PATH)
ref_audio_tensor = torchaudio.functional.resample(ref_audio_tensor.squeeze(0), orig_freq=sr, new_freq=generator.sample_rate)

context_segment = Segment(
    text=REF_TRANSCRIPT,
    speaker=0,
    audio=ref_audio_tensor
)

# ------------- 2. Flask endpoint for generating cloned speech from any input text -------------
@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Missing 'text'."}), 400

    # Duration: auto-estimate based on text length in words
    avg_wpm = 170
    n_words = len(text.strip().split())
    seconds = max(2, int(n_words / (avg_wpm / 60)))
    max_audio_length_ms = seconds * 1000 + 1000  # 1s buffer

    try:
        with torch.inference_mode():
            start = time.time()
            audio = generator.generate(
                text=text,
                speaker=0,
                context=[context_segment],  # Your reference segment for cloning
                max_audio_length_ms=max_audio_length_ms
            )
            end = time.time()
            print(f"Generation time: {end - start:.2f} seconds",flush=True)

        timestamp = datetime.now().strftime("%H-%M-%S")
        output_path = f"voices/server/server_{timestamp}.wav"
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        print(f"Saved audio to {output_path}")

        return send_file(output_path, mimetype="audio/wav", as_attachment=True, download_name="output.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

