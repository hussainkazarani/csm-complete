from flask import Flask, request, send_file, jsonify, after_this_request
from flask_cors import CORS
import torch
import torchaudio
from generator import load_csm_1b, Segment
import time
from datetime import datetime
import warnings
from pydub import AudioSegment
import os
import io

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # /home/ubuntu/csm/customModels
PROJECT_ROOT = os.path.dirname(BASE_DIR)              # /home/ubuntu/csm

# Skip Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model and reference segment...")
generator = load_csm_1b(device=device)  # Model loads ONCE
generator._model = torch.compile(generator._model)

# ------------- 1. Load and prepare your reference ("cloned") voice once -------------
REF_AUDIO_PATH = "voices/test/base_voice.wav"  # <-- replace with your file
REF_TRANSCRIPT = "Hey there, I would really love to talk with you."  # Write the exact spoken words

# Load and resample reference audio to sample rate expected by model
ref_audio_tensor, sr = torchaudio.load(REF_AUDIO_PATH)
ref_audio_tensor = torchaudio.functional.resample(ref_audio_tensor.squeeze(0), orig_freq=sr, new_freq=generator.sample_rate)

print("Loading reference segment...")
context_segment = Segment(
    text=REF_TRANSCRIPT,
    speaker=0,
    audio=ref_audio_tensor
)

def estimate_duration(text: str, wpm=170, buffer_secs=2):
    n_words = len(text.strip().split())
    base_seconds = n_words / (wpm / 60)
    extra_per_word = 0.15 * n_words  # extra time to cover longer phrasing
    total_seconds = base_seconds + buffer_secs + extra_per_word
    total_seconds = max(total_seconds, 2.0)
    return int(total_seconds * 1000)

# ------------- 2. Flask endpoint for generating cloned speech from any input text -------------
@app.route("/csm-basic", methods=["POST"])
def tts():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Missing 'text'."}), 400

    # Duration: auto-estimate based on text length in words
#    avg_wpm = 170
#    n_words = len(text.strip().split())
#    seconds = max(2, int(n_words / (avg_wpm / 60)))
#    max_audio_length_ms = seconds * 1000 + 1000  # 1s buffer

    max_audio_length_ms = estimate_duration(text)

    try:
        with torch.inference_mode():
            start = time.time()
            audio = generator.generate(
                text=text,
                speaker=0,
                context=[context_segment],  # Your reference segment for cloning
                max_audio_length_ms=max_audio_length_ms
            )
            inference_time = time.time() - start
            print(f"Inference generation time: {inference_time:.2f} seconds", flush=True)

#        buffer = io.BytesIO()
#        torchaudio.save(buffer, audio.unsqueeze(0).cpu(), generator.sample_rate, format="wav")
#        buffer.seek(0)
#        return send_file(buffer, mimetype="audio/wav", as_attachment=True, download_name="output.wav")        
        
        wav_io = io.BytesIO()
        torchaudio.save(wav_io, audio.unsqueeze(0).cpu(), generator.sample_rate, format="wav")
        wav_io.seek(0)
        
        # Optional disk save
        output_path = f"voices/optimize/optimize_{datetime.now().strftime('%H-%M-%S')}.wav"
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        print(f"Saved audio to {output_path}")

        sound = AudioSegment.from_file(wav_io, format="wav")
        mp3_io = io.BytesIO()
        sound.export(mp3_io, format="mp3")
        mp3_io.seek(0)

        @after_this_request
        def log_response_time(response):
            print(f"Finished sending response at {time.time()}, total request took {time.time() - start:.2f} seconds", flush=True)
            return response

        return send_file(mp3_io, mimetype="audio/mpeg", as_attachment=True, download_name="output.mp3")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Exception during TTS request:\n{tb}", flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

