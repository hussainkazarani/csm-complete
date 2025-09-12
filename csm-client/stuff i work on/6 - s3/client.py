import requests
import base64
import time
import os
import uuid

# Increase timeout for long texts
TIMEOUT = 30000

# Read texts from list.txt
with open("list.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# Create folder for saving audios
os.makedirs("saved_audios", exist_ok=True)

# Send request to server
try:
    response = requests.post(
        "http://storage.hussainkazarani.site/generate",
        json={
            "texts": texts,
            "voice_id": 0,
        },
        timeout=TIMEOUT,
    )

    if response.status_code == 200:
        result = response.json()
        sample_rate = result["sample_rate"]

        # Save each audio file locally with random names
        for i, audio_base64 in enumerate(result["audio_data"]):
            audio_data = base64.b64decode(audio_base64)
            random_name = f"{uuid.uuid4().hex}.wav"
            file_path = os.path.join("saved_audios", random_name)
            with open(file_path, "wb") as f:
                f.write(audio_data)
            print(f"Saved {file_path}")

    else:
        print(f"Server returned error: {response.status_code} {response.text}")

except requests.exceptions.Timeout:
    print("Request timed out. The server is processing a long text.")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
