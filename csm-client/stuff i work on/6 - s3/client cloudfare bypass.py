import requests
import base64
import json
import os
import uuid

# Read texts from list.txt
with open("list.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# Create folder for saving audios
os.makedirs("saved_audios", exist_ok=True)

print(f"📋 Processing {len(texts)} texts...")
print("🔄 Connecting to server...")

success_count = 0
error_count = 0

try:
    # Send request with streaming
    response = requests.post(
        "http://storage.hussainkazarani.site/generate",
        json={
            "texts": texts,
            "voice_id": 0,
        },
        stream=True,  # Enable streaming
        timeout=300,  # 5 minute timeout
    )

    if response.status_code == 200:
        print("✅ Connected! Starting streaming processing...\n")

        # Process each audio as it streams in
        for line in response.iter_lines():
            if line:
                data = json.loads(line)

                if data["status"] == "success":
                    # Save the audio file
                    audio_data = base64.b64decode(data["audio_data"])
                    filename = f"audio_{data['index']+1}_{uuid.uuid4().hex[:8]}.wav"
                    filepath = os.path.join("saved_audios", filename)

                    with open(filepath, "wb") as f:
                        f.write(audio_data)

                    success_count += 1
                    print(f"✅ Saved: {filename} ({data['index']+1}/{data['total']})")

                elif data["status"] == "error":
                    error_count += 1
                    print(f"❌ Failed: Text {data['index']+1}/{data['total']}")

        # Final summary
        print(f"\n🎉 Processing complete!")
        print(f"✅ Successfully saved: {success_count} files")
        print(f"❌ Failed: {error_count} files")
        print(f"💾 All files saved to 'saved_audios' folder")

    else:
        print(f"❌ Server error: {response.status_code} {response.text}")

except requests.exceptions.Timeout:
    print("⏰ Request timed out. The server took too long to respond.")
except requests.exceptions.RequestException as e:
    print(f"❌ Request failed: {e}")
except KeyboardInterrupt:
    print("\n⏹️ Processing stopped by user")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("\n✨ Client finished")
