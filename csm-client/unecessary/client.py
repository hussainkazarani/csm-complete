# fluttermane predefined text
import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd


async def test_with_local_playback():
    # Connect to localhost (through SSH tunnel)
    async with websockets.connect(
        "ws://csm-stream.hussainkazarani.site/ws"
    ) as websocket:
        print("Connected through SSH tunnel!")

        response = await websocket.recv()
        print("Server:", json.loads(response).get("message"))

        text = """a soft rain fell steadily, blanketing the town in a quiet hush that muted even the loudest thoughts. the streets, slick with water, shimmered under the dull glow of streetlights, casting long reflections that danced with each ripple. cars moved slowly, their tires hissing against the wet pavement, while people hurried under umbrellas, heads down, coats pulled tight. in a small park near the center of town, the trees stood still, their branches heavy with droplets that occasionally slipped off and landed with soft plops on the grass below. a bench sat empty near a winding path, its surface darkened by rain, yet inviting in its solitude."""

        await websocket.send(json.dumps({"type": "text_message", "text": text}))

        stream = None
        sample_rate = None

        try:
            while True:
                response = await websocket.recv()

                if isinstance(response, str):
                    data = json.loads(response)

                    if data.get("type") == "audio_chunk":
                        chunk = np.array(data["audio"], dtype=np.float32)
                        sample_rate = data["sample_rate"]

                        if stream is None:
                            # Start streaming as soon as first chunk arrives
                            stream = sd.OutputStream(
                                samplerate=sample_rate, channels=1, dtype="float32"
                            )
                            stream.start()

                        # Write audio chunk to stream
                        stream.write(chunk)
                        print(f"Played chunk {data['chunk_num']}")

                    elif data.get("type") in ("complete", "error"):
                        print("Server:", data)
                        break
        finally:
            if stream:
                stream.stop()
                stream.close()
            print("Playback finished")


asyncio.run(test_with_local_playback())
