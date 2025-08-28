# test_client.py
import asyncio
import websockets
import json


async def test_connection():
    uri = "ws://csm-stream.hussainkazarani.site/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to server. Waiting for welcome message...")

            # Wait for the server's ready message
            welcome_msg = await websocket.recv()
            print(f"Server says: {welcome_msg}")

            # Send a test message
            test_message = {"type": "text_message", "text": "Hello, who are you?"}
            await websocket.send(json.dumps(test_message))
            print("Sent message: 'Hello, who are you?'")

            # Listen for responses
            while True:
                response = await websocket.recv()
                data = json.loads(response)

                if data["type"] == "llm_response":
                    print(f"\nLLM Text Response: {data['text']}")
                elif data["type"] == "status":
                    print(f"Status: {data['message']}")
                elif data["type"] == "audio_status":
                    print(f"Audio Status: {data['status']}")
                # You would handle 'audio_chunk' here to play the audio

    except Exception as e:
        print(f"Connection failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_connection())
