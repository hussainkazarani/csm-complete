# kive long chunk
import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd
import time


async def send_keep_alive_ping(websocket, interval=15):
    """Send ping messages to keep connection alive"""
    while True:
        try:
            await asyncio.sleep(interval)
            await websocket.ping()
            print("Sent keep-alive ping", end="\r")
        except:
            break  # Exit if connection is closed


async def test_with_local_playback():
    # Connect to localhost (through SSH tunnel)
    try:
        async with websockets.connect(
            "ws://hive.hussainkazarani.site/ws", ping_interval=None
        ) as websocket:
            print("Connected through SSH tunnel!")

            # Start keep-alive ping task
            ping_task = asyncio.create_task(send_keep_alive_ping(websocket, 15))

            # Wait for initial status messages
            response = await websocket.recv()
            response_data = json.loads(response)
            print("Server:", response_data.get("message", "Connected"))

            # Check if server is busy
            if (
                response_data.get("type") == "error"
                and "busy" in response_data.get("message", "").lower()
            ):
                print("Server is busy with another connection. Try again later.")
                ping_task.cancel()
                return

            while True:
                print("\nChoose input method:")
                print("1. Type text directly")
                print("2. Read from file")
                print("3. Quit")

                choice = input("Enter choice (1/2/3): ")

                if choice == "3":
                    break
                elif choice == "1":
                    text = input("Enter text to synthesize: ")
                    if not text.strip():
                        print("Please enter some text.")
                        continue
                elif choice == "2":
                    filename = input("Enter filename (or path): ")
                    try:
                        with open(filename, "r", encoding="utf-8") as f:
                            text = f.read()
                        print(f"Read {len(text)} characters from {filename}")
                    except FileNotFoundError:
                        print(f"File {filename} not found.")
                        continue
                    except Exception as e:
                        print(f"Error reading file: {e}")
                        continue
                else:
                    print("Invalid choice.")
                    continue

                await websocket.send(json.dumps({"type": "text_message", "text": text}))
                print(f"Sent {len(text)} characters to server...")

                stream = None
                sample_rate = None
                chunk_count = 0
                start_time = time.time()

                try:
                    while True:
                        try:
                            # No timeout - wait indefinitely for server response
                            response = await websocket.recv()

                            if isinstance(response, str):
                                data = json.loads(response)

                                if data.get("type") == "audio_chunk":
                                    chunk = np.array(data["audio"], dtype=np.float32)
                                    sample_rate = data["sample_rate"]
                                    chunk_count += 1
                                    part_info = data.get("part")
                                    if part_info:
                                        print(f"Part {part_info} - ", end="")

                                    if stream is None:
                                        # Start streaming as soon as first chunk arrives
                                        stream = sd.OutputStream(
                                            samplerate=sample_rate,
                                            channels=1,
                                            dtype="float32",
                                        )
                                        stream.start()
                                        print(f"Started playback at {sample_rate}Hz")

                                    # Write audio chunk to stream
                                    stream.write(chunk)
                                    elapsed = time.time() - start_time
                                    print(
                                        f"Received chunk {chunk_count} | Elapsed: {elapsed:.1f}s",
                                        end="\r",
                                    )

                                elif data.get("type") == "audio_status":
                                    status = data.get("status", "")
                                    if status == "complete":
                                        elapsed = time.time() - start_time
                                        print(
                                            f"\nAudio generation complete! Received {chunk_count} chunks in {elapsed:.1f}s"
                                        )
                                        break
                                    elif status == "error":
                                        print(
                                            f"\nError: {data.get('message', 'Unknown error')}"
                                        )
                                        break

                                    elif (
                                        data.get("type") == "status"
                                    ):  # ← YOU'RE MISSING THIS HANDLER!
                                        message = data.get("message", "")
                                        print(f"\nServer: {message}")

                                elif data.get("type") == "error":
                                    print(
                                        f"\nServer error: {data.get('message', 'Unknown error')}"
                                    )
                                    break

                        except asyncio.TimeoutError:
                            # This shouldn't happen anymore since we removed the timeout
                            print("\nUnexpected timeout")
                            break

                except Exception as e:
                    print(f"\nError during audio playback: {e}")

                finally:
                    if stream:
                        stream.stop()
                        stream.close()
                        stream = None
                    print("Playback finished")

    except websockets.exceptions.ConnectionClosedError:
        print("Connection closed by server")
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        # Cancel the ping task when we're done
        if "ping_task" in locals():
            ping_task.cancel()


if __name__ == "__main__":
    asyncio.run(test_with_local_playback())
