import asyncio
import websockets
import json
import aiofiles
import time
import numpy as np
import sounddevice as sd
from websockets.exceptions import ConnectionClosed


class SimpleTTSClient:
    def __init__(self):
        self.websocket = None
        self.is_connected = False
        self.ping_task = None
        self.audio_stream = None
        self.keep_alive_interval = 5
        self.last_activity_time = 0

    async def connect(self, uri="ws://hive.hussainkazarani.site/ws"):
        """Connect to the server"""
        try:
            self.websocket = await websockets.connect(
                uri, ping_interval=None, close_timeout=300, max_queue=2048
            )
            self.is_connected = True
            self.last_activity_time = time.time()

            # Get welcome message
            welcome = await self.websocket.recv()
            print(f"Server: {json.loads(welcome).get('message', 'Connected')}")

            # Start continuous keep-alive
            self.ping_task = asyncio.create_task(self._continuous_keep_alive())

            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def _continuous_keep_alive(self):
        """Continuous keep-alive"""
        while self.is_connected:
            try:
                if self.websocket:
                    await self.websocket.ping()
                await asyncio.sleep(self.keep_alive_interval)
            except:
                self.is_connected = False
                break

    async def _play_audio(self, audio_data, sample_rate):
        """Play audio chunk"""
        if self.audio_stream is None:
            self.audio_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            self.audio_stream.start()
        self.audio_stream.write(audio_data)

    async def _cleanup_audio(self):
        """Clean up audio resources"""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

    async def send_text(self, text):
        """Send text to server and play audio response"""
        if not self.is_connected:
            print("Not connected to server!")
            return False

        try:
            # Calculate timeout based on text length
            word_count = len(text.split())
            timeout_seconds = max(30, word_count * 0.3)  # 0.3 seconds per word, min 30s
            print(f"Text has {word_count} words, using {timeout_seconds}s timeout")

            # Send text to server
            await self.websocket.send(
                json.dumps({"type": "text_message", "text": text})
            )
            self.last_activity_time = time.time()
            print("Text sent successfully! Waiting for audio...")

            # Listen for audio chunks
            chunk_count = 0
            sample_rate = None
            audio_complete = False
            start_time = time.time()

            while self.is_connected and not audio_complete:
                try:
                    # Check if we've exceeded overall timeout
                    if time.time() - start_time > timeout_seconds:
                        print(f"\n❌ Timeout after {timeout_seconds} seconds")
                        return False

                    # Check connection health
                    if time.time() - self.last_activity_time > 15.0:
                        print("\n🔍 No activity for 15s, checking connection...")
                        try:
                            await self.websocket.ping()
                            self.last_activity_time = time.time()
                        except:
                            print("❌ Connection lost during health check")
                            self.is_connected = False
                            return False

                    message = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                    self.last_activity_time = time.time()

                    # Parse message (handle both JSON and string formats)
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        print(f"Non-JSON message: {message}")
                        continue

                    if data.get("type") == "audio_chunk":
                        audio_data = np.array(data["audio"], dtype=np.float32)
                        sample_rate = data["sample_rate"]
                        await self._play_audio(audio_data, sample_rate)
                        chunk_count += 1
                        print(f"Playing chunk {chunk_count}", end="\r")

                    elif data.get("type") == "audio_status":
                        status = data.get("status")
                        if status == "complete" or status == "ready_for_next":
                            print(f"\n✅ Audio completed! Played {chunk_count} chunks")
                            audio_complete = True
                        elif status == "generating":
                            print("Server is generating audio...")
                        elif status == "error":
                            print(f"Server error: {data.get('message')}")
                            return False
                        elif status == "first_chunk":
                            print("\n🎵 First audio chunk received...")

                    elif data.get("type") == "completion":
                        if data.get("message") == "audio_generation_complete":
                            print(f"\n✅ Audio completed! Played {chunk_count} chunks")
                            audio_complete = True

                    elif data.get("type") == "status":
                        # Handle status messages
                        message_text = data.get("message", "")
                        print(f"Server: {message_text}")
                        if (
                            "ready" in message_text.lower()
                            and "next" in message_text.lower()
                        ):
                            print(f"\n✅ Audio completed! Played {chunk_count} chunks")
                            audio_complete = True

                    elif data.get("type") == "error":
                        print(f"Error: {data.get('message')}")
                        return False

                    else:
                        print(f"Unknown message type: {data.get('type')}")

                except asyncio.TimeoutError:
                    # Just continue listening, but show we're still waiting
                    if time.time() - self.last_activity_time > 5.0:
                        print("⏳", end="", flush=True)
                    continue

            # Audio completed successfully, connection remains open
            await self._cleanup_audio()
            print(f"🎉 Total processing time: {time.time() - start_time:.1f}s")
            return True

        except ConnectionClosed:
            print("\n❌ Connection closed by server")
            self.is_connected = False
            return False

        except Exception as e:
            print(f"\n❌ Failed during text processing: {e}")
            self.is_connected = False
            return False

        finally:
            await self._cleanup_audio()

    async def disconnect(self):
        """Cleanly disconnect"""
        self.is_connected = False

        if self.ping_task:
            self.ping_task.cancel()
            try:
                await self.ping_task
            except:
                pass

        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        await self._cleanup_audio()
        print("Disconnected from server")


async def main():
    client = SimpleTTSClient()

    print("Connecting to server...")
    if await client.connect():
        print("Connected successfully! ✅")
    else:
        print("Failed to connect ❌")
        return

    try:
        while True:
            print("\n" + "=" * 50)
            print("1. Enter text directly")
            print("2. Read from text file")
            print("3. Check connection")
            print("4. Disconnect & exit")
            print("=" * 50)

            choice = input("Choose option (1-4): ").strip()

            if choice == "1":
                text = input("Enter text to speak: ").strip()
                if text:
                    success = await client.send_text(text)
                    if not success and not client.is_connected:
                        print("Reconnecting...")
                        if await client.connect():
                            print("Reconnected! ✅")
                        else:
                            break
                else:
                    print("Please enter some text!")

            elif choice == "2":
                filename = input("Enter filename: ").strip()
                try:
                    async with aiofiles.open(filename, "r", encoding="utf-8") as f:
                        text = await f.read()
                    print(f"Read {len(text)} characters from {filename}")

                    success = await client.send_text(text)
                    if not success and not client.is_connected:
                        print("Reconnecting...")
                        if await client.connect():
                            print("Reconnected! ✅")
                        else:
                            break
                except Exception as e:
                    print(f"Error reading file: {e}")

            elif choice == "3":
                if client.is_connected:
                    print("✅ Still connected to server")
                    # Test connection with a ping
                    try:
                        await client.websocket.ping()
                        print("✅ Ping successful")
                    except:
                        print("❌ Ping failed - connection may be broken")
                        client.is_connected = False
                else:
                    print("❌ Not connected - need to reconnect")

            elif choice == "4":
                await client.disconnect()
                break

            else:
                print("Invalid choice! Please enter 1-4")

    except KeyboardInterrupt:
        print("\nDisconnecting...")
        await client.disconnect()
    except Exception as e:
        print(f"Unexpected error: {e}")
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
