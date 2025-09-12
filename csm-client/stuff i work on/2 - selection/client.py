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
        self.is_playing_audio = False
        self.available_voices = {}  # Store available voices {id_string: name}
        self.current_voice_id = 0  # Current selected voice ID (integer)

    async def connect(self, uri="ws://hive.hussainkazarani.site/ws"):
        """Connect to the server"""
        try:
            self.websocket = await websockets.connect(
                uri, ping_interval=None, close_timeout=300, max_queue=2048
            )
            self.is_connected = True

            # Get welcome message and available voices
            welcome = await self.websocket.recv()
            welcome_data = json.loads(welcome)

            if welcome_data.get("type") == "available_voices":
                self.available_voices = welcome_data.get("voices", {})
                self.current_voice_id = welcome_data.get("default_voice", 0)
            else:
                print(f"Server: {welcome_data.get('message', 'Connected')}")

            # Start continuous keep-alive
            self.ping_task = asyncio.create_task(self._continuous_keep_alive())

            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    async def _continuous_keep_alive(self):
        """Continuous keep-alive - pauses during audio playback"""
        while self.is_connected:
            try:
                # Don't send pings while audio is playing to avoid interruptions
                if not self.is_playing_audio and self.websocket:
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

        # Ensure audio data is the right shape and type
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data, dtype=np.float32)

        self.audio_stream.write(audio_data)

    async def _cleanup_audio(self):
        """Clean up audio resources"""
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except:
                pass
            finally:
                self.audio_stream = None

    async def send_text(self, text, voice_id=None):
        """Send text to server and play audio response"""
        if not self.is_connected:
            print("❌ Not connected to server!")
            return False

        try:
            # Set flag to indicate audio playback is starting
            self.is_playing_audio = True

            # Use specified voice_id or current voice_id
            if voice_id is None:
                voice_id = self.current_voice_id

            # Send text to server with voice_id
            await self.websocket.send(
                json.dumps({"type": "text_message", "text": text, "voice_id": voice_id})
            )

            # FIXED: Convert integer voice_id to string for dictionary lookup
            current_voice_name = self.available_voices.get(str(voice_id), "Unknown")
            print(f"📤 Sending text to voice {voice_id} ({current_voice_name})...")

            # Listen for audio chunks
            chunk_count = 0
            sample_rate = None
            audio_complete = False
            first_chunk_received = False
            timeout_count = 0

            while self.is_connected and not audio_complete:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(), timeout=10.0
                    )
                    data = json.loads(message)
                    timeout_count = 0  # Reset timeout counter on successful receive

                    if data.get("type") == "audio_chunk":
                        audio_data = np.array(data["audio"], dtype=np.float32)
                        sample_rate = data["sample_rate"]

                        if not first_chunk_received:
                            print("🎵 First audio chunk received! Playing...")
                            first_chunk_received = True

                        await self._play_audio(audio_data, sample_rate)
                        chunk_count += 1
                        part_info = data.get("part", "")
                        if part_info:
                            print(
                                f"▶️ Playing part {part_info}, chunk {chunk_count}",
                                end="\r",
                            )
                        else:
                            print(f"▶️ Playing chunk {chunk_count}", end="\r")

                    elif data.get("type") == "audio_status":
                        status = data.get("status")
                        if status == "first_chunk":
                            print("⏳ First audio chunk generated...")
                        elif status == "generating":
                            print("⚙️ Server is generating audio...")

                    elif data.get("type") == "status":
                        message = data.get("message", "")
                        print(f"ℹ️  {message}")

                    elif data.get("type") == "completion":
                        if data.get("message") == "audio_generation_complete":
                            total_chunks = data.get("chunks_processed", chunk_count)
                            print(f"\n✅ Audio completed! Played {total_chunks} chunks")
                            audio_complete = True

                    elif data.get("type") == "error":
                        error_msg = data.get("message", "Unknown error")
                        print(f"\n❌ Error: {error_msg}")
                        return False

                except asyncio.TimeoutError:
                    timeout_count += 1
                    if timeout_count > 3:  # If we timeout 3 times in a row
                        print(f"\n⏰ Connection seems slow, still waiting...")
                        timeout_count = 0
                    continue

            # Audio completed successfully, connection remains open
            await self._cleanup_audio()
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
            # Reset the flag when audio playback is done
            self.is_playing_audio = False
            await self._cleanup_audio()

    def show_current_voice(self):
        """Show current voice selection"""
        # FIXED: Convert integer voice_id to string for dictionary lookup
        current_voice_name = self.available_voices.get(
            str(self.current_voice_id), "Unknown"
        )
        print(f"\n🔊 Current voice: {self.current_voice_id} - {current_voice_name}")

    def show_all_voices(self):
        """Show all available voices"""
        if not self.available_voices:
            print("❌ No voices available")
            return

        print("\n🎤 Available voices:")
        print("-" * 30)
        for voice_id, voice_name in self.available_voices.items():
            # FIXED: Compare string voice_id with string representation of current_voice_id
            current_indicator = (
                " ← CURRENT" if voice_id == str(self.current_voice_id) else ""
            )
            print(f"  {voice_id}: {voice_name}{current_indicator}")
        print("-" * 30)

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
        print("👋 Disconnected from server")


async def main():
    client = SimpleTTSClient()

    print("🔗 Connecting to server...")
    if await client.connect():
        print("✅ Connected successfully!")
    else:
        print("❌ Failed to connect")
        return

    try:
        while True:
            print("\n" + "=" * 50)
            print("🎯 TTS Client Menu")
            print("=" * 50)
            print("1. 📝 Enter text directly")
            print("2. 📖 Read from text file")
            print("3. 🎤 Show/Change voice")
            print("4. 🔍 Check connection")
            print("5. 🚪 Disconnect & exit")
            print("=" * 50)

            # Show only current voice, not all voices
            client.show_current_voice()
            print("=" * 50)

            choice = input("Choose option (1-5): ").strip()

            if choice == "1":
                text = input("Enter text to speak: ").strip()
                if text:
                    success = await client.send_text(text)
                    if not success and not client.is_connected:
                        print("🔄 Reconnecting...")
                        if await client.connect():
                            print("✅ Reconnected!")
                        else:
                            break
                else:
                    print("❌ Please enter some text!")

            elif choice == "2":
                filename = input("Enter filename: ").strip()
                try:
                    async with aiofiles.open(filename, "r", encoding="utf-8") as f:
                        text = await f.read()
                    print(f"📖 Read {len(text)} characters from {filename}")

                    success = await client.send_text(text)
                    if not success and not client.is_connected:
                        print("🔄 Reconnecting...")
                        if await client.connect():
                            print("✅ Reconnected!")
                        else:
                            break
                except Exception as e:
                    print(f"❌ Error reading file: {e}")

            elif choice == "3":
                # Show all voices first
                client.show_all_voices()

                # Ask if user wants to change voice
                change = input("\nDo you want to change voice? (y/N): ").strip().lower()
                if change in ["y", "yes"]:
                    try:
                        new_voice_id = int(input("Enter voice ID: ").strip())
                        # FIXED: Check if string representation exists in available_voices
                        if str(new_voice_id) in client.available_voices:
                            client.current_voice_id = new_voice_id
                            voice_name = client.available_voices[str(new_voice_id)]
                            print(f"✅ Voice changed to {voice_name}")
                        else:
                            print("❌ Invalid voice ID")
                    except ValueError:
                        print("❌ Please enter a valid number")

            elif choice == "4":
                if client.is_connected:
                    print("✅ Still connected to server")
                    print(f"Available voices: {len(client.available_voices)}")
                    client.show_current_voice()
                else:
                    print("❌ Not connected - need to reconnect")

            elif choice == "5":
                await client.disconnect()
                break

            else:
                print("❌ Invalid choice! Please enter 1-5")

    except KeyboardInterrupt:
        print("\n🛑 Disconnecting...")
        await client.disconnect()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        await client.disconnect()


if __name__ == "__main__":
    print("🎵 TTS Client Starting...")
    print("Press Ctrl+C to exit at any time")
    asyncio.run(main())
