import asyncio
import websockets
import json
import aiofiles
import time
import numpy as np
import sounddevice as sd
from websockets.exceptions import ConnectionClosed
import os
from datetime import datetime


class TTSClient:
    def __init__(self):
        self.websocket = None
        self.is_connected = False
        self.ping_task = None
        self.audio_stream = None
        self.keep_alive_interval = 5
        self.is_playing_audio = False
        self.available_voices = {}
        self.current_voice_id = 0
        self.conversation_history = []
        self.server_uri = "ws://hive.hussainkazarani.site/ws"

    async def connect(self):
        """Connect to the server"""
        try:
            print("🔗 Connecting to server...")
            self.websocket = await websockets.connect(
                self.server_uri, ping_interval=None, close_timeout=300, max_queue=2048
            )
            self.is_connected = True

            # Get initial messages
            await self._handle_initial_messages()

            # Start keep-alive
            self.ping_task = asyncio.create_task(self._continuous_keep_alive())

            print("✅ Connected successfully!")
            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    async def _handle_initial_messages(self):
        """Handle initial server messages"""
        try:
            # Get available voices
            voices_msg = await self.websocket.recv()
            voices_data = json.loads(voices_msg)

            if voices_data.get("type") == "available_voices":
                self.available_voices = voices_data.get("voices", {})
                self.current_voice_id = voices_data.get("default_voice", 0)
                print(f"🎤 Available voices: {len(self.available_voices)}")

            # Get status message
            status_msg = await self.websocket.recv()
            status_data = json.loads(status_msg)
            if status_data.get("type") == "status":
                print(f"ℹ️  {status_data.get('message')}")

        except Exception as e:
            print(f"❌ Error handling initial messages: {e}")

    async def _continuous_keep_alive(self):
        """Continuous keep-alive - pauses during audio playback"""
        while self.is_connected:
            try:
                if self.websocket:
                    await self.websocket.ping()
                await asyncio.sleep(self.keep_alive_interval)
            except:
                self.is_connected = False
                break

    async def _play_audio_chunk(self, audio_data, sample_rate):
        """Play audio chunk"""
        if self.audio_stream is None:
            self.audio_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            self.audio_stream.start()

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

    async def send_text(self, text):
        """Send text to server and handle response"""
        if not self.is_connected:
            print("❌ Not connected to server!")
            return False

        try:
            self.is_playing_audio = True

            # Add user message to conversation
            self._add_to_conversation("User", text)

            # Send text to server
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "text_message",
                        "text": text,
                        "voice_id": self.current_voice_id,
                    }
                )
            )

            print(f"📤 Sending: {text[:50]}...")

            # Process server responses
            return await self._process_server_responses()

        except ConnectionClosed:
            print("\n❌ Connection closed by server")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"\n❌ Error during text processing: {e}")
            return False
        finally:
            self.is_playing_audio = False
            await self._cleanup_audio()

    async def _process_server_responses(self):
        """Process all server responses for a single text input"""
        chunk_count = 0
        llm_response = None
        audio_complete = False

        while self.is_connected and not audio_complete:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
                data = json.loads(message)

                if data.get("type") == "llm_response":
                    llm_response = data.get("text", "")
                    print(f"\n🤖 AI: {llm_response}")
                    self._add_to_conversation("AI", llm_response)

                elif data.get("type") == "audio_chunk":
                    await self._play_audio_chunk(data["audio"], data["sample_rate"])
                    chunk_count += 1
                    print(f"▶️ Chunk {chunk_count}", end="\r")

                elif data.get("type") == "status":
                    print(f"ℹ️  {data.get('message')}")

                elif data.get("type") == "completion":
                    audio_complete = True
                    print(f"\n✅ Audio completed - {chunk_count} chunks")

                elif data.get("type") == "error":
                    print(f"\n❌ Error: {data.get('message')}")
                    return False

            except asyncio.TimeoutError:
                print("\n⏰ Timeout waiting for server response")
                return False

        return True

    def _add_to_conversation(self, speaker, text):
        """Add message to conversation history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.conversation_history.append(
            {"timestamp": timestamp, "speaker": speaker, "text": text}
        )

        # Keep only last 20 messages
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def show_conversation(self):
        """Display conversation history"""
        if not self.conversation_history:
            print("💬 No conversation history yet")
            return

        print("\n" + "=" * 60)
        print("💬 CONVERSATION HISTORY")
        print("=" * 60)

        for msg in self.conversation_history:
            color = (
                "32" if msg["speaker"] == "AI" else "34"
            )  # Green for AI, Blue for User
            print(f"\033[1;{color}m{msg['timestamp']} {msg['speaker']}:\033[0m")
            print(f"   {msg['text']}")
            print("-" * 40)

        print("=" * 60)

    def show_voices(self):
        """Show available voices"""
        if not self.available_voices:
            print("❌ No voices available")
            return

        print("\n🎤 AVAILABLE VOICES")
        print("=" * 30)
        for voice_id, voice_name in self.available_voices.items():
            current = " ← CURRENT" if voice_id == str(self.current_voice_id) else ""
            print(f"  {voice_id}: {voice_name}{current}")
        print("=" * 30)

    def change_voice(self):
        """Change current voice"""
        self.show_voices()

        try:
            new_voice_id = int(input("\nEnter voice ID to change to: ").strip())
            if str(new_voice_id) in self.available_voices:
                self.current_voice_id = new_voice_id
                voice_name = self.available_voices[str(new_voice_id)]
                print(f"✅ Voice changed to: {voice_name}")
            else:
                print("❌ Invalid voice ID")
        except ValueError:
            print("❌ Please enter a valid number")

    async def disconnect(self):
        """Cleanly disconnect from server"""
        print("\n👋 Disconnecting...")

        self.is_connected = False

        # Cancel ping task
        if self.ping_task:
            self.ping_task.cancel()
            try:
                await self.ping_task
            except:
                pass

        # Close websocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        # Cleanup audio
        await self._cleanup_audio()

        print("✅ Disconnected successfully")


async def main():
    client = TTSClient()

    # Try to connect
    if not await client.connect():
        return

    try:
        while True:
            # Clear screen for better UI
            os.system("cls" if os.name == "nt" else "clear")

            print("\n" + "=" * 60)
            print("🎵 TTS CLIENT - Voice Conversation System")
            print("=" * 60)
            print("1. 💬 Send text message")
            print("2. 🎤 Change voice")
            print("3. 📜 Show conversation")
            print("4. 🚪 Disconnect & exit")
            print("=" * 60)

            # Show current status
            voice_name = client.available_voices.get(
                str(client.current_voice_id), "Unknown"
            )
            print(f"🔊 Current voice: {client.current_voice_id} ({voice_name})")
            print(f"🔗 Connected: {'✅' if client.is_connected else '❌'}")
            print(f"💬 Messages: {len(client.conversation_history)}")
            print("=" * 60)

            choice = input("\nChoose option (1-4): ").strip()

            if choice == "1":
                text = input("Enter your message: ").strip()
                if text:
                    await client.send_text(text)
                    input("\nPress Enter to continue...")
                else:
                    print("❌ Please enter some text!")
                    await asyncio.sleep(1)

            elif choice == "2":
                client.change_voice()
                await asyncio.sleep(1)

            elif choice == "3":
                client.show_conversation()
                input("\nPress Enter to continue...")

            elif choice == "4":
                await client.disconnect()
                break

            else:
                print("❌ Invalid choice! Please enter 1-4")
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Keyboard interrupt detected...")
        await client.disconnect()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        await client.disconnect()


if __name__ == "__main__":
    print("🎵 Starting TTS Client...")
    print("Press Ctrl+C to exit at any time")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
