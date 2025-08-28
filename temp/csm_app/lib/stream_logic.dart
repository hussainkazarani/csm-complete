import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'messages.dart';

class StreamLogic {
  WebSocketChannel? _channel;
  FlutterSoundPlayer? _player;
  int? _sampleRate;
  bool _isConnected = false;
  Function(Message)? onNewMessage;
  Function(String)? onStatusUpdate;
  Function(String)? onError;

  void connect({
    Function(Message)? onMessage,
    Function(String)? onStatus,
    Function(String)? onError,
  }) {
    if (_isConnected) return;

    this.onNewMessage = onMessage;
    this.onStatusUpdate = onStatus;
    this.onError = onError;

    _channel = WebSocketChannel.connect(
      Uri.parse('ws://csm-stream.hussainkazarani.site/ws'),
    );

    _channel!.stream.listen(
      _handleMessage,
      onError: (error) {
        print("WebSocket error: $error");
        if (this.onError != null) {
          this.onError!("Connection error: $error");
        }
      },
      onDone: () {
        print("WebSocket disconnected");
        _isConnected = false;
      },
    );

    _isConnected = true;
    print("Connected ✅");
  }

  void sendText(String text) {
    if (_channel == null) return;
    _channel!.sink.add(jsonEncode({"type": "text_message", "text": text}));
  }

  Future<void> _handleMessage(dynamic message) async {
    try {
      final data = jsonDecode(message);

      if (data["type"] == "llm_response") {
        // AI text response
        if (onNewMessage != null) {
          onNewMessage!(Message(text: data["text"], type: MessageType.aiText));
        }
      } else if (data["type"] == "status") {
        // Status message
        if (onStatusUpdate != null) {
          onStatusUpdate!(data["message"]);
        }
      } else if (data["type"] == "error") {
        // Error message
        if (onError != null) {
          onError!(data["message"]);
        }
      } else if (data["type"] == "audio_chunk") {
        // Audio chunk - handle playback
        await _playChunk(data);
      } else if (data["type"] == "audio_status") {
        // Audio status updates
        if (data["status"] == "generating" && onStatusUpdate != null) {
          onStatusUpdate!("Generating audio...");
        } else if (data["status"] == "complete" && onStatusUpdate != null) {
          onStatusUpdate!("Audio complete");
        } else if (data["status"] == "first_chunk" && onStatusUpdate != null) {
          onStatusUpdate!("Playing audio...");
        }
      }
    } catch (e) {
      print("Error handling message: $e");
      if (onError != null) {
        onError!("Failed to process message: $e");
      }
    }
  }

  Future<void> _playChunk(Map<String, dynamic> data) async {
    try {
      // Initialize player on first chunk
      if (_player == null) {
        _sampleRate = data["sample_rate"];
        _player = FlutterSoundPlayer();
        await _player!.openPlayer();
        await _player!.startPlayerFromStream(
          codec: Codec.pcm16,
          sampleRate: _sampleRate!,
          numChannels: 1,
          interleaved: true,
          bufferSize: 4096,
        );
        print("Audio player initialized with sample rate: $_sampleRate");
      }

      // Convert and play the audio chunk
      final pcmData = _convertToPCM16(data["audio"]);
      _player!.feedFromStream(pcmData);

      // Optional: log chunk info
      if (data["chunk_num"] != null) {
        print("Playing audio chunk ${data['chunk_num']}");
      }
    } catch (e) {
      print("Error playing audio chunk: $e");
      if (onError != null) {
        onError!("Audio playback error: $e");
      }
    }
  }

  Uint8List _convertToPCM16(List<dynamic> float32) {
    final buffer = Uint8List(float32.length * 2);
    final byteData = ByteData.view(buffer.buffer);

    for (int i = 0; i < float32.length; i++) {
      double sample = float32[i].toDouble();
      int pcm16 = (sample * 32767).clamp(-32768, 32767).toInt();
      byteData.setUint16(i * 2, pcm16, Endian.little);
    }
    return buffer;
  }

  Future<void> disconnect() async {
    try {
      await _player?.stopPlayer();
      await _player?.closePlayer();
      await _channel?.sink.close();
      _isConnected = false;
      print("Disconnected");
    } catch (e) {
      print("Error during disconnect: $e");
    }
  }

  bool get isConnected => _isConnected;
}
