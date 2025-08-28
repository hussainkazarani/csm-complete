import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class StreamLogic {
  WebSocketChannel? _channel;
  FlutterSoundPlayer? _player;
  int? _sampleRate;
  bool _isConnected = false;

  void connect() {
    if (_isConnected) return;

    _channel = WebSocketChannel.connect(
      Uri.parse('ws://csm-stream.hussainkazarani.site/ws'),
    );

    _channel!.stream.listen(_handleMessage);
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

      if (data["type"] == "audio_chunk") {
        await _playChunk(data);
      }

      if (data["type"] == "complete") {
        print("Done");
        await _player?.stopPlayer();
      }
    } catch (e) {
      print("Error: $e");
    }
  }

  Future<void> _playChunk(Map<String, dynamic> data) async {
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
    }

    // Convert and play
    final pcmData = _convertToPCM16(data["audio"]);
    _player!.feedFromStream(pcmData);
    print("Chunk ${data['chunk_num']}");
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
    await _player?.stopPlayer();
    await _player?.closePlayer();
    await _channel?.sink.close();
    _isConnected = false;
  }
}
