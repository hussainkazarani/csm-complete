// WORKS
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'data.dart';

final channel = WebSocketChannel.connect(
  Uri.parse("ws://hive.hussainkazarani.site/ws"),
);
final player = FlutterSoundPlayer();

void sendToServer(int index) {
  final message = {"type": "text_message", "text": texts[index], "voice_id": 0};
  channel.sink.add(jsonEncode(message));
}

void listenStream() {
  channel.stream.listen((event) {
    final data = jsonDecode(event);

    switch (data["type"]) {
      case "audio_status":
        print("Status: ${data["status"]}");
        break;

      case "audio_chunk":
        handleAudioChunk(data["audio"]);
        break;

      case "completion":
        print("Audio generation complete");
        break;

      default:
        print("Received: $data");
    }
  });
}

Future<void> initPlayer() async {
  await player.openPlayer();
  await player.startPlayerFromStream(
    codec: Codec.pcm16,
    numChannels: 1,
    sampleRate: 24000,
    interleaved: false,
    bufferSize: 1024,
  );
}

void disposeAudio() {
  channel.sink.close();
  player.closePlayer();
}

void handleAudioChunk(List<dynamic> floatList) {
  // Convert Float32 PCM → Int16 PCM
  final floats = Float32List.fromList(floatList.cast<double>());
  final int16 = Int16List(floats.length);

  for (int i = 0; i < floats.length; i++) {
    final v = (floats[i] * 32767.0).clamp(-32768.0, 32767.0);
    int16[i] = v.toInt();
  }

  // Feed into flutter_sound using the updated API
  if (player.uint8ListSink != null) {
    player.uint8ListSink!.add(int16.buffer.asUint8List());
  }
}
