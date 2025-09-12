// import 'dart:convert';
// import 'dart:typed_data';
// import 'package:audioplayers/audioplayers.dart';
// import 'package:flutter_sound/flutter_sound.dart';
// import 'package:flutter_sound/public/flutter_sound_player.dart';
// import 'package:mp_audio_stream/mp_audio_stream.dart';
// import 'package:web_socket_channel/web_socket_channel.dart';
// import 'data.dart';

// final channel = WebSocketChannel.connect(
//   Uri.parse("ws://hive.hussainkazarani.site/ws"),
// );
// final audioPlayer = AudioPlayer();

// void sendToServer(int index) {
//   final message = {"type": "text_message", "text": texts[index], "voice_id": 0};
//   channel.sink.add(jsonEncode(message));
// }

// void listenStream() {
//   channel.stream.listen((event) {
//     final data = jsonDecode(event);

//     switch (data["type"]) {
//       case "audio_status":
//         print("Status: ${data["status"]}");
//         break;

//       case "audio_chunk":
//         handleAudioChunk(data["audio"]);
//         break;

//       case "completion":
//         print("Audio generation complete");
//         break;

//       default:
//         print("Received: $data");
//     }
//   });
// }

// Future<void> initPlayer() async {
//   // Set up for low latency streaming
//   await audioPlayer.setPlayerMode(PlayerMode.lowLatency);
//   await audioPlayer.setReleaseMode(ReleaseMode.stop);
// }

// void disposeAudio() {
//   channel.sink.close();
//   audioPlayer.stop();
//   audioPlayer.dispose();
// }

// void handleAudioChunk(List<dynamic> floatList) {
//   // Convert Float32 PCM → Int16 PCM
//   final int16 = Int16List(floatList.length);

//   for (int i = 0; i < floatList.length; i++) {
//     final sample = floatList[i].toDouble();
//     int16[i] = (sample * 32767.0).clamp(-32768.0, 32767.0).toInt();
//   }

//   // Play immediately - no buffering
//   audioPlayer.play(
//     BytesSource(int16.buffer.asUint8List()),
//     mode: PlayerMode.lowLatency,
//   );
// }
