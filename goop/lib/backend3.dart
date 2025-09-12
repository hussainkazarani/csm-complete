// import 'dart:convert';
// import 'dart:typed_data';
// import 'package:just_audio/just_audio.dart';
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
//   // Initialize with empty source
//   await audioPlayer.setAudioSources([]);
// }

// void disposeAudio() {
//   channel.sink.close();
//   audioPlayer.dispose();
// }

// void handleAudioChunk(List<dynamic> floatList) {
//   // Convert Float32 PCM → Int16 PCM
//   final int16 = Int16List(floatList.length);

//   for (int i = 0; i < floatList.length; i++) {
//     final sample = floatList[i].toDouble();
//     int16[i] = (sample * 32767.0).clamp(-32768.0, 32767.0).toInt();
//   }

//   // Convert to Uint8List and create a data URI
//   final audioData = int16.buffer.asUint8List();
//   final dataUri = Uri.dataFromBytes(
//     audioData,
//     mimeType: 'audio/pcm',
//     parameters: {'rate': '24000', 'channels': '1'},
//   );

//   // Play the audio using the data URI
//   audioPlayer.setAudioSource(AudioSource.uri(dataUri));
// }
