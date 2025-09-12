// import 'dart:convert';
// import 'dart:typed_data';
// import 'package:mp_audio_stream/mp_audio_stream.dart';
// import 'package:web_socket_channel/web_socket_channel.dart';
// import 'data.dart';

// final channel = WebSocketChannel.connect(
//   Uri.parse("ws://hive.hussainkazarani.site/ws"),
// );
// final audioStream = getAudioStream();

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
//         // final List<dynamic> chunk = data["audio"];
//         // Convert JSON array -> Float32List
//         final floats = Float32List.fromList(data["audio"].cast<double>());

//         // Push chunk into player
//         audioStream.push(floats);
//         break;

//       case "completion":
//         print("Audio generation complete");
//         break;

//       default:
//         print("Received: $data");
//     }
//   });
// }

// void initAudio() {
//   // match server: mono @ 24000 Hz
//   audioStream.init(
//     channels: 1,
//     sampleRate: 24000,
//     bufferMilliSec: 1000,
//     waitingBufferMilliSec: 100,
//   );
// }

// void disposeAudio() {
//   audioStream.uninit();
//   channel.sink.close();
// }
