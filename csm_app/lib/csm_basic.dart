import 'dart:io';
import 'dart:typed_data';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

String url = "https://csm-basic.hussainkazarani.site/csm-basic";

class CSMBasic extends StatefulWidget {
  const CSMBasic({super.key});
  @override
  State<CSMBasic> createState() => _CSMBasicState();
}

class _CSMBasicState extends State<CSMBasic> {
  final TextEditingController _controller = TextEditingController();
  bool isLoading = false;
  double? timeTaken; // in seconds
  Uint8List? audioBytes; // to store the downloaded audio
  final AudioPlayer _audioPlayer = AudioPlayer();

  @override
  void dispose() {
    _controller.dispose();
    _audioPlayer.dispose();
    super.dispose();
  }

  //------------ LOGIC ------------
  Future<void> generateClonedVoice() async {
    if (_controller.text.isEmpty) return;

    setState(() {
      isLoading = true;
      timeTaken = null;
      audioBytes = null;
    });

    final start = DateTime.now();

    try {
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: '{"text": "${_controller.text}"}',
      );

      final end = DateTime.now();
      setState(() {
        timeTaken = end.difference(start).inMilliseconds / 1000.0;
      });

      if (response.statusCode == 200) {
        setState(() {
          audioBytes = response.bodyBytes;
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: ${response.statusCode}')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error: $e')));
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  Widget audioContainer() {
    if (isLoading) {
      return Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: const [
          SizedBox(
            width: 20,
            height: 20,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          SizedBox(width: 16),
          Text("Generating audio..."),
        ],
      );
    } else if (audioBytes != null) {
      return Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text("Time: ${timeTaken?.toStringAsFixed(2)}s"),
          IconButton(
            icon: const Icon(Icons.play_arrow),
            onPressed: () async {
              if (audioBytes == null) return;

              // Save bytes to a temporary file
              final tempDir = await getTemporaryDirectory();
              final filePath = '${tempDir.path}/temp_audio.mp3';
              final file = File(filePath);
              await file.writeAsBytes(audioBytes!);

              // Play from file path
              await _audioPlayer.play(DeviceFileSource(filePath));
            },
          ),
        ],
      );
    } else {
      return const SizedBox.shrink(); // empty container if nothing
    }
  }

  //------------ UI ------------
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => FocusScope.of(context).unfocus(),
      child: Scaffold(
        appBar: customAppBar(),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              customTextHeader(),
              const SizedBox(height: 16),
              customTextField(),
              const SizedBox(height: 16),
              customAudio(),
              const SizedBox(height: 16),
              customButton(),
            ],
          ),
        ),
      ),
    );
  }

  //------------ WIDGETS ------------
  AppBar customAppBar() {
    return AppBar(
      title: const Text(
        "Blocking CSM",
        style: TextStyle(color: Colors.white, fontSize: 25),
      ),
      backgroundColor: Colors.black.withValues(alpha: 0.3),
    );
  }

  Widget customTextHeader() {
    return Text(
      "Enter your text here:",
      style: TextStyle(fontWeight: FontWeight.w800, fontSize: 20),
    );
  }

  Widget customTextField() {
    return TextField(
      controller: _controller,
      maxLength: 300,
      maxLines: null,
      textInputAction: TextInputAction.done,
      autocorrect: false,
      enableSuggestions: false,
      keyboardType: TextInputType.text,
      decoration: InputDecoration(
        hintText: "Type something...",
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: Colors.grey),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: Colors.black),
        ),
      ),
    );
  }

  Container customAudio() {
    return Container(
      padding: const EdgeInsets.all(12),
      width: double.infinity,
      height: 65,
      decoration: BoxDecoration(
        color: Colors.grey[200],
        borderRadius: BorderRadius.circular(12),
      ),
      child: audioContainer(),
    );
  }

  Widget customButton() {
    return Center(
      child: TextButton(
        onPressed: generateClonedVoice,
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            color: Colors.black,
          ),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 15, horizontal: 10),
            child: Text(
              "Get Cloned Voice",
              style: TextStyle(color: Colors.white),
            ),
          ),
        ),
      ),
    );
  }
}
