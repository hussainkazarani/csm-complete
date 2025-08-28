import 'package:bubble/bubble.dart';
import 'package:csm_app/messages.dart';
import 'package:flutter/material.dart';
import 'stream_logic.dart';

class CSMStream extends StatefulWidget {
  const CSMStream({super.key});

  @override
  State<CSMStream> createState() => _CSMStreamState();
}

class _CSMStreamState extends State<CSMStream> {
  final streamLogic = StreamLogic();
  final TextEditingController _controller = TextEditingController();
  List<Message> messages = [];
  bool _isGeneratingAudio = false;
  String _currentMessage = '';
  bool _isError = false;

  @override
  void initState() {
    super.initState();
    streamLogic.connect(
      onMessage: handleNewMessage,
      onStatus: handleStatusUpdate,
      onError: handleError,
    );
  }

  @override
  void dispose() {
    streamLogic.disconnect();
    super.dispose();
  }

  void sendText(String text) {
    if (text.trim().isEmpty) return;
    _controller.clear();

    // Add user message immediately
    setState(() {
      messages.add(Message(text: text, type: MessageType.user));
    });

    // Send to server
    streamLogic.sendText(text);
  }

  void handleNewMessage(Message message) {
    setState(() {
      if (message.type == MessageType.aiText) {
        _isGeneratingAudio = true;
        _currentMessage = 'Generating audio...';
        _isError = false;
      }
      messages.add(message);
    });
  }

  void handleStatusUpdate(String status) {
    setState(() {
      _currentMessage = status;
      _isError = false;

      // Handle different status messages
      if (status.contains('complete') || status.isEmpty) {
        _isGeneratingAudio = false;
        _currentMessage = '';
      } else if (status.contains('Generating audio') ||
          status.contains('Processing')) {
        _isGeneratingAudio = true;
      } else if (status.contains('Thinking')) {
        _isGeneratingAudio =
            false; // Thinking doesn't mean audio is generating yet
      }
    });
  }

  void handleError(String error) {
    setState(() {
      _currentMessage = error;
      _isError = true;
      _isGeneratingAudio = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => FocusScope.of(context).unfocus(),
      child: Scaffold(
        appBar: customAppBar(),
        body: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: ListView.builder(
                  itemCount:
                      messages.length + (_currentMessage.isNotEmpty ? 1 : 0),
                  itemBuilder: (context, index) {
                    if (index < messages.length) {
                      final message = messages[index];
                      return _buildMessageBubble(context, message);
                    } else {
                      if (_isError) errorBubble(context, _currentMessage);
                    }
                  },
                ),
              ),
              if (_isGeneratingAudio) loadingAudio(),
              chatRow(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMessageBubble(BuildContext context, Message message) {
    switch (message.type) {
      case MessageType.user:
        return userBubble(context, message.text);
      case MessageType.aiText:
        return aiBubble(context, message.text);
    }
  }

  Padding loadingAudio() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: LinearProgressIndicator(
        backgroundColor: Colors.grey[300],
        valueColor: AlwaysStoppedAnimation<Color>(Colors.deepPurple),
      ),
    );
  }

  //------------ WIDGETS ------------
  AppBar customAppBar() {
    return AppBar(
      title: const Text(
        "Streaming CSM",
        style: TextStyle(color: Colors.white, fontSize: 25),
      ),
      backgroundColor: Colors.black.withValues(alpha: 0.3),
    );
  }

  Widget customTextField() {
    return TextField(
      controller: _controller,
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

  Widget customButton() {
    return IconButton(
      onPressed: () => sendText(_controller.text),
      icon: Container(
        padding: EdgeInsets.all(15),
        decoration: BoxDecoration(
          color: Colors.deepPurple,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(Icons.send, color: Colors.white),
      ),
    );
  }

  Widget userBubble(BuildContext context, String text) {
    return Bubble(
      alignment: Alignment.topRight,
      nip: BubbleNip.rightTop,
      nipRadius: 4.0,
      nipWidth: 15,
      nipHeight: 15,
      color: Color.fromRGBO(225, 255, 199, 1.0),
      padding: BubbleEdges.all(12),
      margin: BubbleEdges.only(
        left: MediaQuery.of(context).size.width * 0.10,
        bottom: 15,
      ),
      child: Text(text),
    );
  }

  Widget aiBubble(BuildContext context, String text) {
    return Bubble(
      alignment: Alignment.topLeft,
      nip: BubbleNip.leftTop,
      nipRadius: 4.0,
      nipWidth: 15,
      nipHeight: 15,
      color: Color.fromRGBO(212, 234, 244, 1.0),
      padding: BubbleEdges.all(12),
      margin: BubbleEdges.only(
        right: MediaQuery.of(context).size.width * 0.10,
        bottom: 15,
      ),
      child: Text(text),
    );
  }

  Widget statusBubble(BuildContext context, String text) {
    return Bubble(
      alignment: Alignment.center,
      nip: BubbleNip.no,
      color: Colors.grey[200],
      padding: const BubbleEdges.all(12),
      margin: const BubbleEdges.only(bottom: 15),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(
            width: 16,
            height: 16,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          const SizedBox(width: 8),
          Text(
            text,
            style: const TextStyle(
              fontStyle: FontStyle.italic,
              color: Colors.grey,
            ),
          ),
        ],
      ),
    );
  }

  Widget errorBubble(BuildContext context, String text) {
    return Bubble(
      alignment: Alignment.center,
      nip: BubbleNip.no,
      color: const Color.fromRGBO(255, 200, 200, 1.0),
      padding: const BubbleEdges.all(12),
      margin: const BubbleEdges.only(bottom: 10),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(
            Icons.error,
            color: Color.fromARGB(255, 169, 46, 37),
            size: 16,
          ),
          const SizedBox(width: 8),
          Text(
            text,
            style: const TextStyle(color: Color.fromARGB(255, 169, 46, 37)),
          ),
        ],
      ),
    );
  }

  Widget chatRow() {
    return Padding(
      padding: EdgeInsets.only(bottom: 12, left: 15, right: 10),
      child: Row(
        children: [Expanded(child: customTextField()), customButton()],
      ),
    );
  }
}
