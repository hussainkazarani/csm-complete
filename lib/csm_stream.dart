import 'package:bubble/bubble.dart';
import 'package:csm_app/messages.dart';
import 'package:csm_app/stream_logic.dart';
import 'package:flutter/material.dart';

class CSMStream extends StatefulWidget {
  const CSMStream({super.key});

  @override
  State<CSMStream> createState() => _CSMStreamState();
}

class _CSMStreamState extends State<CSMStream> {
  final streamLogic = StreamLogic();
  final TextEditingController _controller = TextEditingController();
  List<Message> messages = [];

  void sendText(String text) {
    if (text.trim().isEmpty) return;
    _controller.clear();

    setState(() {
      messages.add(Message(text: text, isUser: true));
    });

    streamLogic.sendText(text);
  }

  @override
  void initState() {
    super.initState();
    streamLogic.connect(); // Connect once at the start
  }

  @override
  void dispose() {
    streamLogic.disconnect(); // Clean up on exit
    super.dispose();
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
                  itemCount: messages.length,
                  itemBuilder: (context, index) {
                    final message = messages[index];
                    if (message.isUser) {
                      return userBubble(context, message.text);
                    } else {
                      return aiBubble(context, message.text);
                    }
                  },
                ),
              ),
              chatRow(),
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

  Bubble userBubble(BuildContext context, String text) {
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

  Bubble aiBubble(BuildContext context, String text) {
    return Bubble(
      alignment: Alignment.topLeft,
      nip: BubbleNip.leftTop,
      nipRadius: 4.0,
      nipWidth: 15,
      nipHeight: 15,
      color: Color.fromRGBO(212, 234, 244, 1.0),
      padding: BubbleEdges.all(12),
      margin: BubbleEdges.only(right: MediaQuery.of(context).size.width * 0.10),
      child: Text(text),
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
