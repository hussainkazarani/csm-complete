import 'dart:convert';
import 'package:flutter/material.dart';
import 'backend.dart';
import 'data.dart';

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  int currentText = 1;

  @override
  void initState() {
    // initAudio();
    initPlayer();
    listenStream();
    sendToServer(currentText);
    super.initState();
  }

  @override
  void dispose() {
    disposeAudio();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        appBar: AppBar(
          backgroundColor: Colors.black,
          title: Text('Demo', style: TextStyle(color: Colors.white)),
          centerTitle: true,
        ),
        body: Column(
          children: [
            Expanded(
              child: Center(
                child: Padding(
                  padding: const EdgeInsets.all(15.0),
                  child: Text(texts[currentText], textAlign: TextAlign.justify),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(bottom: 30),
              child: Column(
                children: [
                  TextButton(
                    onPressed: () {
                      if (currentText < texts.length - 1) {
                        setState(() {
                          currentText++;
                          sendToServer(currentText);
                        });
                      } else {
                        setState(() {
                          currentText = 1;
                          sendToServer(currentText);
                        });
                      }
                    },
                    child: Container(
                      padding: EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 5,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.green.shade900,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        "next",
                        style: TextStyle(color: Colors.white, fontSize: 24),
                      ),
                    ),
                  ),
                  Text("$currentText/${texts.length - 1}"),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
