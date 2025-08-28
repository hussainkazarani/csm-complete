import 'package:csm_app/csm_basic.dart';
import 'package:csm_app/csm_stream.dart';
import 'package:flutter/material.dart';

class Home extends StatelessWidget {
  const Home({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'CSM',
          style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white),
        ),
        backgroundColor: Colors.blueGrey.shade700,
      ),
      body: PageView(
        scrollDirection: Axis.horizontal,
        children: [CSMBasic(), CSMStream()],
      ),
    );
  }
}
