class Message {
  final String text;
  final MessageType type;

  Message({required this.text, required this.type});
}

enum MessageType {
  user, // User message
  aiText, // AI text response
}
