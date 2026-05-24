"""
Demonstrating the chatbot's functionality and validating its core components.
"""

import unittest
from Simple_chatbot import TraditionalChatbot


class TestChatbotFunctionality(unittest.TestCase):
    """Test cases for the traditional chatbot."""
    
    def setUp(self):
        """Set up a fresh chatbot instance for each test."""
        self.chatbot = TraditionalChatbot()
    
    def test_greeting_intent(self):
        """Test that greeting intents are correctly recognized."""
        response = self.chatbot.chat("hello")
        self.assertIsNotNone(response)
        self.assertGreater(len(response), 0)
    
    def test_name_extraction(self):
        """Test that user name is correctly extracted."""
        self.chatbot.chat("my name is alice")
        self.assertEqual(self.chatbot.user_name, "Alice")
    
    def test_name_in_response(self):
        """Test that extracted name appears in subsequent responses."""
        self.chatbot.chat("my name is bob")
        response = self.chatbot.chat("how are you doing")
        # The response should be personalized
        self.assertIsNotNone(response)
    
    def test_preprocessing(self):
        """Test input preprocessing."""
        processed = self.chatbot.preprocess_input("  HELLO  ")
        self.assertEqual(processed, "hello")
    
    def test_intent_recognition(self):
        """Test intent recognition for various inputs."""
        intent, _ = self.chatbot.recognize_intent("hello")
        self.assertEqual(intent, "greeting")
        
        intent, _ = self.chatbot.recognize_intent("goodbye")
        self.assertEqual(intent, "farewell")
        
        intent, _ = self.chatbot.recognize_intent("what's your name")
        self.assertEqual(intent, "name_inquiry")
    
    def test_conversation_history(self):
        """Test that conversation history is properly maintained."""
        self.chatbot.chat("hello")
        self.chatbot.chat("my name is charlie")
        self.chatbot.chat("goodbye")
        
        history = self.chatbot.get_conversation_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["user"], "hello")
        self.assertIn("intent", history[0])
    
    def test_empty_input(self):
        """Test handling of empty input."""
        response = self.chatbot.chat("")
        self.assertEqual(response, "Please say something!")
    
    def test_default_intent(self):
        """Test fallback to default intent for unknown input."""
        response = self.chatbot.chat("xyzabc random nonsense")
        self.assertIsNotNone(response)
        self.assertGreater(len(response), 0)
    
    def test_reset_functionality(self):
        """Test that reset clears history and name."""
        self.chatbot.chat("my name is diana")
        self.chatbot.chat("hello")
        self.chatbot.reset()
        
        self.assertIsNone(self.chatbot.user_name)
        self.assertEqual(len(self.chatbot.get_conversation_history()), 0)


class TestIntentPatterns(unittest.TestCase):
    """Test individual intent pattern matching."""
    
    def setUp(self):
        """Set up a fresh chatbot instance."""
        self.chatbot = TraditionalChatbot()
    
    def test_greeting_variations(self):
        """Test various greeting patterns."""
        greetings = ["hello", "hi", "hey", "good morning", "greetings"]
        for greeting in greetings:
            intent, _ = self.chatbot.recognize_intent(greeting)
            self.assertEqual(intent, "greeting", f"Failed for '{greeting}'")
    
    def test_farewell_variations(self):
        """Test various farewell patterns."""
        farewells = ["bye", "goodbye", "see you", "farewell"]
        for farewell in farewells:
            intent, _ = self.chatbot.recognize_intent(farewell)
            self.assertEqual(intent, "farewell", f"Failed for '{farewell}'")
    
    def test_name_extraction_patterns(self):
        """Test various name extraction patterns."""
        patterns = [
            ("my name is john", "John"),
            ("i'm jane", "Jane"),
            ("call me jack", "Jack"),
            ("i am jill", "Jill")
        ]
        for text, expected_name in patterns:
            chatbot = TraditionalChatbot()
            chatbot.chat(text)
            self.assertEqual(chatbot.user_name, expected_name, f"Failed for '{text}'")


def run_interactive_demo():
    """Run an interactive demonstration of the chatbot."""
    print("\n" + "=" * 60)
    print("Interactive Chatbot Demo")
    print("=" * 60 + "\n")
    
    chatbot = TraditionalChatbot()
    demo_inputs = [
        "hello",
        "my name is Emma",
        "how are you doing",
        "thank you",
        "what's your name",
        "what can you do",
        "i don't understand",
        "the weather is nice today",
        "goodbye"
    ]
    
    for user_input in demo_inputs:
        response = chatbot.chat(user_input)
        print(f"User: {user_input}")
        print(f"Bot:  {response}\n")
    
    print("\n" + "-" * 60)
    print("Conversation Statistics:")
    print("-" * 60)
    history = chatbot.get_conversation_history()
    print(f"Total turns: {len(history)}")
    print(f"User name recognized: {chatbot.user_name}")
    
    # Count intents
    intent_counts = {}
    for turn in history:
        intent = turn['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"\nIntent Distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_interactive_demo()
    else:
        # Run unit tests
        unittest.main(verbosity=2)
