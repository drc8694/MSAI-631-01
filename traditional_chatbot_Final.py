import re
import json
import random
from typing import Dict, List, Tuple
from datetime import datetime

class TraditionalChatbot:
    """A rule-based chatbot using pattern matching and intent recognition."""

    def __init__(self):
        """Initialize the chatbot with intents, patterns, and responses."""
        self.conversation_history = []
        self.user_name = None
        self.intents = self._load_intents()


    def _load_intents(self) -> Dict:
        """Load intent definitions with patterns and response templates."""
        return {
            "greeting": {
                "patterns": [
                    r"(hello|hi|hey|heyo|hello there|hi there|greetings|good morning|good afternoon|good evening)",
                    r"what's up",
                    r"howdy"
                ],
                "responses": [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Greetings! What brings you here?",
                    "Hey! Great to see you. How can I assist?"
                ]
            },
            "farewell": {
                "patterns": [
                    r"(bye|goodbye|see you|farewell|exit|quit)",
                    r"(goodbye|bye) now",
                    r"take care"
                ],
                "responses": [
                    "Goodbye! Have a great day!",
                    "See you later!",
                    "Thanks for chatting with me. Take care!",
                    "Farewell! Feel free to come back anytime."
                ]
            },
            "name_introduction": {
                "patterns": [
                    r"my name is ([a-z]+)",
                    r"i'm ([a-z]+)",
                    r"call me ([a-z]+)",
                    r"i am ([a-z]+)"
                ],
                "responses": [
                    "Nice to meet you, {name}! I'll remember that.",
                    "Great to meet you, {name}!",
                    "Hello {name}! That's a great name."
                ]
            },
            "how_are_you": {
                "patterns": [
                    r"how are you",
                    r"how's it going",
                    r"how're you doing",
                    r"what's your status"
                ],
                "responses": [
                    "I'm doing great, thanks for asking!",
                    "All systems operational! How about you?",
                    "I'm functioning perfectly. How are you?",
                    "Excellent! Ready to help you with whatever you need."
                ]
            },
            "help_request": {
                "patterns": [
                    r"(help|assist|support|aid)",
                    r"can you help",
                    r"i need help",
                    r"what can you do"
                ],
                "responses": [
                    "I can help you with greeting exchanges, answer questions about myself, and have a conversation!",
                    "I'm here to chat and help! You can greet me, ask how I'm doing, or just have a conversation.",
                    "I can assist with conversation, answer questions, and help with basic information."
                ]
            },
            "name_inquiry": {
                "patterns": [
                    r"what's your name",
                    r"who are you",
                    r"what are you called",
                    r"do you have a name"
                ],
                "responses": [
                    "I'm ChatBot, a traditional rule-based conversational AI!",
                    "You can call me ChatBot. I'm here to chat with you!",
                    "I'm ChatBot, a simple pattern-matching chatbot."
                ]
            },
            "thanks": {
                "patterns": [
                    r"(thank you|thanks|appreciate it|cheers)",
                    r"thanks[!]",
                    r"much obliged"
                ],
                "responses": [
                    "You're welcome!",
                    "Happy to help!",
                    "Anytime! Feel free to reach out again.",
                    "My pleasure!"
                ]
            },
            "confused": {
                "patterns": [
                    r"(confused|don'?t understand|what)",
                    r"what do you mean",
                    r"can you repeat",
                    r"i didn't get that"
                ],
                "responses": [
                    "I apologize for any confusion. Could you rephrase that?",
                    "Let me clarify. Could you say that again?",
                    "I'm not sure I understood. Can you try explaining differently?"
                ]
            },
            "time_inquiry": {
                "patterns": [
                    r"what's the time",
                    r"what time is it",
                    r"tell me the time",
                    r"current time"
                ],
                "responses": [
                    "The current time is {time}.",
                    "It's currently {time}.",
                    "Right now it's {time}."
                ]
            },
            "default": {
                "patterns": [],
                "responses": [
                    "That's interesting! Tell me more.",
                    "I see. Can you elaborate?",
                    "I'm not entirely sure how to respond to that, but I'm listening!",
                    "Interesting point. What else is on your mind?"
                ]
            }
        }

print('TraditionalChatbot class created!')

# Add methods to TraditionalChatbot class

def preprocess_input(self, user_input: str) -> str:
    """Preprocess user input: convert to lowercase and remove extra whitespace."""
    return user_input.lower().strip()

def recognize_intent(self, user_input: str) -> Tuple[str, List[str]]:
    """Recognize user intent by matching patterns against input."""
    for intent_name, intent_data in self.intents.items():
        if intent_name == "default":
            continue
        for pattern in intent_data["patterns"]:
            match = re.search(pattern, user_input)
            if match:
                return intent_name, list(match.groups())
    return "default", []

def extract_name(self, user_input: str, match_groups: List[str]) -> None:
    """Extract and store the user's name if provided."""
    if match_groups and match_groups[0]:
        self.user_name = match_groups[0].capitalize()

def generate_response(self, intent_name: str, match_groups: List[str]) -> str:
    """Generate a response based on recognized intent."""
    intent_responses = self.intents[intent_name]["responses"]
    response = random.choice(intent_responses)

    if "{name}" in response and self.user_name:
        response = response.format(name=self.user_name)

    if "{time}" in response:
        current_time = datetime.now().strftime("%I:%M %p")
        response = response.format(time=current_time)

    return response

def chat(self, user_input: str) -> str:
    """Main chatbot interaction method."""
    if not user_input.strip():
        return "Please say something!"

    processed_input = self.preprocess_input(user_input)
    intent_name, match_groups = self.recognize_intent(processed_input)

    if intent_name == "name_introduction":
        self.extract_name(processed_input, match_groups)

    response = self.generate_response(intent_name, match_groups)

    self.conversation_history.append({
        "timestamp": datetime.now().isoformat(),
        "user": user_input,
        "bot": response,
        "intent": intent_name
    })

    return response

def get_conversation_history(self) -> List[Dict]:
    """Retrieve the conversation history."""
    return self.conversation_history

def reset(self) -> None:
    """Reset the chatbot's conversation history and user name."""
    self.conversation_history = []
    self.user_name = None

# Attach methods to the class
TraditionalChatbot.preprocess_input = preprocess_input
TraditionalChatbot.recognize_intent = recognize_intent
TraditionalChatbot.extract_name = extract_name
TraditionalChatbot.generate_response = generate_response
TraditionalChatbot.chat = chat
TraditionalChatbot.get_conversation_history = get_conversation_history
TraditionalChatbot.reset = reset

print('All methods added to TraditionalChatbot class!')

"""Let's create an instance and test it with some sample inputs."""

# Initialize the chatbot
chatbot = TraditionalChatbot()
print('Chatbot initialized and ready to chat!\n')

# Run demo conversation
demo_inputs = [
    "Hello!",
    "My name is Alice",
    "How are you doing?",
    "What's your name?",
    "What can you do?",
    "Thank you!",
    "Goodbye!"
]

print("="*60)
print("DEMO CONVERSATION")
print("="*60 + "\n")

for user_input in demo_inputs:
    response = chatbot.chat(user_input)
    print(f"You: {user_input}")
    print(f"Bot: {response}\n")

print("="*60)
print(f"Conversation Statistics:")
print(f"  Total turns: {len(chatbot.get_conversation_history())}")
print(f"  User name recognized: {chatbot.user_name}")
print("="*60)

# Analyze intent distribution
history = chatbot.get_conversation_history()
intent_counts = {}

for turn in history:
    intent = turn['intent']
    intent_counts[intent] = intent_counts.get(intent, 0) + 1

print("\nIntent Distribution:")
print("-" * 40)
for intent, count in sorted(intent_counts.items()):
    print(f"  {intent:20s}: {count} occurrence(s)")
print("-" * 40)

# INTERACTIVE CHAT MODE
# Uncomment the code below and run this cell to chat interactively
# Type 'quit' or 'bye' to exit

# Create a fresh chatbot for interaction
interactive_chatbot = TraditionalChatbot()

print("\n" + "="*60)
print("INTERACTIVE CHATBOT")
print("="*60)
print("\nChat with the bot! (Type your message below)")
print("\nExample inputs to try:")
print("  - 'Hello'")
print("  - 'My name is [Your Name]'")
print("  - 'How are you?'")
print("  - 'What's your name?'")
print("  - 'What can you do?'")
print("  - 'Thank you'")
print("  - 'Goodbye'")
print("\n" + "="*60 + "\n")

# This allows manual input
# For Google Colab, input() will prompt you to type
# For Jupyter, you may need to uncomment individual cells below

# Example 1: Greeting
user_msg = "Hello there!"
response = interactive_chatbot.chat(user_msg)
print(f"You: {user_msg}")
print(f"Bot: {response}\n")

# Example 2: Name introduction
user_msg = "My name is Sarah"
response = interactive_chatbot.chat(user_msg)
print(f"You: {user_msg}")
print(f"Bot: {response}\n")

# Example 3: Status inquiry
user_msg = "How are you doing?"
response = interactive_chatbot.chat(user_msg)
print(f"You: {user_msg}")
print(f"Bot: {response}\n")

# Example 4: Name inquiry
user_msg = "What is your name?"
response = interactive_chatbot.chat(user_msg)
print(f"You: {user_msg}")
print(f"Bot: {response}\n")

# Example 5: Help request
user_msg = "What can you do?"
response = interactive_chatbot.chat(user_msg)
print(f"You: {user_msg}")
print(f"Bot: {response}\n")

# Example 6: Gratitude
user_msg = "Thank you so much!"
response = interactive_chatbot.chat(user_msg)
print(f"You: {user_msg}")
print(f"Bot: {response}\n")

# Example 7: Farewell
user_msg = "Goodbye!"
response = interactive_chatbot.chat(user_msg)
print(f"You: {user_msg}")
print(f"Bot: {response}\n")

# Display full conversation history
history = interactive_chatbot.get_conversation_history()
print("\nFULL CONVERSATION HISTORY")
print("="*60)

for i, turn in enumerate(history, 1):
    print(f"\nTurn {i}:")
    print(f"  Intent: {turn['intent']}")
    print(f"  You: {turn['user']}")
    print(f"  Bot: {turn['bot']}")

print(f"\n" + "="*60)
print(f"Total turns: {len(history)}")
print(f"User name: {interactive_chatbot.user_name}")
print("="*60)
