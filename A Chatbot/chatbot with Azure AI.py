"""
Traditional Rule-Based Chatbot with Azure AI Sentiment Analysis.
A simple chatbot implementing pattern matching, intent recognition, and template-based responses.
"""

import re
import json
import random
from typing import Dict, List, Tuple
from datetime import datetime
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

#---------------------------------------
# Azure AI  Configuration
#---------------------------------------
endpoint = "https://chitti-chatbot-language-service.cognitiveservices.azure.com/"
api_key = "7t8vjX1LUpqPPyOF153aRvGovBA9VpdWQ3ifxAzs9EIqAiOzf5ijJQQJ99CEACYeBjFXJ3w3AAAaACOG7eB8"

credential = AzureKeyCredential(api_key)

text_analytics_client = TextAnalyticsClient(
    endpoint=endpoint,
    credential=credential
)

def analyze_sentiment(text):
    """
    Analyze user message sentiment using Azure AI Language Service.
    Returns positive, negative, neutral, mixed, or unavailable.
    """
    try:
        response = text_analytics_client.analyze_sentiment(documents=[text])[0]
        return response.sentiment
    except Exception as e:
        return "unavailable"

class TraditionalChatbot:
    """
    A rule-based chatbot that uses pattern matching and predefined intents to generate responses and
    Azure AI sentiment analysis.
    """
    
    def __init__(self):
        """Initialize the chatbot with intents, patterns, and responses."""
        self.conversation_history = []
        self.user_name = None
        self.intents = self._load_intents()
        
    def _load_intents(self) -> Dict:
        """
        Load intent definitions with patterns and response templates.
        
        Returns:
            Dict: Dictionary containing intent configurations
        """
        return {
            "greeting": {
                "patterns": [
                    r"(hello|hi|hey|greetings|good morning|good afternoon|good evening)",
                    r"what'?s up",
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
                    r"i'?m ([a-z]+)",
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
                    r"how'?s it going",
                    r"how'?re you doing",
                    r"what'?s your status"
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
                    "I can help you with greeting exchanges, answer questions about myself, and have a conversation! What would you like to know?",
                    "I'm here to chat and help! You can greet me, ask how I'm doing, or just have a conversation.",
                    "I can assist with conversation, answer questions, and help with basic information."
                ]
            },
            "name_inquiry": {
                "patterns": [
                    r"what's your name",
                    r"what is your name",
                    r"who are you",
                    r"what are you called",
                    r"do you have a name"
                ],
                "responses": [
                    "I'm Chitti, a traditional rule-based conversational AI!",
                    "You can call me Chitti. I'm here to chat with you!",
                    "I'm Chitti, a simple pattern-matching chatbot."
                ]
            },
            "thanks": {
                "patterns": [
                    r"(thank you|thanks|appreciate it|cheers)",
                    r"thanks[!]?",
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
                    r"i didn'?t get that"
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
    
    def preprocess_input(self, user_input: str) -> str:
        return user_input.lower().strip()
    
    def recognize_intent(self, user_input: str) -> Tuple[str, List[str]]:
        """
        Recognize user intent by matching patterns against input.

        Args:
            user_input: Preprocessed user input

        Returns:
            Tuple: (intent_name, match_groups) or ('default', []) if no match
        """
        for intent_name, intent_data in self.intents.items():
            if intent_name == "default":
                continue
            
            for pattern in intent_data["patterns"]:
                match = re.search(pattern, user_input)
                if match:
                    return intent_name, list(match.groups())
        
        return "default", []
    
    def extract_name(self, user_input: str, match_groups: List[str]) -> None:
        """
        Extract and store the user's name if provided.
        
        Args:
            user_input: Original user input
            match_groups: Regex match groups from pattern matching
        """
        if match_groups and match_groups[0]:
            self.user_name = match_groups[0].capitalize()
    
    def generate_response(self, intent_name: str, match_groups: List[str]) -> str:
        """
        Generate a response based on recognized intent.
        
        Args:
            intent_name: Name of recognized intent
            match_groups: Regex match groups
            
        Returns:
            str: Generated response
        """
        import random
        
        intent_responses = self.intents[intent_name]["responses"]
        response = random.choice(intent_responses)
        
        # Handle special template variables
        if "{name}" in response and self.user_name:
            response = response.format(name=self.user_name)
        
        if "{time}" in response:
            current_time = datetime.now().strftime("%I:%M %p")
            response = response.format(time=current_time)
        
        return response
    
    def add_sentiment_message(self, response: str, sentiment: str) -> str:
        """
        Add Azure sentiment result to chatbot response.
        """
        if sentiment == "positive":
            response += " I also detected a positive tone in your message."
        elif sentiment == "negative":
            response += " I also detected a negative tone in your message. I hope I can help."
        elif sentiment == "neutral":
            response += " I detected a neutral tone in your message."
        elif sentiment == "mixed":
            response += " I detected mixed emotions in your message."
        else:
            response += " Sentiment analysis was unavailable for this message."

        return response

    def chat(self, user_input: str) -> str:
        if not user_input.strip():
            return "Please say something!"
        
        # Preprocess input
        processed_input = self.preprocess_input(user_input)
        
        # Recognize intent
        intent_name, match_groups = self.recognize_intent(processed_input)
        
        # Extract name if applicable
        if intent_name == "name_introduction":
            self.extract_name(processed_input, match_groups)
        
        # Generate response
        response = self.generate_response(intent_name, match_groups)

        # Azure AI sentiment analysis happens here
        sentiment = analyze_sentiment(user_input)

        # Add sentiment message to chatbot response
        response = self.add_sentiment_message(response, sentiment)
        
        # Store in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "bot": response,
            "intent": intent_name,
            "sentiment": analyze_sentiment
        })
        
        return response
    
    def get_conversation_history(self) -> List[Dict]:
        """
        Retrieve the conversation history.
        
        Returns:
            List[Dict]: List of conversation turns
        """
        return self.conversation_history
    
    def save_conversation(self, filepath: str) -> None:
        """
        Save conversation history to a JSON file.
        
        Args:
            filepath: Path to save the conversation
        """
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def reset(self) -> None:
        """Reset the chatbot's conversation history and user name."""
        self.conversation_history = []
        self.user_name = None


def main():
    """Main function to run the chatbot in interactive mode."""
    print("=" * 60)
    print("Traditional Rule-Based Chatbot")
    print("=" * 60)
    print("\nHello! I'm a traditional rule-based chatbot.")
    print("Type 'quit' or 'bye' to exit.")
    print("Type 'history' to see conversation history.")
    print("Type 'reset' to start a new conversation.")
    print("-" * 60 + "\n")
    
    chatbot = TraditionalChatbot()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "history":
                print("\n--- Conversation History ---")
                for turn in chatbot.get_conversation_history():
                    print(f"You: {turn['user']}")
                    print(f"Bot: {turn['bot']}")
                    print()
                continue
            
            if user_input.lower() == "reset":
                chatbot.reset()
                print("Conversation reset!\n")
                continue
            
            response = chatbot.chat(user_input)
            print(f"Bot: {response}\n")
            
            # Check for exit conditions
            if any(word in user_input.lower() for word in ["bye", "quit", "exit", "goodbye"]):
                print("Chatbot: Thanks for chatting! Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\nChatbot: Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
    