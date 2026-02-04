"""
Google Gemini Client
Provides functions to interact with Google's Gemini API for text generation.
Uses the latest google.genai SDK.
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()


class GeminiClient:
    """Client for Google Gemini API interactions."""
    
    def __init__(self, model_name: str = "gemini-3-pro-preview"):
        """
        Initialize the Gemini client with credentials from environment.
        
        Args:
            model_name: The Gemini model to use. Options include:
                       - "gemini-3-pro-preview" (latest, most capable - default)
                       - "gemini-2.5-pro" (stable pro model)
                       - "gemini-2.5-flash" (fast, efficient)
                       - "gemini-2.0-flash" (balanced)
        """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment variables")
        
        # Initialize the client
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self._chat_session = None
        self._chat_history = []
    
    def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        top_p: float = 0.95,
        top_k: int = 40,
        **kwargs
    ) -> str:
        """
        Generate content based on a prompt.
        
        Args:
            prompt: The input prompt/question.
            temperature: Sampling temperature (0-2). Higher = more creative.
            max_output_tokens: Maximum tokens in the response.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            **kwargs: Additional parameters.
        
        Returns:
            Generated text response.
        
        Example:
            client = GeminiClient()
            response = client.generate_content("Explain quantum computing")
            print(response)
        """
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        
        if max_output_tokens:
            config.max_output_tokens = max_output_tokens
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        return response.text
    
    def generate_with_system_prompt(
        self,
        prompt: str,
        system_instruction: str,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate content with a system instruction.
        
        Args:
            prompt: The user's input prompt.
            system_instruction: System-level instruction for the model.
            temperature: Sampling temperature.
            **kwargs: Additional parameters.
        
        Returns:
            Generated text response.
        
        Example:
            client = GeminiClient()
            response = client.generate_with_system_prompt(
                prompt="What should I cook tonight?",
                system_instruction="You are a professional chef specializing in Italian cuisine."
            )
            print(response)
        """
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            **kwargs
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        return response.text
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Have a conversation with the model.
        
        Args:
            message: The user's message.
            history: Optional conversation history. List of dicts with 'role' and 'content'.
                    Example: [{"role": "user", "content": "Hi"}, {"role": "model", "content": "Hello!"}]
        
        Returns:
            The model's response text.
        
        Example:
            client = GeminiClient()
            response = client.chat("What's the capital of France?")
            print(response)
            response = client.chat("What's its population?")  # Continues conversation
            print(response)
        """
        if history is not None:
            # Reset and use provided history
            self._chat_history = []
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content") or msg.get("parts", "")
                self._chat_history.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=content)]
                    )
                )
        
        # Add user message to history
        self._chat_history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)]
            )
        )
        
        # Generate response with history
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=self._chat_history
        )
        
        # Add assistant response to history
        self._chat_history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=response.text)]
            )
        )
        
        return response.text
    
    def reset_chat(self):
        """Reset the chat session to start a new conversation."""
        self._chat_history = []
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Get the current chat history.
        
        Returns:
            List of conversation messages.
        """
        return self._chat_history
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The text to count tokens for.
        
        Returns:
            Number of tokens.
        
        Example:
            client = GeminiClient()
            count = client.count_tokens("Hello, how are you?")
            print(f"Token count: {count}")
        """
        response = self.client.models.count_tokens(
            model=self.model_name,
            contents=text
        )
        return response.total_tokens
    
    def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        temperature: float = 0.7
    ) -> str:
        """
        Generate content based on an image and text prompt.
        
        Args:
            prompt: Text prompt describing what to do with the image.
            image_path: Path to the image file.
            temperature: Sampling temperature.
        
        Returns:
            Generated text response.
        
        Example:
            client = GeminiClient()
            response = client.generate_with_image(
                prompt="Describe what you see in this image",
                image_path="photo.jpg"
            )
            print(response)
        """
        import PIL.Image
        
        image = PIL.Image.open(image_path)
        
        config = types.GenerateContentConfig(
            temperature=temperature
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image],
            config=config
        )
        
        return response.text
    
    def list_available_models(self) -> List[str]:
        """
        List all available Gemini models.
        
        Returns:
            List of model names.
        """
        models = []
        for model in self.client.models.list():
            models.append(model.name)
        return models


# Convenience function for quick usage
def get_gemini_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gemini-3-pro-preview"
) -> str:
    """
    Quick function to get a response from Gemini.
    
    Args:
        prompt: User's message/question.
        system_prompt: Optional system instruction.
        model: Model to use.
    
    Returns:
        The model's response text.
    
    Example:
        from gemini_client import get_gemini_response
        answer = get_gemini_response("What is the meaning of life?")
        print(answer)
    """
    client = GeminiClient(model_name=model)
    
    if system_prompt:
        return client.generate_with_system_prompt(prompt, system_prompt)
    
    return client.generate_content(prompt)


if __name__ == "__main__":
    # Test the client
    print("Testing Gemini Client...")
    
    try:
        client = GeminiClient()
        
        # Test basic generation
        print("\n--- Basic Generation Test ---")
        response = client.generate_content(
            "Say 'Hello from Gemini!' in exactly those words."
        )
        print(f"Response: {response}")
        
        # Test with system prompt
        print("\n--- System Prompt Test ---")
        response = client.generate_with_system_prompt(
            prompt="Introduce yourself briefly.",
            system_instruction="You are a friendly pirate. Speak like a pirate."
        )
        print(f"Response: {response}")
        
        # Test token counting
        print("\n--- Token Count Test ---")
        test_text = "This is a test sentence for counting tokens."
        token_count = client.count_tokens(test_text)
        print(f"Token count for '{test_text}': {token_count}")
        
        # Test chat
        print("\n--- Chat Test ---")
        response1 = client.chat("My name is Alice.")
        print(f"Response 1: {response1}")
        response2 = client.chat("What's my name?")
        print(f"Response 2: {response2}")
        
        print("\n✓ Gemini Client is working correctly!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
