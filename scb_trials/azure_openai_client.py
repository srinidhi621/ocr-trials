"""
Azure OpenAI Client
Provides functions to interact with Azure OpenAI API for chat completions and embeddings.
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()


class AzureOpenAIClient:
    """Client for Azure OpenAI API interactions."""
    
    def __init__(self):
        """Initialize the Azure OpenAI client with credentials from environment."""
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.embeddings_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        
        if not all([self.endpoint, self.api_key]):
            raise ValueError("Missing required Azure OpenAI credentials in environment variables")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_completion_tokens: Optional[int] = None,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """
        Generate a chat completion response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                      Example: [{"role": "user", "content": "Hello!"}]
            model: Deployment name to use (defaults to AZURE_OPENAI_DEPLOYMENT_NAME)
            temperature: Sampling temperature (0-2). Higher = more creative.
            max_completion_tokens: Maximum tokens in the response (for GPT-5.x models).
            top_p: Nucleus sampling parameter.
            stream: Whether to stream the response.
            **kwargs: Additional parameters to pass to the API.
        
        Returns:
            ChatCompletion object or stream iterator if stream=True.
        
        Example:
            client = AzureOpenAIClient()
            response = client.chat_completion([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"}
            ])
            print(response.choices[0].message.content)
        """
        deployment = model or self.deployment_name
        
        params = {
            "model": deployment,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            **kwargs
        }
        
        if max_completion_tokens:
            params["max_completion_tokens"] = max_completion_tokens
        
        response = self.client.chat.completions.create(**params)
        return response
    
    def get_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to embed.
            model: Embedding deployment name (defaults to AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT)
        
        Returns:
            List of floats representing the embedding vector.
        
        Example:
            client = AzureOpenAIClient()
            embedding = client.get_embedding("Hello, world!")
            print(f"Embedding dimension: {len(embedding)}")
        """
        deployment = model or self.embeddings_deployment
        
        if not deployment:
            raise ValueError("No embeddings deployment specified")
        
        response = self.client.embeddings.create(
            model=deployment,
            input=text
        )
        
        return response.data[0].embedding
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts.
        
        Args:
            texts: List of texts to embed.
            model: Embedding deployment name (defaults to AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT)
        
        Returns:
            List of embedding vectors.
        
        Example:
            client = AzureOpenAIClient()
            embeddings = client.get_embeddings_batch(["Hello", "World"])
            print(f"Generated {len(embeddings)} embeddings")
        """
        deployment = model or self.embeddings_deployment
        
        if not deployment:
            raise ValueError("No embeddings deployment specified")
        
        response = self.client.embeddings.create(
            model=deployment,
            input=texts
        )
        
        return [item.embedding for item in response.data]
    
    def simple_chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Simplified chat function for quick single-turn conversations.
        
        Args:
            prompt: User's message/question.
            system_prompt: Optional system instruction.
        
        Returns:
            The assistant's response text.
        
        Example:
            client = AzureOpenAIClient()
            answer = client.simple_chat("What is 2+2?")
            print(answer)
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat_completion(messages)
        return response.choices[0].message.content


# Convenience function for quick usage
def get_azure_openai_response(
    prompt: str,
    system_prompt: Optional[str] = None
) -> str:
    """
    Quick function to get a response from Azure OpenAI.
    
    Args:
        prompt: User's message/question.
        system_prompt: Optional system instruction.
    
    Returns:
        The assistant's response text.
    
    Example:
        from azure_openai_client import get_azure_openai_response
        answer = get_azure_openai_response("Explain machine learning in simple terms")
        print(answer)
    """
    client = AzureOpenAIClient()
    return client.simple_chat(prompt, system_prompt)


if __name__ == "__main__":
    # Test the client
    print("Testing Azure OpenAI Client...")
    
    try:
        client = AzureOpenAIClient()
        
        # Test chat completion
        print("\n--- Chat Completion Test ---")
        response = client.simple_chat(
            "Say 'Hello from Azure OpenAI!' in exactly those words.",
            system_prompt="You are a helpful assistant. Follow instructions exactly."
        )
        print(f"Response: {response}")
        
        # Test embedding (if deployment is configured)
        if client.embeddings_deployment:
            print("\n--- Embedding Test ---")
            embedding = client.get_embedding("Test embedding text")
            print(f"Embedding dimension: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
        
        print("\n✓ Azure OpenAI Client is working correctly!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
