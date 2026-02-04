"""
Test Suite for Azure OpenAI and Gemini Clients
Tests both text and multi-modal inputs for each client.
"""

import os
import sys
import time
import base64
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from azure_openai_client import AzureOpenAIClient, get_azure_openai_response
from gemini_client import GeminiClient, get_gemini_response


class TestResult:
    """Container for test results."""
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} | {self.name} ({self.duration:.2f}s) - {self.message}"


class ClientTestSuite:
    """Test suite for API clients."""
    
    def __init__(self):
        self.results = []
        self.test_image_path = self._create_test_image()
    
    def _create_test_image(self) -> str:
        """Create a simple test image for multi-modal testing."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a simple test image with text
            img = Image.new('RGB', (400, 200), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw some shapes and text
            draw.rectangle([20, 20, 180, 80], fill='blue', outline='darkblue')
            draw.ellipse([220, 20, 380, 80], fill='red', outline='darkred')
            draw.text((50, 120), "Test Image", fill='black')
            draw.text((50, 150), "Blue Rectangle + Red Circle", fill='gray')
            
            # Save test image
            test_image_path = os.path.join(os.path.dirname(__file__), 'test_image.png')
            img.save(test_image_path)
            return test_image_path
            
        except Exception as e:
            print(f"Warning: Could not create test image: {e}")
            return None
    
    def run_test(self, test_func, test_name: str) -> TestResult:
        """Run a single test and capture results."""
        start_time = time.time()
        try:
            result_message = test_func()
            duration = time.time() - start_time
            result = TestResult(test_name, True, result_message or "Success", duration)
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(test_name, False, str(e), duration)
        
        self.results.append(result)
        print(result)
        return result
    
    # ==================== AZURE OPENAI TESTS ====================
    
    def test_azure_basic_chat(self) -> str:
        """Test basic chat completion with Azure OpenAI."""
        client = AzureOpenAIClient()
        response = client.simple_chat(
            "What is 2 + 2? Reply with just the number.",
            system_prompt="You are a helpful assistant. Be concise."
        )
        assert response is not None and len(response) > 0
        return f"Response: {response.strip()[:50]}"
    
    def test_azure_chat_completion(self) -> str:
        """Test full chat completion API with Azure OpenAI."""
        client = AzureOpenAIClient()
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function that adds two numbers. Keep it simple."}
        ]
        response = client.chat_completion(messages, temperature=0.3, max_completion_tokens=150)
        content = response.choices[0].message.content
        assert "def" in content or "return" in content
        return f"Generated code snippet ({len(content)} chars)"
    
    def test_azure_conversation(self) -> str:
        """Test multi-turn conversation with Azure OpenAI."""
        client = AzureOpenAIClient()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant with memory of the conversation."},
            {"role": "user", "content": "My favorite color is blue."},
        ]
        response1 = client.chat_completion(messages)
        
        messages.append({"role": "assistant", "content": response1.choices[0].message.content})
        messages.append({"role": "user", "content": "What is my favorite color?"})
        response2 = client.chat_completion(messages)
        
        answer = response2.choices[0].message.content.lower()
        assert "blue" in answer
        return "Conversation memory working correctly"
    
    def test_azure_embeddings(self) -> str:
        """Test embeddings generation with Azure OpenAI."""
        client = AzureOpenAIClient()
        
        if not client.embeddings_deployment:
            return "Skipped - No embeddings deployment configured"
        
        text = "This is a test sentence for embedding generation."
        embedding = client.get_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        return f"Embedding dimension: {len(embedding)}"
    
    def test_azure_batch_embeddings(self) -> str:
        """Test batch embeddings with Azure OpenAI."""
        client = AzureOpenAIClient()
        
        if not client.embeddings_deployment:
            return "Skipped - No embeddings deployment configured"
        
        texts = [
            "First test sentence",
            "Second test sentence",
            "Third test sentence"
        ]
        embeddings = client.get_embeddings_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) > 0 for e in embeddings)
        return f"Generated {len(embeddings)} embeddings"
    
    def test_azure_convenience_function(self) -> str:
        """Test the convenience function for Azure OpenAI."""
        response = get_azure_openai_response(
            "Say 'Hello' and nothing else.",
            system_prompt="Follow instructions exactly."
        )
        assert response is not None
        return f"Response: {response.strip()[:30]}"
    
    # ==================== GEMINI TESTS ====================
    
    def test_gemini_basic_generation(self) -> str:
        """Test basic text generation with Gemini."""
        client = GeminiClient()
        response = client.generate_content(
            "What is 2 + 2? Reply with just the number."
        )
        assert response is not None and len(response) > 0
        return f"Response: {response.strip()[:50]}"
    
    def test_gemini_with_system_prompt(self) -> str:
        """Test generation with system instruction."""
        client = GeminiClient()
        response = client.generate_with_system_prompt(
            prompt="Introduce yourself in one sentence.",
            system_instruction="You are a friendly robot named Robo-3000."
        )
        assert response is not None
        return f"Response: {response.strip()[:60]}..."
    
    def test_gemini_chat_session(self) -> str:
        """Test multi-turn chat with Gemini."""
        client = GeminiClient()
        
        # First message
        response1 = client.chat("My name is TestUser and I live in Tokyo.")
        
        # Follow-up that requires memory
        response2 = client.chat("What city do I live in?")
        
        assert "tokyo" in response2.lower()
        return "Chat memory working correctly"
    
    def test_gemini_token_count(self) -> str:
        """Test token counting with Gemini."""
        client = GeminiClient()
        
        text = "This is a test sentence for counting tokens in Gemini."
        count = client.count_tokens(text)
        
        assert isinstance(count, int)
        assert count > 0
        return f"Token count: {count}"
    
    def test_gemini_multimodal_image(self) -> str:
        """Test multi-modal (image + text) input with Gemini."""
        if not self.test_image_path or not os.path.exists(self.test_image_path):
            return "Skipped - Test image not available"
        
        client = GeminiClient()
        response = client.generate_with_image(
            prompt="Describe the shapes and colors you see in this image. Be specific.",
            image_path=self.test_image_path
        )
        
        # Check if it detected our shapes
        response_lower = response.lower()
        detected_features = []
        if "blue" in response_lower or "rectangle" in response_lower:
            detected_features.append("blue rectangle")
        if "red" in response_lower or "circle" in response_lower or "ellipse" in response_lower:
            detected_features.append("red circle")
        
        assert len(detected_features) > 0, "Model should detect shapes in the image"
        return f"Detected: {', '.join(detected_features)}"
    
    def test_gemini_convenience_function(self) -> str:
        """Test the convenience function for Gemini."""
        response = get_gemini_response(
            "Say 'Hello' and nothing else."
        )
        assert response is not None
        return f"Response: {response.strip()[:30]}"
    
    def test_gemini_list_models(self) -> str:
        """Test listing available Gemini models."""
        client = GeminiClient()
        models = client.list_available_models()
        
        assert isinstance(models, list)
        # Filter to just show gemini models
        gemini_models = [m for m in models if 'gemini' in m.lower()]
        return f"Found {len(gemini_models)} Gemini models"
    
    # ==================== RUN ALL TESTS ====================
    
    def run_all_tests(self):
        """Run all tests and display summary."""
        print("=" * 70)
        print("API CLIENT TEST SUITE")
        print("=" * 70)
        
        # Azure OpenAI Tests
        print("\n" + "-" * 35)
        print("AZURE OPENAI TESTS (GPT-5.2)")
        print("-" * 35)
        
        self.run_test(self.test_azure_basic_chat, "Azure: Basic Chat")
        self.run_test(self.test_azure_chat_completion, "Azure: Chat Completion API")
        self.run_test(self.test_azure_conversation, "Azure: Multi-turn Conversation")
        self.run_test(self.test_azure_embeddings, "Azure: Single Embedding")
        self.run_test(self.test_azure_batch_embeddings, "Azure: Batch Embeddings")
        self.run_test(self.test_azure_convenience_function, "Azure: Convenience Function")
        
        # Gemini Tests
        print("\n" + "-" * 35)
        print("GEMINI TESTS (gemini-3-pro-preview)")
        print("-" * 35)
        
        self.run_test(self.test_gemini_basic_generation, "Gemini: Basic Generation")
        self.run_test(self.test_gemini_with_system_prompt, "Gemini: System Prompt")
        self.run_test(self.test_gemini_chat_session, "Gemini: Chat Session")
        self.run_test(self.test_gemini_token_count, "Gemini: Token Counting")
        self.run_test(self.test_gemini_multimodal_image, "Gemini: Multi-modal (Image)")
        self.run_test(self.test_gemini_convenience_function, "Gemini: Convenience Function")
        self.run_test(self.test_gemini_list_models, "Gemini: List Models")
        
        # Summary
        self._print_summary()
        
        # Cleanup
        self._cleanup()
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        total_time = sum(r.duration for r in self.results)
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed:      {passed} ✓")
        print(f"Failed:      {failed} ✗")
        print(f"Total Time:  {total_time:.2f}s")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
        
        print("\n" + "=" * 70)
    
    def _cleanup(self):
        """Clean up test artifacts."""
        if self.test_image_path and os.path.exists(self.test_image_path):
            try:
                os.remove(self.test_image_path)
                print("Cleaned up test image.")
            except:
                pass


def main():
    """Run the test suite."""
    print("\nInitializing test suite...")
    print("Testing Azure OpenAI (GPT-5.2) and Google Gemini (gemini-3-pro-preview)")
    print()
    
    suite = ClientTestSuite()
    suite.run_all_tests()


if __name__ == "__main__":
    main()
