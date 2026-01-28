#!/usr/bin/env python3
"""
Client test script for chained generation feature.
Requires a running vLLM server with the chained generation feature.

Start server first:
    python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8000

Then run this script:
    python test_chained_generation_client.py
"""

from litellm.types.utils import ModelResponse
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
import requests
import json
import sys

# Model configuration - uncomment the model you want to use
# MODEL = "facebook/opt-125m"  # 125M params - fast but basic
# MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B params - small chat model
MODEL = "Qwen/Qwen2.5-3B-Instruct"  # 3B params - smart, good for factual questions


def test_simple_chained_generation():
    """Test simple chained generation with one follow-up."""
    print("=" * 60)
    print("Test 1: Simple Chained Generation")
    print("=" * 60)
    
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "vllm_xargs": {
            "after_generation_messages": json.dumps([
                {"role": "user", "content": "How many people live there?"}
            ]),
            "max_chained_turns": 5,
            "return_intermediate_outputs": 1  # Use 1 for True
        },
        "stream": False,
        "max_tokens": 20
    }
    
    print("\n1. Sending request...")
    print(f"   Initial: {payload['messages'][0]['content']}")
    after_msgs = json.loads(payload['vllm_xargs']['after_generation_messages'])
    print(f"   Follow-up: {after_msgs[0]['content']}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n2. Response received!")
        print(f"   Total choices: {len(result['choices'])}")
        print(f"   Usage: {result['usage']}")
        
        print("\n3. Conversation flow:")
        for i, choice in enumerate(result['choices']):
            print(f"\n   Turn {i}:")
            print(f"   Assistant: {choice['message']['content']}")
            print(f"   Finish reason: {choice['finish_reason']}")
        
        print("\n✅ Test 1 PASSED!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to vLLM server")
        print("   Please start the server first:")
        print("   python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        return False
    except requests.exceptions.Timeout:
        print("\n❌ Test 1 FAILED: Request timed out")
        print("   The server took too long to respond (>60s)")
        print("   This might indicate an infinite loop or performance issue")
        return False
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False


def test_multiple_followups():
    """Test with multiple follow-up messages."""
    print("\n" + "=" * 60)
    print("Test 2: Multiple Follow-ups")
    print("=" * 60)
    
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "vllm_xargs": {
            "after_generation_messages": json.dumps([
                {"role": "user", "content": "And what is 3+3?"},
                {"role": "user", "content": "Great! Now what is 5+5?"}
            ]),
            "max_chained_turns": 10,
            "return_intermediate_outputs": 1
        },
        "stream": False,
        "max_tokens": 15
    }
    
    print("\n1. Sending request with 3 follow-ups...")
    
    try:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n2. Full conversation:")
        print(f"   User: {payload['messages'][0]['content']}")
        
        after_msgs = json.loads(payload['vllm_xargs']['after_generation_messages'])
        for i, choice in enumerate(result['choices']):
            print(f"   Assistant (Turn {i}): {choice['message']['content']}")
            if i < len(after_msgs):
                print(f"   User: {after_msgs[i]['content']}")
        
        print(f"\n3. Total tokens used: {result['usage']['total_tokens']}")
        print("\n✅ Test 2 PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        return False


def test_final_output_only():
    """Test returning only the final output."""
    print("\n" + "=" * 60)
    print("Test 3: Return Final Output Only")
    print("=" * 60)
    
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Count: 1"}
        ],
        "vllm_xargs": {
            "after_generation_messages": json.dumps([
                {"role": "user", "content": "Continue: 2"},
                {"role": "user", "content": "Continue: 3"}
            ]),
            "max_chained_turns": 5,
            "return_intermediate_outputs": 0  # Only final (0 = False)
        },
        "stream": False,
        "max_tokens": 10
    }
    
    print("\n1. Sending request (return_intermediate_outputs=False)...")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        print(f"\n2. Received {len(result['choices'])} choice(s)")
        print("   Expected: 1 (final only)")
        
        if len(result['choices']) == 1:
            print(f"\n3. Final response: {result['choices'][0]['message']['content']}")
            print("\n✅ Test 3 PASSED!")
            return True
        else:
            print(f"\n❌ Expected 1 choice, got {len(result['choices'])}")
            return False
        
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        return False


def test_with_litellm():
    """Test using LiteLLM client."""
    print("\n" + "=" * 60)
    print("Test 4: LiteLLM Integration")
    print("=" * 60)
    
    try:
        import litellm
        
        print("\n1. Using LiteLLM client...")
        
        response: ModelResponse | CustomStreamWrapper = litellm.completion(
            model=f"openai/{MODEL}",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            api_base="http://localhost:8000/v1",
            extra_body={
                "vllm_xargs": {
                    "after_generation_messages": json.dumps([
                        {"role": "user", "content": "How many people live there? Answer only with a number"}
                    ]),
                    "max_chained_turns": 5,
                    "return_intermediate_outputs": 1
                }
            },
            max_tokens=20
        )
        
        print(f"\n2. Response received with {len(response.choices)} choices")
        for i, choice in enumerate(response.choices):
            print(f"   Turn {i}: {choice.message.content}")
        
        print("\n✅ Test 4 PASSED!")
        return True
        
    except ImportError:
        print("\n⚠️  LiteLLM not installed, skipping test")
        print("   Install with: pip install litellm")
        return None
    except Exception as e:
        print(f"\n❌ Test 4 FAILED: {e}")
        return False


def main():
    """Run all client tests."""
    print("\n" + "=" * 60)
    print("CHAINED GENERATION CLIENT TESTS")
    print("=" * 60)
    print("\nMake sure vLLM server is running:")
    print(f"  python -m vllm.entrypoints.openai.api_server --model {MODEL}")
    print()
    
    results = []
    
    # Run tests
    results.append(("Simple chained generation", test_simple_chained_generation()))
    
    if results[0][1]:  # Only continue if server is accessible
        results.append(("Multiple follow-ups", test_multiple_followups()))
        results.append(("Final output only", test_final_output_only()))
        results.append(("LiteLLM integration", test_with_litellm()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    
    for name, result in results:
        status = "✅ PASSED" if result is True else "❌ FAILED" if result is False else "⚠️  SKIPPED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0 and passed > 0:
        print("\n🎉 All tests passed!")
        return 0
    elif failed > 0:
        print("\n❌ Some tests failed")
        return 1
    else:
        print("\n⚠️  No tests could run (server not accessible)")
        return 2


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
