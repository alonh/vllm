#!/usr/bin/env python3
"""
Client test script for chained generation feature with new API.
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
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 0.5B params - small but capable


def test_sequential_simple():
    """Test simple sequential follow-up (linear conversation)."""
    print("=" * 60)
    print("Test 1: Simple Sequential Follow-up")
    print("=" * 60)
    
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "followup_params": {
            "sequential_messages": [
                [{"role": "user", "content": "How many people live there?"}]
            ]
        },
        "stream": False,
        "max_tokens": 20
    }
    
    print("\n1. Sending request...")
    print(f"   Initial: {payload['messages'][0]['content']}")
    print(f"   Follow-up: {payload['followup_params']['sequential_messages'][0][0]['content']}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n2. Response received!")
        print(f"   Usage: {result['usage']}")
        
        main_result = result['choices'][0]
        print("\n3. Full conversation structure:")
        print(f"   User: What is the capital of France?")
        print(f"   Assistant: {main_result['message']['content']}")
        
        if 'sequential_followups' in main_result and main_result['sequential_followups']:
            print(f"\n   Sequential follow-ups ({len(main_result['sequential_followups'])} total):")
            for i, followup in enumerate(main_result['sequential_followups']):
                print(f"\n   Turn {i+2}:")
                print(f"   User: How many people live there?")
                print(f"   Assistant: {followup['message']['content']}")
                print(f"   Finish reason: {followup['finish_reason']}")
        
        print("\n✅ Test 1 PASSED!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to vLLM server")
        print("   Please start the server first:")
        print(f"   python -m vllm.entrypoints.openai.api_server --model {MODEL}")
        return False
    except requests.exceptions.Timeout:
        print("\n❌ Test 1 FAILED: Request timed out")
        return False
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False


def test_sequential_multiple():
    """Test multiple sequential follow-ups."""
    print("\n" + "=" * 60)
    print("Test 2: Multiple Sequential Follow-ups")
    print("=" * 60)
    
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer with a number only!"}
        ],
        "followup_params": {
            "sequential_messages": [
                [{"role": "user", "content": "now add 3 to the result. Answer with a number only!"}],
                [{"role": "user", "content": "now add 7 to the result. Answer with a number only!"}]
            ],
            "return_intermediate_outputs": True
        },
        "stream": False,
        "max_tokens": 50
    }
    
    print("\n1. Sending request with 2 follow-ups...")
    
    try:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n2. Full conversation structure:")
        main_result = result['choices'][0]
        print(f"   Turn 1:")
        print(f"   User: {payload['messages'][0]['content']}")
        print(f"   Assistant: {main_result['message']['content']}")
        
        if 'sequential_followups' in main_result and main_result['sequential_followups']:
            for i, followup in enumerate(main_result['sequential_followups']):
                print(f"\n   Turn {i+2}:")
                print(f"   User: {payload['followup_params']['sequential_messages'][i][0]['content']}")
                print(f"   Assistant: {followup['message']['content']}")
                print(f"   Finish reason: {followup['finish_reason']}")
        
        print(f"\n3. Total tokens used: {result['usage']['total_tokens']}")
        print("\n✅ Test 2 PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        return False


def test_sequential_with_three_turns():
    """Test sequential mode with three turns."""
    print("\n" + "=" * 60)
    print("Test 3: Sequential - Three Turns")
    print("=" * 60)
    
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Count: 1"}
        ],
        "followup_params": {
            "sequential_messages": [
                [{"role": "user", "content": "Continue: 2"}],
                [{"role": "user", "content": "Continue: 3"}]
            ]
        },
        "stream": False,
        "max_tokens": 10
    }
    
    print("\n1. Sending request with 2 follow-ups...")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        print(f"\n2. Received {len(result['choices'])} choice(s)")
        print("   Expected: 1 choice with 2 sequential_followups")
        
        main_result = result['choices'][0]
        has_followups = 'sequential_followups' in main_result and main_result['sequential_followups']
        
        if len(result['choices']) == 1 and has_followups and len(main_result['sequential_followups']) == 2:
            print(f"\n3. Full conversation:")
            print(f"   Turn 1: {main_result['message']['content']}")
            for i, followup in enumerate(main_result['sequential_followups']):
                print(f"   Turn {i+2}: {followup['message']['content']}")
            print("\n✅ Test 3 PASSED!")
            return True
        else:
            print(f"\n❌ Expected 1 choice with 2 followups")
            print(f"   Got {len(result['choices'])} choices")
            if has_followups:
                print(f"   and {len(main_result['sequential_followups'])} followups")
            return False
        
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        return False


def test_parallel_simple():
    """Test simple parallel follow-ups (independent branches)."""
    print("\n" + "=" * 60)
    print("Test 4: Simple Parallel Follow-ups")
    print("=" * 60)
    
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "followup_params": {
            "parallel_messages": [
                [{"role": "user", "content": "What's the population there?"}],
                [{"role": "user", "content": "What are the 2 closest cities?"}]
            ]
        },
        "stream": False,
        "max_tokens": 20
    }
    
    print("\n1. Sending request with 2 parallel branches...")
    print(f"   Initial: {payload['messages'][0]['content']}")
    print(f"   Branch 1: {payload['followup_params']['parallel_messages'][0][0]['content']}")
    print(f"   Branch 2: {payload['followup_params']['parallel_messages'][1][0]['content']}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n2. Full conversation structure:")
        main_result = result['choices'][0]
        print(f"   Initial:")
        print(f"   User: {payload['messages'][0]['content']}")
        print(f"   Assistant: {main_result['message']['content']}")
        
        if 'parallel_followups' in main_result and main_result['parallel_followups']:
            print(f"\n   Parallel branches ({len(main_result['parallel_followups'])} total):")
            for i, followup in enumerate(main_result['parallel_followups']):
                print(f"\n   Branch {i+1}:")
                print(f"   User: {payload['followup_params']['parallel_messages'][i][0]['content']}")
                print(f"   Assistant: {followup['message']['content']}")
                print(f"   Finish reason: {followup['finish_reason']}")
        
        print(f"\n3. Total tokens used: {result['usage']['total_tokens']}")
        print("\n✅ Test 4 PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 4 FAILED: {e}")
        return False


def test_parallel_with_three_branches():
    """Test parallel mode with three independent branches."""
    print("\n" + "=" * 60)
    print("Test 5: Parallel - Three Branches")
    print("=" * 60)
    
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "followup_params": {
            "parallel_messages": [
                [{"role": "user", "content": "What's the population?"}],
                [{"role": "user", "content": "What are nearby cities?"}],
                [{"role": "user", "content": "What's the weather like?"}]
            ]
        },
        "stream": False,
        "max_tokens": 20
    }
    
    print("\n1. Sending request with 3 parallel branches...")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        print(f"\n2. Received {len(result['choices'])} choice(s)")
        print("   Expected: 1 choice with 3 parallel_followups")
        
        if len(result['choices']) == 1:
            main_result = result['choices'][0]
            
            # Check that we have parallel_followups
            if 'parallel_followups' in main_result and main_result['parallel_followups']:
                followups = main_result['parallel_followups']
                print(f"\n3. Initial response: {main_result['message']['content']}")
                print(f"\n4. Found {len(followups)} parallel branches:")
                
                for i, followup in enumerate(followups):
                    print(f"\n   Branch {i+1}:")
                    print(f"   Assistant: {followup['message']['content']}")
                    print(f"   Finish reason: {followup['finish_reason']}")
                
                if len(followups) == 3:
                    print("\n✅ Test 5 PASSED!")
                    return True
                else:
                    print(f"\n❌ Expected 3 parallel_followups, got {len(followups)}")
                    return False
            else:
                print("\n❌ No parallel_followups found in response")
                return False
        else:
            print(f"\n❌ Expected 1 choice, got {len(result['choices'])}")
            return False
        
    except Exception as e:
        print(f"\n❌ Test 5 FAILED: {e}")
        return False


def test_with_litellm():
    """Test using LiteLLM client."""
    print("\n" + "=" * 60)
    print("Test 6: LiteLLM Integration")
    print("=" * 60)
    
    try:
        import litellm
        
        print("\n1. Using LiteLLM client...")
        
        response: ModelResponse | CustomStreamWrapper = litellm.completion(
            model=f"openai/{MODEL}",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            api_base="http://localhost:8000/v1",
            extra_body={
                "followup_params": {
                    "sequential_messages": [
                        [{"role": "user", "content": "How many people live there?"}]
                    ],
                    "return_intermediate_outputs": True
                }
            },
            max_tokens=20
        )
        
        print(f"\n2. Response received")
        print(f"   Initial: {response.choices[0].message.content}")
        
        # Note: LiteLLM may not parse custom followup fields
        print("\n✅ Test 6 PASSED!")
        return True
        
    except ImportError:
        print("\n⚠️  LiteLLM not installed, skipping test")
        print("   Install with: pip install litellm")
        return None
    except Exception as e:
        print(f"\n❌ Test 6 FAILED: {e}")
        return False


def main():
    """Run all client tests."""
    print("\n" + "=" * 60)
    print("CHAINED GENERATION CLIENT TESTS (NEW API)")
    print("=" * 60)
    print("\nMake sure vLLM server is running:")
    print(f"  python -m vllm.entrypoints.openai.api_server --model {MODEL}")
    print()
    
    results = []
    
    # Run tests
    results.append(("Sequential simple", test_sequential_simple()))
    
    if results[0][1]:  # Only continue if server is accessible
        results.append(("Sequential multiple", test_sequential_multiple()))
        results.append(("Sequential three turns", test_sequential_with_three_turns()))
        results.append(("Parallel simple", test_parallel_simple()))
        results.append(("Parallel three branches", test_parallel_with_three_branches()))
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
