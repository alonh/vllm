#!/bin/bash
# Start vLLM server for testing chained generation with multiple models

cd /Users/alonhal/PycharmProjects/vllm
source venv/bin/activate

echo "Starting vLLM server on port 8000 with multiple models..."
echo "  - facebook/opt-125m (125M params - fast but basic)"
echo "  - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B params - small chat model)"
echo "  - Qwen/Qwen2.5-3B-Instruct (3B params - smart, good for M4 Mac)"

# Start server with three models
# Note: OPT model needs a chat template since it doesn't have one built-in
# Qwen and TinyLlama have built-in chat templates
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8000 \
    --max-model-len 2048 \
    --chat-template "{% for message in messages %}{{ message['content'] }} {% endfor %}" \
    2>&1 | tee server.log &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
echo $SERVER_PID > server.pid

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Server is ready!"
        exit 0
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo "❌ Server failed to start within 60 seconds"
exit 1

# Made with Bob
