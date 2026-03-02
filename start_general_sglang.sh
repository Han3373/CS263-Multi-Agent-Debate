#!/usr/bin/env bash
#
# start_general_sglang.sh — Launch a general-purpose SGLang inference server
#
# USAGE:
#   [CUDA_VISIBLE_DEVICES=<gpus>] [PORT=<port>] [TP=<n>] ./start_general_sglang.sh <model-name-or-path> [extra sglang args...]
#
# ARGUMENTS:
#   <model-name-or-path>   HuggingFace model ID or local path to the model weights (required)
#   [extra sglang args]    Any additional flags passed directly to sglang.launch_server (optional)
#
# ENVIRONMENT VARIABLES:
#   CUDA_VISIBLE_DEVICES   Comma-separated GPU indices to use (e.g. "0,1,2,3").
#                          Tensor-parallelism degree (TP) is inferred from the count
#                          of listed GPUs unless TP is set explicitly.
#   PORT                   Port the server listens on. Default: 8000
#   TP                     Tensor-parallelism degree. Overrides the GPU-count inference
#                          when set explicitly.
#
# EXAMPLES:
#   # Single GPU, default port
#   ./start_general_sglang.sh Qwen/Qwen2.5-7B-Instruct
#
#   # Four GPUs, custom port, with reasoning parser
#   CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=8080 ./start_general_sglang.sh /data/models/qwen3-72b --reasoning-parser qwen3
#
#   # With extended context length
#   ./start_general_sglang.sh /data/models/llama3-8b --context-length 131072
#
# OUTPUT:
#   Server stdout/stderr is redirected to server.log in the current directory.
#   The script blocks until Ctrl+C; killing the script also stops the server.
#

source ~/miniconda3/etc/profile.d/conda.sh && conda activate sglang

MODEL=$1
if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model-name-or-path> [extra sglang args...]"
    exit 1
fi
EXTRA_ARGS="${@:2}"

#get the port, default to 8000
PORT=${PORT:-8000}

#infer the TP from the CUDA_VISIBLE_DEVICES, default to 1
if [ -z "$TP" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    else
        TP=1
    fi
fi

model_command="python -m sglang.launch_server --model-path ${MODEL} --port ${PORT} --tp-size ${TP} --mem-fraction-static 0.8 ${EXTRA_ARGS}"
# #if we are disabling thinking
# if [ "$DISABLE_THINKING" = "1" ]; then
#     #per the docs: temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
#     model_command+=" --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --presence-penalty 1.5 --repetition-penalty 1.0"
#     model_command+=" --chat-template-kwargs '{\"enable_thinking\": false}'"
# else
#    echo "Thinking mode has not been setup yet"
#    exit 1
# fi
    
echo "Running command: $model_command"

eval "$model_command" > server_${PORT}.log 2>&1 &   

SERVER_PID=$!

# Wait for server to be healthy
echo "Waiting for server to be ready..."
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    # Exit early if server process died
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: SGLang server process exited unexpectedly." >&2
        exit 1
    fi
    echo "  Waiting for server..."; sleep 3
done

echo ""
echo "SGLang server ready at http://${HOST}:${PORT}"
echo "  Run: python debate_pipeline_sglang.py --server-url http://${HOST}:${PORT} --model <model-name>"
echo ""

# Keep script running so the server stays alive (Ctrl+C to stop)
wait "$SERVER_PID"
