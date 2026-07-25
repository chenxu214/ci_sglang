export SGLANG_NPU_USE_MULTI_STREAM=1
export USE_VLLM_CUSTOM_ALLREDUCE=1

export PYTHONPATH=/home/zkk/sglang/python:$PYTHONPATH

python -m sglang.launch_server \
    --model-path /home/weights/Kimi-Linear-48B-A3B-Instruct \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --host 0.0.0.0 \
    --port 6660 \
    --chunked-prefill-size 8192 \
    --tp-size 2 \
    --base-gpu-id 0 \
    --page-size 128 \
    --device npu \
    --attention-backend ascend \
    --max-total-tokens 65536 \
    --watchdog-timeout 9000 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --max-running-requests 16 \
    --mamba-ssm-dtype bfloat16