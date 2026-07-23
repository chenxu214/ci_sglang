#!/bin/bash

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1

MODEL_PATH=/home/zkk/weights/Kimi-K3-int4-layer10

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
export SGLANG_MAMBA_CONV_DTYPE=bfloat16
export SGLANG_DEBUG_KIMI_K3_WEIGHT_LOAD=2

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export STREAMS_PER_DEVICE=32

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=1600
export HCCL_OP_EXPANSION_MODE=AIV

export RT_ASCEND_VISIBLE_DEVICES="8,9,10,11,12,13,14,15"

export PYTHONPATH=/home/hanwlax/workspace/sglang/python:$PYTHONPATH

sglang serve \
    --model-path $MODEL_PATH \
    --tokenizer-path $MODEL_PATH\
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --dtype bfloat16 \
    --tp-size 8 \
    --mem-fraction-static 0.7 \
    --max-total-tokens 65536 \
    --page-size 1 \
    --chunked-prefill-size -1 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --disable-cuda-graph \
    --disable-radix-cache \
    --host 0.0.0.0 \
    --port 8880 \
    --skip-server-warmup \
    --enable-multimodal --mm-enable-dp-encoder --mm-attention-backend ascend_attn

exit 1
