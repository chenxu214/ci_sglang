#!/bin/bash

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
MODEL_PATH=/home/l00519189/kk
export ASCEND_USE_FIA=1
#export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/usr/local/memfabric_hybrid/latest/aarch64-linux/lib64/
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
export SGLANG_MAMBA_CONV_DTYPE=bfloat16
export SGLANG_W4A8_MXFP4_MOE=1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# NOTE: expandable_segments:True causes PyTorch caching allocator to use
# virtual address mapping (e.g. 0x4082...), which AICore's DataCopyPad
# cannot access (requires physical HBM addresses 0x1200...). This breaks
# acc_offload sparse_copy. Disabled for MoE DRAM offload.
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export STREAMS_PER_DEVICE=32

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=1200
export HCCL_OP_EXPANSION_MODE=AIV

export PYTHONPATH=/home/l00519189/sglang-offload/python:$PYTHONPATH

sglang serve \
    --model-path $MODEL_PATH \
    --trust-remote-code \
    --tokenizer-path $MODEL_PATH\
    --attention-backend ascend \
    --device npu \
    --quantization compressed-tensors \
    --dtype bfloat16 \
    --tp-size 4 \
    --mem-fraction-static 0.8 \
    --max-total-tokens 65536 \
    --page-size 128 \
    --chunked-prefill-size -1 \
    --disable-cuda-graph \
    --host 0.0.0.0 \
    --port 8880 \
    --moe-dram-offload \
    --moe-dram-pool-size-gb 20 \
    --disable-radix-cache \
    --skip-server-warmup

exit 1
#    --moe-a2a-backend deepep \
#    --moe-dram-offload \
