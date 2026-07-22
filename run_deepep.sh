#!/bin/bash

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1

MODEL_PATH=/home/chenxu/k3_weights/

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
export SGLANG_MAMBA_CONV_DTYPE=bfloat16

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export STREAMS_PER_DEVICE=32

# export DEEP_NORMAL_MODE_USE_INT8_QUANT=0，解决报错：
# Weight quant case with x dtype [DT_INT8] and weight dtype [DT_FLOAT4_E2M1] is not supported.
# 在w4a16_mxfp4_gmm_npu方法中调用的torch.ops.npu.npu_grouped_matmul
# 不支持input类型为DT_INT8 + weight类型为DT_FLOAT4_E2M1的组合
export DEEP_NORMAL_MODE_USE_INT8_QUANT=0

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=1200
export HCCL_OP_EXPANSION_MODE=AIV

export PYTHONPATH=/home/z00937177/sglang/python:$PYTHONPATH

export ASCEND_RT_VISIBLE_DEVICES=0,3,6,7

# export ASCEND_USE_FIA=1，解决报错：
# the current working operator name is PagedAttentionOperation.
# mla 使用flash attention而不是paged attention
export ASCEND_USE_FIA=1

# --page-size 128，解决报错：
# call aclnnFusedInferAttentionScoreV5 failed
# blockSize(1) should be a multiple of 16, and should be in range of [16, 1024]

# --mamba-scheduler-strategy extra_buffer，解决报错（开启radix cache后）：
# MambaComponent requires page_size=1 when mamba_extra_buffer is disabled
# assert cache.page_size == 1

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
    --host 127.0.0.1 \
    --port 9903 \
    --mamba-scheduler-strategy extra_buffer \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --disable-cuda-graph \

exit 1