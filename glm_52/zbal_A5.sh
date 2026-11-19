#!/bin/bash
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1

#HCCL 
#export DEEPEP_HCCL_BUFFSIZE=1500
export HCCL_BUFFSIZE=500
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_INTRA_DMA_ENABLE=1

# skip gpu branch
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False
export SGLANG_RAGGED_VERIFY_MODE="static"

#export MODEL_PATH=/mnt/share/weight/DeepSeek-V4-Flash-0731-W8A8/DeepSeek-V4-Flash-0731-w8a8
export MODEL_PATH=/mnt/share/weight/DeepSeek-V4-Flash-0731
export PYTHONPATH=/mnt/share/chenxu/codes/zbal_fused/ci_sglang/python:$PYTHONPATH

# mtp
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

# perfermance

export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1

#export DEEPEP_HCCL_BUFFSIZE=2600
#export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
#export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=8192
#export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export SGLANG_ZBAL_DFC_MOE_LAYERS=43
export SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8=1

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    --page-size 128 \
    --tp-size 4 \
    --trust-remote-code \
    --attention-backend dsv4 \
    --device npu \
    --watchdog-timeout 9000 \
    --host 141.61.49.195 --port 31000 \
    --mem-fraction-static 0.72 \
    --max-running-requests 32 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 65536 \
    --cuda-graph-bs-decode 1 2 4 8 10 12 \
    --kv-cache-dtype auto \
    --enable-dp-lm-head \
    --disable-radix-cache \
    --moe-a2a-backend zbal_fuseep \
    --deepep-mode auto \
    --attn-dp-size 4 \
    --reasoning-parser deepseek-v4 \
