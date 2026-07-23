#!/bin/bash

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1

MODEL_PATH=/home/weights/Kimi-K3-int4

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export STREAMS_PER_DEVICE=32

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=4200
export HCCL_OP_EXPANSION_MODE=AIV

export PYTHONPATH=/home/l00890003/codes/sglang-0724/python:$PYTHONPATH

D_IP=('192.168.25.209' '192.168.25.212' '192.168.25.216' '192.168.25.217')
LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "Decode -> ${D_IP[$i]}"

        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0

        sglang serve \
            --model-loader-extra-config '{"enable_multithread_load": true}' \
            --dist-init-addr 192.168.25.209:5000 --nnodes 4 --node-rank $i \
            --model-path $MODEL_PATH \
            --tokenizer-path /home/weights/Kimi-K3-int4 \
            --trust-remote-code \
            --attention-backend ascend \
            --device npu \
            --quantization modelslim \
            --dtype bfloat16 \
            --tp-size 64 \
	        --enable-dp-attention --dp-size 4 --enable-dp-lm-head \
            --mem-fraction-static 0.8 \
            --chunked-prefill-size 32768 \
            --cuda-graph-bs 16 \
            --disable-radix-cache \
            --max-total-tokens 16384 \
            --max-running-requests 64 \
            --host 0.0.0.0 \
            --port 30000 \
	        --moe-a2a-backend deepep \
    	    --deepep-mode auto \
            # --disable-cuda-graph

        exit 1
    fi
done

exit 1

python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 1 \
  --disable-ignore-eos \
  --random-range-ratio 1 \
  --warmup-request 0


curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/weights/Kimi-K3-int4",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "max_tokens": 20,
    "temperature": 0
  }'
