# 单机混布
# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑核
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTHONPATH=/home/luochen/glm/sglang_4_15_poc/sglang/python:$PYTHONPATH

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# pd传输, IP设置为p节点首节点
export ASCEND_MF_STORE_URL="tcp://61.47.19.68:24707"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200


# p节点IP
P_IP=('61.47.19.68' '61.47.19.67')
# D节点IP D节点首节点IP
D_IP=('61.47.19.70' '61.47.19.71')


# MODEL_PATH=/home/weights/GLM-5-w4a8-new-fix
# MODEL_PATH=/home/weights/GLM-5-w8a8-new-fix
MODEL_PATH=/home/weights/GLM-5.1-w4a8

# enable mlapo
#export SGLANG_NPU_USE_MLAPO=1

#export SGLANG_USE_FIA_NZ=1
#export SGLANG_SPEC_ENABLE_OVERLAP_REFLOW=1

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"
#export USE_DEEPEP_INT8=1
# prefill
for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
    #    export SGLANG_USE_AG_AFTER_QLORA=1 # ??
        export HCCL_BUFFSIZE=1200
    #    export DEEPEP_NORMAL_LONG_SEQ_ROUND=4
    #    export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
        export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export TASK_QUEUE_ENABLE=2
        export ENABLE_PROFILING=0
        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0
        ### eplb
        # export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/home/luochen/hot_map

        # ### zero buffer
        # # zbal
        # # export ZBAL_HCCL_OP="_allgather_base,allgather"
        # export HCCL_BUFFSIZE=128
        # unset PYTORCH_NPU_ALLOC_CONF
        # export SGLANG_ZBAL_LOCAL_MEM_SIZE=61184
        # export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
        # # # zbal if use mix alloc （开启混合分配减少内存碎片）
        # export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        # export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
        # export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://61.47.19.68:24672"
        # # zbal if support graph（need custom pta） （开启图下沉支持）
        # # export ZBAL_ENABLE_GRAPH=1

        # P节点
        python -m sglang.launch_server --model-path ${MODEL_PATH}  --disaggregation-mode prefill --host ${P_IP[$i]} \
        --port 8000 --disaggregation-bootstrap-port 8998 --dist-init-addr 61.47.19.68:5000 --trust-remote-code --nnodes 2 --node-rank $i \
        --tp-size 32 --mem-fraction-static 0.72 --attention-backend ascend --device npu --quantization modelslim \
        --disaggregation-transfer-backend ascend --max-running-requests 16 \
        --served-model-name glm-5 --chunked-prefill-size 16384 --max-prefill-tokens 180000 --moe-a2a-backend deepep --deepep-mode normal \
        --disable-shared-experts-fusion --disable-cuda-graph --dtype bfloat16 \
        --speculative-draft-model-quantization unquant \
        --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
        --dp-size 1 --enable-dp-attention --load-balance-method round_robin \
        --enable-nsa-prefill-context-parallel \
        --nsa-prefill-cp-mode in-seq-split \
        --attn-cp-size 32 \
        --enable-dp-lm-head --moe-dense-tp 1 
        # --eplb-rebalance-num-iterations 2 --expert-distribution-recorder-buffer-size 2048 --expert-distribution-recorder-mode stat --ep-dispatch-algorithm static
        #--dp-size 2 --enable-dp-attention --disable-shared-experts-fusion --disable-cuda-graph --dtype bfloat16
        #        --enable-attn-tp-input-scattered
        # --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2  \
        # --tool-call-parser glm47 --reasoning-parser glm45 --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2  \
        NODE_RANK=$i
        break
    fi
done

# decode
for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        export SGLANG_SPEC_ENABLE_OVERLAP_REFLOW=1
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export HCCL_BUFFSIZE=200
        # export ENABLE_PROFILING=1
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
        export TASK_QUEUE_ENABLE=0
        # export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
        # export ENABLE_FUSED_MOE=1
        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0

        ###p不开双流 / p回退到native算子
        export SGLANG_NPU_USE_MULTI_STREAM=1

        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode --host ${D_IP[$i]} \
        --port 8003 --trust-remote-code --dist-init-addr 61.47.19.70:5000 --nnodes 2 --node-rank $i --tp-size 32 --dp-size 32 --enable-dp-attention --ep-size 32 \
        --mem-fraction-static 0.85 --max-running-requests 32 --attention-backend ascend --device npu --quantization modelslim \
        --served-model-name glm-5 --moe-a2a-backend deepep --deepep-mode low_latency \
        --cuda-graph-bs 1 2 3 --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 180000 \
        --tokenizer-worker-num 4 --prefill-round-robin-balance --disable-shared-experts-fusion --dtype bfloat16  --load-balance-method round_robin \
        --speculative-draft-model-quantization unquant \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 

        # --enable-dp-lm-head --moe-dense-tp 1  --cuda-graph-bs 1 2 4 8 10 12 14 16 export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
        # --ep-dispatch-algorithm static --init-expert-location /home/luochen/hot_map/expert_distribution_recorder_1774942092.5829542.pt
        # --speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3  \
        # --tool-call-parser glm47 --reasoning-parser glm45 --speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3  \
        NODE_RANK=$i
        break
    fi
done

exit 1
