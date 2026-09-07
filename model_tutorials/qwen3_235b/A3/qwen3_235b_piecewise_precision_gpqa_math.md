# 服务拉起脚本

```shell
#!/bin/bash
set -e

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_USE_V1=1;
export HCCL_OP_EXPANSION_MODE="AIV";
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True;
export HCCL_BUFFSIZE=1024;
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1;
export VLLM_ENGINE_READY_TIMEOUT_S=3600

vllm serve /mnt/weight/Qwen3-235B-A22B-w8a8-QuaRot \
    --served-model-name qwen3_235b \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --data-parallel-size-local 4 \
    --data-parallel-start-rank 0 \
    --data-parallel-address 192.168.13.0 \
    --data-parallel-rpc-port 2345 \
    --max-num-seqs 40 \
    --max-model-len 40960 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --enable-expert-parallel \
    --quantization "ascend" \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
```

# aisbench 脚本

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="/mnt/weight/Qwen3-235B-A22B-w8a8-QuaRot",
        model="qwen3_235b",
        stream=False,
        request_rate = 0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip = "192.168.13.0",
        host_port = 8000,
        url="",
        max_out_len = 32768,
        batch_size = 64,
        trust_remote_code = False,
        generation_kwargs=dict(
            top_k = 20,
            top_p = 0.95,
            temperature = 0.6,
            ignore_eos = False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

```shell
ais_bench --models vllm_api_general_chat --datasets gpqa_gen_0_shot_cot_chat_prompt --debug
ais_bench --models vllm_api_general_chat --datasets math500_gen_0_shot_cot_chat_prompt --debug
```
