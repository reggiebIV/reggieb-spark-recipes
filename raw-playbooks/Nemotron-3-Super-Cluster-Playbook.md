# Nemotron 3 Super on 2-Node DGX Spark — Playbook

Target image: `nvcr.io/nvidia/vllm:26.04-py3` (vLLM 0.19.0+nv26.04).
Topology: 2× DGX Spark, one GB10 GPU each.

## Helpful commands and information for setting up the network:

To tail the vllm logs on the head container

```bash
export HEAD_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$' | head -1)
docker exec $HEAD_CONTAINER tail -f /tmp/vllm-serve.log
```

Ray:
http://10.0.0.245:8265/

To get the status of the CX7 ports
```bash
ibdev2netdev 
```

To get IP addresses of the CX7 (high speed) ports
```bash
ip addr show enp1s0f1np1 
```
---

## 1 — If not already pulled, pull the latest docker image confirmed to work - Check for updates at https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/overview.html

Must be run on Head and Workier
```bash
docker pull nvcr.io/nvidia/vllm:26.04-py3
```

---

## 2 — Get + patch `run_cluster.sh` on both nodes - don't skip this. It seems you have to get a new copy and repatch it

vLLM dropped Ray; the script needs a one-line edit so the container installs Ray on startup before `ray start` fires.
Must be run on head and worker. Pinning ray, testing for staibility

sed installs Ray, since vLLM removed it from their docker image. Then add the --dashboard host line to allow the dashboard to be visible on the network

```bash
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/ray_serving/run_cluster.sh
chmod +x run_cluster.sh
sed -i 's|RAY_START_CMD="ray start --block"|RAY_START_CMD="pip install -q '\''ray[default]==2.54.1'\'' \&\& ray start --block"|' run_cluster.sh
sed -i 's|--head --node-ip-address|--head --dashboard-host=0.0.0.0 --dashboard-port=8265 --node-ip-address|' run_cluster.sh
sed -i '/^trap cleanup EXIT$/d' run_cluster.sh
```


(Optional optimization for repeat runs: build a derivative image once with Ray baked in — `FROM nvcr.io/nvidia/vllm:26.04-py3` + `RUN pip install -q 'ray[default]'`. Then you can revert to the unpatched `run_cluster.sh`. Skips the ~30s install at every container start.)

---

## 3 — Variables
Run on head and worker

```bash
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.04-py3
export MN_IF_NAME=enp1s0f1np1            # high-speed interface
export HF_TOKEN=<your-hf-token>
```

Plus, on the head's terminal 1:
```bash
export WORKER_HOST=192.168.0.145
```

Plus, on the worker (terminal 2):
```bash
export HEAD_NODE_IP=192.168.0.245
```

---

## 4 — Launch the head (terminal 1, stays foreground)

```bash
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
echo "Head IP: $VLLM_HOST_IP"

bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
-d \
  --restart unless-stopped \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$VLLM_HOST_IP \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e HF_TOKEN=$HF_TOKEN \
  -e VLLM_USE_RAY_COMPILED_DAG=0
```

---

## 5 — Launch the worker (terminal 2, ssh'd to worker, stays foreground)

```bash
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
echo "Worker IP: $VLLM_HOST_IP, head: $HEAD_NODE_IP"

bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
-d \
  --restart unless-stopped \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$HEAD_NODE_IP \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e HF_TOKEN=$HF_TOKEN \
  -e VLLM_USE_RAY_COMPILED_DAG=0
```

---

## 6 — Capture container name and verify cluster (terminal 3 on head)

```bash
export HEAD_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$' | head -1)
echo "Head container: $HEAD_CONTAINER"

docker logs --tail 80 $HEAD_CONTAINER     # confirm "Ray runtime started"
docker exec $HEAD_CONTAINER ray status    # expect 2 nodes / 2 GPUs

```

Expect 2 nodes / 2 GPUs. If you see 1/1 the worker hasn't joined — check networking before serving.

Notes for accessing the Ray Dashboard, might have to run. Also, it will name the CX7 ip address in the output from Ray, but you need to access it via the management IP. 

```bash
sudo ufw allow 8265/tcp
```

This will open the firewall on the head itself. 


## 7 - Pull the reasoning parser if it's not already on the head and worker, if it's there, skil the wget step and just copy it to the running docker container

```bash
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py
```

This runs on the HEAD
```bash
docker cp super_v3_reasoning_parser.py $HEAD_CONTAINER:/usr/local/lib/python3.12/dist-packages/vllm/reasoning/super_v3_reasoning_parser.py

docker exec $HEAD_CONTAINER bash -c '
INIT=/usr/local/lib/python3.12/dist-packages/vllm/reasoning/__init__.py
grep -q "super_v3" "$INIT" || cat >> "$INIT" <<'\''PY'\''

ReasoningParserManager.register_lazy_module(
    "super_v3",
    "vllm.reasoning.super_v3_reasoning_parser",
    "SuperV3ReasoningParser",
)
PY
'
```

This runs on the WORKER
```bash
export WORKER_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$' | head -1)
echo "Worker container: $WORKER_CONTAINER"

docker cp super_v3_reasoning_parser.py $WORKER_CONTAINER:/usr/local/lib/python3.12/dist-packages/vllm/reasoning/super_v3_reasoning_parser.py

docker exec $WORKER_CONTAINER bash -c '
INIT=/usr/local/lib/python3.12/dist-packages/vllm/reasoning/__init__.py
grep -q "super_v3" "$INIT" || cat >> "$INIT" <<'\''PY'\''

ReasoningParserManager.register_lazy_module(
    "super_v3",
    "vllm.reasoning.super_v3_reasoning_parser",
    "SuperV3ReasoningParser",
)
PY
'
```
---

## 8 — Launch `vllm serve`

```bash
docker exec -d $HEAD_CONTAINER /bin/bash -c '
  vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super \
    --port 8000 \
    --dtype auto \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --data-parallel-size 1 \
    --distributed-executor-backend ray \
    --trust-remote-code \
    --gpu-memory-utilization 0.80 \
    --enable-chunked-prefill \
    --max-num-seqs 8 \
    --max-model-len 1000000 \
    --mamba_ssm_cache_dtype float32 \
    --quantization modelopt \
    --reasoning-parser super_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    > /tmp/vllm-serve.log 2>&1
'
```

during detached startup, you can tail the logs:

```bash
docker exec $HEAD_CONTAINER tail -f /tmp/vllm-serve.log
```
---

## 9 — Smoke test

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-3-super",
    "messages": [{"role":"user","content":"Call the write_file tool to create test.txt with content hello"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {"type":"string"},
            "content": {"type":"string"}
          },
          "required": ["path","content"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 300
  }' | python3 -m json.tool
```

---
