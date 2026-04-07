# Training ANY LLM to Claude Code

## Requirements
1. install [agentlightning](https://microsoft.github.io/agent-lightning/stable/tutorials/installation/)
2. `(uv) pip install swebench` for evaluation
3. At least one server having privilege to execute docker containers, and at least one GPU server to support RL training. The two servers may or may not be colocated.


## ⚠️ Important Notice (Temporary Version)

⚠️ This is a temporary implementation.


The current algorithm implementation is based on
/examples/spider


Actual rollouts are specified by the runner, so this does NOT affect the core algorithm logic
All related code will be migrated to /examples/cc later

## Execution

#### 1. Update Training Configuration
Please update your configuration in:

``` shell
/examples/spider/train_sql_agent.py
```


#### 2. Network & Port Requirements
Ensure the following ports are reachable between CPU and GPU servers:
Default Port
AGL Store  4747
Runner 8765

#### 3. Start Store (GPU Server)
On the GPU server, start the Agent Lightning store:
``` shell
agl store --port 4747
```

#### 4. Start Algorithm (GPU Server)
On the same GPU server, start the RL algorithm process:
``` shell
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=algorithm \
    python train_sql_agent.py qwen
```

#### 5. Start Runner (CPU Server)
On the CPU server (Docker-enabled), start the rollout runner:
``` shell
python rollout_runner.py
```

## Temporary Full Installation Commands
``` shell
git clone https://github.com/Yizhxy/agent-lightning.git
cd agent-lightning
git checkout sync_main
uv sync --frozen \
    --extra apo \
    --extra verl \
    --group dev \
    --group torch-gpu-stable \
    --group trl \
    --group agents \
    --no-default-groups
source .venv/bin/activate

uv pip install 'litellm[proxy]'==1.80.16

cd examples/cc
uv pip install -r requirements.txt

cd ..
cd spider
uv pip install "langgraph<1.0" "langchain[openai]<1.0" "langchain-community" "langchain-text-splitters<1.0" "sqlparse" "nltk"
```