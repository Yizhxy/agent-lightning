# Copyright (c) Microsoft. All rights reserved.

"""Train a coding agent on the SWE-bench dataset using Agent-lightning.

This module provides a training script for coding agents using different model configurations.
The script supports multiple training configurations:

1. 'fast' - A lightweight configuration optimized for CI testing with reduced epochs
2. 'qwen' - Standard configuration using Qwen3-4B-Instruct model
3. 'llama' - Configuration using LLaMA model with JSON formatting
4. 'npu' - Configuration for training with NPU

Usage:
    python train_cc_agent.py fast    # Fast training for CI/testing
    python train_cc_agent.py qwen    # Standard Qwen model training
    python train_cc_agent.py llama   # LLaMA model training

The script uses reinforcement learning with VERL framework
to train agents on the SWE-bench dataset for code generation tasks.
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from cc_agent import CodingAgent

import agentlightning as agl

RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "/mnt/input/swe_verl/swe_train_2.parquet",
        "val_files": "/mnt/input/swe_verl/swe_eval_2.parquet",
        "train_batch_size": 2,
        "max_prompt_length": 258048,
        "max_response_length": 4096,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 8,
            "n": 8,
            "log_prob_micro_batch_size_per_gpu": 1,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.8,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 4,
            "ppo_micro_batch_size_per_gpu": 1,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
            "ulysses_sequence_parallel_size": 8,
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 1,
            "fsdp_config": {"param_offload": True},
            "ulysses_sequence_parallel_size": 8,
        },
        "model": {
            "path": "Qwen/Qwen3-4B-Instruct-2507",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 8,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "cc_qwen3_4b_instruct",
        "nnodes": 4,
        "test_freq": 32,
        "total_epochs": 2,
        "balance_batch": False,
    },
}


def config_train_fast() -> Dict[str, Any]:
    """A fast training run for CI testing purposes."""

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_NAME = f"cc_{timestamp}"

    PROJECT_NAME = "AgentLightningCI"

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"project_name={PROJECT_NAME}\n")
            f.write(f"run_name={EXPERIMENT_NAME}\n")

    print("Set environment variables:")
    print(f"PROJECT_NAME={PROJECT_NAME}")
    print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
    config["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    config["data"]["val_files"] = "data/test_dev.parquet"
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["experiment_name"] = EXPERIMENT_NAME
    config["trainer"]["project_name"] = PROJECT_NAME
    config["trainer"]["test_freq"] = 1
    return config


def config_train_qwen() -> Dict[str, Any]:
    """A configuration for training with Qwen."""

    config = deepcopy(RL_TRAINING_CONFIG)
    return config


def config_train_npu() -> Dict[str, Any]:
    """A configuration for training with NPU."""

    config = deepcopy(RL_TRAINING_CONFIG)
    del config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["enable_auto_tool_choice"]
    del config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["tool_call_parser"]
    del config["trainer"]["logger"][1]
    config["actor_rollout_ref"]["actor"]["use_torch_compile"] = False
    config["trainer"]["val_before_train"] = False
    config["trainer"]["save_freq"] = 256
    config["trainer"]["device"] = "npu"
    return config


def config_train_llama() -> Dict[str, Any]:
    """A configuration for training with LLaMA.

    You will need a `HF_TOKEN` set to run with this config.
    """

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["multi_turn"]["format"] = "llama3_json"
    config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["tool_call_parser"] = "llama3_json"
    config["actor_rollout_ref"]["model"]["path"] = "meta-llama/Llama-3.2-1B-Instruct"
    return config


def train(config: Dict[str, Any], active_agent: Optional[str]) -> None:
    """Train the coding agent with the given configuration."""

    agent = CodingAgent()
    algorithm = agl.VERL(config)

    client = agl.LightningStoreClient("http://localhost:4747")
    trainer = agl.Trainer(store=client, n_runners=10, algorithm=algorithm, adapter={"agent_match": active_agent})
    print("Adapter agent match acknowledged:", trainer.adapter.agent_match)  # type: ignore

    train_data = pd.read_parquet(config["data"]["train_files"]).to_dict(orient="records")  # type: ignore
    val_data = pd.read_parquet(config["data"]["val_files"]).to_dict(orient="records")  # type: ignore
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore


def main() -> None:
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train a coding agent on the SWE-bench dataset using different model configurations"
    )

    parser.add_argument(
        "config",
        choices=["fast", "qwen", "llama", "npu"],
        help="Training configuration: 'fast' (CI testing), 'qwen' (Qwen3-4B), 'llama' (LLaMA), 'npu' (Train with NPU)",
    )

    parser.add_argument(
        "--active-agent", type=str, help="Override the active agent name (default: auto-generated based on config)"
    )

    args = parser.parse_args()

    config_functions = {
        "fast": config_train_fast,
        "qwen": config_train_qwen,
        "llama": config_train_llama,
        "npu": config_train_npu,
    }
    config = config_functions[args.config]()

    active_agent = args.active_agent

    print(f"Starting training with '{args.config}' configuration...")
    print(f"Active agent: {active_agent}")

    train(config, active_agent)


if __name__ == "__main__":
    main()
