import asyncio
import sys
import tempfile
import os
import socket
import json

# Note: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is incompatible with SGLang's TorchMemorySaver
# If memory fragmentation is an issue, try other approaches like reducing batch sizes

import requests
import ray
import fastapi
import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse
import verl

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from hydra import compose, initialize_config_dir
from verl.workers.rollout.replica import get_rollout_replica_class

from openai import AsyncOpenAI 

from verl.tools.kimina_tool import KiminaTool

from pprint import pprint

ray.init()
verl_config_dir = os.path.join(os.path.dirname(verl.__file__), "trainer/config")


snapshot_download(
    repo_id="AI-MO/NuminaMath-LEAN",
    repo_type="dataset",
    local_dir=os.path.expanduser("~/NuminaMath-LEAN"),
)
snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    repo_type="model",
    local_dir=os.path.expanduser("~/Qwen/Qwen3-1.7B"),
)

model_path = os.path.expanduser("~/Qwen/Qwen3-1.7B")
train_file = os.path.expanduser("~/data/numina-prover/train.parquet")
test_file = os.path.expanduser("~/data/numina-prover/test.parquet")

rollout_name = "sglang"

with initialize_config_dir(config_dir=verl_config_dir):
    config = compose(
        config_name="ppo_trainer",
        overrides=[
            "actor_rollout_ref.rollout.name=" + rollout_name,
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.model.path=" + model_path,
            "actor_rollout_ref.rollout.response_length=4096",
            "actor_rollout_ref.rollout.skip_tokenizer_init=False",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes",
            "+actor_rollout_ref.rollout.engine_kwargs.sglang.tool_call_parser=qwen25",
        ],
    )

rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
rollout_server = rollout_server_class(
    replica_rank=0,
    config=config.actor_rollout_ref.rollout,
    model_config=config.actor_rollout_ref.model,
)

# Create the tool instance for testing
kimina_tool = KiminaTool(
    config={
        "kimina_server_url": "http://localhost:8000",  # or your Kimina server URL
        "kimina_api_key": None,  # Optional API key if your server requires it
    },
    tool_schema=None
)

# Create tool config JSON file for training
tool_config = {
    "tools": [
        {
            "class_name": "verl.tools.kimina_tool.KiminaTool",
            "config": {
                "type": "native",
                "kimina_server_url": "http://localhost:8000",
                "kimina_api_key": None,
            },
        },
    ],
}

tool_config_path = "kimina_tool_config.json"
with open(tool_config_path, "w") as f:
    json.dump(tool_config, f, indent=2)
print(f"Created tool config at: {tool_config_path}")

async def main():
    await rollout_server.init_standalone()
    client = AsyncOpenAI(
        api_key="dummy",
        base_url=f"http://{rollout_server._server_address}/v1",
    )
    # Test 1: Execute a valid Lean 4 check
    print("=== Test 1: Valid Lean 4 check ===")
    valid_proof = "#check Nat"
    result = await kimina_tool.execute(instance_id="", parameters={"proof": valid_proof})
    print(f"Result: {result[0].text}")
    print()

    # Test 2: Execute invalid Lean 4 code (should show error)
    print("=== Test 2: Invalid Lean 4 code ===")
    invalid_proof = "#check undefined_symbol_xyz"
    result = await kimina_tool.execute(instance_id="", parameters={"proof": invalid_proof})
    print(f"Result: {result[0].text}")
    print()

    # Test 3: Test with a simple theorem
    print("=== Test 3: Simple theorem ===")
    theorem = """
    #check 2 + 2 = 4
    """
    result = await kimina_tool.execute(instance_id="", parameters={"proof": theorem})
    print(f"Result: {result[0].text}")
    print()

    # Test 4: Test ReAct agent loop with Kimina tool
    print("=== Test 4: ReAct agent loop with Kimina tool ===")
    messages = [{"role": "user", "content": "Can you verify that 1 + 1 = 2 in Lean 4?"}]

    while True:
        # 1. Chat with the model
        completion = await client.chat.completions.create(
            model=config.actor_rollout_ref.model.path,
            messages=messages,
            tools=[kimina_tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)],
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        messages.append(message)
        pprint(messages[-1])

        # 2. Call tools
        finish_reason = completion.choices[0].finish_reason
        if finish_reason != "tool_calls":
            print(f"No tool calls, finish_reason: {finish_reason}")
            break

        try:
            tool_calls = completion.choices[0].message.tool_calls[0]
            args = json.loads(tool_calls.function.arguments)
            result, _, _ = await kimina_tool.execute("", args)
        except Exception as e:
            print(f"Error: {e}")
            break

        # 3. Add tool response to messages
        messages.append(
            {
                "role": "tool",
                "content": result.text,
            }
        )

    print("\n=== Final conversation ===")
    print(messages)


def setup_training_config():
    """Setup and return training configuration for LEAN4 proof training."""
    # Check if parquet files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training parquet file not found: {train_file}\n"
            "Please run the preprocessing script first:\n"
            "python examples/data_preprocess/numina_prover.py --local_save_dir ~/data/numina-prover"
        )
    if not os.path.exists(test_file):
        print(f"Warning: Test parquet file not found: {test_file}")
        print("Training will proceed without validation dataset.")
        print("To create test dataset, run preprocessing with test_split_ratio > 0:\n"
              "python examples/data_preprocess/numina_prover.py --local_save_dir ~/data/numina-prover --test_split_ratio 0.2")
    
    with initialize_config_dir(config_dir=verl_config_dir):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                # Data configuration
                "algorithm.adv_estimator=grpo",
                "data.train_files=" + train_file,
                "data.val_files=" + (test_file if os.path.exists(test_file) else ""),
                "data.return_raw_chat=True",
                "data.train_batch_size=16",  # Reduced from 32 to 16 for memory efficiency
                "data.max_prompt_length=2048",  # LEAN4 theorems can be very long
                "data.max_response_length=4096",
                "data.truncation=right",  # Truncate from right if still too long
                "+data.apply_chat_template_kwargs.enable_thinking=False",
                
                # Actor configuration
                "actor_rollout_ref.model.path=" + model_path,
                "actor_rollout_ref.model.enable_gradient_checkpointing=True",  # Critical for long sequences to reduce memory
                "actor_rollout_ref.model.enable_activation_offload=True",  # Offload activations to CPU to save GPU memory
                "actor_rollout_ref.actor.ppo_mini_batch_size=8",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",  # Reduced to 2 for very long sequences (6144 tokens)
                "actor_rollout_ref.actor.use_kl_loss=True",  # Required for GRPO
                "actor_rollout_ref.actor.kl_loss_coef=0.001",
                "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
                # Reference model configuration (required for GRPO with use_kl_loss)
                "actor_rollout_ref.ref.fsdp_config.param_offload=True",  # Offload ref model params to CPU
                "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",  # Small batch for ref model
                
                # Rollout configuration with tool agent loop
                "actor_rollout_ref.rollout.name=" + rollout_name,
                "actor_rollout_ref.rollout.mode=async",
                "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                "actor_rollout_ref.rollout.n=8",
                "actor_rollout_ref.rollout.multi_turn.tool_config_path=" + tool_config_path,
                "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
                "actor_rollout_ref.rollout.prompt_length=2048",  # Match max_prompt_length
                "actor_rollout_ref.rollout.response_length=4096",
                "actor_rollout_ref.rollout.skip_tokenizer_init=False",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",  # Reduce rollout engine memory usage
                "actor_rollout_ref.rollout.free_cache_engine=False",  # Disable to reduce wake_up calls (partial workaround for pidfd_getfd)
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True",
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes",
                "+actor_rollout_ref.rollout.engine_kwargs.sglang.tool_call_parser=qwen25",
                
                # Reward function configuration
                "custom_reward_function.path=" + os.path.join(os.path.dirname(verl.__file__), "utils/reward_score/lean4_reward.py"),
                "custom_reward_function.name=compute_score",
                "+custom_reward_function.reward_kwargs.kimina_server_url=http://localhost:8000",
                
                # Trainer configuration
                "trainer.val_before_train=False",  # Disabled due to pidfd_getfd permission issue
                "trainer.log_val_generations=10",
                "trainer.n_gpus_per_node=1",
                "trainer.test_freq=-1",
                "trainer.total_training_steps=100",  # Adjust as needed
                "trainer.logger=['console','tensorboard', 'wandb']",
                "trainer.project_name=verl-lean4",
                "trainer.experiment_name=numina-lean4-training",
            ],
        )
    
    return config


if __name__ == "__main__":
    config = setup_training_config()
    from verl.trainer.main_ppo import main
    main(config)