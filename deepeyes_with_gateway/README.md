# DeepEyes-with-Gateway

This directory contains the DeepEyes recipe for the gateway-based agent path on
`main_ppo_sync.py`.

It is the multimodal counterpart to the legacy `recipe/deepeyes/` recipe:

- `recipe/deepeyes/` targets the legacy `main_ppo.py` path.
- `recipe/deepeyes_with_gateway/` targets the gateway + TransferQueue rollout
  path on `main_ppo_sync.py`.

## What lives here

- `dataset.py`: rewrites DeepEyes samples into gateway-ready multimodal
  messages.
- `agent_runner.py`: runs the in-process multi-turn tool loop through the
  gateway.
- `configs/deepeyes_gateway_grpo.yaml`: points `reward.custom_reward_function`
  at `recipe/deepeyes/deepeyes.py::compute_score` and wires
  `verl.agent.framework.entry.AgentFrameworkRolloutAdapter` as the
  `agent_loop_manager_class`.
- `run_deepeyes_gateway_grpo.sh`: example launch script for real-data training.

## Prerequisites

- Launch the LLM-as-a-judge service on GPU 7.
- Reserve GPUs 0-6 for training.
- Set `VERL_FORCE_TQ_NESTED_READBACK=1`.
- Point `LLM_AS_A_JUDGE_BASE` to the judge endpoint.
- Prepare the DeepEyes parquet, for example
  `/data1/datasets/deepeyes/data/data_0.1.2_visual_toolbox_v2.parquet`.

Judge example:

```bash
CUDA_VISIBLE_DEVICES=7 \
python3 -m vllm.entrypoints.openai.api_server \
  --model /data1/models/Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 \
  --port 18901 \
  --served-model-name qwen3-4b-judge \
  --dtype float16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager
```

## Example launch

From the verl repo root:

```bash
bash recipe/deepeyes_with_gateway/run_deepeyes_gateway_grpo.sh
```

The example script matches the full-shape run that completed 50/50 steps in
this redesign round:

- `data.train_batch_size=14`
- `actor_rollout_ref.rollout.n=4`
- `actor_rollout_ref.rollout.response_length=1024`
- `actor_rollout_ref.rollout.custom.agent_framework.agent_runner_kwargs.max_turns=5`
- `trainer.total_training_steps=50`

You can override the main inputs through environment variables before launch:

```bash
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
PROJECT_NAME=my_project \
EXPERIMENT_NAME=my_run \
TOTAL_TRAINING_STEPS=20 \
bash recipe/deepeyes_with_gateway/run_deepeyes_gateway_grpo.sh
```

## Notes

- This recipe keeps the tool loop in-process; it does not use the removed
  `external/` subprocess path.
- Runtime warnings from malformed tool calls or invalid crop boxes can still
  appear in logs; the recipe does not hide them.
- Reward computation still depends on the judge endpoint being reachable.
