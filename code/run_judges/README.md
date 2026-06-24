# Generating Judge Outputs

This directory contains the code used to call a judge model and generate prediction files for the benchmark.

## Components

```text
code/run_judges/deepresearch/     # DeepResearch evaluator
code/run_judges/agenticcoding/    # AgenticCoding evaluator
```

Convenience wrappers:

```text
code/run_judges/run_deepresearch_judge.sh
code/run_judges/run_agenticcoding_judge.sh
```

---

## Dependencies

For OpenAI-compatible API endpoints:

```bash
python3 -m pip install -r requirements.txt
```

If you use provider-specific SDK paths such as GLM/Zhipu, install and configure those separately.

---

## Environment variables

```bash
export JUDGE_MODEL="your-model-id-or-marker"
export JUDGE_BASE_URL="https://your-openai-compatible-endpoint/v1"
export JUDGE_API_KEY="your-api-key"
export MAX_WORKERS=4
export BATCH_SIZE=1
export PROMPT_STYLE=standard
export LIMIT=0
```

For stochastic self-voting runs, you may also set:

```bash
export VERIBENCH_JUDGE_TEMPERATURE=1
export VERIBENCH_DISABLE_JUDGE_CACHE=1
```

---

## DeepResearch

Run:

```bash
bash code/run_judges/run_deepresearch_judge.sh
```

Default inputs:

```text
MODEL_RESPONSE_FILE=data/benchmark/deepresearch_responses.json
RUBRICS_FILE=data/benchmark/deepresearch_dataset.json
```

Default output:

```text
outputs/generated_predictions/deepresearch/rubric_eval/<judge_model>/deepresearch_responses_evaluation_results.json
```

If you want to use the output in the leaderboard directory, save or rename the resulting JSON to:

```text
data/predictions/main_leaderboard/deepresearch/<model_id>.json
```

---

## AgenticCoding

Run:

```bash
bash code/run_judges/run_agenticcoding_judge.sh
```

Default inputs:

```text
TRAJECTORIES=data/benchmark/agenticcoding_trajectories.jsonl
DATA_FILE=data/benchmark/agenticcoding_dataset.jsonl
```

Default output:

```text
outputs/generated_predictions/agenticcoding/<JUDGE_MODEL>.json
```

If you want to use the output in the leaderboard directory, save or rename the resulting JSON to:

```text
data/predictions/main_leaderboard/agenticcoding/<model_id>.json
```

---

## Strategy experiments

The same evaluators can be reused by the strategy scripts in `code/strategies/`.

Typical use cases:

- prompt variants: set `PROMPT_STYLE=standard|semantic|strict|evidence_first`
- batch experiments: set `BATCH_SIZE`
- self-voting: repeat runs with stochastic sampling and aggregate the outputs
