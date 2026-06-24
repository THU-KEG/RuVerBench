# Run Steps

Run all commands from the repository root.

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 2. Inspect Exported Result Summaries

### Main Leaderboard

Exported leaderboard tables are included in:

```text
results/main_leaderboard/main_leaderboard.json
results/main_leaderboard/main_leaderboard.csv
results/main_leaderboard/main_leaderboard.md
results/main_leaderboard/main_leaderboard_table.tex
```

To compute leaderboard results for generated predictions, save the prediction
files under `data/predictions/main_leaderboard/`, then run:

```bash
python3 code/main_leaderboard/compute_main_leaderboard.py
```

Example prediction files are available under:

```text
data/predictions/examples/deepresearch/prediction_example.json
data/predictions/examples/agenticcoding/prediction_example.json
```

### Strategy Results

Exported strategy tables are included in:

```text
results/strategies/prompt/prompt_sensitivity_table.csv
results/strategies/prompt/prompt_sensitivity_table.json
results/strategies/prompt/prompt_sensitivity_table.md
results/strategies/batch/batch_size_trend.csv
results/strategies/batch/batch_size_trend.json
results/strategies/voting/self_voting_gain.csv
results/strategies/voting/self_voting_gain.json
results/dataset/fixed20_category_distribution.csv
results/dataset/fixed20_category_distribution.json
results/dataset/fixed20_category_distribution.md
```

The fixed subset files used by the strategy analyses are packaged in:

```text
data/strategy_fixed20_subset/
```

### Figures

```bash
python3 code/figures/plot_paper_figures.py
```

Outputs:

```text
results/figures/category_distribution_pie.pdf
results/figures/category_distribution_pie.png
results/figures/category_bacc_boxplot.pdf
results/figures/category_bacc_boxplot.png
results/figures/batch_size_trend.pdf
results/figures/batch_size_trend.png
results/figures/self_voting_gain.pdf
results/figures/self_voting_gain.png
```

## 3. Generate New Judge Predictions

Set an OpenAI-compatible endpoint:

```bash
export JUDGE_MODEL="your-model-id"
export JUDGE_BASE_URL="https://your-endpoint/v1"
export JUDGE_API_KEY="your-api-key"
export MAX_WORKERS=4
export BATCH_SIZE=1
export PROMPT_STYLE=standard
```

DeepResearch:

```bash
bash code/run_judges/run_deepresearch_judge.sh
```

AgenticCoding:

```bash
bash code/run_judges/run_agenticcoding_judge.sh
```

Save or move generated prediction files under:

```text
data/predictions/main_leaderboard/deepresearch/<model_id>.json
data/predictions/main_leaderboard/agenticcoding/<model_id>.json
```

Then rerun:

```bash
python3 code/main_leaderboard/compute_main_leaderboard.py
```

## 4. Run Strategy Generation Scripts

Fresh strategy runs require an OpenAI-compatible endpoint and can differ from
the exported strategy tables when model versions, endpoint behavior, prompt
settings, sampling settings, or run dates differ.

Batch-size and prompt-style generation on the fixed strategy subset:

```bash
bash code/strategies/run_batch_size_experiment.sh \
  --judge-model "$JUDGE_MODEL" \
  --judge-base-url "$JUDGE_BASE_URL" \
  --judge-api-key "$JUDGE_API_KEY" \
  --batch-sizes 1,2,4 \
  --prompt-style standard
```

Voting aggregation:

```bash
bash code/strategies/run_voting.sh \
  --domain deepresearch \
  --voter-files voter1.json voter2.json voter3.json \
  --gold data/strategy_fixed20_subset/deepresearch_subset_labels.json \
  --output outputs/generated_predictions/voting/deepresearch/majority.json
```

The aggregation script reads the compact fixed-subset labels. For DeepResearch,
rubric weights are recovered from `data/strategy_fixed20_subset/deepresearch_subset_dataset.json`.

## 5. Validate The Package

```bash
PYTHONPYCACHEPREFIX=/tmp/ruverbench_pycache \
  python3 -m py_compile $(find code -name '*.py')
find code -name '*.sh' -print -exec bash -n {} \;
```
