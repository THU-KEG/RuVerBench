# RuVerBench

RuVerBench is a rubric verification benchmark for evaluating LLM-as-a-Judge reliability in long agentic scenarios.

The benchmark asks a judge model to decide whether an output satisfies an individual rubric. Each judgment is evaluated against a human gold label.

Code and data are licensed separately. See `LICENSE` for code and
`DATA_LICENSE.md` for dataset terms and third-party notices.

## Domains

- DeepResearch: question-report pairs.
- AgenticCoding: serialized agent trajectories.

The scored benchmark contains 494 cases and 2,458 rubric-verification instances:

- DeepResearch: 284 scored cases, 1,615 rubric points.
- AgenticCoding: 210 cases, 843 checklist items.

The packaged DeepResearch source files contain 298 question-report records. The
main benchmark and leaderboard use the 284 records that have final rubric
taxonomy assignments.

Released benchmark files contain normalized inputs, rubrics/checklists, labels,
responses/trajectories, and taxonomy assignments. The fixed strategy subset
uses the same compact label schema as the main benchmark.

## Repository Layout

```text
RuVerBench/
  README.md
  RUN_STEPS.md
  DATASET_CARD.md
  DATA_LICENSE.md
  requirements.txt

  data/
    benchmark/                  benchmark inputs, labels, and taxonomy files
    predictions/examples/       example judge-prediction files
    strategy_fixed20_subset/    fixed subset files for strategy analyses

  code/
    main_leaderboard/           leaderboard recomputation
    run_judges/                 judge-output generation wrappers
    strategies/                 strategy generation and aggregation scripts
    figures/                    paper-result figure generation
    prompts/                    judge prompts

  results/
    main_leaderboard/           recomputed leaderboard tables
    dataset/                    dataset/category summaries
    strategies/                 prompt, batch, and voting summaries
    error_analysis/             model-error overlap files
    figures/                    generated paper-result figures
```

## Local Reproduction

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Regenerate figures from exported result tables:

```bash
python3 code/figures/plot_paper_figures.py
```

This command uses exported result tables and does not call external model APIs.

The repository includes exported leaderboard tables under
`results/main_leaderboard/`. Generate new prediction files with the judge
wrappers if you want to compute leaderboard results for additional models.
Example prediction files are provided under `data/predictions/examples/`.

Strategy tables are exported under `results/strategies/`. The fixed subset used
by the strategy analyses is packaged under `data/strategy_fixed20_subset/`.
Fresh API-based strategy runs are supported by the scripts in `code/strategies/`,
but exact raw regeneration can change with endpoint behavior, model version,
prompt settings, sampling settings, and run date.

## API-Based Regeneration

Fresh judge predictions require an OpenAI-compatible API endpoint:

```bash
export JUDGE_MODEL="your-model-id"
export JUDGE_BASE_URL="https://your-endpoint/v1"
export JUDGE_API_KEY="your-api-key"
```

Then run:

```bash
bash code/run_judges/run_deepresearch_judge.sh
bash code/run_judges/run_agenticcoding_judge.sh
```

## Reproducibility Boundary

This release supports:

- inspecting the exported main leaderboard tables;
- inspecting exported paper strategy tables;
- inspecting the fixed subset files used by the strategy analyses;
- regenerating result figures from packaged tables.

Fresh model generation is outside the deterministic reproduction path. Results
from new API calls depend on endpoint behavior, model version, prompt settings,
sampling settings, and run date.

## Main Files To Read

1. `DATASET_CARD.md`
2. `RUN_STEPS.md`
3. `results/main_leaderboard/main_leaderboard.md`
4. `results/strategies/`
5. `results/error_analysis/`
