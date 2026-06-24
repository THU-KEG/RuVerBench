# Dataset Card

## Scope

RuVerBench evaluates whether an LLM judge can verify individual rubrics for long agentic outputs.

The released benchmark covers two domains:

- DeepResearch: 284 scored cases and 1,615 rubric points.
- AgenticCoding: 210 cases and 843 checklist items.

Together, the benchmark contains 494 cases and 2,458 rubric-verification instances.

The packaged DeepResearch source files contain 298 question-report records. The
scored benchmark uses the 284 records that have final rubric taxonomy
assignments.

## Files

### DeepResearch

- `data/benchmark/deepresearch_dataset.json`: task prompts and rubric points.
- `data/benchmark/deepresearch_responses.json`: model responses to be judged.
- `data/benchmark/deepresearch_labels.json`: final human labels.
- `data/benchmark/deepresearch_taxonomy.json`: category assignments for rubric points.

### AgenticCoding

- `data/benchmark/agenticcoding_dataset.jsonl`: task/checklist definitions.
- `data/benchmark/agenticcoding_trajectories.jsonl`: serialized agent trajectories.
- `data/benchmark/agenticcoding_labels.json`: final human labels.
- `data/benchmark/agenticcoding_taxonomy.json`: category assignments for checklist items.

### Exported Leaderboard Results

`results/main_leaderboard/` contains exported main-leaderboard tables.
`data/predictions/examples/` contains example judge-prediction files that show
the expected output format.

### Strategy Analysis Files

`results/strategies/` contains exported prompt, batch, and voting tables.

`data/strategy_fixed20_subset/` contains the compact fixed-subset files used by
the strategy analyses. The subset labels mirror the main benchmark label
schema; any rubric weights needed for aggregation are read from the matching
subset dataset file.

## Label Meaning

Each rubric-verification instance has a binary label:

- `1`, `true`, or `success`: the output satisfies the rubric.
- `0`, `false`, or `fail`: the output does not satisfy the rubric.

The leaderboard reports category-level Balanced Accuracy and macro-averaged category BAcc.

## Reproducibility Boundary

The repository supports local recomputation of:

- paper result figures from packaged result tables.

Generating judge predictions requires an OpenAI-compatible API endpoint. Fresh
runs can differ when model versions, endpoint behavior, prompts, sampling
settings, or run dates differ.

## Data Provenance

The benchmark is assembled from DeepResearch and AgenticCoding task sources used
in the RuVerBench project. This public package keeps the release data compact:
the main benchmark and fixed-subset labels expose only the final label schema,
while source-file provenance is documented in the repository structure and
upstream data terms. AgenticCoding files include serialized trajectories,
tool-call traces, project constraints, and model-generated reasoning text.
Review the upstream data terms before redistributing derived versions.

## Recommended Use

Use this dataset for research on LLM-as-a-Judge, rubric verification, meta-evaluation, and category-level judge reliability. Do not treat exported leaderboard results as fresh model evaluations unless the model identity, endpoint, prompt, and run date are documented.
