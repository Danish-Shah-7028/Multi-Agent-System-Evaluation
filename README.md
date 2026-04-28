# Multi-Agent vs Single-Agent LLM Study

An empirical comparison of three LLM-based systems for multi-step task automation:

- **System A**: single-agent end-to-end solver
- **System B**: planner-executor-reviewer pipeline
- **System C**: planner-executor-reviewer with cross-task memory

The study evaluates all three systems on **25 tasks** across five categories: reasoning, planning, coding, summarization, and information extraction. Success is defined by a **semantic similarity score of at least 0.8** against the reference answer.

## Key Results

The full run completed successfully. System C performed best overall, followed by System B, with System A trailing.

| System | Success Rate | Avg. Similarity | Avg. Tokens | Total Tokens |
| --- | ---: | ---: | ---: | ---: |
| A | 16.0% | 0.683 | 505.3 | 12,632 |
| B | 44.0% | 0.792 | 574.8 | 14,370 |
| C | 56.0% | 0.836 | 645.5 | 16,137 |

Category-level success rates showed the same pattern in most task groups, with System C leading reasoning, planning, and summarization tasks. Coding and information extraction remained harder for all systems.

## What This Project Does

This repository implements a controlled experiment to answer whether multi-agent systems outperform single-agent systems for multi-step automation tasks in terms of success rate, cost, and reliability.

The three systems share the same model choice, **Groq `groq/compound-mini`**, so the comparison is not confounded by model changes. When multiple Groq credentials are provided, requests are rotated across up to four API keys to reduce rate-limit pressure while keeping the model fixed.

## Repository Structure

- `run_experiments.py` - main entry point for the full study
- `config.yaml` - experiment settings, thresholds, and system configuration
- `data/tasks_dataset.json` - the 25-task benchmark dataset
- `src/` - core experiment logic, system implementations, evaluator, and memory store
- `results/` - generated CSV, JSON, logs, and memory summaries
- `scripts/` - helper scripts for model discovery and simulation

## Requirements

- Python 3.14+
- Groq API access
- Packages listed in `requirements.txt`

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root. use Groq's API key:

```env
GROQ_API_KEY=your_api_key
```

# Keep the same model across all systems for a fair comparison
GROQ_MODEL=groq/compound-mini
```

## How to Run

Run the complete experiment:

```bash
python run_experiments.py
```

Useful options:

```bash
python run_experiments.py --tasks 2
python run_experiments.py --clear-memory
python run_experiments.py --config config.yaml --data data/tasks_dataset.json
```

## Output Files

After a run, results are written to `results/`:

- `experiment_results.csv` - per-task, per-system comparison table
- `experiment_results.json` - JSON version of the same results
- `experiment.log` - full execution log
- `raw_logs.json` - detailed outputs for the first tasks
- `memory_summary.json` - System C memory state

## Methodology

Each task is run independently through all three systems. Outputs are scored with a local semantic similarity evaluator, which avoids extra API calls and keeps the experiment focused on generation quality.

Success is counted when the similarity score reaches or exceeds `0.8`.

## Notes on Fair Comparison

- All three systems use the same model: `groq/compound-mini`
- API key rotation is only for operational stability and quota management
- The evaluator is local, so scoring does not consume Groq requests
- System C can reuse prior task strategies through memory, which is part of the experimental design

## Reproducibility

To reproduce the study, keep the dataset, configuration, model, and success threshold unchanged, then run the experiment again with the same `.env` setup.

