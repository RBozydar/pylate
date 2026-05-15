# ReasonIR Mixed LR Sweep GPT-4 BRIGHT Comparison

This document compares the completed mixed-data GPT-4 BRIGHT evals for the lower-LR sweep against the current mixed baseline.

Result files:

- mixed baseline: [reasonir-mixed-bs2048-lr8e5-gpt4-full.json](/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5-gpt4-full.json)
- `lr5e-5`: [reasonir-mixed-lr-bs2048-lr5e-5-gpt4-full.json](/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr5e-5-gpt4-full.json)
- `lr3e-5`: [reasonir-mixed-lr-bs2048-lr3e-5-gpt4-full.json](/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr3e-5-gpt4-full.json)
- `lr5e-6`: [reasonir-mixed-lr-bs2048-lr5e-6-gpt4-full.json](/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr5e-6-gpt4-full.json)
- `lr1e-5`: [reasonir-mixed-lr-bs2048-lr1e-5-gpt4-full.json](/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr1e-5-gpt4-full.json)

Reference control:

- recorded base `ColBERT-Zero` full GPT-4 BRIGHT mean: `26.51`
- local full base-model GPT-4 JSON was not present in this checkout when this comparison was written

## Overall Ranking

All runs below are complete 12-task GPT-4 BRIGHT evals.

| Rank | Run | Full mean | Delta vs base `26.51` | Delta vs mixed baseline `27.12` |
| --- | --- | ---: | ---: | ---: |
| 1 | mixed baseline `lr8e5` | 27.12 | +0.61 | +0.00 |
| 2 | mixed LR `5e-5` | 26.89 | +0.38 | -0.23 |
| 3 | mixed LR `3e-5` | 26.81 | +0.30 | -0.31 |
| 4 | mixed LR `5e-6` | 26.60 | +0.09 | -0.52 |
| 5 | mixed LR `1e-5` | 26.26 | -0.25 | -0.86 |

Summary:

- The sweep winner is `lr5e-5`.
- None of the lower-LR runs beat the existing mixed baseline `lr8e5`.
- Three of the four lower-LR runs still beat the recorded base control overall.

## Per-Task NDCG@10

Values are percentages.

| Task | Baseline `lr8e5` | `lr5e-5` | `lr3e-5` | `lr5e-6` | `lr1e-5` | Best run |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AoPS | 7.88 | 6.95 | 7.20 | 7.54 | 7.25 | baseline `lr8e5` |
| Biology | 54.98 | 53.70 | 53.69 | 50.72 | 51.42 | baseline `lr8e5` |
| Earth Science | 57.47 | 56.56 | 55.96 | 55.47 | 54.87 | baseline `lr8e5` |
| Economics | 24.72 | 24.60 | 25.04 | 23.67 | 23.09 | `lr3e-5` |
| Leetcode | 28.92 | 28.10 | 27.44 | 31.52 | 31.38 | `lr5e-6` |
| Pony | 3.94 | 3.49 | 3.05 | 5.61 | 3.82 | `lr5e-6` |
| Psychology | 32.34 | 33.72 | 35.23 | 34.21 | 33.17 | `lr3e-5` |
| Robotics | 19.12 | 19.70 | 19.89 | 20.60 | 20.07 | `lr5e-6` |
| Stackoverflow | 29.73 | 29.13 | 27.96 | 24.46 | 25.00 | baseline `lr8e5` |
| Sustainable Living | 22.72 | 23.12 | 24.41 | 21.37 | 21.54 | `lr3e-5` |
| TheoremQA Questions | 26.70 | 26.69 | 26.29 | 25.59 | 26.05 | baseline `lr8e5` |
| TheoremQA Theorems | 16.86 | 16.94 | 15.50 | 18.47 | 17.48 | `lr5e-6` |

## Delta Vs Mixed Baseline

Positive values mean the sweep run beat the mixed baseline on that task.

| Task | `lr5e-5` | `lr3e-5` | `lr5e-6` | `lr1e-5` |
| --- | ---: | ---: | ---: | ---: |
| AoPS | -0.93 | -0.68 | -0.33 | -0.63 |
| Biology | -1.29 | -1.29 | -4.26 | -3.56 |
| Earth Science | -0.91 | -1.51 | -2.00 | -2.60 |
| Economics | -0.12 | +0.32 | -1.05 | -1.63 |
| Leetcode | -0.82 | -1.48 | +2.59 | +2.46 |
| Pony | -0.45 | -0.89 | +1.67 | -0.12 |
| Psychology | +1.38 | +2.90 | +1.88 | +0.84 |
| Robotics | +0.58 | +0.78 | +1.48 | +0.95 |
| Stackoverflow | -0.60 | -1.77 | -5.27 | -4.73 |
| Sustainable Living | +0.40 | +1.68 | -1.36 | -1.19 |
| TheoremQA Questions | -0.01 | -0.41 | -1.11 | -0.65 |
| TheoremQA Theorems | +0.07 | -1.36 | +1.61 | +0.62 |

## Interpretation

- `lr5e-5` wins overall because it loses less than the other lower-LR runs on the large StackExchange-heavy tasks while still improving `psychology`, `robotics`, `sustainable_living`, and `theoremqa_theorems`.
- `lr3e-5` is the strongest run on `economics`, `psychology`, and `sustainable_living`, but it gives back too much on `stackoverflow`, `earth_science`, and `biology` to beat `lr5e-5` overall.
- `lr5e-6` is best on `leetcode`, `pony`, `robotics`, and `theoremqa_theorems`, but its `stackoverflow` and `biology` regressions are too large.
- `lr1e-5` is the weakest of the completed sweep runs overall.

## Recommendation

- Keep `reasonir-mixed-bs2048-lr8e5` as the current best mixed-model checkpoint for GPT-4 BRIGHT.
- If a lower-LR follow-up is still desired, `reasonir-mixed-lr-bs2048-lr5e-5` is the best next checkpoint from this sweep.
