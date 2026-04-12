# ReasonIR Mixed Temperature Sweep GPT-4 BRIGHT Comparison

This document compares the completed mixed-data temperature sweep against the prior mixed baseline at the same batch size and learning rate.

Reference runs:

- mixed baseline `bs=2048 lr=8e-5 temp=1.0`: [reasonir-mixed-bs2048-lr8e5-gpt4-full.json](/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5-gpt4-full.json)
- best lower-LR sweep result `bs=2048 lr=5e-5 temp=1.0`: [reasonir-mixed-lr-bs2048-lr5e-5-gpt4-full.json](/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr5e-5-gpt4-full.json)

Temperature sweep result files:

- `temp=0.05`: [reasonir-mixed-temp-bs2048-lr8e-5-temp005-gpt4-full.json](/home/rbw/repo/pylate/output/reasonir-mixed-temp-bs2048-lr8e-5-temp005-gpt4-full.json)
- `temp=0.1`: [reasonir-mixed-temp-bs2048-lr8e-5-temp01-gpt4-full.json](/home/rbw/repo/pylate/output/reasonir-mixed-temp-bs2048-lr8e-5-temp01-gpt4-full.json)
- `temp=0.25`: [reasonir-mixed-temp-bs2048-lr8e-5-temp025-gpt4-full.json](/home/rbw/repo/pylate/output/reasonir-mixed-temp-bs2048-lr8e-5-temp025-gpt4-full.json)
- `temp=0.5`: [reasonir-mixed-temp-bs2048-lr8e-5-temp05-gpt4-full.json](/home/rbw/repo/pylate/output/reasonir-mixed-temp-bs2048-lr8e-5-temp05-gpt4-full.json)

Reference control:

- recorded base `ColBERT-Zero` full GPT-4 BRIGHT mean: `26.51`
- local full base-model GPT-4 JSON was not present in this checkout when this comparison was written

## Overall Ranking

All rows below are complete 12-task GPT-4 BRIGHT evals.

| Rank | Run | Full mean | Delta vs base `26.51` | Delta vs mixed baseline `27.12` |
| --- | --- | ---: | ---: | ---: |
| 1 | mixed temp `0.5` | 27.72 | +1.21 | +0.60 |
| 2 | mixed temp `0.1` | 27.56 | +1.05 | +0.44 |
| 3 | mixed temp `0.25` | 27.48 | +0.97 | +0.36 |
| 4 | mixed baseline `temp=1.0` | 27.12 | +0.61 | +0.00 |
| 5 | mixed temp `0.05` | 26.91 | +0.40 | -0.21 |
| 6 | best lower-LR sweep `lr=5e-5 temp=1.0` | 26.89 | +0.38 | -0.23 |

Summary:

- The winner of this round is `temp=0.5`.
- `temp=0.5`, `0.1`, and `0.25` all beat the prior mixed baseline `temp=1.0`.
- `temp=0.05` underperformed the prior mixed baseline and only barely stayed ahead of the best lower-LR sweep winner.

## Group Means

The BRIGHT summaries also break out StackExchange, coding, and theorem subsets.

| Run | Mean StackExchange | Mean coding | Mean theorem | Full mean |
| --- | ---: | ---: | ---: | ---: |
| mixed baseline `temp=1.0` | 33.87 | 13.58 | 21.78 | 27.12 |
| mixed temp `0.05` | 34.05 | 13.55 | 21.96 | 26.91 |
| mixed temp `0.1` | 34.34 | 14.64 | 23.19 | 27.56 |
| mixed temp `0.25` | 34.44 | 14.48 | 22.61 | 27.48 |
| mixed temp `0.5` | 34.89 | 14.46 | 22.51 | 27.72 |

## Per-Task NDCG@10

Values are percentages.

| Task | Baseline `temp=1.0` | `temp=0.05` | `temp=0.1` | `temp=0.25` | `temp=0.5` | Best run |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AoPS | 7.88 | 7.10 | 8.02 | 7.31 | 7.43 | `temp=0.1` |
| Biology | 54.98 | 53.27 | 51.70 | 51.46 | 53.75 | baseline `temp=1.0` |
| Earth Science | 57.47 | 55.85 | 56.26 | 56.50 | 56.39 | baseline `temp=1.0` |
| Economics | 24.72 | 23.52 | 24.48 | 25.69 | 26.16 | `temp=0.5` |
| Leetcode | 28.92 | 28.88 | 30.38 | 29.92 | 29.59 | `temp=0.1` |
| Pony | 3.94 | 4.67 | 5.53 | 6.22 | 6.35 | `temp=0.5` |
| Psychology | 32.34 | 32.63 | 33.63 | 34.74 | 34.63 | `temp=0.25` |
| Robotics | 19.12 | 21.57 | 22.21 | 21.36 | 20.52 | `temp=0.1` |
| Stackoverflow | 29.73 | 26.93 | 28.27 | 28.65 | 29.29 | baseline `temp=1.0` |
| Sustainable Living | 22.72 | 24.57 | 23.80 | 22.69 | 23.50 | `temp=0.05` |
| TheoremQA Questions | 26.70 | 28.21 | 28.75 | 28.26 | 28.51 | `temp=0.1` |
| TheoremQA Theorems | 16.86 | 15.71 | 17.64 | 16.97 | 16.51 | `temp=0.1` |

## Delta Vs Mixed Baseline `temp=1.0`

Positive values mean the temperature sweep run beat the prior mixed baseline on that task.

| Task | `temp=0.05` | `temp=0.1` | `temp=0.25` | `temp=0.5` |
| --- | ---: | ---: | ---: | ---: |
| AoPS | -0.77 | +0.14 | -0.57 | -0.45 |
| Biology | -1.72 | -3.28 | -3.52 | -1.24 |
| Earth Science | -1.62 | -1.21 | -0.97 | -1.08 |
| Economics | -1.20 | -0.24 | +0.97 | +1.44 |
| Leetcode | -0.04 | +1.46 | +0.99 | +0.67 |
| Pony | +0.73 | +1.59 | +2.28 | +2.41 |
| Psychology | +0.29 | +1.29 | +2.41 | +2.29 |
| Robotics | +2.45 | +3.10 | +2.25 | +1.40 |
| Stackoverflow | -2.80 | -1.46 | -1.09 | -0.44 |
| Sustainable Living | +1.85 | +1.08 | -0.03 | +0.78 |
| TheoremQA Questions | +1.51 | +2.05 | +1.56 | +1.81 |
| TheoremQA Theorems | -1.16 | +0.78 | +0.11 | -0.36 |

## Interpretation

- `temp=0.5` wins overall because it combines the strongest `economics` result, the best `pony` result, a strong `psychology` score, and the least damaging `stackoverflow` drop among the sub-1.0 temperatures.
- `temp=0.1` is the sharpest specialist setting. It is best on `AoPS`, `leetcode`, `robotics`, `theoremqa_questions`, and `theoremqa_theorems`, but its `biology` regression is too large to win overall.
- `temp=0.25` is strongest on `psychology` and also improves `economics`, but it gives back too much on `biology`.
- `temp=0.05` is too cold for the full benchmark. It helps `sustainable_living` and `robotics`, but the `stackoverflow`, `biology`, and `earth_science` losses outweigh those gains.

## Recommendation

- Promote `reasonir-mixed-temp-bs2048-lr8e-5-temp05` as the new best mixed-model checkpoint for GPT-4 BRIGHT.
- Keep `temp=0.1` as the strongest alternative if future work wants to bias toward coding/theorem-style tasks.
- If another sweep is needed, a focused refinement around the winning region would be more justified than another broad LR sweep. The obvious next grid would be around `temp=0.25` to `0.75`.
