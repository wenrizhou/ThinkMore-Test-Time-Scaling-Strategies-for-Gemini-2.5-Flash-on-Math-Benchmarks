# ThinkMore: Test-Time Scaling Strategies for Gemini-2.5-Flash on Math Benchmarks

## Overview
This repository contains the code and experiment results for an NLP project that evaluates **test-time scaling** strategies on **Gemini-2.5-Flash** for math reasoning. We compare five inference-time methods—**Chain-of-Thought (CoT)**, **Self-Consistency (SC)**, **Self-Reflection**, **Planned Reasoning (Plan-and-Solve)**, and **Program-of-Thought (PoT)**—against a direct-answer baseline, and report both **accuracy** and **end-to-end wall-clock time**.

项目研究 **test-time scaling（测试时扩展）**：在不进行训练/微调的情况下，通过在推理阶段引入更多结构化推理与计算（提示、采样聚合、自我校验、规划、工具/程序执行等）来提升数学推理性能。我们在 **Gemini-2.5-Flash** 上对比 5 种推理策略（CoT、Self-Consistency、Self-Reflection、Planned Reasoning、Program-of-Thought），并与直接作答的 baseline 进行对比，统计 **准确率** 与 **端到端耗时**。

---

## Methods
All methods use the same base model and differ only in inference-time procedures:
- **Baseline**: direct answer.
- **CoT**: generate step-by-step reasoning before the final answer.
- **Self-Consistency (K=5)**: sample multiple CoT paths (temperature 0.7) and vote.
- **Self-Reflection**: produce an initial solution, then verify and revise in a second pass.
- **Planned Reasoning**: plan first, then solve using the plan.
- **Program-of-Thought (PoT)**: generate Python code for computation, execute it, and output the final answer.

所有方法使用同一个基础模型，差异仅在 **推理阶段** 的流程：
- **Baseline**：直接给出答案。
- **CoT**：先生成逐步推理过程，再输出最终答案。
- **Self-Consistency（K=5）**：以温度 0.7 采样多条 CoT 推理路径，并进行投票聚合。
- **Self-Reflection**：先给出初解，再进行二次审查与修正输出。
- **Planned Reasoning**：先生成解题计划，再依据计划求解。
- **Program-of-Thought（PoT）**：生成可执行的 Python 计算代码，运行得到数值结果并输出答案。

---

## Datasets
- **AIME24&25**: problems from the 2024 and 2025 American Invitational Mathematics Examination (AIME).
- **MAMO**: optimization / linear-programming word problems with numeric answers.

- **AIME24&25**：来自 2024 与 2025 年 AIME（American Invitational Mathematics Examination）数学竞赛题目。
- **MAMO**：偏优化/线性规划风格的数学文字题（数值答案）。

---

## Main Results (Accuracy %)
| Method | AIME | MAMO |
|---|---:|---:|
| Baseline | 66.6 | 19.9 |
| CoT prompt | 76.6 | 27.9 |
| Self-Consistency (5 samples) | 81.6 | 30.8 |
| Self-Reflection | **85.0** | 27.4 |
| Planned Reasoning | 73.3 | 14.6 |
| Program-of-Thought (PoT) | 68.3 | **44.0** |

下表为主要实验结果（准确率 %）：
| 方法 | AIME | MAMO |
|---|---:|---:|
| Baseline | 66.6 | 19.9 |
| CoT | 76.6 | 27.9 |
| Self-Consistency（5 次采样） | 81.6 | 30.8 |
| Self-Reflection | **85.0** | 27.4 |
| Planned Reasoning | 73.3 | 14.6 |
| Program-of-Thought（PoT） | 68.3 | **44.0** |

---

## Runtime (Wall-clock seconds)
| Method | AIME | MAMO |
|---|---:|---:|
| Baseline | 32.4 | 30.2 |
| CoT prompt | 73.2 | 71.8 |
| Self-Consistency (5 samples) | 364.5 | 355.2 |
| Self-Reflection | 138.8 | 116.7 |
| Planned Reasoning | 70.6 | 67.0 |
| Program-of-Thought (PoT) | **22.9** | **20.5** |

下表为端到端耗时（秒，wall-clock time）：
| 方法 | AIME | MAMO |
|---|---:|---:|
| Baseline | 32.4 | 30.2 |
| CoT | 73.2 | 71.8 |
| Self-Consistency（5 次采样） | 364.5 | 355.2 |
| Self-Reflection | 138.8 | 116.7 |
| Planned Reasoning | 70.6 | 67.0 |
| Program-of-Thought（PoT） | **22.9** | **20.5** |

---

## Key Takeaways
- **AIME**: **Self-Reflection** achieves the best accuracy (85.0% vs 66.6% baseline).
- **MAMO**: **PoT** yields the largest accuracy gain (44.0% vs 19.9% baseline), and is also the fastest in our measured wall-clock time.

- **AIME**：**Self-Reflection** 准确率最高（85.0%，相比 baseline 的 66.6% 提升明显）。
- **MAMO**：**PoT** 提升最大（44.0% vs 19.9%），并且在本次测量中端到端耗时也最低。

---

## Repository Structure
- `code/` — evaluation scripts, prompting pipelines, PoT execution, utilities.
- `results_AIME/` — outputs and aggregated results for AIME24&25.
- `results_mamo/` — outputs and aggregated results for MAMO.

- `code/` —— 核心代码（评测脚本、提示策略流程、PoT 执行与工具函数等）。
- `results_AIME/` —— AIME24&25 的输出与结果汇总。
- `results_mamo/` —— MAMO 的输出与结果汇总。

---

## Quick Start (Example)
> The exact commands depend on the entry scripts inside `code/`.

```bash
# (optional) create environment
pip install -r requirements.txt

# set your API key if needed
export GEMINI_API_KEY="YOUR_KEY"

# run experiments (examples)
python code/run_aime.py
python code/run_mamo.py
