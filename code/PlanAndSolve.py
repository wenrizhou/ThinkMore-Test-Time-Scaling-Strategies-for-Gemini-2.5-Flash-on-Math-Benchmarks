import os
import json
import csv
import re
from typing import Optional, Tuple

from google import genai
from tqdm import tqdm

# ========= 配置 =========
MODEL_NAME = "gemini-2.5-flash"

INPUT_JSONL = "mamo_complex_lp.jsonl"
OUTPUT_JSONL = "mamo_complex_lp_plan_results.jsonl"
OUTPUT_CSV = "mamo_complex_lp_plan_results.csv"

os.environ["GEMINI_API_KEY"] = "AIzaSyDMAx50WY_6b5kpHaa8w3evwAcs5MhVnQ8"

client = genai.Client()


# ========= Prompts (Two-Stage) =========

def make_plan_prompt(question: str) -> str:
    """
    第一阶段：生成计划 (Planning)
    只要求生成步骤，不要直接给出最终数字答案。
    """
    return f"""You are a strategic mathematical planner.
Your goal is to create a detailed, step-by-step plan to solve the linear programming problem below.

Requirements for the Plan:
1. Break down the problem into logical sub-tasks (e.g., Define variables, Formulate Objective, List Constraints, Solve System, etc.).
2. Do NOT calculate the final numerical answer yet.
3. Just list the steps clearly.

Problem:
{question}

Output the plan now:
"""

def make_execution_prompt(question: str, plan: str) -> str:
    """
    第二阶段：执行计划 (Execution)
    基于上一步生成的 Plan 进行具体计算。
    """
    return f"""You are a precise mathematical solver.
I will provide you with a problem and a plan to solve it.

Task:
1. Execute the plan step by step strictly.
2. Show all calculations clearly.
3. After finishing the execution, output the final numerical answer on a NEW line.

Format:
FINAL_ANSWER: <number>
(Round to exactly ONE decimal place)

---
Problem:
{question}

---
Plan to follow:
{plan}

---
Now, execute the plan and provide the final answer:
"""


# ========= 工具函数 (复用) =========

def call_gemini(prompt: str) -> str:
    """
    通用调用函数。
    Plan-and-Solve 通常不需要高随机性，保持默认或低温度即可。
    """
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            # 这里的 config 可以选填，不填默认 temp 这里的模型通常是适中的
        )
        return (resp.text or "").strip()
    except Exception as e:
        print(f"[WARN] API call failed: {e}")
        return ""

_final_answer_pattern = re.compile(r"FINAL_ANSWER:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def extract_final_answer(text: Optional[str]) -> Optional[float]:
    if not text: return None
    m = _final_answer_pattern.search(text)
    if m:
        try: return float(m.group(1))
        except: pass
    nums = _number_pattern.findall(text)
    if not nums: return None
    try: return float(nums[-1])
    except: return None

def to_one_decimal(x: Optional[float]) -> Optional[float]:
    if x is None: return None
    return round(x, 1)

# ========= 核心逻辑：Plan -> Execute =========

def run_plan_and_solve(question: str) -> Tuple[str, str, Optional[float]]:
    """
    执行 PS 流程
    Returns:
        plan_text: 生成的计划文本
        exec_text: 执行过程文本
        final_val: 提取出的数值
    """
    # 1. 生成计划
    plan_prompt = make_plan_prompt(question)
    plan_text = call_gemini(plan_prompt)
    
    if not plan_text:
        return "", "Error: Plan generation failed", None

    # 2. 执行计划
    exec_prompt = make_execution_prompt(question, plan_text)
    exec_text = call_gemini(exec_prompt)
    
    # 3. 提取答案
    val = to_one_decimal(extract_final_answer(exec_text))
    
    return plan_text, exec_text, val


# ========= 主流程 =========

def main():
    # 统计行数
    total_lines = 0
    if os.path.exists(INPUT_JSONL):
        with open(INPUT_JSONL, "r", encoding="utf-8") as f:
            for _ in f: total_lines += 1

    print(f"Total examples: {total_lines}")
    print(f"Mode: Plan-and-Solve (2-Stage Prompting)")

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fjson_out, \
         open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fcsv_out:

        csv_writer = csv.writer(fcsv_out)
        csv_writer.writerow([
            "id",
            "question",
            "gold_answer_num",
            "pred_answer_num",
            "correct",
            "plan_text",        # 保存生成的计划
            "execution_text"    # 保存执行的过程
        ])
        fcsv_out.flush()

        for line in tqdm(fin, total=total_lines, desc="Processing (PS)"):
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
            except:
                continue

            qid = data.get("id")
            question = data.get("Question")
            gold_raw = data.get("Answer")
            
            gold_num = None
            if gold_raw is not None:
                try: gold_num = to_one_decimal(float(gold_raw))
                except: gold_num = to_one_decimal(extract_final_answer(str(gold_raw)))

            # === 执行 Plan-and-Solve ===
            plan_text, exec_text, pred_num = run_plan_and_solve(question)

            is_right = None
            if gold_num is not None and pred_num is not None:
                is_right = (abs(gold_num - pred_num) <= 1e-6)

            # ---- 写入 JSONL ----
            record = {
                "id": qid,
                "question": question,
                "gold_answer": gold_raw,
                "gold_answer_num": gold_num,
                "pred_answer_num": pred_num,
                "correct": is_right,
                "generated_plan": plan_text,
                "execution_text": exec_text
            }
            fjson_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            fjson_out.flush()

            # ---- 写入 CSV ----
            # CSV 中截断文本防止文件过大
            csv_writer.writerow([
                qid,
                question,
                gold_num,
                pred_num,
                is_right,
                plan_text[:100].replace("\n", " ") + "...", 
                exec_text[:100].replace("\n", " ") + "..."
            ])
            fcsv_out.flush()

    print("Done (Plan-and-Solve).")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"CSV:   {OUTPUT_CSV}")

if __name__ == "__main__":
    main()