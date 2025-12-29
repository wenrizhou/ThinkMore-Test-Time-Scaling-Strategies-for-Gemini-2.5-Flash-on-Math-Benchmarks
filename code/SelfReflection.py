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
OUTPUT_JSONL = "mamo_complex_lp_reflection_results.jsonl"
OUTPUT_CSV = "mamo_complex_lp_reflection_results.csv"

# 这里的 Key 还是用你的
os.environ["GEMINI_API_KEY"] = "AIzaSyDMAx50WY_6b5kpHaa8w3evwAcs5MhVnQ8"

client = genai.Client()


# ========= Prompts =========

def make_initial_cot_prompt(question: str) -> str:
    """
    第一步：标准 CoT 提示，生成初始答案
    """
    return f"""You are a helpful assistant specialized in mathematical optimization.

Please solve the following problem step by step.
Output the final answer strictly in the format: FINAL_ANSWER: <number>

Question:
{question}
"""

def make_reflection_prompt(question: str, initial_response: str) -> str:
    """
    第二步：反思提示
    要求模型审查上一步的回答（initial_response），找出潜在错误并修正。
    """
    return f"""You are a rigorous mathematical reviewer.

Your task is to review the solution provided below for a specific optimization problem.
1. Check the reasoning for logical loopholes.
2. Verify the calculations carefully.
3. If the previous solution is correct, output the same answer.
4. If there are errors, correct them and provide the right answer.

---
Original Question:
{question}

---
Proposed Solution (to be reviewed):
{initial_response}

---
Now, provide your critique and the CORRECTED final answer.
End your response with a new line exactly in this format:
FINAL_ANSWER: <number>

Requirements:
- The number must be rounded to ONE decimal place.
"""


def call_gemini(prompt: str) -> str:
    """
    通用调用函数
    """
    try:
        # 反思环节建议 temperature 设低一点（如 0.0 或 0.2），让模型更理性严谨
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return (resp.text or "").strip()
    except Exception as e:
        print(f"[WARN] API call failed: {e}")
        return ""


# ========= 数值提取 (复用) =========

_final_answer_pattern = re.compile(r"FINAL_ANSWER:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def extract_final_answer(text: Optional[str]) -> Optional[float]:
    if not text: return None
    # 优先找 FINAL_ANSWER
    m = _final_answer_pattern.search(text)
    if m:
        try: return float(m.group(1))
        except ValueError: pass
    # Fallback: 找最后一个数字
    nums = _number_pattern.findall(text)
    if not nums: return None
    try: return float(nums[-1])
    except ValueError: return None

def to_one_decimal(x: Optional[float]) -> Optional[float]:
    if x is None: return None
    return round(x, 1)

def is_correct(gold: Optional[float], pred: Optional[float], tol: float = 1e-6) -> Optional[bool]:
    if gold is None or pred is None: return None
    return abs(gold - pred) <= tol


# ========= 核心逻辑：Generate -> Reflect =========

def run_reflection_pipeline(question: str) -> Tuple[str, Optional[float], str, Optional[float]]:
    """
    执行两步流程：
    1. 生成初始解
    2. 自我反思修正
    
    Returns:
        resp_1 (str): 初始回答文本
        ans_1 (float): 初始数值
        resp_2 (str): 反思后的回答文本
        ans_2 (float): 反思后的数值 (最终结果)
    """
    # Step 1: Generate
    prompt_1 = make_initial_cot_prompt(question)
    resp_1 = call_gemini(prompt_1)
    ans_1 = to_one_decimal(extract_final_answer(resp_1))
    
    # Step 2: Reflect (Critique)
    # 如果第一步连回答都没生成，就没法反思了，直接重试或跳过，这里简单处理为传空
    prompt_2 = make_reflection_prompt(question, resp_1 if resp_1 else "No solution provided.")
    resp_2 = call_gemini(prompt_2)
    ans_2 = to_one_decimal(extract_final_answer(resp_2))
    
    # 如果反思步骤没提取到答案（比较少见），可以策略性回退到 ans_1，
    # 但为了验证反思方法的效果，我们这里严格返回 ans_2（即使是 None）
    return resp_1, ans_1, resp_2, ans_2


# ========= 主流程 =========

def main():
    # 统计行数
    total_lines = 0
    if os.path.exists(INPUT_JSONL):
        with open(INPUT_JSONL, "r", encoding="utf-8") as f:
            for _ in f: total_lines += 1
            
    print(f"Total examples: {total_lines}")
    print(f"Mode: Self-Reflection (Generate -> Critique)")

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fjson_out, \
         open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fcsv_out:

        csv_writer = csv.writer(fcsv_out)
        csv_writer.writerow([
            "id",
            "question",
            "gold_answer_num",
            "initial_ans_num",   # 第一步生成的答案
            "final_ans_num",     # 第二步反思后的答案 (最终预测)
            "changed",           # 反思是否修改了答案 (True/False)
            "correct",           # 最终答案是否正确
            "initial_response",  # (可选) 记录第一步文本，方便debug
            "reflection_response"
        ])
        fcsv_out.flush()

        for line in tqdm(fin, total=total_lines, desc="Processing (Reflection)"):
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
            except:
                continue

            qid = data.get("id")
            question = data.get("Question")
            gold_raw = data.get("Answer")
            
            # 解析标准答案
            gold_num = None
            if gold_raw is not None:
                try: gold_num = to_one_decimal(float(gold_raw))
                except: gold_num = to_one_decimal(extract_final_answer(str(gold_raw)))

            # === 执行反思流程 ===
            resp_1, ans_1, resp_2, ans_2 = run_reflection_pipeline(question)

            # 判断是否被修改
            # 注意处理 None 的情况
            if ans_1 is None and ans_2 is None:
                changed = False
            elif ans_1 is None or ans_2 is None:
                changed = True
            else:
                # 浮点数比较
                changed = (abs(ans_1 - ans_2) > 1e-6)

            # 最终正确性以 ans_2 为准
            is_right = is_correct(gold_num, ans_2)

            # ---- 写入 JSONL ----
            record = {
                "id": qid,
                "question": question,
                "gold_answer": gold_raw,
                "gold_answer_num": gold_num,
                "initial_answer_num": ans_1,
                "reflection_answer_num": ans_2, # 最终结果
                "initial_response": resp_1,
                "reflection_response": resp_2,
                "changed_by_reflection": changed,
                "correct": is_right
            }
            fjson_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            fjson_out.flush()

            # ---- 写入 CSV ----
            csv_writer.writerow([
                qid,
                question,
                gold_num,
                ans_1,
                ans_2,
                changed,
                is_right,
                resp_1[:50] + "...", # CSV只存开头，防爆
                resp_2[:50] + "..."
            ])
            fcsv_out.flush()

    print("Done (Self-Reflection).")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"CSV:   {OUTPUT_CSV}")

if __name__ == "__main__":
    main()