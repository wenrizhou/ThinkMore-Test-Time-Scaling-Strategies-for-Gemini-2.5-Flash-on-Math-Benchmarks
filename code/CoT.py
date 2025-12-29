import os
import json
import csv
import re
from typing import Optional

from google import genai
from tqdm import tqdm

# ========= 配置 =========
MODEL_NAME = "gemini-2.5-flash"

INPUT_JSONL = "mamo_complex_lp.jsonl"                 # 换成你的实际路径
OUTPUT_JSONL = "mamo_complex_lp_cot_results.jsonl"    # CoT 结果 jsonl
OUTPUT_CSV = "mamo_complex_lp_cot_results.csv"        # CoT 结果 csv

# 如果没在环境变量中设置 key，可以在这里写（不要传到仓库）
os.environ["GEMINI_API_KEY"] = "AIzaSyDMAx50WY_6b5kpHaa8w3evwAcs5MhVnQ8"

client = genai.Client()   # 默认读 GEMINI_API_KEY


# ========= Prompt（CoT 版本） =========

def make_cot_prompt(question: str) -> str:
    """
    CoT 提示：
    - 先要求模型一步步推理（思维链）
    - 最后要求用特定格式输出最终答案：
        FINAL_ANSWER: <number with one decimal place>
    方便我们从长文本中稳定地提取最终答案。
    """
    prompt = f"""You are a helpful assistant specialized in mathematical optimization
and linear programming problems.

Your task:
1. Carefully analyze and solve the following problem step by step.
2. Explicitly show your chain-of-thought reasoning in English.
3. After finishing all reasoning, output the final numerical answer on a NEW line
   with EXACTLY the following format:

   FINAL_ANSWER: <number>

Requirements for the final number:
- It MUST be a single real number.
- It MUST be rounded to exactly ONE decimal place.
- Do NOT add any extra text or units on that line.
- Examples of valid final lines:
  - FINAL_ANSWER: 3.5
  - FINAL_ANSWER: -12.0
  - FINAL_ANSWER: 0.0

Question:
{question}

Now think step by step, and then give the final answer line in the required format.
"""
    return prompt


def call_gemini(prompt: str) -> str:
    """
    调用 Gemini 2.5 flash，返回模型完整文本输出（包含推理过程 + FINAL_ANSWER 那一行）
    """
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return (resp.text or "").strip()


# ========= 数值提取 / 比较 =========

# 匹配 FINAL_ANSWER: 3.5 这样的格式
_final_answer_pattern = re.compile(
    r"FINAL_ANSWER:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# 通用数字模式（fallback 用）
_number_pattern = re.compile(
    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
)


def extract_final_answer(text: Optional[str]) -> Optional[float]:
    """
    从模型输出中提取最终数值：
    1. 优先从 'FINAL_ANSWER: xxx' 中提取
    2. 如果没有这个格式，则退回到：取文本中出现的最后一个数字
    """
    if not text:
        return None

    # 1) 优先用 FINAL_ANSWER
    m = _final_answer_pattern.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # 2) fallback：找所有数字，取最后一个
    nums = _number_pattern.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def to_one_decimal(x: Optional[float]) -> Optional[float]:
    """
    把数字统一保留一位小数；None 原样返回 None
    """
    if x is None:
        return None
    return round(x, 1)


def is_correct(gold: Optional[float], pred: Optional[float], tol: float = 1e-6) -> Optional[bool]:
    """
    判断预测是否正确：
    - gold 或 pred 任一为 None，则返回 None（无法比较）
    - 否则比较两者差值是否在 tol 内
    """
    if gold is None or pred is None:
        return None
    return abs(gold - pred) <= tol


# ========= 主流程 =========

def count_lines(path: str) -> int:
    """
    统计 jsonl 行数，用于 tqdm 显示总进度
    """
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            cnt += 1
    return cnt


def main():
    total_lines = count_lines(INPUT_JSONL)
    print(f"Total examples: {total_lines}")

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fjson_out, \
         open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fcsv_out:

        csv_writer = csv.writer(fcsv_out)
        csv_writer.writerow([
            "id",
            "question",
            "gold_answer_raw",
            "gold_answer_num",
            "pred_answer_raw",    # 包括推理过程 + FINAL_ANSWER
            "pred_answer_num",
            "correct",            # True / False / None
        ])
        fcsv_out.flush()

        # tqdm 进度条
        for line_no, line in enumerate(tqdm(fin, total=total_lines, desc="Processing (CoT)"), start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error at line {line_no}: {e}")
                continue

            qid = data.get("id")
            question = data.get("Question")
            gold_answer_raw = data.get("Answer")

            # jsonl 中的 Answer 是一个一位小数的数值（str or number）
            gold_answer_num = None
            if gold_answer_raw is not None:
                try:
                    gold_answer_num = to_one_decimal(float(gold_answer_raw))
                except (ValueError, TypeError):
                    # 极端情况下，如果格式异常，可以 fallback 到正则提取
                    gold_answer_num = to_one_decimal(
                        extract_final_answer(str(gold_answer_raw))
                    )

            prompt = make_cot_prompt(question)

            try:
                pred_answer_raw = call_gemini(prompt)
            except Exception as e:
                print(f"[ERROR] Gemini call failed for id={qid}: {e}")
                pred_answer_raw = None

            pred_answer_num = to_one_decimal(
                extract_final_answer(pred_answer_raw)
            )

            correct_flag = is_correct(gold_answer_num, pred_answer_num)

            # ---- 实时写入 JSONL ----
            out_record = {
                "id": qid,
                "question": question,
                "gold_answer": gold_answer_raw,
                "gold_answer_num": gold_answer_num,
                "pred_answer": pred_answer_raw,
                "pred_answer_num": pred_answer_num,
                "correct": correct_flag,
            }
            fjson_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            fjson_out.flush()

            # ---- 实时写入 CSV ----
            csv_writer.writerow([
                qid,
                question,
                gold_answer_raw,
                gold_answer_num,
                pred_answer_raw,
                pred_answer_num,
                correct_flag,
            ])
            fcsv_out.flush()

    print("Done (CoT).")
    print(f"JSONL results saved to: {OUTPUT_JSONL}")
    print(f"CSV   results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
