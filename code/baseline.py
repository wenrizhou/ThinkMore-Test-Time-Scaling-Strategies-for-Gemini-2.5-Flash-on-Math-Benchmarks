import os
import json
import csv
import re
from typing import Optional

from google import genai
from tqdm import tqdm

# ========= 配置 =========
MODEL_NAME = "gemini-2.5-flash"

INPUT_JSONL = "mamo_complex_lp.jsonl"          # 换成你的实际路径
OUTPUT_JSONL = "mamo_complex_lp_baseline_results.jsonl"
OUTPUT_CSV = "mamo_complex_lp_baseline_results.csv"

# 如果没在环境变量中设置 key，可以在这里写（不要提交到仓库）
os.environ["GEMINI_API_KEY"] = "AIzaSyDMAx50WY_6b5kpHaa8w3evwAcs5MhVnQ8"

client = genai.Client()   # 默认读 GEMINI_API_KEY


# ========= Prompt =========

def make_baseline_prompt(question: str) -> str:
    """
    baseline 提示：解题 + 严格要求只输出一位小数的数值
    """
    prompt = f"""You are a helpful assistant specialized in mathematical optimization
and linear programming problems.

Solve the following problem and output ONLY the final numerical answer.

Requirements:
- Output ONLY a single real number.
- The number MUST be rounded to exactly ONE decimal place.
- Do NOT output any explanation, text, units.
- The output should look like: 3.5 or -12.0 or 0.0

Question:
{question}

Answer (a single number with exactly one decimal place):"""
    return prompt


def call_gemini(prompt: str) -> str:
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return (resp.text or "").strip()


# ========= 数值提取 / 比较 =========

_number_pattern = re.compile(
    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
)

def extract_number(text: Optional[str]) -> Optional[float]:
    """
    从文本中提取第一个数字为 float；失败返回 None
    """
    if not text:
        return None
    m = _number_pattern.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
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
            "pred_answer_raw",
            "pred_answer_num",
            "correct",          # True / False / None
        ])
        fcsv_out.flush()

        # 用 tqdm 包裹文件迭代，显示进度条
        for line_no, line in enumerate(tqdm(fin, total=total_lines, desc="Processing"), start=1):
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

            # jsonl 中的 Answer 就是一个一位小数的数值（str or number）
            gold_answer_num = None
            if gold_answer_raw is not None:
                try:
                    gold_answer_num = to_one_decimal(float(gold_answer_raw))
                except (ValueError, TypeError):
                    # 极端情况下，如果格式异常，可以 fallback 到正则提取
                    gold_answer_num = to_one_decimal(
                        extract_number(str(gold_answer_raw))
                    )

            prompt = make_baseline_prompt(question)

            try:
                pred_answer_raw = call_gemini(prompt)
            except Exception as e:
                print(f"[ERROR] Gemini call failed for id={qid}: {e}")
                pred_answer_raw = None

            pred_answer_num = to_one_decimal(
                extract_number(pred_answer_raw)
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
            fjson_out.flush()   # 逐题 flush，确保实时落盘

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
            fcsv_out.flush()    # 逐题 flush

    print("Done.")
    print(f"JSONL results saved to: {OUTPUT_JSONL}")
    print(f"CSV   results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()