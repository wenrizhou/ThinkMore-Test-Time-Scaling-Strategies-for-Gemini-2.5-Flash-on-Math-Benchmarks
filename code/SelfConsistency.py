import os
import json
import csv
import re
from typing import Optional, List, Tuple
from collections import Counter

from google import genai
from google.genai import types  # 需要引入 types 来配置 temperature
from tqdm import tqdm

# ========= 配置 =========
MODEL_NAME = "gemini-2.5-flash"

INPUT_JSONL = "mamo_complex_lp.jsonl"                 
OUTPUT_JSONL = "mamo_complex_lp_sc_results.jsonl"     # SC 结果 jsonl
OUTPUT_CSV = "mamo_complex_lp_sc_results.csv"         # SC 结果 csv

# Self-Consistency 特有配置
SC_NUM_SAMPLES = 5      # 对每个问题采样的次数 (通常 5-10 次)
SC_TEMPERATURE = 0.7    # 温度 > 0 以增加多样性

os.environ["GEMINI_API_KEY"] = "AIzaSyDMAx50WY_6b5kpHaa8w3evwAcs5MhVnQ8"

client = genai.Client()


# ========= Prompt（复用 CoT 版本） =========
# Self-Consistency 的基础依然是 CoT，只是执行方式不同

def make_cot_prompt(question: str) -> str:
    """
    CoT 提示：要求思维链 + 最终格式
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

Question:
{question}

Now think step by step, and then give the final answer line in the required format.
"""
    return prompt


def call_gemini_sc_sample(prompt: str, temperature: float) -> str:
    """
    调用 Gemini 生成单个采样，带有温度参数
    """
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
            )
        )
        return (resp.text or "").strip()
    except Exception as e:
        # 如果单个请求失败，返回空字符串，不中断整个程序
        print(f"[WARN] API call failed: {e}")
        return ""


# ========= 数值提取 / 比较 (复用) =========

_final_answer_pattern = re.compile(r"FINAL_ANSWER:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def extract_final_answer(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = _final_answer_pattern.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    nums = _number_pattern.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None

def to_one_decimal(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return round(x, 1)

def is_correct(gold: Optional[float], pred: Optional[float], tol: float = 1e-6) -> Optional[bool]:
    if gold is None or pred is None:
        return None
    return abs(gold - pred) <= tol


# ========= Self-Consistency 核心逻辑 =========

def run_self_consistency(question: str) -> Tuple[str, Optional[float], dict]:
    """
    执行自洽性采样：
    1. 生成 SC_NUM_SAMPLES 个回复
    2. 提取每个回复的数值并归一化（保留一位小数）
    3. 进行投票，选出众数
    
    Returns:
        best_text: 对应于众数答案的那条完整文本（用于展示推理过程）
        final_voted_answer: 投票选出的数字
        vote_stats: 投票详情，例如 {'3.5': 4, '4.0': 1}
    """
    prompt = make_cot_prompt(question)
    
    raw_responses = []
    extracted_nums = []
    
    # 1. 循环采样
    for _ in range(SC_NUM_SAMPLES):
        resp_text = call_gemini_sc_sample(prompt, temperature=SC_TEMPERATURE)
        raw_responses.append(resp_text)
        
        # 提取并立即归一化 (这一步很关键，否则 3.5 和 3.5001 会被当成不同答案)
        num = to_one_decimal(extract_final_answer(resp_text))
        extracted_nums.append(num)

    # 2. 投票
    # 过滤掉提取失败 (None) 的结果
    valid_nums = [n for n in extracted_nums if n is not None]
    
    if not valid_nums:
        # 如果所有尝试都失败
        return raw_responses[0] if raw_responses else "", None, {}

    # 使用 Counter 找众数
    counter = Counter(valid_nums)
    most_common = counter.most_common(1)  # [(number, count)]
    winner_num, count = most_common[0]
    
    # 3. 找回对应的文本
    # 我们倾向于返回生成了该众数答案的第一条文本，作为推理过程的代表
    best_text = ""
    for txt, num in zip(raw_responses, extracted_nums):
        if num == winner_num:
            best_text = txt
            break
            
    # 将 Counter 转为 dict 方便 JSON 序列化 (key 转为 str)
    vote_stats = {str(k): v for k, v in counter.items()}
    
    return best_text, winner_num, vote_stats


# ========= 主流程 =========

def count_lines(path: str) -> int:
    cnt = 0
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f: cnt += 1
    return cnt

def main():
    total_lines = count_lines(INPUT_JSONL)
    print(f"Total examples: {total_lines}")
    print(f"Mode: Self-Consistency (Samples={SC_NUM_SAMPLES}, Temp={SC_TEMPERATURE})")

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fjson_out, \
         open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fcsv_out:

        csv_writer = csv.writer(fcsv_out)
        csv_writer.writerow([
            "id",
            "question",
            "gold_answer_raw",
            "gold_answer_num",
            "pred_answer_raw",  # 选中的那个答案的完整文本
            "pred_answer_num",  # 投票后的最终结果
            "vote_stats",       # 投票分布情况
            "correct",
        ])
        fcsv_out.flush()

        for line_no, line in enumerate(tqdm(fin, total=total_lines, desc="Processing (SC)"), start=1):
            line = line.strip()
            if not line: continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = data.get("id")
            question = data.get("Question")
            gold_answer_raw = data.get("Answer")

            # 处理标准答案
            gold_answer_num = None
            if gold_answer_raw is not None:
                try:
                    gold_answer_num = to_one_decimal(float(gold_answer_raw))
                except (ValueError, TypeError):
                    gold_answer_num = to_one_decimal(
                        extract_final_answer(str(gold_answer_raw))
                    )

            # === 执行 Self-Consistency ===
            best_text_response, pred_answer_num, vote_stats = run_self_consistency(question)

            # 判断正确性
            correct_flag = is_correct(gold_answer_num, pred_answer_num)

            # ---- 写入 JSONL ----
            out_record = {
                "id": qid,
                "question": question,
                "gold_answer": gold_answer_raw,
                "gold_answer_num": gold_answer_num,
                "pred_answer": best_text_response,
                "pred_answer_num": pred_answer_num,
                "vote_stats": vote_stats, # 记录这一项很有用，可以看出模型是否“犹豫”
                "correct": correct_flag,
            }
            fjson_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            fjson_out.flush()

            # ---- 写入 CSV ----
            csv_writer.writerow([
                qid,
                question,
                gold_answer_raw,
                gold_answer_num,
                best_text_response[:100] + "...", # CSV里截断一下长文本
                pred_answer_num,
                json.dumps(vote_stats), # 将字典转为字符串存入 CSV
                correct_flag,
            ])
            fcsv_out.flush()

    print("Done (Self-Consistency).")
    print(f"JSONL results saved to: {OUTPUT_JSONL}")
    print(f"CSV   results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()