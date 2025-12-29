import os
import json
import csv
import re
import sys
import io
import contextlib
import traceback
from typing import Optional, Tuple, Any

from google import genai
from tqdm import tqdm

# ========= 配置 =========
MODEL_NAME = "gemini-2.5-flash"

INPUT_JSONL = "mamo_complex_lp.jsonl"
OUTPUT_JSONL = "mamo_complex_lp_pot_results.jsonl"
OUTPUT_CSV = "mamo_complex_lp_pot_results.csv"

# 这里的 Key 还是用你的
os.environ["GEMINI_API_KEY"] = "AIzaSyDMAx50WY_6b5kpHaa8w3evwAcs5MhVnQ8"

client = genai.Client()


# ========= 1. Prompt (要求生成代码) =========

def make_pot_prompt(question: str) -> str:
    """
    PoT 提示：要求模型写 Python 代码来解决问题，而不是直接推理。
    特别针对线性规划问题，建议使用 scipy.optimize.linprog
    """
    return f"""You are a Python programming expert specialized in Operations Research.

Task:
Write a Python script to solve the following linear programming (or math) problem.
1. Use the `scipy.optimize.linprog` library if it's a linear programming problem.
2. If `scipy` is not applicable, use standard Python math.
3. The code MUST print the final numerical answer.
4. Wrap the code inside a markdown code block (```python ... ```).
5. Do NOT try to calculate the answer in the text. ONLY write code.

Problem:
{question}

Requirements for the Code:
- Define the objective function vectors and inequality/equality constraint matrices carefully.
- Note that `linprog` minimizes by default. If the problem is maximization, negate the objective coefficients.
- At the very end of the script, print the optimal value (objective function value) directly.
- Example of the final print statement: print(result.fun) or print(-result.fun)
"""


# ========= 2. 代码提取与执行引擎 (核心部分) =========

def extract_code_block(text: str) -> str:
    """
    从回答中提取 ```python ... ``` 之间的代码
    """
    pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1)
    
    # 如果没有写 python 标签，尝试找 ``` ... ```
    pattern_fallback = re.compile(r"```\s*(.*?)\s*```", re.DOTALL)
    match_fallback = pattern_fallback.search(text)
    if match_fallback:
        return match_fallback.group(1)
    
    return ""


def execute_python_code(code: str) -> Tuple[str, str]:
    """
    执行 Python 代码并捕获标准输出 (stdout)。
    警告：exec() 存在安全风险，仅在本地受控环境处理可信数据时使用。
    
    Returns:
        stdout_output: 代码打印的内容
        error_message: 如果出错，返回错误信息
    """
    # 创建一个用来捕获输出的缓冲区
    stdout_buffer = io.StringIO()
    
    try:
        # 重定向 stdout 到缓冲区
        with contextlib.redirect_stdout(stdout_buffer):
            # 定义一个局部命名空间，防止污染全局
            local_scope = {}
            exec(code, local_scope, local_scope)
            
        return stdout_buffer.getvalue(), ""
        
    except Exception:
        # 捕获所有运行时错误（语法错误、库缺失、维度不匹配等）
        return stdout_buffer.getvalue(), traceback.format_exc()


# ========= 3. 数值处理 =========

def to_one_decimal(x: Optional[float]) -> Optional[float]:
    if x is None: return None
    return round(x, 1)

def parse_output_to_number(output_text: str) -> Optional[float]:
    """
    从代码的打印结果中提取最后一个数字
    """
    if not output_text:
        return None
    
    # 找文本中的所有数字
    # 处理像 "Optimization terminated successfully. 12.5" 这样的情况
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", output_text)
    if nums:
        try:
            return float(nums[-1])
        except:
            return None
    return None


# ========= 主流程 =========

def run_pot_pipeline(question: str) -> Tuple[str, str, str, Optional[float]]:
    """
    执行 PoT 流程:
    Text -> LLM -> Code -> Python Exec -> Number
    """
    prompt = make_pot_prompt(question)
    
    # 1. LLM 生成代码
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        llm_response = (resp.text or "").strip()
    except Exception as e:
        return "", "", f"API Error: {e}", None

    # 2. 提取代码
    code = extract_code_block(llm_response)
    if not code:
        return llm_response, "", "No code block found", None
        
    # 3. 执行代码
    stdout_output, error_msg = execute_python_code(code)
    
    # 4. 从执行结果提取数字
    # 如果有报错，优先记录报错，但有时候报错前也打印了结果，可以尝试提取
    final_val = parse_output_to_number(stdout_output)
    
    return llm_response, code, stdout_output + ("\n[ERROR]\n" + error_msg if error_msg else ""), final_val


def main():
    total_lines = 0
    if os.path.exists(INPUT_JSONL):
        with open(INPUT_JSONL, "r", encoding="utf-8") as f:
            for _ in f: total_lines += 1

    print(f"Total examples: {total_lines}")
    print(f"Mode: Program-of-Thought (PoT) / Code Execution")

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
            "code_snippet",       # 生成的代码
            "execution_output",   # 代码运行的打印结果
            "llm_raw_response"
        ])
        fcsv_out.flush()

        for line in tqdm(fin, total=total_lines, desc="Processing (PoT)"):
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
                except: gold_num = None # PoT 不需要从正则提取标准答案，只要数值

            # === 执行 PoT ===
            llm_resp, code, exec_output, pred_num = run_pot_pipeline(question)
            
            # 归一化
            pred_num = to_one_decimal(pred_num)
            
            is_right = None
            if gold_num is not None and pred_num is not None:
                is_right = (abs(gold_num - pred_num) <= 1e-6)

            # ---- 写入 JSONL ----
            record = {
                "id": qid,
                "question": question,
                "gold_answer_num": gold_num,
                "pred_answer_num": pred_num,
                "correct": is_right,
                "generated_code": code,
                "execution_output": exec_output,
                "llm_response": llm_resp
            }
            fjson_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            fjson_out.flush()

            # ---- 写入 CSV ----
            csv_writer.writerow([
                qid,
                question,
                gold_num,
                pred_num,
                is_right,
                code[:100].replace("\n", "\\n") + "...", 
                exec_output[:100].replace("\n", "\\n") + "...",
                llm_resp[:50] + "..."
            ])
            fcsv_out.flush()

    print("Done (PoT).")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"CSV:   {OUTPUT_CSV}")

if __name__ == "__main__":
    main()