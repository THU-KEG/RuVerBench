#!/usr/bin/env python3
"""
轨迹评估器 - 评估 convert 后的轨迹 (支持并发、单Check评估、增量保存)

用法:
    python evaluate.py \
        --trajectories ./results/converted_for_training.jsonl \
        --data data_checklist.jsonl \
        --output ./results/scores.json \
        --limit 1 \
        --workers 4
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from copy import deepcopy

from concurrent.futures import ThreadPoolExecutor, as_completed

_RUBRICBENCH_ROOT = str(Path(__file__).resolve().parent.parent)
if _RUBRICBENCH_ROOT not in sys.path:
    sys.path.insert(0, _RUBRICBENCH_ROOT)

os.environ.setdefault("DETECTOR_API_BACKEND", "openai")

try:
    from model import APIModel
except Exception:
    # Fallback to DeepResearch shared APIModel implementation.
    _FALLBACK_MODEL_ROOT = Path(__file__).resolve().parents[2] / "DeepResearch" / "veribench-deep-research"
    if _FALLBACK_MODEL_ROOT.is_dir():
        sys.path.insert(0, str(_FALLBACK_MODEL_ROOT))
        from model import APIModel
    else:
        raise


class TokenTracker:
    """Thread-safe token usage tracker."""
    _lock = threading.Lock()
    _stats = {
        "total_requests": 0,
        "total_input_chars": 0,
        "total_output_chars": 0,
        "estimated_input_tokens": 0,
        "estimated_output_tokens": 0,
        "actual_input_tokens": 0,
        "actual_output_tokens": 0,
        "actual_cached_input_tokens": 0,
        "actual_uncached_input_tokens": 0,
        "requests_with_usage": 0,
        "requests_without_usage": 0,
        "local_cache_hits": 0,
        "effective_input_tokens": 0,
        "effective_output_tokens": 0,
        "effective_cached_input_tokens": 0,
        "effective_uncached_input_tokens": 0,
        "estimated_cache_input_tokens_from_lcp": 0,
    }
    _last_input_text_for_lcp = ""
    _stats_file = ""
    _save_every = 10
    _track_count = 0

    @classmethod
    def setup(cls, stats_file: str, save_every: int = 10, resume: bool = False):
        cls._stats_file = stats_file
        cls._save_every = save_every
        if resume:
            cls._load_from_file()

    @classmethod
    def _load_from_file(cls):
        if not cls._stats_file or not os.path.exists(cls._stats_file):
            return
        try:
            with open(cls._stats_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for k in cls._stats:
                if k in saved:
                    cls._stats[k] = saved[k]
        except Exception:
            pass

    @classmethod
    def _save_to_file(cls):
        if not cls._stats_file:
            return
        try:
            tmp = cls._stats_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cls._stats, f, indent=2)
            os.replace(tmp, cls._stats_file)
        except Exception:
            pass

    @classmethod
    def reset(cls):
        with cls._lock:
            for k in cls._stats:
                cls._stats[k] = 0
            cls._last_input_text_for_lcp = ""
            cls._track_count = 0

    @classmethod
    def _estimate_tokens(cls, text: str) -> int:
        cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f')
        ascii_chars = len(text) - cjk
        return int(cjk * 1.5 + ascii_chars * 0.25)

    @classmethod
    def _safe_int(cls, value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    @classmethod
    def _extract_usage_tokens(cls, usage: dict):
        if not isinstance(usage, dict):
            return 0, 0, 0
        input_tokens = cls._safe_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        output_tokens = cls._safe_int(usage.get("completion_tokens", usage.get("output_tokens", 0)))

        cached_input_tokens = cls._safe_int(usage.get("cache_input_tokens", usage.get("cached_tokens", 0)))
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached_input_tokens = max(cached_input_tokens, cls._safe_int(prompt_details.get("cached_tokens", 0)))
        elif prompt_details is not None and hasattr(prompt_details, "cached_tokens"):
            cached_input_tokens = max(cached_input_tokens, cls._safe_int(getattr(prompt_details, "cached_tokens", 0)))
        cached_input_tokens = max(cached_input_tokens, cls._safe_int(usage.get("cache_read_input_tokens", 0)))
        cached_input_tokens = max(cached_input_tokens, cls._safe_int(usage.get("cache_creation_input_tokens", 0)))
        return input_tokens, cached_input_tokens, output_tokens

    @classmethod
    def _longest_common_prefix_len(cls, a: str, b: str) -> int:
        max_len = min(len(a), len(b))
        i = 0
        while i < max_len and a[i] == b[i]:
            i += 1
        return i

    @classmethod
    def track(cls, input_text: str, output_text: str, usage: dict | None = None, cache_hit: bool = False):
        in_chars = len(input_text)
        out_chars = len(output_text)
        est_in = cls._estimate_tokens(input_text)
        est_out = cls._estimate_tokens(output_text)
        input_tokens, cached_input_tokens, output_tokens = cls._extract_usage_tokens(usage or {})
        has_usage = (input_tokens > 0) or (output_tokens > 0) or (cached_input_tokens > 0)
        lcp_cached_est = 0

        with cls._lock:
            if not has_usage and not cache_hit and cls._last_input_text_for_lcp:
                lcp_chars = cls._longest_common_prefix_len(cls._last_input_text_for_lcp, input_text)
                lcp_cached_est = cls._estimate_tokens(input_text[:lcp_chars])
            cls._last_input_text_for_lcp = input_text

            if cache_hit:
                effective_input = est_in
                effective_output = est_out
                effective_cached = est_in
            elif has_usage:
                effective_input = input_tokens
                effective_output = output_tokens
                effective_cached = cached_input_tokens
            else:
                effective_input = est_in
                effective_output = est_out
                effective_cached = lcp_cached_est

            cls._stats["total_requests"] += 1
            cls._stats["total_input_chars"] += in_chars
            cls._stats["total_output_chars"] += out_chars
            cls._stats["estimated_input_tokens"] += est_in
            cls._stats["estimated_output_tokens"] += est_out
            if cache_hit:
                cls._stats["local_cache_hits"] += 1
            if has_usage:
                cls._stats["requests_with_usage"] += 1
                cls._stats["actual_input_tokens"] += input_tokens
                cls._stats["actual_output_tokens"] += output_tokens
                cls._stats["actual_cached_input_tokens"] += cached_input_tokens
                cls._stats["actual_uncached_input_tokens"] += max(input_tokens - cached_input_tokens, 0)
            else:
                cls._stats["requests_without_usage"] += 1
                cls._stats["estimated_cache_input_tokens_from_lcp"] += lcp_cached_est
            cls._stats["effective_input_tokens"] += effective_input
            cls._stats["effective_output_tokens"] += effective_output
            cls._stats["effective_cached_input_tokens"] += effective_cached
            cls._stats["effective_uncached_input_tokens"] += max(effective_input - effective_cached, 0)

            cls._track_count += 1
            if cls._track_count % cls._save_every == 0:
                cls._save_to_file()

    @classmethod
    def get_stats(cls) -> dict:
        with cls._lock:
            return cls._stats.copy()

    @classmethod
    def print_summary(cls):
        s = cls.get_stats()
        print(f"\n{'='*60}")
        print("Token Usage Summary")
        print(f"{'='*60}")
        print(f"Total API requests:        {s['total_requests']}")
        print(f"Local cache hits:          {s['local_cache_hits']}")
        print(f"Requests w/ usage fields:  {s['requests_with_usage']}")
        print(f"Requests w/o usage fields: {s['requests_without_usage']}")
        print("")
        print("Server-reported tokens")
        print(f"Input tokens:              {s['actual_input_tokens']}")
        print(f"Cache input tokens:        {s['actual_cached_input_tokens']}")
        print(f"Uncached input tokens:     {s['actual_uncached_input_tokens']}")
        print(f"Output tokens:             {s['actual_output_tokens']}")
        print("")
        print("Effective tokens (server-first, chars fallback)")
        print(f"Input tokens:              {s['effective_input_tokens']}")
        print(f"Cache input tokens:        {s['effective_cached_input_tokens']}")
        print(f"Uncached input tokens:     {s['effective_uncached_input_tokens']}")
        print(f"Output tokens:             {s['effective_output_tokens']}")
        print("")
        print("Estimated tokens (fallback)")
        print(f"Total input chars:         {s['total_input_chars']}")
        print(f"Total output chars:        {s['total_output_chars']}")
        print(f"Estimated input tokens:    {s['estimated_input_tokens']}")
        print(f"Estimated output tokens:   {s['estimated_output_tokens']}")
        print(f"Estimated cache tokens(LCP): {s['estimated_cache_input_tokens_from_lcp']}")
        print(f"{'='*60}")




# VERIBENCH_VLLM_FIXED20_FEWSHOT_AC_V1
def _veribench_fewshot_instruction() -> str:
    import json
    import os

    path = os.getenv("VERIBENCH_FEWSHOT_EXAMPLES_FILE", "").strip()
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return f"\n\n--------------------------------------------------\nFew-shot examples\n--------------------------------------------------\n[Unable to load few-shot examples: {exc}]"
    examples = payload.get("agenticcoding", [])
    if not examples:
        return ""
    lines = [
        "",
        "--------------------------------------------------",
        "Few-shot examples",
        "--------------------------------------------------",
        "以下紧凑人工标注样例仅用于校准 success/fail 判定；当前输入中的 Check 才是需要评估的对象。",
    ]
    for i, item in enumerate(examples, 1):
        lines.extend([
            f"Example {i} [{item.get('category', 'unknown')}]:",
            f"User query snippet: {str(item.get('user_query_snippet', ''))[:500]}",
            f"Check: {str(item.get('check', ''))[:700]}",
            f"Human label: {item.get('label', '')}",
            f"Reason: {str(item.get('reason', ''))[:600]}",
        ])
    return "\n" + "\n".join(lines)

def _veribench_original_get_prompt_style_instruction(prompt_style):
    prompt_style = (prompt_style or "standard").strip().lower()
    if prompt_style == "standard":
        return ""
    if prompt_style == "strict":
        return """
--------------------------------------------------
Prompt Style
--------------------------------------------------
Strict Verification Prompt: 判定 "success" 必须要求 assistant 明确、完整地满足该 Check；如果只是隐含、模糊、部分满足或缺少关键细节，判定为 "fail"。
"""
    if prompt_style in {"semantic", "soft", "semantic_equivalence"}:
        return """
--------------------------------------------------
Prompt Style
--------------------------------------------------
Semantic Equivalence Prompt: 重点判断 assistant 是否以语义等价方式满足该 Check；允许不同措辞、顺序或表达形式，不要求关键词完全一致。
"""
    if prompt_style in {"evidence_first", "evidence-first"}:
        return """
--------------------------------------------------
Prompt Style
--------------------------------------------------
Evidence-First Prompt: 先在 assistant 的消息、reasoning_content 或工具调用中定位最相关证据，再判断 success/fail；reasoning 中简要写出证据。
"""
    raise ValueError(f"Unknown prompt_style: {prompt_style}")


def get_prompt_style_instruction(prompt_style):
    return _veribench_original_get_prompt_style_instruction(prompt_style) + _veribench_fewshot_instruction()


# 评估 Prompt - 针对单个 Check 修改
EVAL_PROMPT_SINGLE = """
你是一个「Agent 指令遵循」评审模型。

你的任务是：根据给定的 **单个 Check 项**，评估 assistant 在对话中的表现。

=====INPUT CONVERSATION=====
====TOOLS===
{tools}
====TOOLS===
====MESSAGES===
{messages}
====MESSAGES===
=====INPUT CONVERSATION=====

=====CHECK TO EVALUATE=====
Category: {category}
Check ID: {check_id}
Description: {check_description}
=====CHECK TO EVALUATE=====

--------------------------------------------------
评估规则
--------------------------------------------------

1. **评估依据**：检查所有 `role == "assistant"` 的消息，包括：
   - 自然语言输出
   - 内部推理（reasoning_content，如有）
   - 工具调用（tool_calls）

2. **判定标准**：
   - `"success"`：assistant 明确遵循了该要求
   - `"fail"`：assistant 明显违背了该要求，或在应该遵循时未遵循

3. **reasoning 字段**：
   - 对于 "fail"：简要说明 assistant 如何违背该要求
   - 对于 "success"：简要说明 assistant 如何遵循该要求

--------------------------------------------------
输出格式（必须为合法 JSON）
--------------------------------------------------

输出一个 JSON 对象，包含 check_id, reasoning 和 result 字段：

{{
  "check_id": "{check_id}",
  "reasoning": "简要说明判定依据",
  "result": "success 或 fail"
}}

--------------------------------------------------
注意事项
--------------------------------------------------

1. result 只能是 "success" 或 "fail"（小写），不允许其他值
2. 输出必须是合法 JSON，不要在 JSON 外添加任何文字
3. 确保输出的 check_id 与输入一致

请严格按照 Check 描述进行评估。{prompt_style_instruction}
"""
EVAL_PROMPT_BATCH = """
你是一个「Agent 指令遵循」评审模型。

你的任务是：根据给定的 **多个 Check 项**，逐项评估 assistant 在对话中的表现。

=====INPUT CONVERSATION=====
====TOOLS===
{tools}
====TOOLS===
====MESSAGES===
{messages}
====MESSAGES===
=====INPUT CONVERSATION=====

=====CHECKS TO EVALUATE=====
{checks}
=====CHECKS TO EVALUATE=====

--------------------------------------------------
评估规则
--------------------------------------------------

1. **评估依据**：检查所有 `role == "assistant"` 的消息，包括：
   - 自然语言输出
   - 内部推理（reasoning_content，如有）
   - 工具调用（tool_calls）

2. **判定标准**：
   - `"success"`：assistant 明确遵循了该要求
   - `"fail"`：assistant 明显违背了该要求，或在应该遵循时未遵循

3. **reasoning 字段**：
   - 对于 "fail"：简要说明 assistant 如何违背该要求
   - 对于 "success"：简要说明 assistant 如何遵循该要求

--------------------------------------------------
输出格式（必须为合法 JSON）
--------------------------------------------------

输出一个 JSON 数组，数组长度必须等于 Check 数量。每个对象必须包含输入中的精确 check_id、reasoning 和 result 字段：

[
  {{
    "check_id": "...",
    "reasoning": "简要说明判定依据",
    "result": "success 或 fail"
  }}
]

--------------------------------------------------
注意事项
--------------------------------------------------

1. result 只能是 "success" 或 "fail"（小写），不允许其他值
2. 输出必须是合法 JSON，不要在 JSON 外添加任何文字
3. 确保每个输入 check_id 在输出中恰好出现一次，不要依赖顺序省略或改写 check_id

请严格按照每个 Check 描述进行评估。{prompt_style_instruction}
"""

def load_trajectory(filepath, session_id=None):
    """加载 convert 后的轨迹，返回主轨迹"""
    records = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except FileNotFoundError:
        return None
    
    if not records:
        return None
    
    if session_id:
        filtered = [r for r in records if r.get("meta", {}).get("session_id") == session_id]
        if filtered:
            records = filtered
    
    with_tools = [r for r in records if r.get("tools")]
    
    if with_tools:
        return max(with_tools, key=lambda r: len(r.get("messages", [])))
    
    return records[-1] if records else None


def get_truncated_context(record, max_total_chars=0):
    max_tool_content_length = 5000
    max_assistant_content_length = 50000
    max_assistant_reasoning_content_length = 50000

    messages = record.get("messages", [])
    tools = record.get("tools", [])
    
    truncated_messages = []
    
    for message in messages:
        msg = deepcopy(message)
        role = msg.get("role")
        
        if role == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > max_tool_content_length:
                msg["content"] = (
                    content[:max_tool_content_length//2] + 
                    "\n\n[content too long, truncated]\n\n" + 
                    content[-max_tool_content_length//2:]
                )
        elif role == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > max_assistant_content_length:
                msg["content"] = (
                    content[:max_assistant_content_length//2] + 
                    "\n\n[content too long, truncated]\n\n" + 
                    content[-max_assistant_content_length//2:]
                )
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if len(text) > max_assistant_content_length:
                            item["text"] = (
                                text[:max_assistant_content_length//2] + 
                                "\n\n[content too long, truncated]\n\n" + 
                                text[-max_assistant_content_length//2:]
                            )
            
            reasoning = msg.get("reasoning_content", "")
            if isinstance(reasoning, str) and len(reasoning) > max_assistant_reasoning_content_length:
                msg["reasoning_content"] = (
                    reasoning[:max_assistant_reasoning_content_length//2] + 
                    "\n\n[reasoning too long, truncated]\n\n" + 
                    reasoning[-max_assistant_reasoning_content_length//2:]
                )
        
        truncated_messages.append(msg)
    
    tools_str = "\n".join([json.dumps(t, ensure_ascii=False) for t in tools])
    messages_str = "\n".join([json.dumps(m, ensure_ascii=False) for m in truncated_messages])

    max_total_chars_limit = max_total_chars
    if max_total_chars_limit > 0:
        combined_len = len(tools_str) + len(messages_str)
        if combined_len > max_total_chars_limit:
            if len(messages_str) > len(tools_str):
                budget = max_total_chars_limit - len(tools_str)
                messages_str = (
                    messages_str[:budget // 2]
                    + "\n\n[...messages truncated due to context limit...]\n\n"
                    + messages_str[-budget // 2:]
                )
            else:
                budget = max_total_chars_limit - len(messages_str)
                tools_str = (
                    tools_str[:budget // 2]
                    + "\n\n[...tools truncated due to context limit...]\n\n"
                    + tools_str[-budget // 2:]
                )

    return tools_str, messages_str


def format_prompt_for_single_check(tools_str, messages_str, category, check_item, prompt_style="standard"):
    """构造单个 Check 的评估 Prompt"""
    return EVAL_PROMPT_SINGLE.format(
        tools=tools_str,
        messages=messages_str,
        category=category,
        check_id=check_item["check_id"],
        check_description=check_item["description"],
        prompt_style_instruction=get_prompt_style_instruction(prompt_style),
    )


def call_llm(prompt, llm_config):
    api_model = llm_config.get("_api_model")
    if api_model is None:
        raise RuntimeError("APIModel not initialized in llm_config")

    max_prompt_chars = llm_config.get("max_prompt_chars", 0)
    if max_prompt_chars and max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
        half = max_prompt_chars // 2
        prompt = (
            prompt[:half]
            + "\n\n[...prompt truncated due to context limit...]\n\n"
            + prompt[-half:]
        )

    usage = {}
    cache_hit = False
    if llm_config.get("cache_only_missing_as_fail"):
        result = api_model.cache.check_prompt(prompt)
        cache_hit = result is not None
    elif hasattr(api_model, "generate_chat_with_metadata"):
        messages = [{"role": "user", "content": prompt}]
        metadata = api_model.generate_chat_with_metadata(messages, max_tokens=1024, temperature=0.0)
        result = metadata.get("response")
        usage = metadata.get("usage", {}) or {}
        cache_hit = bool(metadata.get("cache_hit", False))
    else:
        result = api_model.generate(prompt, max_tokens=1024, temperature=0.0)
    if result is not None:
        TokenTracker.track(prompt, result, usage=usage, cache_hit=cache_hit)
    return result


def call_llm_with_retry(prompt, llm_config, retries=5, delay=2):
    """调用 LLM 进行评估，增加重试机制"""
    if llm_config.get("cache_only_missing_as_fail"):
        return call_llm(prompt, llm_config)
    for attempt in range(retries):
        response = call_llm(prompt, llm_config)
        if response is not None:
            return response
        print(f"[WARN] LLM调用失败，尝试第 {attempt + 1} 次重试...")
        time.sleep(delay)
    return None

# def call_llm_with_retry(prompt, llm_config, retries=0, delay=2):
#     """调用 LLM 进行评估，增加重试机制"""
#     # for attempt in range(retries):
#     response = call_llm(prompt, llm_config)
#     if response is not None:
#         return response
#     print(f"[WARN] LLM调用失败")
#     return None

def parse_eval_result(response_text, check_id):
    """解析 LLM 返回的单个 Check 评估结果"""
    try:
        # 如果响应为空或仅包含非 JSON 内容，返回默认错误
        if not response_text or not response_text.strip():
            return {
                "check_id": check_id,
                "result": "fail",
                "reasoning": "LLM 响应为空或无有效内容",
                "raw_response": response_text
            }
        
        # 处理并去除响应中的非 JSON 部分，如 Markdown 格式或附加文本
        json_start_index = response_text.find("{")
        json_end_index = response_text.rfind("}") + 1
        if json_start_index == -1 or json_end_index == -1:
            return {
                "check_id": check_id,
                "result": "fail",
                "reasoning": "LLM 响应不包含有效 JSON",
                "raw_response": response_text
            }

        # 提取有效 JSON 部分
        json_text = response_text[json_start_index:json_end_index]

        # 解析 JSON
        data = json.loads(json_text.strip())
        
        # 校验与修正
        if "result" not in data:
            data["result"] = "fail"
        if data["result"] not in ["success", "fail"]:
            data["result"] = "fail"
        if "reasoning" not in data:
            data["reasoning"] = "No reasoning provided"
            
        data["check_id"] = check_id  # 强制对齐 ID
        return data
        
    except json.JSONDecodeError as e:
        return {
            "check_id": check_id,
            "result": "fail",
            "reasoning": f"JSON Parse Error: {str(e)}",
            "raw_response": response_text
        }
    except Exception as e:
        return {
            "check_id": check_id,
            "result": "fail",
            "reasoning": f"Unknown Error: {str(e)}",
            "raw_response": response_text
        }


def calculate_reward(eval_result):
    """计算 reward 分数"""
    if "error" in eval_result:
        return 0.0
    
    total = 0
    success = 0
    
    for category, data in eval_result.items():
        checklist = data.get("checks", [])
        for item in checklist:
            total += 1
            if item.get("result") == "success":
                success += 1
    
    if total == 0:
        return 0.0
    
    return round(success / total, 3)


def save_results(filepath, results_dict, lock):
    """保存结果到文件 (线程安全)"""
    with lock:
        results_list = list(results_dict.values())
        
        # 计算全局统计
        total_instances = len(results_list)
        success_count = sum(1 for r in results_list if r.get("success"))
        avg_reward = round(sum(r.get("reward", 0) for r in results_list) / total_instances, 3) if total_instances else 0
        pass_count = sum(1 for r in results_list if r.get("binary_reward") == 1)
        
        output_data = {
            "results": results_list,
            "summary": {
                "total": total_instances,
                "success_count": success_count,
                "avg_reward": avg_reward,
                "pass_count": pass_count
            }
        }
        
        # 原子写入
        temp_path = filepath + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, filepath)

def format_prompt_for_check_batch(tools_str, messages_str, check_items, prompt_style="standard"):
    checks = []
    for index, check_item in enumerate(check_items, start=1):
        checks.append(
            f'{index}. Category: {check_item["category"]}\nCheck ID: {check_item["check_id"]}\nDescription: {check_item["description"]}'
        )
    return EVAL_PROMPT_BATCH.format(
        tools=tools_str,
        messages=messages_str,
        checks="\n\n".join(checks),
        prompt_style_instruction=get_prompt_style_instruction(prompt_style),
    )


def parse_eval_batch_result(response_text, check_items):
    try:
        if not response_text or not response_text.strip():
            raise ValueError("LLM 响应为空或无有效内容")
        json_start_index = response_text.find("[")
        json_end_index = response_text.rfind("]") + 1
        if json_start_index == -1 or json_end_index <= json_start_index:
            raise ValueError("LLM 响应不包含有效 JSON 数组")
        parsed = json.loads(response_text[json_start_index:json_end_index].strip())
        if not isinstance(parsed, list):
            raise ValueError("LLM 响应 JSON 不是数组")
    except Exception as e:
        return [
            {
                "check_id": check_item["check_id"],
                "result": "fail",
                "reasoning": f"Batch parse error: {str(e)}",
                "raw_response": response_text,
            }
            for check_item in check_items
        ]

    outputs = []
    expected_ids = [str(check_item["check_id"]) for check_item in check_items]
    expected_set = set(expected_ids)
    by_id = {}
    duplicate_ids = set()
    malformed_items = 0
    for item in parsed:
        if not isinstance(item, dict):
            malformed_items += 1
            continue
        check_id = item.get("check_id")
        if check_id is None:
            malformed_items += 1
            continue
        check_id = str(check_id)
        if check_id in by_id:
            duplicate_ids.add(check_id)
            continue
        by_id[check_id] = item

    unexpected_ids = sorted(check_id for check_id in by_id if check_id not in expected_set)
    global_errors = []
    if len(parsed) != len(check_items):
        global_errors.append(f"expected {len(check_items)} objects, got {len(parsed)}")
    if malformed_items:
        global_errors.append(f"{malformed_items} malformed object(s)")
    if unexpected_ids:
        global_errors.append(f"unexpected check_id values: {unexpected_ids}")

    for check_item in check_items:
        check_id = str(check_item["check_id"])
        if check_id in duplicate_ids:
            outputs.append({
                "check_id": check_id,
                "result": "fail",
                "reasoning": f"Batch parse error: duplicate check_id {check_id}",
                "raw_response": response_text,
            })
            continue
        item = by_id.get(check_id)
        if item is None:
            outputs.append({
                "check_id": check_id,
                "result": "fail",
                "reasoning": "Batch response missing this check_id",
                "raw_response": response_text,
            })
            continue
        result = item.get("result", "fail")
        reasoning = item.get("reasoning", "No reasoning provided")
        if result not in ["success", "fail"]:
            result = "fail"
            reasoning = f"Batch parse error: invalid result for check_id {check_id}. {reasoning}"
        elif global_errors:
            reasoning = f"Batch parse error: {'; '.join(global_errors)}. {reasoning}"
        outputs.append({
            "check_id": check_id,
            "result": result,
            "reasoning": reasoning,
        })
    return outputs


def process_check_batch(instance_id, check_items, tools_str, messages_str, llm_config, prompt_style="standard"):
    check_ids = [item["check_id"] for item in check_items]
    print(f"[EVAL] 开始批量评估: {instance_id} - Checks {check_ids}")
    prompt = format_prompt_for_check_batch(tools_str, messages_str, check_items, prompt_style=prompt_style)
    try:
        response = call_llm_with_retry(prompt, llm_config)
        parsed_items = parse_eval_batch_result(response, check_items)
    except Exception as e:
        parsed_items = [
            {
                "check_id": check_item["check_id"],
                "result": "fail",
                "reasoning": f"异常: {str(e)}",
            }
            for check_item in check_items
        ]
    outputs = []
    category_by_id = {str(item["check_id"]): item["category"] for item in check_items}
    for item in parsed_items:
        outputs.append({
            "instance_id": instance_id,
            "category": category_by_id.get(str(item["check_id"]), ""),
            "check_id": item["check_id"],
            "result": item.get("result", "fail"),
            "reasoning": item.get("reasoning", "LLM 解析失败"),
        })
        print(f"[EVAL] 完成批量评估: {instance_id} - Check {item['check_id']}: {item.get('result')}")
    return outputs


def process_check_item(instance_id, category, check_item, tools_str, messages_str, llm_config, prompt_style="standard"):
    """处理单个 Check 的评估任务"""
    check_id = check_item["check_id"]
    print(f"[EVAL] 开始评估: {instance_id} - Check {check_id}")

    # 初始化单个 Check 的评估结果
    eval_result = {
        "instance_id": instance_id,
        "category": category,
        "check_id": check_id,
        "result": "fail",  # 默认失败
        "reasoning": "未评估",
    }
    
    # 构造评估 Prompt
    prompt = format_prompt_for_single_check(tools_str, messages_str, category, check_item, prompt_style=prompt_style)

    # 调用 LLM 进行评估
    try:
        # 调用带重试机制的 LLM
        response = call_llm_with_retry(prompt, llm_config)
        # 解析 LLM 的 JSON
        llm_data = parse_eval_result(response, check_id) 
        
        # 【关键修改】：将 LLM 结果合并回包含元数据的字典
        eval_res = {
            "instance_id": instance_id,
            "category": category,
            "check_id": check_id,
            "result": llm_data.get("result", "fail"),
            "reasoning": llm_data.get("reasoning", "LLM 解析失败")
        }
    except Exception as e:
        eval_res = {
            "instance_id": instance_id,
            "category": category,
            "check_id": check_id,
            "result": "fail",
            "reasoning": f"异常: {str(e)}"
        }
    print(f"[EVAL] 完成评估: {instance_id} - Check {check_id}: {eval_res.get('result')}")
    return eval_res


def process_instance_checks(instance_id, case_data, traj_file, llm_config, results_dict, lock, workers, batch_size, prompt_style="standard"):
    """处理单个实例下的所有 Check 评估任务"""
    print(f"[EVAL] 开始评估实例: {instance_id}")
    
    # 加载轨迹
    record = load_trajectory(traj_file, session_id=instance_id)
    if not record:
        print(f"[EVAL] 轨迹文件为空或不存在: {instance_id}")
        return

    # 获取 Checklist
    checklist = case_data.get("checklist", {})
    if not checklist:
        print(f"[EVAL] case 中没有 checklist: {instance_id}")
        return

    # 提前格式化上下文 (只需一次)
    try:
        tools_str, messages_str = get_truncated_context(record, max_total_chars=llm_config.get("max_prompt_chars", 0))
    except Exception as e:
        print(f"[EVAL] 轨迹格式化失败: {str(e)}")
        return

    # 初始化并发执行器
    futures = []
    check_items = []
    for category, cat_data in checklist.items():
        for check in cat_data.get("checks", []):
            check_item = dict(check)
            check_item["category"] = category
            check_items.append(check_item)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch_start in range(0, len(check_items), batch_size):
            check_batch = check_items[batch_start:batch_start + batch_size]
            if batch_size == 1:
                future = executor.submit(
                    process_check_item,
                    instance_id, check_batch[0]["category"], check_batch[0],
                    tools_str, messages_str, llm_config, prompt_style
                )
            else:
                future = executor.submit(
                    process_check_batch,
                    instance_id, check_batch,
                    tools_str, messages_str, llm_config, prompt_style
                )
            futures.append(future)

        # 等待所有任务完成并收集结果
        for future in as_completed(futures):
            result = future.result()
            eval_results = result if isinstance(result, list) else [result]
            for eval_res in eval_results:
                if eval_res:
                    # 保存每个 Check 的结果
                    with lock:
                        # 【关键修改】：初始化时存入 instance_id
                        if instance_id not in results_dict:
                            results_dict[instance_id] = {
                                "instance_id": instance_id,
                                "eval_result": {}
                            }
                        category = eval_res.get("category", "")
                        if category not in results_dict[instance_id]:
                            results_dict[instance_id][category] = {
                                "checks": []
                            }
                        results_dict[instance_id][category]["checks"].append(eval_res)
            
    print(f"[EVAL] 完成实例评估: {instance_id}")
def process_instance(instance_id, case_data, traj_file, llm_config, results_dict, output_file, lock):
    """处理单个 Session 的评估任务"""
    print(f"[EVAL] 开始评估: {instance_id}")
    
    # 初始化结果结构
    instance_result = {
        "instance_id": instance_id,
        "success": False,
        "eval_result": {},
        "reward": 0.0,
        "binary_reward": 0
    }
    
    # 预占位，确保即时写入
    with lock:
        results_dict[instance_id] = instance_result
    save_results(output_file, results_dict, lock)

    # 加载轨迹
    record = load_trajectory(traj_file, session_id=instance_id)
    if not record:
        instance_result["error"] = "轨迹文件为空或不存在"
        with lock:
            results_dict[instance_id] = instance_result
        save_results(output_file, results_dict, lock)
        return

    # 获取 Checklist
    checklist = case_data.get("checklist", {})
    if not checklist:
        instance_result["error"] = "case 中没有 checklist"
        with lock:
            results_dict[instance_id] = instance_result
        save_results(output_file, results_dict, lock)
        return

    # 提前格式化上下文 (只需一次)
    try:
        tools_str, messages_str = get_truncated_context(record, max_total_chars=llm_config.get("max_prompt_chars", 0))
    except Exception as e:
        instance_result["error"] = f"轨迹格式化失败: {str(e)}"
        with lock:
            results_dict[instance_id] = instance_result
        save_results(output_file, results_dict, lock)
        return

    # 逐个 Check 评估
    try:
        for category, cat_data in checklist.items():
            if category not in instance_result["eval_result"]:
                instance_result["eval_result"][category] = {
                    "description": cat_data.get("description", ""),
                    "checks": []
                }
            
            for check_item in cat_data.get("checks", []):
                check_id = check_item["check_id"]
                
                # 构造 Prompt
                prompt = format_prompt_for_single_check(tools_str, messages_str, category, check_item)
                
                # 调用 LLM
                try:
                    response = call_llm(prompt, llm_config)
                    eval_res = parse_eval_result(response, check_id)
                except Exception as e:
                    eval_res = {
                        "check_id": check_id,
                        "result": "fail",
                        "reasoning": f"LLM调用失败: {str(e)}"
                    }

                # 更新结果
                instance_result["eval_result"][category]["checks"].append(eval_res)
                instance_result["reward"] = calculate_reward(instance_result["eval_result"])
                instance_result["binary_reward"] = 1 if instance_result["reward"] == 1.0 else 0
                instance_result["success"] = True # 标记为处理成功（即使check失败）

                # 实时写入
                with lock:
                    results_dict[instance_id] = instance_result
                save_results(output_file, results_dict, lock)
                print(f"[EVAL] {instance_id} - Check {check_id}: {eval_res.get('result')}")

    except Exception as e:
        instance_result["error"] = f"评估过程异常: {str(e)}"
        instance_result["success"] = False
        with lock:
            results_dict[instance_id] = instance_result
        save_results(output_file, results_dict, lock)

    print(f"[EVAL] 完成: {instance_id}, Reward: {instance_result['reward']}")


def main():
    parser = argparse.ArgumentParser(description="轨迹评估器 - 通过 APIModel 调用 LLM")
    parser.add_argument("--trajectories", required=True, help="轨迹文件或目录")
    parser.add_argument("--data", required=True, help="测试用例文件 (JSONL)")
    parser.add_argument("--output", default="./scores.json", help="输出文件")
    parser.add_argument(
        "--model",
        default=os.environ.get("VERIBENCH_MODEL", "openai-compatible-judge"),
        help="model_marker 字符串",
    )
    parser.add_argument("--case", help="指定 instance_id")
    parser.add_argument("--limit", type=int, default=0, help="限制评估的Session数量 (0为不限制)")
    parser.add_argument("--workers", type=int, default=1, help="并发数")
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get("JUDGE_BASE_URL", os.environ.get("OPENAI_BASE_URL", "")),
        help="OpenAI-compatible base URL (e.g. http://127.0.0.1:8000/v1 for local vLLM)",
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.environ.get("JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")),
        help="API key for OpenAI-compatible endpoint. Default: JUDGE_API_KEY/OPENAI_API_KEY/EMPTY",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="每次 LLM 调用评估的 Check 数量")
    parser.add_argument(
        "--prompt-style",
        default="standard",
        choices=["standard", "strict", "semantic", "evidence_first"],
        help="Prompt variant for check verification (默认: standard)",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=int(os.environ.get("MAX_PROMPT_CHARS", "0")),
        help="Max prompt chars to send to LLM. 0 = no limit (default). Truncates head+tail if exceeded.",
    )
    parser.add_argument(
        "--cache-only-missing-as-fail",
        action="store_true",
        help="Only use local cache; cache misses are parsed as fail without API calls.",
    )
    parser.add_argument(
        "--skip-final-cache-flush",
        action="store_true",
        help="Do not flush cache at shutdown; useful after a cache-only recovery run.",
    )
    args = parser.parse_args()

    if args.judge_base_url:
        os.environ["DETECTOR_API_BACKEND"] = "openai"

    # 初始化 APIModel（与 test_api.py 用法一致）
    cache_dir = os.path.join(_RUBRICBENCH_ROOT, "cache", args.model)
    api_model = APIModel(
        cache_url=cache_dir,
        base_url=args.judge_base_url,
        model_name=args.model,
        api_key=args.judge_api_key,
    )
    print(f"[INFO] 使用模型: {args.model}，缓存目录: {cache_dir}")
    print(f"[INFO] Check batch size: {args.batch_size}")
    print(f"[INFO] Prompt style: {args.prompt_style}")
    if args.judge_base_url:
        print(f"[INFO] Judge base URL: {args.judge_base_url}")

    llm_config = {
        "_api_model": api_model,
        "max_prompt_chars": args.max_prompt_chars,
        "cache_only_missing_as_fail": args.cache_only_missing_as_fail,
    }
    if args.cache_only_missing_as_fail:
        print("[INFO] Cache-only recovery enabled: cache misses will be recorded as fail without API calls.")

    # 加载测试用例
    with open(args.data, encoding="utf-8") as f:
        cases = {json.loads(line)["instance_id"]: json.loads(line) for line in f if line.strip()}
    
    print(f"[INFO] 加载了 {len(cases)} 个测试用例")

    # 映射 Session ID 到文件
    traj_path = Path(args.trajectories)
    session_to_file = {}
    
    if traj_path.is_file():
        with open(traj_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    session_id = record.get("meta", {}).get("session_id", "")
                    if session_id and session_id not in session_to_file:
                        session_to_file[session_id] = str(traj_path)
        print(f"[INFO] 从文件中找到 {len(session_to_file)} 个 session")
    else:
        trajectory_files = list(traj_path.glob("*.jsonl"))
        session_to_file = {f.stem: str(f) for f in trajectory_files}
        print(f"[INFO] 找到 {len(trajectory_files)} 个轨迹文件")

    # 筛选待评估任务
    tasks = []
    for instance_id, case_data in cases.items():
        if args.case and instance_id != args.case:
            continue
        if instance_id in session_to_file:
            tasks.append((instance_id, case_data, session_to_file[instance_id]))
    
    # 应用 Limit
    if args.limit > 0:
        tl = len(tasks)
        print(f"[INFO] 总数据量: {tl}")
        tasks = tasks[:args.limit]
        print(f"[INFO] 根据限制，将评估前 {len(tasks)} 条数据")

    results_dict = {}
    lock = threading.Lock()

    token_stats_file = str(Path(args.output).with_suffix(".token_stats.json"))
    TokenTracker.setup(token_stats_file, save_every=10)

    try:
        # 并发执行每个实例的所有 Check
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for instance_id, case_data, traj_file in tasks:
                future = executor.submit(
                    process_instance_checks,
                    instance_id, case_data, traj_file,
                    llm_config, results_dict, lock, args.workers, max(1, args.batch_size), args.prompt_style
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] 线程执行异常: {e}")
    finally:
        TokenTracker._save_to_file()
    # 保存最终结果
    save_results(args.output, results_dict, lock)
    if not args.skip_final_cache_flush:
        api_model.save_cache()
    print(f"\n[DONE] 评估完成，结果已保存至 {args.output}")
    TokenTracker.print_summary()


if __name__ == "__main__":
    main()
