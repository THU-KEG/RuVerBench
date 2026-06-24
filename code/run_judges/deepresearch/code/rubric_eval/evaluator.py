#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core evaluation logic for AI model responses using rubrics (Parallel Version)
Enhanced with thread-safe operations and optimized for concurrent processing.
Uses APIModel from the parent project for LLM calls.
"""

import os
import time
import threading
import json
from typing import List, Dict, Any, Tuple

import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

rubricbench_root = os.path.dirname(project_root)
if rubricbench_root not in sys.path:
    sys.path.insert(0, rubricbench_root)

from code.prompt import get_single_rubric_evaluation_prompt, get_prompt_style_instruction


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: CJK chars ~1.5 tokens each, ASCII ~0.25 tokens per char."""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f')
    ascii_chars = len(text) - cjk
    return int(cjk * 1.5 + ascii_chars * 0.25)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _extract_usage_tokens(usage: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Return (input_tokens, cached_input_tokens, output_tokens) from heterogeneous usage shapes.
    """
    if not isinstance(usage, dict):
        return 0, 0, 0

    input_tokens = _safe_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    output_tokens = _safe_int(usage.get("completion_tokens", usage.get("output_tokens", 0)))

    cached_input_tokens = _safe_int(usage.get("cache_input_tokens", usage.get("cached_tokens", 0)))
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        cached_input_tokens = max(cached_input_tokens, _safe_int(prompt_details.get("cached_tokens", 0)))
    elif prompt_details is not None and hasattr(prompt_details, "cached_tokens"):
        cached_input_tokens = max(cached_input_tokens, _safe_int(getattr(prompt_details, "cached_tokens", 0)))

    # Some APIs report cache read/write under separate keys.
    cached_input_tokens = max(
        cached_input_tokens,
        _safe_int(usage.get("cache_read_input_tokens", 0)),
    )
    cached_input_tokens = max(
        cached_input_tokens,
        _safe_int(usage.get("cache_creation_input_tokens", 0)),
    )
    return input_tokens, cached_input_tokens, output_tokens


def _longest_common_prefix_len(a: str, b: str) -> int:
    max_len = min(len(a), len(b))
    i = 0
    while i < max_len and a[i] == b[i]:
        i += 1
    return i


class ThreadSafeRubricEvaluator:
    """线程安全的标准评估器，使用 APIModel 调用 LLM"""

    _print_lock = threading.Lock()
    _stats_lock = threading.Lock()
    _global_stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0
    }
    _token_lock = threading.Lock()
    _token_stats = {
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
        # Hybrid/fallback totals: always populated.
        "effective_input_tokens": 0,
        "effective_output_tokens": 0,
        "effective_cached_input_tokens": 0,
        "effective_uncached_input_tokens": 0,
        # Heuristic cache estimate from longest common prefix when usage is unavailable.
        "estimated_cache_input_tokens_from_lcp": 0,
    }
    _lcp_lock = threading.Lock()
    _last_input_text_for_lcp = ""

    def __init__(self, api_model, max_retries: int = 3, rubric_batch_size: int = 1, prompt_style: str = "standard"):
        """
        Parameters:
        api_model: APIModel instance (shared across threads, thread-safe via Cache)
        max_retries (int): 最大重试次数
        rubric_batch_size (int): 每次 LLM 调用评估的 rubric 数量
        prompt_style (str): Prompt variant. "standard" preserves the original prompt.
        """
        self.api_model = api_model
        self.judge_model = api_model.model_name
        self.max_retries = max_retries
        self.rubric_batch_size = max(1, int(rubric_batch_size))
        self.prompt_style = prompt_style
        self.thread_id = threading.current_thread().ident

        self.thread_stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0
        }

    def _safe_print(self, message: str):
        with self._print_lock:
            print(f"[Thread-{self.thread_id}] {message}")

    def _update_global_stats(self, success: bool):
        with self._stats_lock:
            self._global_stats["total_requests"] += 1
            if success:
                self._global_stats["successful_requests"] += 1
            else:
                self._global_stats["failed_requests"] += 1

    def _track_tokens(self, messages: List[Dict], response: str, usage: Dict[str, Any], cache_hit: bool):
        input_text = " ".join(m.get("content", "") for m in messages)
        input_chars = len(input_text)
        output_chars = len(response)
        est_in = _estimate_tokens(input_text)
        est_out = _estimate_tokens(response)
        input_tokens, cached_input_tokens, output_tokens = _extract_usage_tokens(usage)
        has_usage = (input_tokens > 0) or (output_tokens > 0) or (cached_input_tokens > 0)

        # Fallback heuristic for cache tokens when server usage is missing:
        # estimate by longest common prefix against the previous request text.
        lcp_cached_est = 0
        if not has_usage:
            with self._lcp_lock:
                prev = self._last_input_text_for_lcp
                if prev:
                    lcp_chars = _longest_common_prefix_len(prev, input_text)
                    lcp_cached_est = _estimate_tokens(input_text[:lcp_chars])
                self._last_input_text_for_lcp = input_text

        effective_input = input_tokens if has_usage else est_in
        effective_output = output_tokens if has_usage else est_out
        effective_cached = cached_input_tokens if has_usage else lcp_cached_est

        with self._token_lock:
            self._token_stats["total_input_chars"] += input_chars
            self._token_stats["total_output_chars"] += output_chars
            self._token_stats["estimated_input_tokens"] += est_in
            self._token_stats["estimated_output_tokens"] += est_out
            if cache_hit:
                self._token_stats["local_cache_hits"] += 1
            if has_usage:
                self._token_stats["requests_with_usage"] += 1
                self._token_stats["actual_input_tokens"] += input_tokens
                self._token_stats["actual_output_tokens"] += output_tokens
                self._token_stats["actual_cached_input_tokens"] += cached_input_tokens
                self._token_stats["actual_uncached_input_tokens"] += max(input_tokens - cached_input_tokens, 0)
            else:
                self._token_stats["requests_without_usage"] += 1
                self._token_stats["estimated_cache_input_tokens_from_lcp"] += lcp_cached_est
            self._token_stats["effective_input_tokens"] += effective_input
            self._token_stats["effective_output_tokens"] += effective_output
            self._token_stats["effective_cached_input_tokens"] += effective_cached
            self._token_stats["effective_uncached_input_tokens"] += max(effective_input - effective_cached, 0)

    def _make_api_request(self, messages: List[Dict], operation_name: str) -> str:
        for retry_count in range(self.max_retries):
            try:
                self.thread_stats["requests"] += 1
                usage = {}
                cache_hit = False
                if hasattr(self.api_model, "generate_chat_with_metadata"):
                    metadata = self.api_model.generate_chat_with_metadata(messages)
                    result = metadata.get("response")
                    usage = metadata.get("usage", {}) or {}
                    cache_hit = bool(metadata.get("cache_hit", False))
                else:
                    result = self.api_model.generate_chat(messages)
                if result is None:
                    raise RuntimeError("APIModel returned None")
                result = result.strip()
                self.thread_stats["successes"] += 1
                self._update_global_stats(True)
                self._track_tokens(messages, result, usage, cache_hit)
                return result
            except Exception as e:
                self.thread_stats["failures"] += 1
                error_msg = f"{operation_name} 失败 (尝试 {retry_count + 1}/{self.max_retries}): {e}"
                if retry_count == self.max_retries - 1:
                    self._safe_print(f"Reached maximum retry limit, {operation_name} failed")
                    self._update_global_stats(False)
                    raise e
                else:
                    self._safe_print(error_msg + " - Retrying...")
                    time.sleep(min(2 ** retry_count, 60))
    
    def evaluate_single_rubric(self, question: str, rubric: Dict[str, Any], 
                              ai_response: str) -> Tuple[bool, str]:
        """
        评估单个标准项是否在AI回答中得到覆盖
        
        Parameters:
        question (str): 用户提出的研究问题
        rubric (dict): 单个标准项信息
        ai_response (str): 完整的AI回答文本
        
        Returns:
        tuple: (是否覆盖, 理由说明)
        """
        rubric_content = rubric['point']
        rubric_weight = rubric.get('weight', 1)
        
        # 获取评估提示词
        coverage_prompt = get_single_rubric_evaluation_prompt(
            question, rubric_content, rubric_weight, ai_response, prompt_style=self.prompt_style
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that helps determine if a rubric is covered in an AI response."},
            {"role": "user", "content": coverage_prompt}
        ]
        
        try:
            result = self._make_api_request(messages, "标准项覆盖评估")
            
            # 解析是否/否答案和理由说明
            if result.lower().startswith("yes:") or result.lower().startswith("yes "):
                is_covered = True
                justification = result[4:].strip() if len(result) > 3 and result[3] == ":" else result[4:].strip()
                return is_covered, justification
            elif result.lower().startswith("no:") or result.lower().startswith("no "):
                is_covered = False
                justification = result[3:].strip() if len(result) > 2 and result[2] == ":" else result[3:].strip()
                return is_covered, justification
            else:
                # 如果不以yes/no开头，检查是否包含yes/no
                if "yes" in result.lower():
                    return True, result
                elif "no" in result.lower():
                    return False, result
                else:
                    self._safe_print("Evaluation result does not contain clear yes/no answer, defaulting to not covered")
                    return False, "Unable to get clear answer, defaulting to not covered"
                        
        except Exception as e:
            error_msg = f"Error evaluating rubric coverage: {str(e)}"
            self._safe_print(error_msg)
            return False, error_msg
    
    def evaluate_rubric_batch(self, question: str, rubrics: List[Dict[str, Any]], ai_response: str, batch_start: int) -> List[Tuple[bool, str]]:
        rubric_lines = []
        expected_indices = []
        for offset, rubric in enumerate(rubrics):
            point_index = batch_start + offset
            expected_indices.append(point_index)
            point_text = str(rubric["point"]).replace("\n", " ")
            rubric_lines.append(f'point_index={point_index}: {point_text}')
        example_items = [
            {
                "point_index": expected_indices[0],
                "answer": "yes",
                "covered": True,
                "justification": "The response clearly addresses this criterion by explaining [specific detail]...",
            }
        ]
        if len(expected_indices) > 1:
            example_items.append(
                {
                    "point_index": expected_indices[-1],
                    "answer": "no",
                    "covered": False,
                    "justification": "While the response mentions [related concept], it fails to address [specific aspect] of the criterion...",
                }
            )
        example_json = json.dumps(example_items, ensure_ascii=False, indent=2)
        prompt = f"""# Rubric Coverage Evaluation

## Task
Determine whether the AI response adequately covers each specific criterion provided. For each criterion, answer with "yes" or "no" followed by a brief justification.

## Input Materials
<Question>: {question}
<AI Response>: {ai_response}
<criteria>
{chr(10).join(rubric_lines)}
</criteria>

## Evaluation Criteria
- Answer "yes" if the AI response clearly includes or adequately expresses the main content of the criterion
- Answer "yes" if the response conveys the same meaning as the criterion, even if using different terminology or phrasing
- Answer "no" if the AI response only partially addresses or completely fails to mention the content of the criterion
- Consider semantic equivalence, not just keyword matching
- Pay special attention to technical details, numerical values, and specific claims in the criterion{get_prompt_style_instruction(self.prompt_style)}

## Output Format
Return only a valid JSON array with exactly {len(rubrics)} objects. Each object must repeat the exact point_index shown above:
{example_json}

Do not use positional order as the only identifier. Every expected point_index must appear exactly once.
Note: Your assessment will be used to calculate recall metrics, so accuracy is critical."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that helps determine if rubrics are covered in an AI response."},
            {"role": "user", "content": prompt},
        ]
        try:
            result = self._make_api_request(messages, "批量标准项覆盖评估")
            start = result.find("[")
            end = result.rfind("]") + 1
            if start == -1 or end <= start:
                raise ValueError("Batch response does not contain a JSON array")
            parsed = json.loads(result[start:end])
            if not isinstance(parsed, list):
                raise ValueError("Batch response JSON is not an array")

            by_index: Dict[int, Dict[str, Any]] = {}
            duplicate_indices = set()
            malformed_items = 0
            for item in parsed:
                if not isinstance(item, dict):
                    malformed_items += 1
                    continue
                if "point_index" not in item:
                    malformed_items += 1
                    continue
                try:
                    point_index = int(item["point_index"])
                except Exception:
                    malformed_items += 1
                    continue
                if point_index in by_index:
                    duplicate_indices.add(point_index)
                    continue
                by_index[point_index] = item

            expected_set = set(expected_indices)
            unexpected_indices = sorted(idx for idx in by_index if idx not in expected_set)
            length_mismatch = len(parsed) != len(rubrics)
            global_errors = []
            if length_mismatch:
                global_errors.append(f"expected {len(rubrics)} objects, got {len(parsed)}")
            if malformed_items:
                global_errors.append(f"{malformed_items} malformed object(s)")
            if unexpected_indices:
                global_errors.append(f"unexpected point_index values: {unexpected_indices}")

            outputs: List[Tuple[bool, str]] = []
            for point_index in expected_indices:
                if point_index in duplicate_indices:
                    outputs.append((False, f"Batch parse error: duplicate point_index {point_index}"))
                    continue
                item = by_index.get(point_index)
                if item is None:
                    outputs.append((False, f"Batch response missing point_index {point_index}"))
                    continue

                answer = str(item.get("answer", "")).strip().lower()
                covered_value = item.get("covered")
                if answer.startswith("yes"):
                    covered = True
                elif answer.startswith("no"):
                    covered = False
                elif isinstance(covered_value, bool):
                    covered = covered_value
                else:
                    outputs.append((False, f"Batch parse error: invalid answer for point_index {point_index}"))
                    continue

                justification = str(item.get("justification", item.get("reasoning", "No justification provided")))
                if global_errors:
                    justification = f"Batch parse error: {'; '.join(global_errors)}. {justification}"
                outputs.append((covered, justification))
            return outputs
        except Exception as e:
            error_msg = f"Error evaluating rubric batch: {str(e)}"
            self._safe_print(error_msg)
            return [(False, error_msg) for _ in rubrics]

    def evaluate_coverage(self, question: str, reference_rubrics: List[Dict[str, Any]],
                         ai_response: str) -> Tuple[list, int, int]:
        """
        评估AI回答对参考标准项的覆盖情况
        
        Parameters:
        question (str): 用户提出的研究问题
        reference_rubrics (list): 参考标准项列表
        ai_response (str): 完整的AI回答文本
        
        Returns:
        tuple: (覆盖结果, 加权覆盖数, 总权重)
        """
        coverage_results = []
        m_weighted = 0
        total_weight = 0
        
        self._safe_print(f"Starting rubric evaluation, total {len(reference_rubrics)} rubric items...")
        
        for batch_start in range(0, len(reference_rubrics), self.rubric_batch_size):
            rubric_batch = reference_rubrics[batch_start:batch_start + self.rubric_batch_size]
            if self.rubric_batch_size == 1:
                batch_outputs = [self.evaluate_single_rubric(question, rubric_batch[0], ai_response)]
            else:
                batch_outputs = self.evaluate_rubric_batch(question, rubric_batch, ai_response, batch_start)

            for batch_offset, rubric in enumerate(rubric_batch):
                i = batch_start + batch_offset
                weight = rubric.get("weight", 1)
                total_weight += weight
                is_covered, justification = batch_outputs[batch_offset]

                coverage_results.append({
                    "point": rubric["point"],
                    "weight": weight,
                    "covered": is_covered,
                    "justification": justification
                })

                if is_covered:
                    m_weighted += weight
                    self._safe_print(f"Rubric item {i+1}/{len(reference_rubrics)} covered (weight: {weight})")
                else:
                    self._safe_print(f"Rubric item {i+1}/{len(reference_rubrics)} not covered (weight: {weight})")
        
        self._safe_print(f"Coverage evaluation completed. Covered weight: {m_weighted}, Total weight: {total_weight}")
        
        return coverage_results, m_weighted, total_weight
    
    def evaluate_response(self, question_id: int, model_name: str, question: str, 
                         ai_response: str, reference_rubrics: List[Dict[str, Any]]) -> Dict:
        """
        评估单个模型回答并返回结果
        
        Parameters:
        question_id (int): 问题ID
        model_name (str): 模型名称
        question (str): 研究问题
        ai_response (str): 要评估的AI生成回答文本
        reference_rubrics (list): 参考标准项列表
        
        Returns:
        dict: 包含评估结果的JSON对象
        """
        self._safe_print(f"Starting evaluation for question {question_id}, using evaluator: {self.judge_model}")
        
        # 评估标准覆盖情况
        coverage_results, m_weighted, total_weight = self.evaluate_coverage(
            question, reference_rubrics, ai_response
        )
        
        # 计算指标
        recall = m_weighted / total_weight if total_weight > 0 else 0
        
        self._safe_print(f"Question {question_id} evaluation completed - Coverage: {recall:.4f}")
        
        # 创建结果对象
        result_json = {
            "id": question_id,
            "model": model_name,
            "metric": "Evaluation",
            "evaluator": self.judge_model,
            "result": {
                "m_weighted": m_weighted,
                "total_weight": total_weight,
                "recall": round(recall, 4),
                "coverage_results": coverage_results
            },
            "thread_stats": self.thread_stats.copy()  # 包含线程级统计信息
        }
        
        return result_json
    
    @classmethod
    def get_global_stats(cls) -> Dict:
        with cls._stats_lock:
            return cls._global_stats.copy()

    @classmethod
    def get_token_stats(cls) -> Dict:
        with cls._token_lock:
            return cls._token_stats.copy()

    @classmethod
    def reset_global_stats(cls):
        with cls._stats_lock:
            cls._global_stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0
            }
        with cls._token_lock:
            cls._token_stats = {
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
        with cls._lcp_lock:
            cls._last_input_text_for_lcp = ""


RubricEvaluator = ThreadSafeRubricEvaluator
