#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main evaluation script for AI model responses using rubrics (Parallel Version)
Enhanced with concurrent processing for improved performance.
Uses APIModel from the parent project for LLM calls.
"""

import os
import sys
import argparse
import time
import json
import re
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

rubricbench_root = os.path.dirname(project_root)
if rubricbench_root not in sys.path:
    sys.path.insert(0, rubricbench_root)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

os.environ.setdefault("DETECTOR_API_BACKEND", "openai")

from code.utils import (
    load_rubrics, load_model_responses, save_json_file, 
    create_output_directory, get_model_name_from_file,
    validate_rubric_format, validate_response_format
)
from evaluator import RubricEvaluator
from model import APIModel
from pathlib import Path


def _safe_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "__", value.strip())
    return sanitized.strip("._") or "model"


def _is_failed_result(result_item: Dict) -> bool:
    """
    判断单个问题的评估结果是否明显失败（如评估过程中一直报错）。
    当前通过以下启发式规则判断：
      - 任意 rubric 的 justification 以 'Error evaluating rubric coverage:' 开头
      - 或 thread_stats 中 successes 为 0 且 failures > 0
    """
    try:
        result = result_item.get("result", {})
        coverage_results = result.get("coverage_results", [])
        for item in coverage_results:
            justification = item.get("justification", "")
            if isinstance(justification, str) and justification.startswith("Error evaluating rubric coverage:"):
                return True
        thread_stats = result_item.get("thread_stats", {})
        if isinstance(thread_stats, dict):
            successes = thread_stats.get("successes", 0)
            failures = thread_stats.get("failures", 0)
            if successes == 0 and failures > 0:
                return True
    except Exception:
        # 出现异常时宁可视作“非失败”，避免误删或覆盖结果
        pass
    return False


class ParallelEvaluationCoordinator:
    """协调并行评估任务的类"""
    
    def __init__(self, max_workers: int = 4, rubric_batch_size: int = 1, prompt_style: str = "standard"):
        """
        初始化并行评估协调器

        Parameters:
        max_workers (int): 最大并行工作线程数
        """
        self.max_workers = max_workers
        self.rubric_batch_size = max(1, int(rubric_batch_size))
        self.prompt_style = prompt_style
        self.print_lock = Lock()  # For synchronized print output
        
    def safe_print(self, message: str):
        """线程安全的打印函数"""
        with self.print_lock:
            print(message)
    
    def evaluate_single_question(self, args: Tuple) -> Dict:
        (question_id, response_data, rubric_data, model_name, api_model, thread_id) = args

        try:
            self.safe_print(f"[Thread-{thread_id}] Starting evaluation for question {question_id}...")

            evaluator = RubricEvaluator(
                api_model=api_model,
                rubric_batch_size=self.rubric_batch_size,
                prompt_style=self.prompt_style,
            )

            question = response_data["question"]
            ai_response = response_data["response"]
            reference_rubrics = rubric_data["rubric"]

            result = evaluator.evaluate_response(
                question_id=question_id,
                model_name=model_name,
                question=question,
                ai_response=ai_response,
                reference_rubrics=reference_rubrics
            )

            self.safe_print(f"[Thread-{thread_id}] Question {question_id} evaluation completed")
            return result

        except Exception as e:
            error_result = {
                "question_id": question_id,
                "error": str(e),
                "status": "failed"
            }
            self.safe_print(f"[Thread-{thread_id}] Question {question_id} evaluation failed: {e}")
            return error_result


def evaluate_single_model_parallel(model_file: str, rubrics_dict: Dict, result_dir: str,
                                 api_model=None, max_workers: int = 4,
                                 rerun_failed_only: bool = False,
                                 limit: int = 0,
                                 rubric_batch_size: int = 1,
                                 prompt_style: str = "standard") -> str:
    """
    并行评估单个模型的响应

    Parameters:
    model_file (str): 模型响应文件路径
    rubrics_dict (Dict): 按问题ID组织的标准字典
    result_dir (str): 结果输出目录
    api_model: APIModel instance for LLM calls
    max_workers (int): 最大并行工作线程数
    limit (int): 只评估前N个问题，0=全部

    Returns:
    str: 输出文件路径
    """
    print(f"\n{'='*60}")
    print(f"Parallel evaluation for model: {model_file}")
    print(f"Maximum parallel workers: {max_workers}")
    print(f"Rubric batch size: {rubric_batch_size}")
    print(f"Prompt style: {prompt_style}")
    if limit > 0:
        print(f"LIMIT MODE: only evaluating first {limit} question(s)")
    print(f"{'='*60}")
    
    # 加载模型响应
    model_responses = load_model_responses(model_file)
    if not model_responses:
        print(f"Unable to load responses from {model_file}")
        return None
    
    # 从文件名提取模型名称
    model_name = get_model_name_from_file(model_file)
    
    judge_model = _safe_path_component(api_model.model_name)
    model_result_dir = os.path.join(result_dir, "rubric_eval", judge_model)
    if not create_output_directory(model_result_dir):
        print(f"Unable to create model output directory: {model_result_dir}")
        return None
    output_file = os.path.join(model_result_dir, f"{model_name}_evaluation_results.json")
    
    # 如果需要“只重跑失败条目”，先读取已有结果并找出失败的问题
    existing_results_by_id: Dict[int, Dict] = {}
    failed_question_ids = set()
    if rerun_failed_only:
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                if isinstance(existing_data, list):
                    for item in existing_data:
                        qid = item.get("id")
                        if qid is None:
                            continue
                        existing_results_by_id[qid] = item
                        if _is_failed_result(item):
                            failed_question_ids.add(qid)
                print(f"Found existing results for {len(existing_results_by_id)} questions "
                      f"(failed={len(failed_question_ids)}) in {output_file}")
            except Exception as e:
                print(f"Warning: unable to load existing results from {output_file}: {e}")
                # 如果旧结果读失败，为安全起见退回到全量重跑
                rerun_failed_only = False
        else:
            print(f"No existing result file found at {output_file}, "
                  f"--rerun_failed_only will be ignored and full evaluation will run.")
            rerun_failed_only = False
    
    # 初始化并行评估协调器
    coordinator = ParallelEvaluationCoordinator(
        max_workers=max_workers,
        rubric_batch_size=rubric_batch_size,
        prompt_style=prompt_style,
    )
    
    # 准备评估任务
    evaluation_tasks = []
    thread_counter = 0
    
    for question_id, response_data in model_responses.items():
        # 若只重跑失败条目且该问题不在失败列表中，则跳过
        if rerun_failed_only and failed_question_ids and question_id not in failed_question_ids:
            continue
        
        # 验证响应格式
        if not validate_response_format(response_data):
            print(f"Skipping question {question_id}, invalid format")
            continue
        
        # 检查是否存在对应的标准
        if question_id not in rubrics_dict:
            print(f"Question {question_id} has no corresponding rubric, skipping...")
            continue
        
        rubric_data = rubrics_dict[question_id]
        
        # 验证标准格式
        if not validate_rubric_format(rubric_data):
            print(f"Skipping question {question_id}, invalid rubric format")
            continue
        
        task_args = (question_id, response_data, rubric_data, model_name, api_model, thread_counter)
        evaluation_tasks.append(task_args)
        thread_counter += 1

        if limit > 0 and thread_counter >= limit:
            print(f"Reached limit of {limit} question(s), stopping task preparation.")
            break
    
    if not evaluation_tasks:
        if rerun_failed_only:
            print("No failed questions to re-evaluate. Existing results are kept unchanged.")
            return model_result_dir
        print("No valid evaluation tasks")
        return None
    
    print(f"Preparing parallel evaluation for {len(evaluation_tasks)} questions...")
    
    # 执行并行评估
    all_results = []
    failed_tasks = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(coordinator.evaluate_single_question, task_args): task_args[0] 
            for task_args in evaluation_tasks
        }
        
        # 收集结果
        completed_count = 0
        for future in as_completed(future_to_task):
            question_id = future_to_task[future]
            try:
                result = future.result()
                if "error" in result:
                    failed_tasks.append(result)
                else:
                    all_results.append(result)
                
                completed_count += 1
                print(f"Progress: {completed_count}/{len(evaluation_tasks)} completed")
                
            except Exception as e:
                error_result = {
                    "question_id": question_id,
                    "error": str(e),
                    "status": "exception"
                }
                failed_tasks.append(error_result)
                print(f"Task exception - Question {question_id}: {e}")
    
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    # 打印执行统计
    print(f"\n{'='*50}")
    print("Parallel Evaluation Statistics")
    print(f"{'='*50}")
    print(f"Total tasks: {len(evaluation_tasks)}")
    print(f"Successfully completed: {len(all_results)}")
    print(f"Failed tasks: {len(failed_tasks)}")
    print(f"Total time: {evaluation_time:.2f} seconds")
    print(f"Average time per question: {evaluation_time/len(evaluation_tasks):.2f} seconds")
    print(f"{'='*50}")
    
    # 如果有失败的任务，打印详情
    if failed_tasks:
        print("\nFailed task details:")
        for failed_task in failed_tasks:
            print(f"- Question {failed_task['question_id']}: {failed_task['error']}")
    
    # 保存结果
    # 1) 默认情况：全量重跑，直接保存本次结果
    # 2) 只重跑失败条目：读取旧结果并按 question_id 合并
    data_to_save = all_results
    if rerun_failed_only and existing_results_by_id:
        merged = existing_results_by_id.copy()
        for item in all_results:
            qid = item.get("id")
            if qid is None:
                continue
            merged[qid] = item
        # 按问题 id 排序，便于后续查看
        data_to_save = sorted(merged.values(), key=lambda x: x.get("id", 0))
    
    if save_json_file(data_to_save, output_file):
        print(f"\nEvaluation results saved to: {model_result_dir}")
        
        # 打印汇总统计
        if all_results:
            print(f"\n{'='*50}")
            print("Evaluation Results Summary")
            print(f"{'='*50}")
            
            # 计算平均分数
            total_recall = sum(r["result"]["recall"] for r in all_results)
            count = len(all_results)
            
            avg_recall = total_recall / count
            
            print(f"Model: {model_name}")
            print(f"Successfully evaluated questions: {count}")
            print(f"Average coverage: {avg_recall:.4f}")
            print(f"{'='*50}")
            # 保存到txt文件
            with open(os.path.join(model_result_dir, f"{model_name}_evaluation_results.txt"), "w") as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Average coverage: {avg_recall:.4f}\n")
                f.write(f"{'='*50}\n")
        
        return model_result_dir
    else:
        print(f"Failed to save results for {model_name}")
        return None


def main():
    parser = argparse.ArgumentParser(description='并行评估AI模型响应 (增强版标准评估)')

    parser.add_argument('--model_file', type=str, required=True,
                        help='模型响应文件路径')
    parser.add_argument('--rubrics_file', type=str, default='data/eval_data/rubric.json',
                        help='标准JSON文件路径')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='评估结果输出目录')
    parser.add_argument('--judge_model', type=str, default='your-model-id',
                       help='Judge model name for the OpenAI-compatible endpoint.')
    parser.add_argument('--judge_base_url', type=str, default=os.getenv("JUDGE_BASE_URL", ""),
                        help='Optional OpenAI-compatible base URL for the judge model, e.g. http://127.0.0.1:8000/v1')
    parser.add_argument('--judge_api_key', type=str, default=os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
                        help='Optional API key for the OpenAI-compatible judge endpoint. Defaults to JUDGE_API_KEY / OPENAI_API_KEY / EMPTY.')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='最大并行工作线程数 (默认: 4)')
    parser.add_argument('--rerun_failed_only', action='store_true',
                        help='仅重跑已有结果文件中判定为失败的问题，其它问题沿用旧结果')
    parser.add_argument('--limit', type=int, default=0,
                        help='只评估前 N 个问题，0=全部')
    parser.add_argument('--batch_size', '--batch-size', type=int, default=1,
                        help='每次 LLM 调用评估的 rubric 数量 (默认: 1)')
    parser.add_argument('--prompt_style', '--prompt-style', type=str, default='standard',
                        choices=['standard', 'strict', 'semantic', 'evidence_first'],
                        help='Prompt variant for rubric verification (默认: standard)')

    args = parser.parse_args()

    print("Loading evaluation rubrics...")
    rubrics_dict = load_rubrics(args.rubrics_file)
    if not rubrics_dict:
        print("Failed to load rubrics, exiting program.")
        return

    if not create_output_directory(args.result_dir):
        print("Failed to create output directory, exiting program.")
        return

    if args.judge_base_url:
        os.environ["DETECTOR_API_BACKEND"] = "openai"

    safe_judge_model = _safe_path_component(args.judge_model)
    cache_dir = os.path.join(rubricbench_root, "cache", safe_judge_model)
    print(
        f"Creating APIModel with model_marker={args.judge_model}, "
        f"base_url={args.judge_base_url or '<not set>'}"
    )
    api_model = APIModel(
        cache_url=cache_dir,
        base_url=args.judge_base_url,
        model_name=args.judge_model,
        api_key=args.judge_api_key,
    )

    RubricEvaluator.reset_global_stats()

    try:
        result_file = evaluate_single_model_parallel(
            model_file=args.model_file,
            rubrics_dict=rubrics_dict,
            result_dir=args.result_dir,
            api_model=api_model,
            max_workers=args.max_workers,
            rerun_failed_only=args.rerun_failed_only,
            limit=args.limit,
            rubric_batch_size=args.batch_size,
            prompt_style=args.prompt_style,
        )
    finally:
        api_model.save_cache()

    print(f"\n{'='*60}")
    print("Final Evaluation Summary")
    print(f"{'='*60}")

    if result_file:
        print(f"Evaluation completed successfully!")
        print(f"Results saved to: {result_file}")
        print(f"Using parallel workers: {args.max_workers}")
    else:
        print(f"Evaluation failed!")
        print(f"Please check the error logs above for detailed information.")

    token_stats = RubricEvaluator.get_token_stats()
    global_stats = RubricEvaluator.get_global_stats()
    print(f"\n{'='*60}")
    print("Token Usage Summary")
    print(f"{'='*60}")
    print(f"Total API requests:        {global_stats['total_requests']}")
    print(f"Successful requests:       {global_stats['successful_requests']}")
    print(f"Failed requests:           {global_stats['failed_requests']}")
    print(f"Local cache hits:          {token_stats['local_cache_hits']}")
    print(f"Requests w/ usage fields:  {token_stats['requests_with_usage']}")
    print(f"Requests w/o usage fields: {token_stats['requests_without_usage']}")
    print("")
    print("Server-reported tokens")
    print(f"Input tokens:              {token_stats['actual_input_tokens']}")
    print(f"Cache input tokens:        {token_stats['actual_cached_input_tokens']}")
    print(f"Uncached input tokens:     {token_stats['actual_uncached_input_tokens']}")
    print(f"Output tokens:             {token_stats['actual_output_tokens']}")
    print("")
    print("Effective tokens (server-first, chars fallback)")
    print(f"Input tokens:              {token_stats['effective_input_tokens']}")
    print(f"Cache input tokens:        {token_stats['effective_cached_input_tokens']}")
    print(f"Uncached input tokens:     {token_stats['effective_uncached_input_tokens']}")
    print(f"Output tokens:             {token_stats['effective_output_tokens']}")
    print("")
    print("Estimated tokens (fallback)")
    print(f"Total input chars:         {token_stats['total_input_chars']}")
    print(f"Total output chars:        {token_stats['total_output_chars']}")
    print(f"Estimated input tokens:    {token_stats['estimated_input_tokens']}")
    print(f"Estimated output tokens:   {token_stats['estimated_output_tokens']}")
    print(f"Estimated cache tokens(LCP): {token_stats['estimated_cache_input_tokens_from_lcp']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
