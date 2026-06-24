#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

VALID_AC_LABELS = {"success", "fail"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split())


def majority_bool(values: list[bool]) -> bool | None:
    if not values:
        return None
    return sum(1 for value in values if value) >= 2


def majority_ac(values: list[str]) -> str | None:
    values = [value for value in values if value in VALID_AC_LABELS]
    if not values:
        return None
    return "success" if values.count("success") >= 2 else "fail"


def load_dr_gold(gold_path: Path) -> list[dict[str, Any]]:
    rows = []
    dataset_weights: dict[str, list[float]] = {}
    if gold_path.name.endswith("_labels.json"):
        dataset_path = gold_path.with_name(gold_path.name.replace("_labels.json", "_dataset.json"))
        if dataset_path.exists():
            dataset = load_json(dataset_path)
            if isinstance(dataset, list):
                for row in dataset:
                    qid = str(row["id"])
                    dataset_weights[qid] = [point.get("weight", 1) for point in row.get("rubric", [])]
    for row in load_json(gold_path):
        qid = str(row["id"])
        coverage = row.get("result", {}).get("coverage_results", [])
        weights = dataset_weights.get(qid, [])
        for idx, point in enumerate(coverage):
            rows.append({
                "question_id": qid,
                "point_index": idx,
                "point": point.get("point", ""),
                "weight": point.get("weight", weights[idx] if idx < len(weights) else 1),
            })
    return rows


def build_dr_pred_map(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    output: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(data, list):
        return output
    for row in data:
        qid = str(row.get("id", row.get("question_id", "")))
        for item in row.get("result", {}).get("coverage_results", []):
            if "point" in item:
                output[(qid, normalize_text(item["point"]))] = item
    return output


def aggregate_deepresearch(gold_path: Path, voter_files: list[Path], output_path: Path, voter_labels: list[str]) -> None:
    gold_rows = load_dr_gold(gold_path)
    voter_maps = [build_dr_pred_map(path) for path in voter_files]
    by_question: dict[str, list[dict[str, Any]]] = {}

    for row in gold_rows:
        key = (row["question_id"], normalize_text(row["point"]))
        votes = []
        voter_details = []
        for label, path, pred_map in zip(voter_labels, voter_files, voter_maps):
            item = pred_map.get(key)
            covered = item.get("covered") if item else None
            if isinstance(covered, bool):
                votes.append(covered)
            voter_details.append({
                "voter": label,
                "file": str(path),
                "covered": covered if isinstance(covered, bool) else None,
                "justification": item.get("justification", "missing") if item else "missing",
            })
        covered = majority_bool(votes)
        if covered is None:
            covered = False
        by_question.setdefault(row["question_id"], []).append({
            "point": row["point"],
            "weight": row["weight"],
            "covered": covered,
            "justification": "majority vote: " + ", ".join(
                f"{detail['voter']}={detail['covered']}" for detail in voter_details
            ),
            "voter_details": voter_details,
        })

    results = []
    for qid, coverage_results in by_question.items():
        total_weight = sum(float(item.get("weight", 1)) for item in coverage_results)
        covered_weight = sum(float(item.get("weight", 1)) for item in coverage_results if item.get("covered"))
        results.append({
            "id": qid,
            "model": "majority_vote",
            "metric": "Evaluation",
            "evaluator": "majority_vote",
            "result": {
                "m_weighted": covered_weight,
                "total_weight": total_weight,
                "recall": covered_weight / total_weight if total_weight else 0.0,
                "coverage_results": coverage_results,
            },
        })
    results.sort(key=lambda item: item["id"])
    write_json(output_path, results)


def load_ac_gold(gold_path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(gold_path)
    gold: dict[str, dict[str, Any]] = {}
    for item in data.get("results", []):
        instance_id = str(item.get("instance_id"))
        if instance_id:
            gold[instance_id] = item
    return gold


def build_ac_pred_map(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    data = load_json(path)
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not isinstance(data, dict):
        return output
    for item in data.get("results", []):
        instance_id = item.get("instance_id")
        if not instance_id:
            continue
        for group_name, group in item.items():
            if not isinstance(group, dict) or "checks" not in group:
                continue
            for check in group.get("checks", []):
                key = (str(instance_id), str(check.get("category", group_name)), str(check.get("check_id")))
                output[key] = check
    return output


def aggregate_agenticcoding(gold_path: Path, voter_files: list[Path], output_path: Path, voter_labels: list[str]) -> None:
    gold = load_ac_gold(gold_path)
    voter_maps = [build_ac_pred_map(path) for path in voter_files]
    results = []

    for instance_id, gold_item in sorted(gold.items()):
        output_item = {"instance_id": instance_id, "eval_result": {}}
        total = 0
        success = 0
        for group_name, group in gold_item.items():
            if group_name in {"instance_id", "eval_result"} or not isinstance(group, dict) or "checks" not in group:
                continue
            output_checks = []
            for check in group.get("checks", []):
                raw_category = str(check.get("category", group_name))
                check_id = str(check.get("check_id"))
                key = (instance_id, raw_category, check_id)
                votes = []
                voter_details = []
                for label, path, pred_map in zip(voter_labels, voter_files, voter_maps):
                    item = pred_map.get(key)
                    result = item.get("result") if item else None
                    if result in VALID_AC_LABELS:
                        votes.append(result)
                    voter_details.append({
                        "voter": label,
                        "file": str(path),
                        "result": result if result in VALID_AC_LABELS else None,
                        "reasoning": item.get("reasoning", "missing") if item else "missing",
                    })
                result = majority_ac(votes) or "fail"
                total += 1
                success += int(result == "success")
                new_check = deepcopy(check)
                new_check["result"] = result
                new_check["reasoning"] = "majority vote: " + ", ".join(
                    f"{detail['voter']}={detail['result']}" for detail in voter_details
                )
                new_check["voter_details"] = voter_details
                output_checks.append(new_check)
            if output_checks:
                output_item[group_name] = {"checks": output_checks}
        reward = round(success / total, 3) if total else 0.0
        output_item["reward"] = reward
        output_item["binary_reward"] = 1 if reward >= 1.0 else 0
        output_item["success"] = bool(total and success == total)
        results.append(output_item)

    payload = {
        "results": results,
        "summary": {
            "total": len(results),
            "success_count": sum(1 for item in results if item.get("success")),
            "avg_reward": round(sum(item.get("reward", 0) for item in results) / len(results), 3) if results else 0,
            "pass_count": sum(1 for item in results if item.get("binary_reward") == 1),
        },
    }
    write_json(output_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate three voter outputs into majority-vote prediction files.")
    parser.add_argument("--domain", choices=["deepresearch", "agenticcoding"], required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--voter-files", nargs=3, type=Path, required=True)
    parser.add_argument("--voter-labels", nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.voter_files:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Voter file missing or empty: {path}")
    if args.domain == "deepresearch":
        aggregate_deepresearch(args.gold.resolve(), [p.resolve() for p in args.voter_files], args.output.resolve(), args.voter_labels)
    else:
        aggregate_agenticcoding(args.gold.resolve(), [p.resolve() for p in args.voter_files], args.output.resolve(), args.voter_labels)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
