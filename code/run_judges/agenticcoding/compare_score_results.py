#!/usr/bin/env python3
"""Compare rubric results across two score JSON files and build review-friendly diffs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json_file(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_file(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_check(category_name, check):
    return {
        "category": check.get("category", category_name),
        "check_id": check["check_id"],
        "result": check.get("result"),
        "reasoning": check.get("reasoning"),
    }


def normalize_user_query(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [part.strip() for part in value if isinstance(part, str) and part.strip()]
        return "\n\n".join(parts) if parts else None
    return str(value)


def extract_message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part for part in parts if part.strip())
    return ""


def build_preview(text, head_chars, tail_chars):
    if not text:
        return None
    if len(text) <= head_chars + tail_chars:
        return {
            "head": text,
            "tail": "",
            "omitted": False,
            "total_chars": len(text),
        }
    return {
        "head": text[:head_chars],
        "tail": text[-tail_chars:],
        "omitted": True,
        "total_chars": len(text),
    }


def index_results(data):
    indexed = {}
    for item in data["results"]:
        instance_id = item["instance_id"]
        checks = {}
        for category_name, category_value in item.items():
            if isinstance(category_value, dict) and "checks" in category_value:
                for raw_check in category_value["checks"]:
                    check = normalize_check(category_name, raw_check)
                    checks[(check["category"], check["check_id"])] = check
        indexed[instance_id] = checks
    return indexed


def extract_checklist_context(checklist):
    category_map = {}
    check_map = {}

    if not isinstance(checklist, dict):
        return {"categories": category_map, "checks": check_map}

    for category_name, category_payload in checklist.items():
        if not isinstance(category_payload, dict):
            continue

        category_map[category_name] = {
            "category": category_name,
            "description": category_payload.get("description"),
        }

        for check in category_payload.get("checks", []):
            check_id = check.get("check_id")
            if not check_id:
                continue
            check_map[(category_name, check_id)] = {
                "category": category_name,
                "category_description": category_payload.get("description"),
                "check_id": check_id,
                "check_description": check.get("description"),
                "check_type": check.get("check_type"),
            }

    return {"categories": category_map, "checks": check_map}


def load_benchmark_context(path):
    if not path or not Path(path).exists():
        return {}

    contexts = {}
    for record in load_jsonl_file(path):
        instance_id = record.get("instance_id")
        if not instance_id:
            continue
        checklist_context = extract_checklist_context(record.get("checklist"))
        contexts[instance_id] = {
            "user_query": normalize_user_query(record.get("user_query")),
            "benchmark_category": record.get("category"),
            "expected_skill": record.get("expected_skill"),
            "rubric_categories": checklist_context["categories"],
            "rubric_checks": checklist_context["checks"],
        }
    return contexts


def extract_trajectory_context(record):
    assistant_messages = []
    user_messages = []

    for message in record.get("messages", []):
        role = message.get("role")
        text = extract_message_text(message.get("content"))
        if not text.strip():
            continue
        if role == "assistant":
            assistant_messages.append(text.strip())
        elif role == "user":
            user_messages.append(text.strip())

    response_text = "\n\n".join(assistant_messages).strip() if assistant_messages else None
    final_response = assistant_messages[-1] if assistant_messages else None
    fallback_user_query = user_messages[0] if user_messages else None

    return {
        "assistant_message_count": len(assistant_messages),
        "response_text": response_text,
        "final_response_text": final_response,
        "fallback_user_query": fallback_user_query,
    }


def load_trajectory_context(path):
    if not path or not Path(path).exists():
        return {}

    contexts = {}
    for record in load_jsonl_file(path):
        meta = record.get("meta", {})
        instance_id = meta.get("session_id") or record.get("instance_id")
        if not instance_id:
            continue
        candidate = extract_trajectory_context(record)
        existing = contexts.get(instance_id)
        if existing is None:
            contexts[instance_id] = candidate
            continue

        existing_len = len(existing.get("response_text") or "")
        candidate_len = len(candidate.get("response_text") or "")
        if candidate_len > existing_len:
            contexts[instance_id] = candidate
    return contexts


def resolve_sidecar_path(preferred, left_path, candidates):
    if preferred:
        candidate = Path(preferred)
        return candidate if candidate.exists() else None

    base_dir = Path(left_path).resolve().parent
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return None


def build_instance_records(
    mismatches,
    benchmark_contexts,
    trajectory_contexts,
    response_mode,
    head_chars,
    tail_chars,
):
    mismatches_by_instance = defaultdict(list)
    for mismatch in mismatches:
        mismatches_by_instance[mismatch["instance_id"]].append(mismatch)

    instance_records = []
    for instance_id, rows in sorted(
        mismatches_by_instance.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        benchmark_context = benchmark_contexts.get(instance_id, {})
        trajectory_context = trajectory_contexts.get(instance_id, {})

        user_query = benchmark_context.get("user_query") or trajectory_context.get("fallback_user_query")
        response_text = trajectory_context.get("response_text")
        final_response_text = trajectory_context.get("final_response_text")
        category_counts = Counter(row["category"] for row in rows)

        instance_record = {
            "instance_id": instance_id,
            "mismatch_count": len(rows),
            "mismatch_counts_by_category": dict(
                sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "benchmark_category": benchmark_context.get("benchmark_category"),
            "expected_skill": benchmark_context.get("expected_skill"),
            "user_query": user_query,
            "rubric_categories": benchmark_context.get("rubric_categories"),
            "assistant_message_count": trajectory_context.get("assistant_message_count", 0),
            "mismatches": rows,
        }

        if response_text:
            instance_record["response_preview"] = build_preview(response_text, head_chars, tail_chars)
            if final_response_text and final_response_text != response_text:
                instance_record["final_response_preview"] = build_preview(
                    final_response_text, head_chars, tail_chars
                )
            if response_mode == "full":
                instance_record["response_full"] = response_text

        instance_records.append(instance_record)

    return instance_records


def compare_results(
    left,
    right,
    *,
    left_name,
    right_name,
    left_path,
    right_path,
    benchmark_path,
    trajectory_path,
    response_mode,
    head_chars,
    tail_chars,
):
    left_index = index_results(left)
    right_index = index_results(right)
    benchmark_contexts = load_benchmark_context(benchmark_path)
    trajectory_contexts = load_trajectory_context(trajectory_path)

    all_instance_ids = sorted(set(left_index) | set(right_index))
    missing_instances = []
    mismatches = []
    compared_check_count = 0

    for instance_id in all_instance_ids:
        benchmark_context = benchmark_contexts.get(instance_id, {})
        rubric_checks = benchmark_context.get("rubric_checks", {})

        if instance_id not in left_index or instance_id not in right_index:
            missing_instances.append(
                {
                    "instance_id": instance_id,
                    f"in_{left_name}": instance_id in left_index,
                    f"in_{right_name}": instance_id in right_index,
                }
            )
            continue

        check_keys = sorted(set(left_index[instance_id]) | set(right_index[instance_id]))
        compared_check_count += len(check_keys)

        for category, check_id in check_keys:
            left_check = left_index[instance_id].get((category, check_id))
            right_check = right_index[instance_id].get((category, check_id))
            left_result = left_check["result"] if left_check else "<MISSING>"
            right_result = right_check["result"] if right_check else "<MISSING>"

            if left_result == right_result:
                continue

            rubric_context = rubric_checks.get((category, check_id), {})
            mismatches.append(
                {
                    "instance_id": instance_id,
                    "category": category,
                    "category_description": rubric_context.get("category_description"),
                    "check_id": check_id,
                    "check_description": rubric_context.get("check_description"),
                    "check_type": rubric_context.get("check_type"),
                    f"{left_name}_result": left_result,
                    f"{right_name}_result": right_result,
                    f"{left_name}_reasoning": left_check["reasoning"] if left_check else None,
                    f"{right_name}_reasoning": right_check["reasoning"] if right_check else None,
                }
            )

    mismatch_counts_by_instance = Counter(item["instance_id"] for item in mismatches)
    mismatch_counts_by_category = Counter(item["category"] for item in mismatches)
    mismatch_count = len(mismatches)
    matched_check_count = compared_check_count - mismatch_count
    mismatch_ratio = mismatch_count / compared_check_count if compared_check_count else 0.0

    instance_records = build_instance_records(
        mismatches,
        benchmark_contexts,
        trajectory_contexts,
        response_mode,
        head_chars,
        tail_chars,
    )

    instances_with_user_query = sum(1 for item in instance_records if item.get("user_query"))
    instances_with_response = sum(1 for item in instance_records if item.get("response_preview"))

    return {
        "source_files": {
            left_name: str(Path(left_path).resolve()),
            right_name: str(Path(right_path).resolve()),
            "benchmark_context": str(Path(benchmark_path).resolve()) if benchmark_path else None,
            "trajectory_context": str(Path(trajectory_path).resolve()) if trajectory_path else None,
        },
        "summary": {
            f"instance_count_{left_name}": len(left_index),
            f"instance_count_{right_name}": len(right_index),
            "missing_instance_count": len(missing_instances),
            "compared_instance_count": len(all_instance_ids) - len(missing_instances),
            "compared_check_count": compared_check_count,
            "matched_check_count": matched_check_count,
            "mismatch_count": mismatch_count,
            "mismatch_ratio": round(mismatch_ratio, 6),
            "mismatch_percentage": round(mismatch_ratio * 100, 2),
            "instances_with_mismatches": len(mismatch_counts_by_instance),
            "instances_with_user_query": instances_with_user_query,
            "instances_with_response_preview": instances_with_response,
            "response_mode": response_mode,
        },
        "missing_instances": missing_instances,
        "mismatch_counts_by_instance": dict(
            sorted(mismatch_counts_by_instance.items(), key=lambda item: (-item[1], item[0]))
        ),
        "mismatch_counts_by_category": dict(
            sorted(mismatch_counts_by_category.items(), key=lambda item: (-item[1], item[0]))
        ),
        "instances": instance_records,
        "mismatches": mismatches,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare rubric results in two score JSON files.")
    parser.add_argument("left", help="First score JSON file")
    parser.add_argument("right", help="Second score JSON file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file path. Defaults to <left_stem>_vs_<right_stem>.diff.json in the current directory.",
    )
    parser.add_argument(
        "--left-name",
        default="left",
        help="Short label for the first file, used in output field names.",
    )
    parser.add_argument(
        "--right-name",
        default="right",
        help="Short label for the second file, used in output field names.",
    )
    parser.add_argument(
        "--benchmark-file",
        help="Optional JSONL benchmark file used to enrich the diff with user_query.",
    )
    parser.add_argument(
        "--trajectory-file",
        help="Optional trajectory JSONL file used to enrich the diff with assistant response previews.",
    )
    parser.add_argument(
        "--response-mode",
        choices=("none", "preview", "full"),
        default="preview",
        help="How much assistant response text to include for each mismatched instance.",
    )
    parser.add_argument(
        "--preview-head-chars",
        type=int,
        default=800,
        help="How many characters from the start of the response preview to keep.",
    )
    parser.add_argument(
        "--preview-tail-chars",
        type=int,
        default=400,
        help="How many characters from the end of the response preview to keep.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Optional Markdown report path for a human-readable diff view.",
    )
    return parser.parse_args()


def default_output_path(left, right):
    left_stem = Path(left).stem
    right_stem = Path(right).stem
    return Path.cwd() / f"{left_stem}_vs_{right_stem}.diff.json"


def default_markdown_path(json_path):
    return Path(json_path).with_suffix(".md")


def format_response_preview(preview):
    if not preview:
        return "_No response preview available._"

    parts = []
    if preview.get("head"):
        parts.append(preview["head"])
    if preview.get("omitted"):
        parts.append("\n...\n")
    if preview.get("tail"):
        parts.append(preview["tail"])

    text = "".join(parts).strip()
    if not text:
        return "_No response preview available._"

    return f"```text\n{text}\n```"


def render_markdown_report(result, left_name, right_name):
    summary = result["summary"]
    lines = [
        "# Rubric Diff Report",
        "",
        f"- Compared checks: {summary['compared_check_count']}",
        f"- Mismatches: {summary['mismatch_count']}",
        f"- Rubric mismatch percentage: {summary['mismatch_percentage']:.2f}%",
        f"- Instances with mismatches: {summary['instances_with_mismatches']}",
        "",
    ]

    for instance in result["instances"]:
        lines.extend(
            [
                f"## {instance['instance_id']}",
                "",
                "### User Query",
                "",
                instance.get("user_query") or "_No user query available._",
                "",
                "### Response Preview",
                "",
                format_response_preview(instance.get("response_preview")),
                "",
            ]
        )

        for idx, mismatch in enumerate(instance["mismatches"], start=1):
            lines.extend(
                [
                    f"### Mismatch {idx}",
                    "",
                    "**Check Description**",
                    "",
                    mismatch.get("check_description") or "_No check description available._",
                    "",
                    f"**{left_name} Result**",
                    "",
                    str(mismatch.get(f"{left_name}_result")),
                    "",
                    f"**{left_name} Reasoning**",
                    "",
                    mismatch.get(f"{left_name}_reasoning") or "_No reasoning provided._",
                    "",
                    f"**{right_name} Result**",
                    "",
                    str(mismatch.get(f"{right_name}_result")),
                    "",
                    f"**{right_name} Reasoning**",
                    "",
                    mismatch.get(f"{right_name}_reasoning") or "_No reasoning provided._",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()

    benchmark_path = resolve_sidecar_path(args.benchmark_file, args.left, ["OctoBench.jsonl"])
    trajectory_path = resolve_sidecar_path(
        args.trajectory_file,
        args.left,
        ["merged_trajectories.jsonl", "merged_trajectories-old.jsonl"],
    )

    left_data = load_json_file(args.left)
    right_data = load_json_file(args.right)
    result = compare_results(
        left_data,
        right_data,
        left_name=args.left_name,
        right_name=args.right_name,
        left_path=args.left,
        right_path=args.right,
        benchmark_path=benchmark_path,
        trajectory_path=trajectory_path,
        response_mode=args.response_mode,
        head_chars=args.preview_head_chars,
        tail_chars=args.preview_tail_chars,
    )

    output_path = Path(args.output) if args.output else default_output_path(args.left, args.right)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    markdown_path = Path(args.markdown_output) if args.markdown_output else default_markdown_path(output_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with markdown_path.open("w", encoding="utf-8") as f:
        f.write(render_markdown_report(result, args.left_name, args.right_name))

    summary = result["summary"]
    print(f"Saved diff JSON to: {output_path}")
    print(f"Saved diff Markdown to: {markdown_path}")
    print(f"Compared checks: {summary['compared_check_count']}")
    print(f"Mismatches: {summary['mismatch_count']}")
    print(f"Rubric mismatch ratio: {summary['mismatch_ratio']:.6f}")
    print(f"Rubric mismatch percentage: {summary['mismatch_percentage']:.2f}%")
    print(f"Instances with user_query: {summary['instances_with_user_query']}")
    print(f"Instances with response preview: {summary['instances_with_response_preview']}")


if __name__ == "__main__":
    main()
