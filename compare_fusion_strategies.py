"""Compare fixed-alpha and confidence-bucketed fusion on saved dev scores."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from calibrate_investigator import load_dev_doc_ids, load_score_csv


def metrics(gold, pred):
    accuracy = float(np.mean(gold == pred))
    f1s = []
    for label in (0, 1):
        tp = np.sum((gold == label) & (pred == label))
        fp = np.sum((gold != label) & (pred == label))
        fn = np.sum((gold == label) & (pred != label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return accuracy, float(np.mean(f1s))


def selection_accuracy(records, scores):
    by_doc = {}
    for record, score in zip(records, scores):
        by_doc.setdefault(record.doc_id, []).append((record.text_key, float(score)))
    correct = total = 0
    for items in by_doc.values():
        if not any(key == "text1" for key, _ in items):
            continue
        total += 1
        correct += max(items, key=lambda item: item[1])[0] == "text1"
    return correct / total if total else 0.0, correct, total


def evaluate_schedule(records, alphas, thresholds=range(0, 101)):
    semantic = np.array([r.semantic_score for r in records])
    qwen_truth = np.array([r.qwen_truth_score for r in records])
    qwen_conf = np.array([r.qwen_confidence for r in records])
    gold = np.array([1 if r.gold_label == "Real" else 0 for r in records])
    bucket = np.select(
        [qwen_conf >= 85, qwen_conf >= 70, qwen_conf >= 55],
        [0, 1, 2],
        default=3,
    )
    row_alpha = np.array(alphas)[bucket]
    scores = row_alpha * semantic + (1.0 - row_alpha) * qwen_truth
    select_acc, select_correct, select_total = selection_accuracy(records, scores)

    best = None
    for threshold in thresholds:
        accuracy, macro_f1 = metrics(gold, (scores >= threshold).astype(int))
        objective = 0.5 * macro_f1 + 0.5 * select_acc
        candidate = {
            "alphas": [round(float(x), 2) for x in alphas],
            "threshold": float(threshold),
            "objective": objective,
            "text_accuracy": accuracy,
            "macro_f1": macro_f1,
            "real_selection_accuracy": select_acc,
            "real_selection_correct": select_correct,
            "real_selection_docs": select_total,
        }
        if best is None or candidate["objective"] > best["objective"]:
            best = candidate
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--json", default="hallu/extr-hallu-final.json")
    parser.add_argument("--dev-size", type=int, default=50)
    parser.add_argument("--out", default="res/fusion_strategy_comparison.json")
    args = parser.parse_args()

    dev_ids = load_dev_doc_ids(Path(args.json), args.dev_size)
    records = load_score_csv(Path(args.scores_csv), dev_ids)
    grid = [i / 20 for i in range(21)]

    original = evaluate_schedule(records, (0.35, 0.40, 0.45, 0.60))
    fixed = max(
        (evaluate_schedule(records, (a, a, a, a)) for a in grid),
        key=lambda result: result["objective"],
    )
    # Preserve the original design principle: as Qwen confidence falls, the
    # semantic-path weight may stay equal or increase.
    schedules = itertools.combinations_with_replacement(grid, 4)
    staircase = max(
        (evaluate_schedule(records, schedule) for schedule in schedules),
        key=lambda result: result["objective"],
    )

    payload = {
        "record_count": len(records),
        "bucket_definition": [">=85", "70-84", "55-69", "<55"],
        "alpha_grid": {"min": 0.0, "max": 1.0, "step": 0.05},
        "threshold_grid": {"min": 0, "max": 100, "step": 1},
        "original_staircase": original,
        "best_fixed": fixed,
        "best_monotone_staircase": staircase,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
