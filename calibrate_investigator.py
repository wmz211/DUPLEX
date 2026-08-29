"""Calibrate Investigator score fusion on a development split.

This script does not call any model APIs. It reads the structured Investigator
CSV emitted by HD (preferred), uses the dataset's known text-key labels as gold, and
grid-searches alpha and threshold for:

    score = alpha * semantic_score + (1 - alpha) * qwen_truth_score

By default, the first 50 documents in hallu/extr-hallu-final.json are used as
the development set.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TEXT_GOLD = {
    "text1": "Real",
    "text4": "Factuality",
    "text5": "Factuality",
    "text6": "Factuality",
}


@dataclass
class ScoreRecord:
    doc_id: str
    text_key: str
    semantic_score: float
    qwen_label: str
    qwen_confidence: float
    old_final_score: float | None = None
    old_label: str | None = None

    @property
    def gold_label(self) -> str:
        return TEXT_GOLD[self.text_key]

    @property
    def qwen_truth_score(self) -> float:
        if self.qwen_label == "Real":
            return self.qwen_confidence
        return 100.0 - self.qwen_confidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="hallu/extr-hallu-final.json")
    parser.add_argument("--log", default="log/phase2c_hd.txt")
    parser.add_argument(
        "--scores-csv",
        default="res/hd_results_calibration.csv",
        help="Structured HD calibration CSV. Falls back to --log if absent.",
    )
    parser.add_argument("--dev-size", type=int, default=50)
    parser.add_argument("--out-csv", default="res/investigator_dev_scores.csv")
    parser.add_argument("--out-json", default="res/investigator_calibration.json")
    args = parser.parse_args()

    dev_doc_ids = load_dev_doc_ids(Path(args.json), args.dev_size)
    scores_csv = Path(args.scores_csv)
    if scores_csv.exists():
        records = load_score_csv(scores_csv, dev_doc_ids)
        source = str(scores_csv)
    else:
        records = parse_log(Path(args.log), dev_doc_ids)
        source = str(args.log)
    if not records:
        raise SystemExit(
            "No usable Investigator score records found. Run HD first to create "
            "<hd-output>_calibration.csv, or provide --scores-csv."
        )

    best = grid_search(records)
    write_records_csv(Path(args.out_csv), records)
    write_summary_json(Path(args.out_json), records, best, args, source)
    print_summary(records, best, args, source)


def load_dev_doc_ids(json_path: Path, dev_size: int) -> set[str]:
    with json_path.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    if not isinstance(docs, list):
        docs = [docs]
    return {str(doc.get("id", "")) for doc in docs[:dev_size]}


def load_score_csv(path: Path, dev_doc_ids: set[str]) -> list[ScoreRecord]:
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            doc_id = row.get("doc_id", "").strip()
            text_key = row.get("text_key", "").strip()
            if doc_id not in dev_doc_ids or text_key not in TEXT_GOLD:
                continue
            try:
                records.append(
                    ScoreRecord(
                        doc_id=doc_id,
                        text_key=text_key,
                        semantic_score=float(row["semantic_score"]),
                        qwen_label=row["qwen_label"].strip(),
                        qwen_confidence=float(row["qwen_confidence"]),
                        old_final_score=_optional_float(row.get("old_final_score")),
                        old_label=(row.get("old_label") or "").strip() or None,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return _dedupe(records)


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _dedupe(records: list[ScoreRecord]) -> list[ScoreRecord]:
    return list({(r.doc_id, r.text_key): r for r in records}.values())


def parse_log(log_path: Path, dev_doc_ids: set[str]) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    current_doc = ""
    current_text = ""
    pending_semantic: dict[tuple[str, str], float] = {}
    latest_key: tuple[str, str] | None = None

    doc_re = re.compile(r"Processing HD group #\d+:\s+(\S+)")
    investigator_re = re.compile(r"Investigator 检测 (text[1-6])")
    semantic_re = re.compile(r"路径1（语义）:\s*([0-9.]+)%")
    qwen_re = re.compile(r"路径2（Qwen）:\s*(Real|Factuality)\s*(?:置信度)?\s*([0-9.]+)%")
    fusion_re = re.compile(r"融合(?:公式)?:.*=\s*([0-9.]+)%")
    final_re = re.compile(r"最终判断:\s*(IsRW|IsFacHal)")

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = strip_timestamp(raw_line)

        doc_match = doc_re.search(line)
        if doc_match:
            current_doc = doc_match.group(1)
            current_text = ""
            latest_key = None
            continue

        investigator_match = investigator_re.search(line)
        if investigator_match:
            current_text = investigator_match.group(1)
            latest_key = (current_doc, current_text)
            continue

        if current_doc not in dev_doc_ids or current_text not in TEXT_GOLD:
            continue

        semantic_match = semantic_re.search(line)
        if semantic_match:
            key = (current_doc, current_text)
            pending_semantic[key] = float(semantic_match.group(1))
            latest_key = key
            continue

        qwen_match = qwen_re.search(line)
        if qwen_match and latest_key in pending_semantic:
            doc_id, text_key = latest_key
            if text_key in TEXT_GOLD:
                records.append(
                    ScoreRecord(
                        doc_id=doc_id,
                        text_key=text_key,
                        semantic_score=pending_semantic[latest_key],
                        qwen_label=qwen_match.group(1),
                        qwen_confidence=float(qwen_match.group(2)),
                    )
                )
            continue

        fusion_match = fusion_re.search(line)
        if fusion_match and records:
            records[-1].old_final_score = float(fusion_match.group(1))
            continue

        final_match = final_re.search(line)
        if final_match and records:
            records[-1].old_label = "Real" if final_match.group(1) == "IsRW" else "Factuality"

    return _dedupe(records)


def strip_timestamp(line: str) -> str:
    if "]" in line and line.startswith("["):
        return line.split("]", 1)[1].strip()
    return line.strip()


def grid_search(records: list[ScoreRecord]) -> dict:
    best = None
    for alpha_i in range(0, 21):
        alpha = alpha_i / 20.0
        for threshold in range(35, 76):
            metrics = evaluate(records, alpha, float(threshold))
            objective = 0.5 * metrics["macro_f1"] + 0.5 * metrics["real_selection_accuracy"]
            candidate = {
                "alpha": alpha,
                "threshold": float(threshold),
                "objective": objective,
                **metrics,
            }
            if best is None or candidate["objective"] > best["objective"]:
                best = candidate
    return best


def evaluate(records: list[ScoreRecord], alpha: float, threshold: float) -> dict:
    y_true = []
    y_pred = []
    by_doc = defaultdict(list)

    for record in records:
        score = fuse(record, alpha)
        pred = "Real" if score >= threshold else "Factuality"
        y_true.append(record.gold_label)
        y_pred.append(pred)
        by_doc[record.doc_id].append((record.text_key, score))

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_f1 = compute_macro_f1(y_true, y_pred)
    real_ok = 0
    real_total = 0
    for doc_id, items in by_doc.items():
        keys = {key for key, _ in items}
        if "text1" not in keys:
            continue
        chosen_key = max(items, key=lambda x: x[1])[0]
        real_total += 1
        if chosen_key == "text1":
            real_ok += 1
    real_selection_accuracy = real_ok / real_total if real_total else 0.0

    return {
        "text_accuracy": accuracy,
        "macro_f1": macro_f1,
        "real_selection_accuracy": real_selection_accuracy,
        "real_selection_docs": real_total,
    }


def fuse(record: ScoreRecord, alpha: float) -> float:
    return alpha * record.semantic_score + (1.0 - alpha) * record.qwen_truth_score


def compute_macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    scores = []
    for label in ("Real", "Factuality"):
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)


def write_records_csv(path: Path, records: list[ScoreRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "text_key",
                "gold_label",
                "semantic_score",
                "qwen_label",
                "qwen_confidence",
                "qwen_truth_score",
                "old_final_score",
                "old_label",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "doc_id": r.doc_id,
                    "text_key": r.text_key,
                    "gold_label": r.gold_label,
                    "semantic_score": f"{r.semantic_score:.3f}",
                    "qwen_label": r.qwen_label,
                    "qwen_confidence": f"{r.qwen_confidence:.3f}",
                    "qwen_truth_score": f"{r.qwen_truth_score:.3f}",
                    "old_final_score": "" if r.old_final_score is None else f"{r.old_final_score:.3f}",
                    "old_label": r.old_label or "",
                }
            )


def write_summary_json(path: Path, records: list[ScoreRecord], best: dict, args, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_records = [r for r in records if r.old_label]
    old_accuracy = None
    old_macro_f1 = None
    if old_records:
        old_accuracy = sum(r.gold_label == r.old_label for r in old_records) / len(old_records)
        old_macro_f1 = compute_macro_f1(
            [r.gold_label for r in old_records],
            [r.old_label or "" for r in old_records],
        )
    payload = {
        "source_scores": source,
        "source_json": args.json,
        "dev_size": args.dev_size,
        "record_count": len(records),
        "doc_count": len({r.doc_id for r in records}),
        "best": best,
        "old_accuracy_on_parsed_records": old_accuracy,
        "old_macro_f1_on_parsed_records": old_macro_f1,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(records: list[ScoreRecord], best: dict, args, source: str) -> None:
    print(f"Parsed records: {len(records)}")
    print(f"Covered docs: {len({r.doc_id for r in records})}")
    print(f"Source scores: {source}")
    print(
        "Best calibration: "
        f"alpha={best['alpha']:.2f}, threshold={best['threshold']:.0f}, "
        f"objective={best['objective']:.4f}"
    )
    print(
        "Metrics: "
        f"text_acc={best['text_accuracy']:.4f}, "
        f"macro_f1={best['macro_f1']:.4f}, "
        f"real_select={best['real_selection_accuracy']:.4f} "
        f"({best['real_selection_docs']} docs)"
    )
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
