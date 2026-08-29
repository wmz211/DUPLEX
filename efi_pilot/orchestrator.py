import logging
import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from efi_pilot.agents.arbiter import ArbiterAgent
from efi_pilot.agents.base import AgentState, GroupState
from efi_pilot.agents.dispatcher import DispatcherAgent
from efi_pilot.agents.investigator import InvestigatorAgent
from efi_pilot.agents.judge import JudgeAgent
from efi_pilot.agents.linguist import LinguistAgent
from efi_pilot.utils.api_clients import make_api_clients, make_qwen_client
from efi_pilot.utils.data_io import (
    append_to_excel,
    filter_docs,
    gather_text_items,
    get_text_from_doc,
    load_hd_results_for_efi,
)
from efi_pilot.utils.embedder import get_shared_embedder
from efi_pilot.utils.logging import ResultCollector, ThreadSafeLogger
from efi_pilot.utils.naming_bridge import efi_to_legacy, to_legacy


def _process_group_hd(
    group_index: int,
    doc: dict,
    api_config: dict,
    embedder,
    logger: ThreadSafeLogger,
    collector: ResultCollector,
    evidence_writer=None,
    calibration_writer=None,
):
    api_clients = make_api_clients(api_config["bocha_key"], api_config["qwen_key"])
    if api_config.get("investigator_calibration"):
        api_clients["investigator_calibration"] = api_config["investigator_calibration"]
    dispatcher = DispatcherAgent(api_clients, logger)
    arbiter = ArbiterAgent(api_clients, logger)
    investigator = InvestigatorAgent(api_clients, logger, embedder)
    judge = JudgeAgent(api_clients, logger)

    doc_id = doc.get("id", "Unknown")
    event = doc.get("events", {}).get("event1", "")

    logger.log(f"\n{'=' * 60}", doc_index=group_index)
    logger.log(f"Processing HD group #{group_index}: {doc_id}", doc_index=group_index)
    logger.log(f"Event: {event}", doc_index=group_index)

    text_items = gather_text_items(doc)
    if not text_items:
        logger.log("No text candidates found.", doc_index=group_index)
        logger.flush_logs(group_index)
        collector.add_result(group_index, {"id": doc_id, "efi_text": ""})
        return

    rng = random.Random(group_index + 42)
    rng.shuffle(text_items)
    logger.log(
        f"Candidate order: {', '.join(k for k, _ in text_items)}",
        doc_index=group_index,
    )

    states = []
    for text_key, text in text_items:
        state = AgentState(
            doc_id=doc_id,
            group_index=group_index,
            text_key=text_key,
            text=text,
            event=event,
        )
        state = dispatcher.run(state)
        state = arbiter.run(state)
        if state.arbiter_output.mapped_label == "IsFaiHal":
            state.hd_label = "FaiHal"
            logger.log(
                f"{text_key}: FaiHal by Arbiter [{state.arbiter_output.raw_label}] ({state.arbiter_output.confidence:.1f}%)",
                doc_index=group_index,
            )
        else:
            logger.log(
                f"{text_key}: passed Arbiter [{state.arbiter_output.raw_label}] ({state.arbiter_output.confidence:.1f}%)",
                doc_index=group_index,
            )
        states.append(state)

    remaining = [s for s in states if s.hd_label is None]
    logger.log(
        f"Arbiter summary: {len(states) - len(remaining)} FaiHal, {len(remaining)} to investigate",
        doc_index=group_index,
    )

    for state in remaining:
        state = investigator.run(state)
        state = judge.run(state)

    efi_text = _select_efi_text(states)

    result = {"id": doc_id}
    for state in states:
        result[state.text_key] = to_legacy(state.hd_label) if state.hd_label else ""
    result["efi_text"] = efi_text

    real_c = sum(1 for s in states if s.hd_label == "Real")
    fai_c = sum(1 for s in states if s.hd_label == "FaiHal")
    fac_c = sum(1 for s in states if s.hd_label == "FacHal")
    logger.log(
        f"HD result for {doc_id}: {real_c} Real, {fai_c} FaiHal, {fac_c} FacHal; efi_text={efi_text}",
        doc_index=group_index,
    )

    if evidence_writer is not None:
        evidence_writer.add_group(group_index, doc_id, event, states, efi_text)
    if calibration_writer is not None:
        calibration_writer.add_group(states)
    logger.flush_logs(group_index)
    collector.add_result(group_index, result)


class EvidenceWriter:
    """Write HD intermediate evidence grouped by document and text key."""

    def __init__(self, output_file: str):
        self.output_file = output_file
        self.lock = threading.Lock()
        self.completed_groups = {}
        self.next_index = 0

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("EFI-Pilot HD Intermediate Evidence\n")
            f.write("=" * 80 + "\n\n")

    def add_group(
        self,
        group_index: int,
        doc_id: str,
        event: str,
        states: list[AgentState],
        efi_text: str,
    ):
        with self.lock:
            self.completed_groups[group_index] = self._format_group(
                group_index, doc_id, event, states, efi_text
            )
            self._try_flush()

    def _try_flush(self):
        with open(self.output_file, "a", encoding="utf-8") as f:
            while self.next_index in self.completed_groups:
                f.write(self.completed_groups.pop(self.next_index))
                self.next_index += 1

    def _format_group(
        self,
        group_index: int,
        doc_id: str,
        event: str,
        states: list[AgentState],
        efi_text: str,
    ) -> str:
        lines = [
            "=" * 80,
            f"doc_id: {doc_id}",
            f"group_index: {group_index}",
            f"event: {event}",
            f"efi_text: {efi_text}",
            "-" * 80,
        ]

        for state in sorted(states, key=lambda s: s.text_key):
            lines.extend(self._format_state(state))
            lines.append("-" * 80)

        lines.append("")
        return "\n".join(lines)

    def _format_state(self, state: AgentState) -> list[str]:
        lines = [
            f"text_key: {state.text_key}",
            f"claim_type: {state.claim_type or ''}",
            f"hd_label: {state.hd_label or ''}",
            "text:",
            state.text,
            "",
            "arbiter_output:",
        ]

        arbiter = state.arbiter_output
        if arbiter is None:
            lines.append("  <missing>")
        else:
            lines.extend(
                [
                    f"  raw_label: {arbiter.raw_label}",
                    f"  mapped_label: {arbiter.mapped_label}",
                    f"  confidence: {arbiter.confidence:.1f}",
                    f"  evidence: {arbiter.evidence}",
                    f"  reasoning: {arbiter.reasoning or ''}",
                ]
            )

        lines.append("")
        lines.append("investigator_output:")
        investigator = state.investigator_output
        if investigator is None:
            lines.append("  <skipped>")
        else:
            lines.extend(
                [
                    f"  label: {investigator.label}",
                    f"  confidence: {investigator.confidence:.1f}",
                    f"  semantic_score: {investigator.semantic_score:.1f}",
                    f"  qwen_label: {investigator.qwen_label}",
                    f"  qwen_confidence: {investigator.qwen_confidence:.1f}",
                    f"  iterations: {investigator.iterations}",
                    f"  stop_reason: {investigator.stop_reason}",
                    f"  evidence: {investigator.evidence}",
                    "  qwen_rounds:",
                ]
            )
            for index, round_data in enumerate(investigator.qwen_rounds, 1):
                lines.extend(
                    [
                        f"    - round: {index}",
                        f"      verdict: {round_data.get('verdict', '')}",
                        f"      confidence: {round_data.get('confidence', '')}",
                        f"      evidence_summary: {round_data.get('evidence_summary', '')}",
                        f"      sources: {round_data.get('sources', [])}",
                        f"      unverified_points: {round_data.get('unverified_points', [])}",
                        f"      next_query: {round_data.get('next_query', '')}",
                        f"      new_evidence: {round_data.get('new_evidence', '')}",
                        f"      retained_evidence: {round_data.get('retained_evidence', '')}",
                        f"      rejected_evidence: {round_data.get('rejected_evidence', '')}",
                    ]
                )

        return lines


class CalibrationScoreWriter:
    """Write one self-contained Investigator row per text.

    Unlike the human-readable log, every row carries its own document and text
    identifiers, so concurrent workers cannot cause records to be attributed to
    the wrong document.
    """

    FIELDNAMES = (
        "doc_id",
        "text_key",
        "semantic_score",
        "qwen_label",
        "qwen_confidence",
        "old_final_score",
        "old_label",
    )

    def __init__(self, output_file: str):
        self.output_file = output_file
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def add_group(self, states: list[AgentState]):
        rows = []
        for state in states:
            output = state.investigator_output
            if output is None:
                continue
            rows.append(
                {
                    "doc_id": state.doc_id,
                    "text_key": state.text_key,
                    "semantic_score": f"{output.semantic_score:.6f}",
                    "qwen_label": output.qwen_label,
                    "qwen_confidence": f"{output.qwen_confidence:.6f}",
                    "old_final_score": f"{output.confidence:.6f}",
                    "old_label": "Real" if output.label == "IsRW" else "Factuality",
                }
            )
        if not rows:
            return
        with self.lock, open(self.output_file, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerows(rows)


def _select_efi_text(states: list[AgentState]) -> str:
    """Select one evidence text for EFI without changing HD labels."""
    real_states = [s for s in states if s.hd_label == "Real"]
    if len(real_states) == 1:
        return real_states[0].text_key
    if len(real_states) > 1:
        return max(real_states, key=_efi_confidence).text_key

    candidates = [s for s in states if s.investigator_output is not None]
    if candidates:
        return max(candidates, key=_efi_confidence).text_key
    return states[0].text_key if states else ""


def _efi_confidence(state: AgentState) -> float:
    if state.investigator_output is not None:
        return state.investigator_output.confidence
    if state.arbiter_output is not None:
        return state.arbiter_output.confidence
    return 0.0


def run_hd(documents: list, config: dict, output_file: str, log_file: str):
    realtime_mode = config.get("realtime_mode", False)
    max_workers = config.get("max_workers", 10)
    evidence_file = config.get("hd_evidence_file") or _default_evidence_file(output_file)
    calibration_file = config.get("hd_calibration_records") or _default_calibration_file(output_file)

    logger = ThreadSafeLogger(log_file, realtime_mode=realtime_mode)
    collector = ResultCollector(output_file, logger)
    evidence_writer = EvidenceWriter(evidence_file)
    calibration_writer = CalibrationScoreWriter(calibration_file)
    logger.log(f"Starting EFI-Pilot HD, max_workers={max_workers}", doc_index=None)
    logger.log(f"HD evidence output: {evidence_file}", doc_index=None)
    logger.log(f"HD calibration records: {calibration_file}", doc_index=None)

    embedder = get_shared_embedder(logger)
    docs_to_process = filter_docs(documents, config.get("start_id"), config.get("end_id"))
    logger.log(f"Documents to process: {len(docs_to_process)}", doc_index=None)

    api_config = {k: config[k] for k in ("bocha_key", "qwen_key")}
    if config.get("investigator_calibration"):
        api_config["investigator_calibration"] = _load_json_config(
            config["investigator_calibration"]
        )
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Worker") as executor:
        futures = {
            executor.submit(
                _process_group_hd,
                i,
                doc,
                api_config,
                embedder,
                logger,
                collector,
                evidence_writer,
                calibration_writer,
            ): (
                i,
                doc.get("id", "Unknown"),
            )
            for i, (_, doc) in enumerate(docs_to_process)
        }
        completed, total = 0, len(docs_to_process)
        for future in as_completed(futures):
            idx, doc_id = futures[future]
            try:
                future.result()
                completed += 1
                if realtime_mode:
                    print(f"Completed: {completed}/{total} ({completed / total * 100:.1f}%) - {doc_id}")
            except Exception as e:
                import traceback

                logger.log(f"Failed processing {doc_id} (#{idx}): {e}", doc_index=None)
                logger.log(traceback.format_exc(), doc_index=None)

    logger.log(
        f"HD finished in {time.time() - start_time:.2f}s, output={output_file}, evidence={evidence_file}, calibration={calibration_file}",
        doc_index=None,
    )


def _default_evidence_file(output_file: str) -> str:
    root, _ = os.path.splitext(output_file)
    return f"{root}_evidence.txt"


def _default_calibration_file(output_file: str) -> str:
    root, _ = os.path.splitext(output_file)
    return f"{root}_calibration.csv"


def _load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_efi(
    hd_excel: str,
    json_data_cn: list,
    json_data_en: list,
    config: dict,
    output_file: str,
    log_file: str,
):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("EFI")

    class _SimpleLogger:
        def log(self, msg, doc_index=None, print_console=True):
            if print_console:
                log.info(msg)

    api_clients = {"qwen": make_qwen_client(config["qwen_key"])}
    linguist = LinguistAgent(api_clients, _SimpleLogger())

    log.info("=" * 60 + "\nStarting EFI-Pilot EFI")
    hd_results, efi_texts = load_hd_results_for_efi(hd_excel)
    doc_ids = _apply_id_filter(list(hd_results.keys()), config, log)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    idx_cn = {d["id"]: d for d in json_data_cn}
    idx_en = {d["id"]: d for d in json_data_en}
    total, success, fail = len(doc_ids), 0, 0

    for i, doc_id in enumerate(doc_ids):
        log.info(f"[{i + 1}/{total}] Processing {doc_id}")
        labels = hd_results[doc_id]
        efi_text = efi_texts.get(doc_id)
        doc = idx_en.get(doc_id) if doc_id.startswith("ED") else idx_cn.get(doc_id)
        if not doc:
            log.warning(f"{doc_id}: not found in JSON")
            fail += 1
            continue

        event = doc.get("events", {}).get("event1", "")
        gt_fac = doc.get("events", {}).get("factuality1", "")

        agent_states = [
            AgentState(
                doc_id=doc_id,
                group_index=i,
                text_key=k,
                text=get_text_from_doc(doc, k),
                event=event,
                hd_label=v,
            )
            for k, v in labels.items()
            if get_text_from_doc(doc, k)
        ]
        if not agent_states:
            log.warning(f"{doc_id}: no text candidates")
            fail += 1
            continue

        if efi_text:
            chosen = next((s for s in agent_states if s.text_key == efi_text), None)
            if chosen is not None:
                agent_states = [chosen]
                chosen.hd_label = "Real"
            else:
                log.warning(f"{doc_id}: efi_text={efi_text} not found, using fallback selection")

        group = GroupState(
            doc_id=doc_id,
            group_index=i,
            event=event,
            ground_truth_factuality=gt_fac,
            documents=agent_states,
        )
        try:
            group = linguist.run(group)
            out = group.linguist_output
            result = {
                "id": doc_id,
                "text_used": out.chosen_text_key,
                "event1_factuality": gt_fac,
                "predict_factuality1": efi_to_legacy(out.efi_label),
            }
            if doc.get("events", {}).get("event2"):
                result["event2_factuality"] = doc["events"].get("factuality2", "")
            append_to_excel(output_file, result, log)
            log.info(f"{doc_id}: chosen={out.chosen_text_key}, efi={out.efi_label}")
            success += 1
        except Exception as e:
            log.error(f"{doc_id}: {e}")
            fail += 1
        time.sleep(0.5)

    log.info(f"EFI finished: {success} success / {fail} failed / {total} total -> {output_file}")


def _apply_id_filter(doc_ids: list, config: dict, log) -> list:
    start_id, end_id = config.get("start_id"), config.get("end_id")
    if start_id:
        try:
            doc_ids = doc_ids[doc_ids.index(start_id) :]
        except ValueError:
            log.warning(f"start_id {start_id} not found")
    if end_id:
        try:
            doc_ids = doc_ids[: doc_ids.index(end_id) + 1]
        except ValueError:
            log.warning(f"end_id {end_id} not found")
    return doc_ids
