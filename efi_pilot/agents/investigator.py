"""Investigator：Bocha 单轮语义先验 + Qwen 自适应累计证据核查。"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field

from efi_pilot.agents.base import AgentState, BaseAgent, InvestigatorOutput
from efi_pilot.prompts.investigator import (
    QWEN_VERIFY_SYSTEM,
    build_qwen_followup_prompt,
    build_qwen_verify_prompt,
    build_semantic_query_prompt,
)
from efi_pilot.config import QWEN_MODEL
EVIDENCE_SUPPORT_THRESHOLD = 45.0
MAX_QWEN_ROUNDS = 2


@dataclass
class FusionCalibrator:
    alpha: float | None = None
    threshold: float = 55.0
    temperature: float = 1.0

    @classmethod
    def from_config(cls, config: dict | None) -> "FusionCalibrator":
        if not config:
            return cls()
        best = config.get("best") if isinstance(config.get("best"), dict) else config
        return cls(
            alpha=float(best["alpha"]) if "alpha" in best else None,
            threshold=float(best.get("threshold", 55.0)),
            temperature=float(best.get("temperature", 1.0)),
        )

    def alpha_for(self, qwen_confidence: float) -> float:
        if self.alpha is not None:
            return self.alpha
        if qwen_confidence >= 85.0:
            return 0.35
        if qwen_confidence >= 70.0:
            return 0.40
        if qwen_confidence >= 55.0:
            return 0.45
        return 0.60

    def qwen_truth_score(self, qwen_label: str, qwen_confidence: float) -> float:
        return qwen_confidence if qwen_label == "Real" else 100.0 - qwen_confidence

    def fuse(self, semantic_score: float, qwen_label: str, qwen_confidence: float) -> float:
        qwen_score = self.qwen_truth_score(qwen_label, qwen_confidence)
        alpha = self.alpha_for(qwen_confidence)
        raw_score = alpha * semantic_score + (1.0 - alpha) * qwen_score
        return max(0.0, min(100.0, self._temperature_calibrate(raw_score)))

    def _temperature_calibrate(self, score: float) -> float:
        if self.temperature <= 0 or abs(self.temperature - 1.0) < 1e-9:
            return score
        probability = max(1e-6, min(1.0 - 1e-6, score / 100.0))
        logit = math.log(probability / (1.0 - probability))
        return 100.0 / (1.0 + math.exp(-(logit / self.temperature)))


@dataclass
class SemanticResult:
    score: float
    evidence: str
    search_query: str
    key_facts: list[str]
    fact_scores: list[float]
    result_count: int

    @property
    def coverage(self) -> float:
        if not self.fact_scores:
            return 0.0
        covered = sum(x >= EVIDENCE_SUPPORT_THRESHOLD for x in self.fact_scores)
        return covered / len(self.fact_scores)


@dataclass
class QwenRoundResult:
    verdict: str
    confidence: float
    evidence_summary: str
    sources: list[str] = field(default_factory=list)
    unverified_points: list[str] = field(default_factory=list)
    need_more_search: bool = False
    next_query: str = ""
    new_evidence: str = ""
    retained_evidence: str = ""
    rejected_evidence: str = ""
    raw_content: str = ""
    parse_valid: bool = True


def _extract_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Qwen response contains no JSON object")
    return json.loads(text[start:end + 1])


def parse_qwen_round(content: str, final: bool = False) -> QwenRoundResult:
    data = _extract_json_object(content)
    verdict_key = "final_verdict" if final else "verdict"
    confidence_key = "final_confidence" if final else "confidence"
    verdict = str(data.get(verdict_key, "")).strip()
    if verdict not in {"Real", "Factuality"}:
        raise ValueError(f"invalid verdict: {verdict!r}")
    confidence = max(0.0, min(100.0, float(data[confidence_key])))
    evidence_key = "final_evidence_summary" if final else "evidence_summary"
    points_key = "remaining_unverified_points" if final else "unverified_points"
    sources = data.get("sources", [])
    points = data.get(points_key, [])
    if not isinstance(sources, list) or not isinstance(points, list):
        raise ValueError("sources and unverified points must be arrays")
    return QwenRoundResult(
        verdict=verdict,
        confidence=confidence,
        evidence_summary=str(data.get(evidence_key, "")).strip(),
        sources=[str(x) for x in sources],
        unverified_points=[str(x) for x in points],
        need_more_search=bool(data.get("need_more_search", False)),
        next_query=str(data.get("next_query", "")).strip(),
        new_evidence=str(data.get("new_evidence", "")).strip(),
        retained_evidence=str(data.get("retained_evidence", "")).strip(),
        rejected_evidence=str(data.get("rejected_evidence", "")).strip(),
        raw_content=content,
    )


def needs_followup(result: QwenRoundResult) -> bool:
    if not result.parse_valid or result.need_more_search or result.unverified_points:
        return True
    if not result.evidence_summary:
        return True
    return result.verdict == "Factuality" and not result.sources


class InvestigatorAgent(BaseAgent):
    def __init__(self, api_clients: dict, logger, embedder):
        super().__init__(api_clients, logger)
        self.embedder = embedder
        self.calibrator = FusionCalibrator.from_config(
            api_clients.get("investigator_calibration")
        )

    def run(self, state: AgentState) -> AgentState:
        self._log(f"      Investigator 检测 {state.text_key}...", state.group_index)
        semantic = self._semantic_similarity_check(state.text, state.group_index)

        first = self._qwen_search_verify(state.text, state.group_index, round_number=1)
        rounds = [first]
        final = first
        stop_reason = "evidence_sufficient"

        if needs_followup(first):
            stop_reason = "max_rounds"
            self._log("      第一轮证据不足，启动第二轮定向联网核查", state.group_index)
            second = self._qwen_search_verify(
                state.text,
                state.group_index,
                round_number=2,
                first_round=first,
            )
            if second.parse_valid:
                rounds.append(second)
                final = second
            else:
                stop_reason = "round2_failed_fallback_round1"
                self._log("      第二轮无有效结果，回退第一轮", state.group_index)

        final_score = self.calibrator.fuse(
            semantic.score, final.verdict, final.confidence
        )
        label = "IsRW" if final_score >= self.calibrator.threshold else "IsFacHal"
        qwen_score = self.calibrator.qwen_truth_score(final.verdict, final.confidence)
        self._log(
            f"      Bocha单轮: semantic={semantic.score:.1f}% coverage={semantic.coverage:.2f}",
            state.group_index,
        )
        self._log(
            f"      Qwen累计结论: {final.verdict} {final.confidence:.1f}% "
            f"→真实性分数 {qwen_score:.1f}% (轮数={len(rounds)})",
            state.group_index,
        )
        self._log(
            f"      最终判断: {label} (V_fac={final_score:.1f}%)",
            state.group_index,
        )
        state.investigator_output = InvestigatorOutput(
            label=label,
            confidence=final_score,
            evidence=final.evidence_summary,
            semantic_score=semantic.score,
            qwen_label=final.verdict,
            qwen_confidence=final.confidence,
            iterations=len(rounds),
            stop_reason=stop_reason,
            qwen_rounds=[asdict(x) for x in rounds],
        )
        return state

    def _semantic_similarity_check(self, text: str, group_index: int) -> SemanticResult:
        from efi_pilot.utils.api_clients import bocha_search

        prompt = build_semantic_query_prompt(text)
        self._log("      路径1: Bocha单轮语义相似度验证...", group_index)
        try:
            response = self.clients["qwen"].chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=False,
            )
            content = response.choices[0].message.content.strip()
            search_query, key_facts = self._parse_semantic_query_response(content)
        except Exception as exc:
            self._log(f"         LLM处理失败: {exc}", group_index)
            search_query, key_facts = text[:50], [text[:100]]
        search_query = search_query or text[:50]
        key_facts = key_facts or [text[:100]]
        results = bocha_search(
            self.clients["bocha_key"], search_query, num_results=3,
            logger=self.logger, group_index=group_index,
        )
        if not results:
            return SemanticResult(30.0, "无法获取搜索结果", search_query, key_facts, [], 0)

        # 延迟导入，纯 mock 离线测试无需安装完整 transformers 依赖。
        from sentence_transformers import util

        contexts = [f"{x['title']} {x['snippet']}" for x in results]
        fact_emb = self.embedder.encode(key_facts, convert_to_tensor=True)
        ctx_emb = self.embedder.encode(contexts, convert_to_tensor=True)
        similarities = util.cos_sim(fact_emb, ctx_emb)
        scores = [float(x) * 100.0 for x in similarities.max(dim=1).values]
        score = sum(scores) / len(scores)
        return SemanticResult(score, "语义验证完成", search_query, key_facts, scores, len(results))

    def _qwen_search_verify(
        self,
        text: str,
        group_index: int,
        round_number: int,
        first_round: QwenRoundResult | None = None,
    ) -> QwenRoundResult:
        final = round_number == MAX_QWEN_ROUNDS
        prompt = (
            build_qwen_followup_prompt(text, asdict(first_round))
            if final and first_round is not None
            else build_qwen_verify_prompt(text)
        )
        try:
            response = self.clients["qwen"].chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": QWEN_VERIFY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                extra_body={"enable_search": True},
            )
            content = response.choices[0].message.content.strip()
            self._log(f"         [DEBUG] Qwen第{round_number}轮返回:\n{content}", group_index, False)
            return parse_qwen_round(content, final=final)
        except Exception as exc:
            self._log(f"         Qwen第{round_number}轮验证失败: {exc}", group_index)
            fallback = first_round or QwenRoundResult(
                "Real", 30.0, f"验证失败: {exc}", parse_valid=False
            )
            if first_round is not None:
                return QwenRoundResult(
                    fallback.verdict, fallback.confidence, fallback.evidence_summary,
                    fallback.sources, fallback.unverified_points,
                    fallback.need_more_search, fallback.next_query,
                    raw_content=str(exc), parse_valid=False,
                )
            return fallback

    def _parse_semantic_query_response(self, content: str):
        query, facts = "", []
        for raw in content.splitlines():
            line = raw.strip()
            if "搜索查询" in line or "search_query" in line or "Search query" in line:
                query = self._value_after_colon(line)
            elif "关键事实" in line or "key_fact" in line or "Key fact" in line:
                fact = self._value_after_colon(line)
                if fact:
                    facts.append(fact)
        return query, facts

    @staticmethod
    def _value_after_colon(line: str) -> str:
        for sep in ("：", ":", "="):
            if sep in line:
                return line.split(sep, 1)[1].strip()
        return line.strip()
