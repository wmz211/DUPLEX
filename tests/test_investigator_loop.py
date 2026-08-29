import json
import unittest
from unittest.mock import Mock

from efi_pilot.agents.base import AgentState, InvestigatorOutput
from efi_pilot.agents.investigator import (
    FusionCalibrator,
    InvestigatorAgent,
    QwenRoundResult,
    SemanticResult,
    needs_followup,
    parse_qwen_round,
)


class QwenRoundParsingTests(unittest.TestCase):
    def test_sufficient_real_result_stops_after_round_one(self):
        content = json.dumps({
            "verdict": "Real", "confidence": 88,
            "evidence_summary": "官方档案与同期报道一致",
            "sources": ["官方档案"], "unverified_points": [],
            "need_more_search": False, "next_query": "",
        }, ensure_ascii=False)
        result = parse_qwen_round(content)
        self.assertEqual(result.verdict, "Real")
        self.assertEqual(result.confidence, 88.0)
        self.assertFalse(needs_followup(result))

    def test_unverified_points_trigger_followup(self):
        content = json.dumps({
            "verdict": "Factuality", "confidence": 91,
            "evidence_summary": "人物任职时间可能错误",
            "sources": ["搜索摘要"],
            "unverified_points": ["核实人物在事件发生日的正式职务"],
            "need_more_search": False,
            "next_query": "人物 官方职务 事件日期",
        }, ensure_ascii=False)
        self.assertTrue(needs_followup(parse_qwen_round(content)))

    def test_factuality_without_source_triggers_followup(self):
        content = json.dumps({
            "verdict": "Factuality", "confidence": 92,
            "evidence_summary": "存在时间错误", "sources": [],
            "unverified_points": [], "need_more_search": False,
            "next_query": "核查新闻核心事实",
        }, ensure_ascii=False)
        self.assertTrue(needs_followup(parse_qwen_round(content)))

    def test_final_round_parses_cumulative_evidence(self):
        content = json.dumps({
            "new_evidence": "第二轮证据", "retained_evidence": "第一轮有效证据",
            "rejected_evidence": "第一轮错误指控",
            "remaining_unverified_points": [],
            "final_verdict": "Real", "final_confidence": 89,
            "final_evidence_summary": "累计证据确认真实",
        }, ensure_ascii=False)
        result = parse_qwen_round(content, final=True)
        self.assertEqual(result.verdict, "Real")
        self.assertEqual(result.rejected_evidence, "第一轮错误指控")


class InvestigatorLoopTests(unittest.TestCase):
    def _state(self):
        return AgentState("CDX", 0, "text1", "新闻正文", "目标事件")

    def _agent(self):
        logger = Mock()
        logger.log = Mock()
        return InvestigatorAgent({
            "bocha_key": "x", "qwen": Mock()
        }, logger, Mock())

    def test_sufficient_evidence_calls_semantic_and_qwen_once(self):
        agent = self._agent()
        semantic = SemanticResult(70.0, "ok", "query", ["fact"], [70.0], 3)
        first = QwenRoundResult("Real", 90, "权威报道一致", ["官方"])
        agent._semantic_similarity_check = Mock(return_value=semantic)
        agent._qwen_search_verify = Mock(return_value=first)
        output = agent.run(self._state()).investigator_output
        agent._semantic_similarity_check.assert_called_once()
        agent._qwen_search_verify.assert_called_once()
        self.assertEqual(output.iterations, 1)

    def test_insufficient_evidence_runs_cumulative_followup(self):
        agent = self._agent()
        semantic = SemanticResult(68.0, "ok", "query", ["fact"], [68.0], 3)
        first = QwenRoundResult(
            "Factuality", 92, "人物身份可能错误", ["搜索摘要"],
            ["核实事件日正式职务"], True, "人物 官方职务 日期"
        )
        second = QwenRoundResult(
            "Real", 89, "累计证据确认人物身份正确",
            retained_evidence="第一轮背景证据", rejected_evidence="身份错误指控"
        )
        agent._semantic_similarity_check = Mock(return_value=semantic)
        agent._qwen_search_verify = Mock(side_effect=[first, second])
        output = agent.run(self._state()).investigator_output
        self.assertEqual(agent._qwen_search_verify.call_count, 2)
        second_call = agent._qwen_search_verify.call_args_list[1]
        self.assertIs(second_call.kwargs["first_round"], first)
        self.assertEqual(output.qwen_label, "Real")
        self.assertEqual(output.qwen_confidence, 89)
        self.assertEqual(output.iterations, 2)


class CompatibilityTests(unittest.TestCase):
    def test_old_output_constructor_remains_valid(self):
        output = InvestigatorOutput("IsRW", 80, "e", 70, "Real", 90)
        self.assertEqual(output.stop_reason, "")
        self.assertEqual(output.qwen_rounds, [])

    def test_fusion_semantics_are_unchanged(self):
        calibrator = FusionCalibrator(alpha=0.4, threshold=55)
        self.assertEqual(calibrator.qwen_truth_score("Real", 90), 90)
        self.assertEqual(calibrator.qwen_truth_score("Factuality", 90), 10)
        self.assertAlmostEqual(calibrator.fuse(70, "Real", 90), 82)
        self.assertAlmostEqual(calibrator.fuse(70, "Factuality", 90), 34)


if __name__ == "__main__":
    unittest.main()
