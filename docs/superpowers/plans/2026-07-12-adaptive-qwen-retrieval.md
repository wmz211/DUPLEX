# Qwen 自适应检索循环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Investigator 改为“Bocha 固定单轮 + Qwen 证据充分性驱动的一至两轮累计核查”，同时保持现有标签、置信度和融合语义兼容。

**Architecture:** Bocha 只生成一次语义相关性与覆盖率；Qwen 第一轮自由核查并返回结构化证据状态，确定性门控仅在证据不足时启动第二轮。第二轮携带第一轮全部证据并输出累计结论，最终继续通过现有 `FusionCalibrator` 与单轮 Bocha 分数融合。

**Tech Stack:** Python 3.11、`dataclasses`、`json`、OpenAI-compatible Qwen API、Bocha Web Search、SentenceTransformer、`unittest`。

## Global Constraints

- Bocha 每篇文本只调用一次，不改写 query，不输出真假判断。
- Qwen 每篇文本最多调用两次，并继续使用 `enable_search=True`。
- Qwen 自由决定核查哪些关键信息，不固定核查点数量，不做完整原子化。
- 二轮只补查第一轮未验证信息或复核第一轮错误指控，并基于累计证据重新评分。
- 无新证据或两轮仍不足时，仍按现有 `Real`/`Factuality` 与 0–100 置信度机制评估；代码不自动降分。
- `qwen_truth_score`、`FusionCalibrator` 公式和外部 `InvestigatorOutput` 核心字段保持兼容。
- 保留工作区中已有的未提交修改，不覆盖与本功能无关的改动。

---

## 文件结构

- Create: `tests/test_investigator_loop.py` — 使用纯 mock 验证解析、门控、调用次数、累计证据和失败回退。
- Modify: `efi_pilot/prompts/investigator.py` — 保留 Bocha prompt，新增 Qwen 第一轮 JSON 协议和第二轮累计核查 prompt。
- Modify: `efi_pilot/agents/base.py` — 为 `InvestigatorOutput` 增加向后兼容的可选审计字段。
- Modify: `efi_pilot/agents/investigator.py` — 定义轮次状态、JSON 解析、证据门控、单轮 Bocha 和两轮 Qwen 主流程。
- Modify: `efi_pilot/orchestrator.py` — 在已有 EvidenceWriter 中输出新增审计字段，不改变 Excel 核心列。

### Task 1: Qwen 轮次协议、解析器与证据充分性门控

**Files:**
- Create: `tests/test_investigator_loop.py`
- Modify: `efi_pilot/prompts/investigator.py`
- Modify: `efi_pilot/agents/investigator.py`

**Interfaces:**
- Produces: `QwenRoundResult`, `parse_qwen_round(content: str, final: bool = False) -> QwenRoundResult`、`needs_followup(result: QwenRoundResult) -> bool`。
- Consumes: 现有 `QWEN_VERIFY_SYSTEM` 和现有置信度定义。

- [ ] **Step 1: 编写失败测试，覆盖第一轮 JSON 解析和门控**

```python
import json
import unittest

from efi_pilot.agents.investigator import needs_followup, parse_qwen_round


class QwenRoundParsingTests(unittest.TestCase):
    def test_sufficient_real_result_stops_after_round_one(self):
        content = json.dumps({
            "verdict": "Real",
            "confidence": 88,
            "evidence_summary": "官方档案与同期报道一致",
            "sources": ["官方档案"],
            "unverified_points": [],
            "need_more_search": False,
            "next_query": "",
        }, ensure_ascii=False)
        result = parse_qwen_round(content)
        self.assertEqual(result.verdict, "Real")
        self.assertEqual(result.confidence, 88.0)
        self.assertFalse(needs_followup(result))

    def test_unverified_points_trigger_followup(self):
        content = json.dumps({
            "verdict": "Factuality",
            "confidence": 91,
            "evidence_summary": "人物任职时间可能错误",
            "sources": ["搜索摘要"],
            "unverified_points": ["核实人物在事件发生日的正式职务"],
            "need_more_search": False,
            "next_query": "人物 官方职务 事件日期",
        }, ensure_ascii=False)
        self.assertTrue(needs_followup(parse_qwen_round(content)))

    def test_factuality_without_concrete_evidence_triggers_followup(self):
        content = json.dumps({
            "verdict": "Factuality",
            "confidence": 92,
            "evidence_summary": "",
            "sources": [],
            "unverified_points": [],
            "need_more_search": False,
            "next_query": "核查新闻核心事实",
        }, ensure_ascii=False)
        self.assertTrue(needs_followup(parse_qwen_round(content)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `C:\Users\wang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.test_investigator_loop.QwenRoundParsingTests -v`

Expected: FAIL，提示无法导入 `needs_followup` 或 `parse_qwen_round`。

- [ ] **Step 3: 实现最小轮次数据结构、JSON 提取与门控**

在 `efi_pilot/agents/investigator.py` 增加：

```python
import json
from dataclasses import dataclass, field


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
    evidence = str(data.get(
        "final_evidence_summary" if final else "evidence_summary", ""
    )).strip()
    unverified = data.get(
        "remaining_unverified_points" if final else "unverified_points", []
    )
    return QwenRoundResult(
        verdict=verdict,
        confidence=confidence,
        evidence_summary=evidence,
        sources=[str(x) for x in data.get("sources", [])],
        unverified_points=[str(x) for x in unverified],
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
    if result.verdict == "Factuality" and not result.sources:
        return True
    return False
```

- [ ] **Step 4: 更新两个 Qwen prompt**

保留 `build_semantic_query_prompt()`，删除运行时使用的 Bocha `build_reformulate_prompt()`。将 `build_qwen_verify_prompt()` 的结尾替换为第一轮 JSON 协议，并增加：

```python
def build_qwen_followup_prompt(text: str, first_round: dict) -> str:
    context = json.dumps(first_round, ensure_ascii=False, indent=2)
    return f"""你正在执行第二轮补充事实核查。

原始新闻：
{text}

第一轮完整状态：
{context}

只检索第一轮未验证的信息，或复核第一轮提出的事实错误。保留仍有效的第一轮证据，明确剔除被新证据推翻的依据，并基于第一轮与第二轮的累计信息重新使用原有 Real/Factuality 和置信度标准评分。没有找到新证据时，也必须根据现有全部信息完成评估，不得由程序化规则自动降低置信度。

只输出一个 JSON 对象：
{{
  "new_evidence": "第二轮新增证据",
  "retained_evidence": "仍有效的第一轮证据",
  "rejected_evidence": "被纠正或剔除的第一轮证据",
  "remaining_unverified_points": [],
  "final_verdict": "Real 或 Factuality",
  "final_confidence": 0,
  "final_evidence_summary": "基于累计证据的综合判断"
}}"""
```

同时在文件顶部增加 `import json`。

- [ ] **Step 5: 运行解析测试**

Run: `C:\Users\wang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.test_investigator_loop.QwenRoundParsingTests -v`

Expected: 3 tests PASS。

- [ ] **Step 6: 提交 Task 1**

```powershell
git add tests/test_investigator_loop.py efi_pilot/agents/investigator.py efi_pilot/prompts/investigator.py
git commit -m "feat: add structured qwen evidence rounds"
```

### Task 2: 单轮 Bocha 与自适应两轮 Qwen 主流程

**Files:**
- Modify: `tests/test_investigator_loop.py`
- Modify: `efi_pilot/agents/investigator.py`

**Interfaces:**
- Consumes: `QwenRoundResult`、`parse_qwen_round()`、`needs_followup()`、`build_qwen_followup_prompt()`。
- Produces: `InvestigatorAgent.run(state: AgentState) -> AgentState`，保证 Bocha 一次、Qwen 一至两次。

- [ ] **Step 1: 编写失败测试，验证调用次数与累计输入**

在测试文件增加 mock logger、embedder 和 chat client，并增加：

```python
from unittest.mock import Mock, patch

from efi_pilot.agents.base import AgentState
from efi_pilot.agents.investigator import InvestigatorAgent, SemanticResult


class InvestigatorLoopTests(unittest.TestCase):
    def _state(self):
        return AgentState("CDX", 0, "text1", "新闻正文", "目标事件")

    def _agent(self):
        logger = Mock()
        logger.log = Mock()
        return InvestigatorAgent({
            "deepseek": Mock(), "bocha_key": "x", "qwen": Mock()
        }, logger, Mock())

    @patch("efi_pilot.agents.investigator.bocha_search")
    def test_sufficient_evidence_calls_bocha_and_qwen_once(self, search):
        agent = self._agent()
        semantic = SemanticResult(70.0, "ok", "query", ["fact"], [70.0], 3)
        agent._semantic_similarity_check = Mock(return_value=semantic)
        agent._qwen_search_verify = Mock(return_value=parse_qwen_round(json.dumps({
            "verdict": "Real", "confidence": 90,
            "evidence_summary": "权威报道一致", "sources": ["官方"],
            "unverified_points": [], "need_more_search": False, "next_query": ""
        }, ensure_ascii=False)))
        agent.run(self._state())
        agent._semantic_similarity_check.assert_called_once()
        agent._qwen_search_verify.assert_called_once()

    def test_insufficient_evidence_runs_followup_with_round_one_state(self):
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
        self.assertEqual(output.qwen_label, "Real")
        self.assertEqual(output.qwen_confidence, 89)
        self.assertEqual(output.iterations, 2)
```

- [ ] **Step 2: 运行测试并确认第二个测试失败**

Run: `C:\Users\wang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.test_investigator_loop.InvestigatorLoopTests -v`

Expected: FAIL，因为当前 `run()` 会循环 Bocha，且 `_qwen_search_verify()` 返回旧元组。

- [ ] **Step 3: 重构 Investigator 主流程**

将 `run()` 改为以下结构，不再调用 `_semantic_similarity_check_reformulate()`：

```python
def run(self, state: AgentState) -> AgentState:
    semantic = self._semantic_similarity_check(state.text, state.group_index)
    first = self._qwen_search_verify(state.text, state.group_index, round_number=1)
    rounds = [first]
    final = first
    stop_reason = "evidence_sufficient"

    if needs_followup(first):
        stop_reason = "max_rounds"
        try:
            final = self._qwen_search_verify(
                state.text,
                state.group_index,
                round_number=2,
                first_round=first,
            )
            rounds.append(final)
        except Exception as exc:
            stop_reason = "round2_failed_fallback_round1"
            self._log(f"第二轮失败，回退第一轮: {exc}", state.group_index)
            final = first

    final_score = self.calibrator.fuse(
        semantic.score, final.verdict, final.confidence
    )
    label = "IsRW" if final_score >= self.calibrator.threshold else "IsFacHal"
    state.investigator_output = InvestigatorOutput(
        label=label,
        confidence=final_score,
        evidence=final.evidence_summary,
        semantic_score=semantic.score,
        qwen_label=final.verdict,
        qwen_confidence=final.confidence,
        iterations=len(rounds),
        stop_reason=stop_reason,
        qwen_rounds=rounds,
    )
    return state
```

将 `_qwen_search_verify()` 改成：第一轮使用 `build_qwen_verify_prompt()`，第二轮使用 `build_qwen_followup_prompt()`，均返回 `QwenRoundResult`。第二轮 prompt 的 `first_round` 使用 `dataclasses.asdict(first_round)` 序列化。

- [ ] **Step 4: 删除不再使用的 Bocha 循环符号**

从 `investigator.py` 删除：

- `MAX_ITERATIONS` 以外所有 Bocha 循环阈值；
- `_semantic_similarity_check_reformulate()`；
- `_should_continue()`；
- `_select_final_round()`；
- `_query_similarity()`；
- `EvidenceRound`；
- `build_reformulate_prompt` import。

保留 `MAX_QWEN_ROUNDS = 2` 作为清晰上限。

- [ ] **Step 5: 运行控制流测试**

Run: `C:\Users\wang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.test_investigator_loop -v`

Expected: 所有测试 PASS；充分证据 Qwen 一次，证据不足 Qwen 两次，Bocha语义函数始终一次。

- [ ] **Step 6: 提交 Task 2**

```powershell
git add tests/test_investigator_loop.py efi_pilot/agents/investigator.py
git commit -m "feat: move adaptive retrieval loop to qwen"
```

### Task 3: 审计状态与 EvidenceWriter 兼容输出

**Files:**
- Modify: `tests/test_investigator_loop.py`
- Modify: `efi_pilot/agents/base.py`
- Modify: `efi_pilot/orchestrator.py`

**Interfaces:**
- Consumes: `QwenRoundResult` 列表和 `stop_reason`。
- Produces: 向后兼容的 `InvestigatorOutput`，以及包含两轮证据的文本日志。

- [ ] **Step 1: 编写失败测试，验证旧构造方式兼容和新字段可写**

```python
from efi_pilot.agents.base import InvestigatorOutput


class InvestigatorOutputCompatibilityTests(unittest.TestCase):
    def test_old_constructor_remains_valid(self):
        output = InvestigatorOutput(
            "IsRW", 80.0, "evidence", 70.0, "Real", 90.0
        )
        self.assertEqual(output.iterations, 1)
        self.assertEqual(output.stop_reason, "")
        self.assertEqual(output.qwen_rounds, [])

    def test_audit_fields_accept_two_rounds(self):
        output = InvestigatorOutput(
            "IsRW", 80.0, "evidence", 70.0, "Real", 90.0,
            iterations=2, stop_reason="max_rounds", qwen_rounds=[{"round": 1}, {"round": 2}]
        )
        self.assertEqual(len(output.qwen_rounds), 2)
```

- [ ] **Step 2: 运行测试并确认新字段不存在**

Run: `C:\Users\wang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.test_investigator_loop.InvestigatorOutputCompatibilityTests -v`

Expected: FAIL，提示未知参数 `stop_reason`。

- [ ] **Step 3: 增加向后兼容字段**

在 `InvestigatorOutput` 末尾增加：

```python
stop_reason: str = ""
qwen_rounds: list = field(default_factory=list)
```

不改变已有字段顺序和默认值。

- [ ] **Step 4: 扩展 EvidenceWriter**

在 `orchestrator.py` 的 `_format_state()` 中，在现有 Investigator 行后追加：

```python
f"  stop_reason: {investigator.stop_reason}",
"  qwen_rounds:",
```

然后逐轮输出 `verdict`、`confidence`、`evidence_summary`、`sources`、`unverified_points`、`next_query`、`new_evidence`、`retained_evidence` 和 `rejected_evidence`。使用 `getattr()` 同时兼容字典和 dataclass 序列化结果，不修改现有 Excel 核心列。

- [ ] **Step 5: 运行全部单元测试与语法检查**

Run: `C:\Users\wang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.test_investigator_loop -v`

Expected: 所有测试 PASS。

Run: `C:\Users\wang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall -q efi_pilot tests`

Expected: exit code 0，无输出。

- [ ] **Step 6: 提交 Task 3**

```powershell
git add tests/test_investigator_loop.py efi_pilot/agents/base.py efi_pilot/orchestrator.py
git commit -m "feat: record qwen retrieval audit state"
```

### Task 4: 回归检查与开发集消融准备

**Files:**
- Modify: `tests/test_investigator_loop.py`
- Modify: `compare_fusion_strategies.py` only if its current input assumptions reject the new single-Bocha/two-Qwen records.

**Interfaces:**
- Consumes: 保持不变的 `semantic_score`、`qwen_label`、`qwen_confidence`、`confidence`。
- Produces: 可用于现有校准脚本的最终累计 Qwen 分数记录。

- [ ] **Step 1: 增加融合语义回归测试**

```python
from efi_pilot.agents.investigator import FusionCalibrator


class FusionCompatibilityTests(unittest.TestCase):
    def test_real_and_factuality_truth_score_semantics_are_unchanged(self):
        calibrator = FusionCalibrator(alpha=0.4, threshold=55.0)
        self.assertEqual(calibrator.qwen_truth_score("Real", 90), 90)
        self.assertEqual(calibrator.qwen_truth_score("Factuality", 90), 10)
        self.assertAlmostEqual(calibrator.fuse(70, "Real", 90), 82)
        self.assertAlmostEqual(calibrator.fuse(70, "Factuality", 90), 34)
```

- [ ] **Step 2: 运行完整离线测试**

Run: `C:\Users\wang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests -v`

Expected: 所有测试 PASS，且无需真实 API key。

- [ ] **Step 3: 检查现有调用方和残留 Bocha 循环**

Run: `Get-ChildItem -Recurse -File efi_pilot | Select-String -Pattern 'build_reformulate_prompt|_semantic_similarity_check_reformulate|_select_final_round|LOW_EVIDENCE_COVERAGE'`

Expected: 无运行时代码命中；若设计文档或历史注释命中则不修改。

Run: `Get-ChildItem -Recurse -File efi_pilot | Select-String -Pattern 'InvestigatorOutput\('`

Expected: 所有现有构造点仍可使用原参数，新增字段均有默认值。

- [ ] **Step 4: 运行 10 个已知错误案例前的只读准备检查**

Run: `Import-Csv res/investigator_dev50_rerun_scores.csv | Where-Object { $_.old_label -ne $_.gold_label } | Select-Object doc_id,text_key,gold_label,semantic_score,qwen_label,qwen_confidence`

Expected: 输出 10 行已知错误，作为联网回归运行的固定 case list。

- [ ] **Step 5: 在获得 API 运行许可后执行开发集实验**

分别运行 One-shot、Proposed 和 Fixed-2 配置，输出独立结果文件。不得覆盖现有 `res/dev50_rerun_*`。比较 Real F1、FacHal F1、Macro-F1、二轮触发率、平均 Qwen 调用次数、平均耗时、正确修正数和有害反转数。

Expected: Bocha 每篇一次、Qwen 每篇一至两次；Macro-F1 不低于当前校准系统，正确修正数多于有害反转数。

- [ ] **Step 6: 提交 Task 4**

```powershell
git add tests/test_investigator_loop.py compare_fusion_strategies.py
git commit -m "test: cover adaptive qwen retrieval regressions"
```

如果 `compare_fusion_strategies.py` 无需修改，则只暂存并提交测试文件。
