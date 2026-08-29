# EFI-Pilot 架构设计文档（修订版 v2）

## 一、当前代码结构分析

### 1.1 关键类与调用关系图

```
qwen_search.py
│
├── ThreadSafeLogger                       共享；线程间通过 doc_index 缓冲后按序刷出
│   ├── log(msg, doc_index, print_console) — 缓存(按序模式)或即时输出
│   └── flush_logs(doc_index)              — 当 next_doc_index 连续时刷出
│
├── ResultCollector                        共享；PriorityQueue 保证 Excel 按序追加
│   ├── add_result(doc_index, result)
│   ├── _try_flush()
│   └── _append_to_excel(result)           — read-then-write（CSV 兜底）
│
├── NewsHallucinationDetector              per-thread 实例，共享 embedder
│   ├── detect_faithfulness_hallucination(text)  → ("Real"|"Faithfulness", conf, evidence)
│   │     DeepSeek，二元判定，内置两个 few-shot 例
│   ├── semantic_similarity_check(text)          → (score, evidence)
│   │     DeepSeek 提取关键事实 → Bocha 搜索 → SentenceTransformer cosine
│   │     注：注释说"并行双验证"，实际为串行调用。Phase 1 忠实复现串行行为。
│   ├── qwen_search_verify(text)                 → ("Real"|"Factuality", conf, evidence)
│   │     Qwen + enable_search=True，内置 1 个长 few-shot 例
│   └── detect_factuality_hallucination(text)    → ("Real"|"Factuality", conf, evidence)
│           串行调用上两个函数 → 加权融合（α·semantic + β·qwen）
│
└── process_document(doc_index, doc, detector, collector) → Dict
      ├── 随机打乱 6 个文本的处理顺序
      ├── Phase1: 对每篇调 detect_faithfulness_hallucination
      │   └── Faithfulness → 直接标记，不进 Phase2     ← 要修复的设计缺陷（Phase 2）
      ├── Phase2: 对剩余文本调 detect_factuality_hallucination
      │   └── 0个Real时：强制取置信度最高的 → Real      ← 要移到 LinguistAgent（Phase 2）
      │   └── ≥2个Real时：只保留置信度最高的，其余→Factuality  ← 同上
      └── 返回 {id, text1: label, ..., text6: label}

predict_factuality.py
│
└── EventFactualityAnalyzer                单线程，使用标准 logging 模块
    ├── judge_factuality(text, event)       → "CT+"|"CT-"|"PS+"|"PS-"|"UU"
    │     Qwen（无联网），7 个 few-shot 例（含巴黎协定 PS- 例）
    ├── load_real_doc_ids(excel)            — 读 HD 输出 Excel，找每文档的 Real 列（取第一个）
    ├── process_document(doc_id, doc, text_col) → Dict
    │     处理 event1（必有）和 event2（可选）
    └── run(excel_input, json_cn, json_en, output, log, start_id, end_id)
          顺序遍历；按 id 前缀 CD/ED 分路 CN/EN JSON
```

### 1.2 数据格式说明

每条 JSON 文档结构固定（EFHallu 数据集，论文 4.1 节 Hallucination Synthesis 定义）：

```
document
├── id: "CD0" / "ED0"（中/英文前缀，CN/EN JSON 分开存放）
├── events: {event1, factuality1, event2?, factuality2?}
├── text1                             ← ground truth: Real
└── hallu_type
    ├── faithfulness: {text2, text3}  ← ground truth: FaiHal（对所有 CN/EN 文档成立）
    └── factuality:   {text4, text5, text6} ← ground truth: FacHal（对所有 CN/EN 文档成立）
```

HD 阶段**不知道**哪篇是 Real；6 篇文本在处理前被打乱顺序。
EFI 阶段读 HD 输出的 Excel，找到被标记为 Real 的列，再从 JSON 取对应文本内容。

### 1.3 当前两大设计缺陷（修复阶段）

| 缺陷位置 | 描述 | Phase 1 处理 | Phase 2 修复 |
|---|---|---|---|
| `process_document` Phase1→Phase2 | FaiHal 文本被丢弃，不进 FacHD | 原样保留（sanity check） | Arbiter+Investigator 对所有 6 篇并行 |
| `process_document` Phase2 末尾 | 0/≥2 个 Real 时强制选最高置信度 | 原样保留，加 TODO 注释 | 移到 LinguistAgent 内部（见 §5） |

---

## 二、新文件目录结构

```
efi_pilot/
├── agents/
│   ├── __init__.py
│   ├── base.py           # AgentState, GroupState, 所有 Output dataclass, BaseAgent
│   ├── dispatcher.py     # DispatcherAgent（Phase1: stub）
│   ├── arbiter.py        # ArbiterAgent
│   ├── investigator.py   # InvestigatorAgent
│   ├── judge.py          # JudgeAgent（per-document）
│   └── linguist.py       # LinguistAgent
├── prompts/
│   ├── __init__.py
│   ├── arbiter.py        # detect_faithfulness_hallucination 的 prompt（直接迁移）
│   ├── investigator.py   # semantic_similarity + qwen_search_verify 的 prompt（直接迁移）
│   ├── linguist.py       # judge_factuality 的 prompt（直接迁移，含全部 7 个 few-shot）
│   └── pairwise_selector.py   # Phase3 新增
├── utils/
│   ├── __init__.py
│   ├── logging.py        # ThreadSafeLogger + ResultCollector（原封不动搬移）
│   ├── api_clients.py    # DeepSeek / Bocha / Qwen 客户端初始化
│   ├── embedder.py       # 共享 SentenceTransformer（单例，线程安全）
│   └── naming_bridge.py  # 新增：输入/输出边界的命名转换（单向硬约束）
├── orchestrator.py       # HD 阶段（ThreadPool）+ EFI 阶段（串行）调度
├── evaluator.py          # 三个指标计算
└── main.py               # CLI 入口（argparse）
```

**不修改的文件**：`qwen_search.py` 和 `predict_factuality.py` 原样保留，永远作为对比基线。

---

## 三、命名规范（单向硬约束）

### 3.1 `utils/naming_bridge.py` 的职责

- **输入边界**（读 JSON / 读原 Excel → 内部代码）：legacy → internal
- **输出边界**（内部代码 → 写 Excel / 评测报告）：internal → legacy
- **内部代码（`agents/*`）只能见到内部命名**

| Legacy（原代码/Excel） | Internal（agents 内部） |
|---|---|
| `"Faithfulness"` | `"FaiHal"` |
| `"Factuality"` | `"FacHal"` |
| `"Real"` | `"Real"` |
| `"UU"` | `"Uu"` |

---

## 四、每个 Agent 的输入/输出接口

### 4.1 数据结构（`agents/base.py`）

```python
# AgentState: HD 阶段使用，per-document（每篇文本一个实例）
# GroupState: EFI 阶段使用，per-group（6 个 AgentState 的聚合）
# Orchestrator 在 HD→EFI 过渡处做 AgentState[] → GroupState 的聚合

@dataclass
class ArbiterOutput:
    raw_label: str        # Phase1: "Real" | "Faithfulness"; Phase2: 4-class
    mapped_label: str     # "IsRW" | "IsFaiHal"
    confidence: float     # 0-100
    evidence: str
    reasoning: Optional[str] = None   # Phase2 开启

@dataclass
class InvestigatorOutput:
    label: str            # "IsRW" | "IsFacHal"
    confidence: float     # 0-100（V_fac 融合分）
    evidence: str
    semantic_score: float
    qwen_label: str
    qwen_confidence: float
    iterations: int = 1   # Phase2 迭代次数

@dataclass
class AgentState:
    doc_id: str
    group_index: int      # 该组在全局的顺序编号（对应 ThreadSafeLogger.doc_index）
    text_key: str         # "text1" ~ "text6"
    text: str
    event: str
    claim_type: Optional[str] = None
    arbiter_output: Optional[ArbiterOutput] = None
    investigator_output: Optional[InvestigatorOutput] = None
    hd_label: Optional[str] = None    # "Real" | "FaiHal" | "FacHal"（内部命名）

@dataclass
class LinguistOutput:
    chosen_text_key: str
    chosen_text: str
    selection_reason: str      # 写入日志，便于 case study
    efi_label: str             # "CT+" | "CT-" | "PS+" | "PS-" | "Uu"（内部命名）
    modal_analysis: Optional[str] = None   # Phase3 模态链推理过程

@dataclass
class GroupState:
    doc_id: str
    group_index: int
    event: str
    ground_truth_factuality: str           # events.factuality1，用于评测
    documents: List[AgentState] = field(default_factory=list)
    linguist_output: Optional[LinguistOutput] = None
```

### 4.2 Agent 接口规范

| Agent | 输入 | 输出 | Phase1 行为 |
|---|---|---|---|
| `DispatcherAgent` | `AgentState` | `AgentState`（`claim_type` 已填） | stub，返回 `"general"` |
| `ArbiterAgent` | `AgentState`（`text`） | `AgentState`（`arbiter_output` 已填） | 封装原二元 prompt，不改 prompt |
| `InvestigatorAgent` | `AgentState`（`text`, `claim_type`） | `AgentState`（`investigator_output` 已填） | 封装原双路径+融合，不改 prompt |
| `JudgeAgent` | `AgentState`（两个 output 已填） | `AgentState`（`hd_label` 已填） | 三行优先级裁决 |
| `LinguistAgent` | `GroupState`（6 个 AgentState） | `GroupState`（`linguist_output` 已填） | 取第一个 Real + 原 judge_factuality |

---

## 五、LinguistAgent 的三阶段选 Real 策略

这是回答问题 A 的关键设计，以下规则**必须写进 `linguist.py` 的注释**：

| HD 输出 Real 数量 | Phase 1（当前） | Phase 2 临时兜底 | Phase 3 最终 |
|---|---|---|---|
| 恰好 1 个 | 直接用 | 直接用 | 直接用 |
| 0 个 | 不会出现（HD 有强制选） | 取 `V_fac`（investigator.confidence）最高的 | pairwise 选最优 |
| ≥2 个 | 不会出现（HD 有强制选） | 取 `V_fac` 最高的 | pairwise 两两投票 |

**实现结构**（三阶段替换只需修改 `_select_real` 方法内部）：

```python
def _select_real(self, group: GroupState) -> AgentState:
    real_docs = [s for s in group.documents if s.hd_label == "Real"]
    if len(real_docs) == 1:
        return real_docs[0]
    # Phase 2+ 临时兜底（HD 移除 force-select 后启用）
    # TODO Phase 3: 替换为 pairwise_compare_and_select
    candidates = real_docs if real_docs else [
        s for s in group.documents if s.investigator_output is not None
    ]
    if not candidates:
        return group.documents[0]
    return max(candidates, key=lambda s: s.investigator_output.confidence)
```

**关键约束**：HD 阶段不做任何后处理；Excel 写入的是 JudgeAgent 原始输出。Phase 2 升级时只需删除 Orchestrator 里的 force-select TODO 块，LinguistAgent 一行代码不动。

---

## 六、多线程框架的复用方式

### 6.1 ThreadSafeLogger 和 ResultCollector 完全复用

两个类**原封不动**移动到 `utils/logging.py`，不修改任何接口。

| 当前代码 | 新架构 |
|---|---|
| `doc_index` = 一个文档组（6 篇）的全局顺序编号 | 不变；`group_index` 对应 `doc_index` |
| `process_document` 末尾调 `result_collector.add_result(doc_index, result)` | `_process_group_hd` 末尾调用，接口不变 |
| `detector.logger.flush_logs(doc_index)` | Orchestrator 每组末尾调用 |
| `ResultCollector._append_to_excel` 格式：`{id, text1, ..., text6}` | HD 阶段输出格式不变（legacy 命名） |

### 6.2 线程拓扑

```
Phase 1（行为等同原代码）：
  ThreadPoolExecutor（外层，per-group，n=MAX_WORKERS）
    └── _process_group_hd(group_index, doc, api_config, embedder, logger, collector)
         ├── 串行 Arbiter(text1..6)，FaiHal→直接标记，剩余→Investigator（原行为）
         └── 末尾 result_collector.add_result + logger.flush_logs

Phase 2（内层并行，串行 vs 并行对比数据写入 ablation）：
  ThreadPoolExecutor（外层，per-group）
    └── _process_group_hd(...)
         ├── inner_executor.submit(arbiter.run + investigator.run) × 6 （并行）
         ├── for each state: judge.run(state)
         └── result_collector.add_result + logger.flush_logs
```

### 6.3 API 客户端的线程安全

每个 group worker 创建自己的 Agent 实例（含独立 OpenAI client），与原代码模式一致：

```python
def _process_group_hd(group_index, doc, api_config, embedder, logger, collector):
    api_clients = make_api_clients(...)  # per-call，线程安全
    arbiter = ArbiterAgent(api_clients, logger)
    investigator = InvestigatorAgent(api_clients, logger, embedder)
    ...
```

---

## 七、阶段 1 具体改动清单

### 新增文件

| 文件 | 说明 | 预估行数 |
|---|---|---|
| `efi_pilot/utils/logging.py` | 原封不动复制 `ThreadSafeLogger` + `ResultCollector` | ~140 |
| `efi_pilot/utils/api_clients.py` | 三个客户端的初始化函数 + `bocha_search` | ~70 |
| `efi_pilot/utils/embedder.py` | 共享 SentenceTransformer 单例，保留原 fallback | ~30 |
| `efi_pilot/utils/naming_bridge.py` | 双向命名映射（输入/输出边界使用） | ~25 |
| `efi_pilot/agents/base.py` | 所有 dataclass + `BaseAgent` + 边界注释 | ~75 |
| `efi_pilot/agents/dispatcher.py` | stub | ~20 |
| `efi_pilot/agents/arbiter.py` | 封装 `detect_faithfulness_hallucination` | ~90 |
| `efi_pilot/agents/investigator.py` | 封装双路径 + 融合 | ~165 |
| `efi_pilot/agents/judge.py` | 三行优先级裁决 | ~30 |
| `efi_pilot/agents/linguist.py` | 封装 `judge_factuality`，三阶段结构就位 | ~90 |
| `efi_pilot/prompts/arbiter.py` | 原 prompt 字符串（`qwen_search.py:207-248`） | ~55 |
| `efi_pilot/prompts/investigator.py` | 两个 prompt 字符串（`qwen_search.py:348-478`） | ~145 |
| `efi_pilot/prompts/linguist.py` | 原 prompt 字符串含全部 7 个 few-shot（`predict_factuality.py:58-116`） | ~75 |
| `efi_pilot/orchestrator.py` | HD 阶段（ThreadPool）+ EFI 阶段（串行） | ~275 |
| `efi_pilot/evaluator.py` | 三个指标（含 Phase1 就实现的 Real Selection Acc） | ~200 |
| `efi_pilot/main.py` | CLI 入口 | ~130 |

### Phase1 中刻意保留的原有行为（sanity check 需要）

1. `random.Random(group_index + 42).shuffle(text_items)`（固定种子，线程安全）
2. FaiHal 文本不进 Investigator（orchestrator 串行路径）
3. "0 个 Real / ≥2 个 Real → force-select"在 Orchestrator HD 阶段保留，加 `# TODO: remove in Phase 2` 注释
4. LinguistAgent Phase1 只走 `len(real_docs) == 1` 分支

---

## 八、Phase 1 Sanity Check 标准

**必须满足，不达标不进 Phase 2**：

1. 取测试集前 20 个文档，分别用原 `qwen_search.py` + `predict_factuality.py` 和新 `efi_pilot` 各跑一遍
2. 对比每篇文本的 HD 标签（Real/Faithfulness/Factuality）
3. 对比每个文档的 EFI 标签（CT+/CT-/PS+/PS-/Uu）
4. 若有不一致，判断是 LLM 随机性（temperature 相同但 API 不保证确定性）还是代码逻辑差异
5. LLM 随机性造成的差异可接受；代码逻辑差异必须找到根因并修复
6. 若三轮跑仍有 ≥10% 差异，停下来排查，不进 Phase 2

**固定参数**：
- `temperature=0.3`（Arbiter / Investigator）/ `temperature=0.1`（Linguist）与原代码一致
- `random.Random(group_index + 42)` 固定 shuffle 顺序
- API 支持 `seed` 参数时固定 seed

---

## 九、评测指标（`evaluator.py`）

### 9.1 三个指标（Phase1 全部实现）

**指标 1：HD per-document F1**（与 Table 2 对比）
- Ground truth：从 JSON 结构推导（text1=Real, text2/3=FaiHal, text4/5/6=FacHal）
- Predicted：HD 输出 Excel（legacy → internal 转换）
- 指标：macro F1，三分类 {Real/FaiHal/FacHal}
- Robust 处理：缺失文本不计入，报告实际参与文本数 / 理论文本数

**指标 2：EFI Real Selection Accuracy**（新指标，Phase1 即可算基线）
- Gold：永远是 `"text1"`（数据集构造保证）
- Predicted：`LinguistOutput.chosen_text_key`（EFI 输出 Excel 的 `text_used` 列）
- Phase1 数值 = 原 DUPLEX "强制选最高置信度"的真实表现，是 Phase2/3 的对比基线

```python
def compute_real_selection_accuracy(predicted_col: str, gold_col: str = "text1") -> bool:
    return predicted_col == gold_col
```

**指标 3：EFI per-event F1**（与 Table 3 对比）
- Ground truth：JSON 的 `events.factuality1`（CT+/CT-/PS+/PS-）
- Predicted：EFI 输出 Excel 的 `predict_factuality1`（含 Uu/UU）
- 指标：macro F1，五分类 {CT+/CT-/PS+/PS-/Uu}
- Robust 处理：缺失 event 不计入

### 9.2 评测脚本输出格式

```
=== HD per-document F1 ===
参与评测: N_actual / N_total 篇文本
              precision  recall  f1-score
Real          ...
FaiHal        ...
FacHal        ...
macro avg     ...

=== EFI Real Selection Accuracy ===
选中 text1 的比例: X/N (XX.X%)

=== EFI per-event F1 ===
参与评测: N 个事件
              precision  recall  f1-score
CT+           ...
CT-           ...
PS+           ...
PS-           ...
Uu            ...
macro avg     ...
```

---

## 十、两阶段数据流设计

**Phase 1-2**：文件解耦，两个独立运行模式：

```
--mode hd   → 输入 JSON → HD 处理 → 输出 HD Excel（legacy 命名）
--mode efi  → 输入 HD Excel + JSON → EFI 处理 → 输出 EFI Excel
--mode eval → 输入 HD Excel + EFI Excel + JSON → 输出三张评测表
```

Phase 2 的 `load_hd_results` 升级为返回所有标签（`Dict[str, Dict[str, str]]`），不只是第一个 Real，使 Phase 2 的多 Real 情况能正确传递给 LinguistAgent。

**注意**：Phase 2 EFI 的 V_fac 兜底需要 `investigator_output.confidence`，但从 Excel 重建时 AgentState 没有 investigator_output。Phase 2 需要在 HD Excel 中额外保存 V_fac 分数列，或改为内存串联模式。此问题记录为 Phase 2 待解决。

---

## 十一、不要动的设计决定

1. 原 `qwen_search.py` 和 `predict_factuality.py` 永远保留为基线
2. 不引入 LangGraph
3. ThreadSafeLogger 和 ResultCollector 接口完全复用
4. 每个 Agent 的 prompt 放在独立的 `prompts/*.py` 文件中
5. Phase 1 所有 prompt 一字不改
6. Phase 2 双路径串行→并行的性能对比数据写入 ablation 附录
