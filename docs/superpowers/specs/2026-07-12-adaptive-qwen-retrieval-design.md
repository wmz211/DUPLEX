# Qwen 自适应检索循环设计

## 目标

在不对长新闻逐句原子化、也不强制所有样本执行两轮搜索的前提下，提高事实性幻觉检测效果。保留现有双路径架构，但把迭代检索从 Bocha 语义路径迁移到 Qwen 事实核查路径。

检索循环只改变 Qwen 可使用的证据上下文，不改变现有 `Real`/`Factuality` 标签定义、置信度评分标准和校准融合公式。

## 现有运行结果提供的依据

最新 `dev50_rerun` 中共有 179 个进入 Investigator 的样本。若暂以 `semantic_score >= 55` 表示 Bocha 语义路径偏向 `Real`，则：

- 54 个样本路径不冲突，最终判断正确；
- 115 个样本路径冲突，但最终判断正确；
- 10 个样本路径冲突，且最终判断错误；
- 没有错误样本出现在路径不冲突的情况中。

10 个错误样本的 Qwen 置信度都很高。其中 9 个真实文本被 Qwen 以 82–92 的置信度误判为 `Factuality`；另 1 个事实性幻觉被 Qwen 以 90 的置信度误判为 `Real`。因此：

- 路径冲突占样本的 69.8%，单独以冲突触发二轮会造成过高成本；
- 低置信度无法识别这些已知错误；
- 是否进入二轮应取决于 Qwen 证据是否充分、可审计，而不能只依赖置信度或路径冲突。

## 总体架构

### Bocha 语义路径

Bocha 改为固定单轮：

1. DeepSeek 生成一个搜索 query 和三个关键事实，保留现有提示词行为；
2. Bocha 返回前三条结果；
3. 使用 SentenceTransformer 余弦相似度比较关键事实与每条结果的标题、摘要；
4. 输出语义相关性、覆盖率、query、关键事实和搜索结果元数据。

该路径不判断证据是支持还是反驳新闻，不输出真假结论，不改写 query，也不执行循环。它的分数继续作为校准融合中的独立语义相关性先验。

### Qwen 事实核查路径

Qwen 使用 `enable_search=True` 执行一轮或两轮核查：

1. 第一轮自由核查完整新闻，由 Qwen 自行决定哪些信息关键，不固定核查点数量，也不做完整原子事实分解；
2. Qwen 按现有评分标准输出判断与置信度，同时输出使用的证据、可提供的来源、重要未验证信息、是否需要继续检索以及定向补充 query；
3. 确定性的证据充分性门控决定是否进入第二轮；
4. 第二轮接收原文和完整的第一轮状态，只检索缺失信息或复核第一轮提出的事实错误；
5. 第二轮综合保留的第一轮证据和新增证据，剔除被证伪的第一轮依据，再输出累计证据下的最终判断与置信度。

每篇文本最多调用 Qwen 两次。

### 证据充分性门控

门控不调用额外模型，也不直接判断新闻真假。满足任一条件时启动第二轮：

- `need_more_search` 为真；
- `unverified_points` 非空；
- Qwen 判断为 `Factuality`，但没有为声称的错误提供具体证据；
- 响应缺少必要的证据摘要；
- 结构化输出无法可靠解析。

置信度本身既不是循环触发条件，也不是停止条件。

## Qwen 数据协议

### 第一轮输出

Qwen 保持自由核查，但通过结构化外壳返回结果：

```json
{
  "verdict": "Real | Factuality",
  "confidence": 88,
  "evidence_summary": "第一轮判断使用的主要证据",
  "sources": ["可获得时填写来源名称或 URL"],
  "unverified_points": ["本轮未验证且可能影响最终判断的重要信息"],
  "need_more_search": true,
  "next_query": "针对最重要证据缺口的补充查询"
}
```

Qwen 自行判断哪些细节重要，不要求枚举长新闻中的全部原子事实。

### 第二轮输入

第二轮接收：

- 原始新闻文本；
- 第一轮判断与置信度；
- 第一轮证据摘要和来源；
- 第一轮未验证信息；
- 第一轮提出的补充 query。

提示词明确要求不要重新完整分析已经验证过的内容。

### 第二轮输出

第二轮输出累计证据下的综合评估：

```json
{
  "new_evidence": "第二轮新增证据",
  "retained_evidence": "仍然有效的第一轮证据",
  "rejected_evidence": "被第二轮纠正或剔除的第一轮证据",
  "remaining_unverified_points": [],
  "final_verdict": "Real | Factuality",
  "final_confidence": 87,
  "final_evidence_summary": "基于当前全部信息的综合判断"
}
```

两轮置信度不做机械算术平均。第二轮以累计证据 `E1 + E2` 重新判断，避免第一轮错误但高置信的结论阻碍有效纠正。

## 评分与融合

现有置信度定义保持不变。如果第二轮没有找到新证据，或者两轮后仍有重要信息未验证，Qwen 仍根据当前所有信息，按现有标准输出 `Real` 或 `Factuality` 及置信度。代码不自动降低置信度，也不赋予中性分数。

最终 Qwen 真实性分数保持为：

```python
qwen_truth_score = (
    confidence
    if verdict == "Real"
    else 100.0 - confidence
)
```

校准融合公式保持为：

```text
V_fac = alpha * S_semantic + (1 - alpha) * S_qwen
```

对于单轮样本，`S_qwen` 来自第一轮。对于二轮样本，`S_qwen` 来自第二轮基于累计证据输出的最终判断与置信度。`S_semantic` 始终来自唯一一次 Bocha 检索。

由于证据流发生变化，需要在开发集上重新校准 `alpha`、分类阈值和温度参数，不能直接沿用当前运行参数。

## 失败处理

- Bocha 失败时保留现有公共接口，并在日志中明确记录失败；实施计划需要在不改变标签和置信度语义的前提下确定兼容兜底；
- Qwen 第一轮失败或无法解析时，允许第二轮作为恢复性尝试；
- Qwen 第二轮失败时，直接回退第一轮有效的判断与置信度，不对其进行修改；
- 两轮 Qwen 均无有效结果时，保留现有技术失败兜底行为，并记录 `verification_failed`；
- API/解析失败与证据不足必须分别记录和处理。

## 状态与兼容性

保留 Judge、Excel 输出和评测代码正在使用的 `InvestigatorOutput` 公共字段：

- `label`
- `confidence`
- `evidence`
- `semantic_score`
- `qwen_label`
- `qwen_confidence`
- `iterations`

增加用于审计的可选内部状态：

- 第一轮 query 与证据；
- 第二轮 query 与证据；
- 未验证信息；
- 停止原因；
- 判断是否发生变化；
- 保留和剔除的证据；
- 可获得的来源引用。

现有下游标签和 EFI 行为保持不变。

## 实施范围

实施仅限 Investigator 相关边界：

- `efi_pilot/agents/investigator.py`
  - 将 Bocha 改为单轮；
  - 将自适应循环迁移至 Qwen；
  - 增加证据充分性门控和累计轮次状态；
  - 删除 Bocha query 改写和基于跨轮 coverage 的结果选择。
- `efi_pilot/prompts/investigator.py`
  - 保留现有 Bocha query 提示词；
  - 扩展 Qwen 第一轮输出协议；
  - 新增基于累计证据的 Qwen 第二轮提示词。
- `efi_pilot/agents/base.py`
  - 只增加新循环需要的可选审计字段，保持现有使用方兼容。
- `efi_pilot/orchestrator.py`
  - 记录两轮 query、证据、未验证信息、停止原因和判断变化。

Arbiter、Judge、Linguist、EFI 标签以及外部三分类 HD 接口不在本次修改范围内。

## 验证方案

### 控制流验证

验证以下行为：

- 第一轮证据充分时只调用一次 Qwen；
- 第一轮证据不足时恰好调用两次 Qwen；
- 每次 Investigator 运行只调用一次 Bocha；
- Qwen 调用次数绝不超过两次；
- 第二轮失败时回退未经修改的第一轮有效结果；
- 第二轮有效时基于累计证据判断，而不是只使用新增证据。

### 已知错误案例

重新运行 `investigator_dev50_rerun_scores.csv` 中的 10 个路径冲突错误：

- 检查 9 个被误判为 `Factuality` 的真实文本，能否通过复核第一轮错误指控得到纠正；
- 检查 `CD36/text4` 能否通过补查未验证信息识别为 FacHal；
- 确认修正来自累计证据，而不是无依据反转标签。

### 回归案例

从 115 个“路径冲突但判断正确”的样本中抽样，验证循环不会频繁把正确的 `Factuality` 判断反转为 `Real`。

### 消融实验

比较以下方案：

1. Current：当前 Bocha 自适应多轮，加 Qwen 重复独立核查；
2. One-shot：Bocha 单轮，加 Qwen 单轮；
3. Proposed：Bocha 单轮，加证据充分性驱动的 Qwen 一至两轮；
4. Fixed-2：Bocha 单轮，所有样本固定执行两轮 Qwen。

报告：

- Real F1、FacHal F1 和 Macro-F1；
- 第二轮触发率；
- 每篇文本平均 Qwen 调用次数；
- 平均耗时；
- 判断变化次数；
- 正确修正数和有害反转数。

## 成功标准

- Macro-F1 不低于当前校准系统；
- 新循环能够修正 10 个已知错误中的一部分；
- 正确修正数多于有害反转数；
- 平均 Qwen 调用次数明显低于固定两轮方案；
- 每篇文本只调用一次 Bocha；
- 现有 Judge、Excel 和 evaluator 使用方保持兼容。
