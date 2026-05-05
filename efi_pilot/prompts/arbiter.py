"""Arbiter prompt —— Phase2: 四分类三角辩论（取代 Phase1 二分类）。

四类输出：
  GenuineContradiction  → IsFaiHal（真实内部自相矛盾，跳过 Investigator）
  NoContradiction       → IsRW（无矛盾，进 Investigator）
  PerspectiveDiversity  → IsRW（多角度陈述，进 Investigator）
  TemporalUpdate        → IsRW（时序更新，进 Investigator）

关键修正：文本内容与外部事实不符 ≠ 内部矛盾，不应归为 GenuineContradiction。
"""

ARBITER_SYSTEM = (
    "你是文本一致性分析法庭的法官。"
    "你的核心任务是判断文本内部句子之间是否存在直接的逻辑矛盾，"
    "而非判断文本与外部现实的符合程度。"
)


def build_arbiter_prompt(text: str) -> str:
    return f"""你是文本逻辑一致性分析法庭的法官。请通过三方辩论分析以下新闻文本。

新闻文本：
{text}

**三方辩论（内部推理，不输出过程）：**
1. 检察官：在文本内部寻找任何句子之间的直接矛盾
2. 辩护人：对指控提出合理解释（不同角度、时序更新、补充关系等）
3. 法官：综合双方，给出以下四类判断之一

**四类判断标准：**
- GenuineContradiction：文本内部存在无法调和的直接自相矛盾
  （同一文本中"A发生了"与"A未发生"同时出现）
- PerspectiveDiversity：不同陈述来自不同来源或视角，是合理的多元观点，非矛盾
- TemporalUpdate：较新陈述更新了较早陈述，是时间线上的信息演进，非矛盾
- NoContradiction：文本内部逻辑一致，无任何矛盾

**判断边界（必须遵守）：**
- 文本内容与外部已知事实不符 → 不是内部矛盾 → 不能判 GenuineContradiction
- 捏造了不存在的机构、协议、术语 → 不是内部矛盾 → 不能判 GenuineContradiction
- 数字、日期、人名有误 → 不是内部矛盾 → 不能判 GenuineContradiction
- 只有文本自身句子之间直接冲突才能判 GenuineContradiction

**示例：**

例1（GenuineContradiction）：
"这款汽车能在沼泽、沙滩、浅海等多种地形行驶...不过由于技术限制，这款汽车实际上只能在平坦路面上行驶，无法适应复杂地形。"
→ GenuineContradiction（"能在多种地形"与"只能在平坦路面"直接矛盾）

例2（NoContradiction，外部事实错误但内部一致）：
"特朗普宣布启动'跨太平洋秘密外交协议'（TPSDE），以整合东亚情报网络。蒂勒森随后确认已收到通知并于3月15日正式离职。"
→ NoContradiction（TPSDE不存在是外部事实错误，文本自身逻辑一致）

例3（PerspectiveDiversity）：
"美方称此次谈判取得积极进展。俄方则表示分歧仍然存在，需要进一步磋商。"
→ PerspectiveDiversity（双方各自表述立场，非矛盾）

例4（TemporalUpdate）：
"事故发生时报道称有3人死亡。经过进一步搜救，最终确认死亡人数上升至7人。"
→ TemporalUpdate（后续数字更新了初始报道，非矛盾）

**置信度定义（0-100）：**
- 非常确定 → 85-95
- 较为确定 → 70-84
- 有一定把握 → 55-69

请严格按以下格式回答（每项占一行）：
判断结果：[GenuineContradiction/PerspectiveDiversity/TemporalUpdate/NoContradiction]
置信度：[0-100的数字]
矛盾证据：[GenuineContradiction时列出矛盾句对；其他类型写明判断依据]"""
