"""DispatcherAgent —— Phase2: claim-type 分类。

四类：numerical / event / entity / temporal
claim_type 写入 AgentState，供后续 Investigator 日志和 ablation 分析使用。
"""
from efi_pilot.agents.base import AgentState, BaseAgent
from efi_pilot.config import QWEN_MODEL

_CLAIM_TYPES = ("numerical", "event", "entity", "temporal")

_PROMPT = """请判断以下新闻文本的主要声明类型，从四类中选一个：
- numerical：核心声明涉及具体数字、统计、比例
- event：核心声明描述一个事件或行为
- entity：核心声明涉及特定机构、人物、组织
- temporal：核心声明涉及时间、日期、顺序

新闻文本：
{text}

只回答一个词（numerical / event / entity / temporal），不要解释。"""


class DispatcherAgent(BaseAgent):

    def run(self, state: AgentState) -> AgentState:
        try:
            response = self.clients["qwen"].chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "user", "content": _PROMPT.format(text=state.text[:300])}],
                temperature=0.0,
                max_tokens=10,
                stream=False,
            )
            raw = response.choices[0].message.content.strip().lower()
            state.claim_type = next((t for t in _CLAIM_TYPES if t in raw), "event")
        except Exception:
            state.claim_type = "event"

        self._log(
            f"      Dispatcher: {state.text_key} → claim_type={state.claim_type}",
            state.group_index,
        )
        return state
