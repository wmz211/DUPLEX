"""DispatcherAgent —— Phase1: stub，返回 claim_type='general'。
Phase2: 实现 4 分类（temporal / numeric / attribution / causal）。
"""
from efi_pilot.agents.base import AgentState, BaseAgent


class DispatcherAgent(BaseAgent):
    def run(self, state: AgentState) -> AgentState:
        # Phase1: stub
        # Phase2: 调用 LLM 对 state.text 做 4 分类，写入 state.claim_type
        state.claim_type = "general"
        return state
