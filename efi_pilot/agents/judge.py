"""JudgeAgent —— per-document HD 标签裁决，不做组内仲裁。

优先级：
  1. Arbiter 说 IsFaiHal  → FaiHal
  2. Investigator 说 IsFacHal → FacHal
  3. 否则                  → Real

Phase1: investigator_output 对 FaiHal 文本为 None（skip 逻辑在 Orchestrator），
        此时由 Arbiter 单独决定（步骤 1 已覆盖）。
"""
from efi_pilot.agents.base import AgentState, BaseAgent


class JudgeAgent(BaseAgent):

    def run(self, state: AgentState) -> AgentState:
        if state.arbiter_output.mapped_label == "IsFaiHal":
            state.hd_label = "FaiHal"
        elif (state.investigator_output is not None
              and state.investigator_output.label == "IsFacHal"):
            state.hd_label = "FacHal"
        else:
            state.hd_label = "Real"

        self._log(
            f"      Judge → {state.text_key}: {state.hd_label}",
            state.group_index,
        )
        return state
