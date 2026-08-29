"""LinguistAgent —— EFI 阶段：选 Real + 事实性判断。

Step A: _select_real
  Phase1  → 取唯一 hd_label=="Real" 的文本（HD 的 force-select 保证恰好 1 个）
  Phase2+ → 0 或 ≥2 个 Real 时，取 investigator.confidence（V_fac）最高的
  Phase3  → 替换为 pairwise_compare_and_select（TODO 注释处）

Step B: _judge_factuality
  Phase1/2: 原 judge_factuality prompt（7 个 few-shot，含巴黎协定 PS- 例）
  Phase3:   改为显式模态链推理（修改 prompts/linguist.py）

关键约束：_select_real 的选择不回写 AgentState.hd_label，
          HD 输出已经报告完毕，选择仅为 EFI 内部预处理。
"""
from efi_pilot.agents.base import AgentState, GroupState, LinguistOutput, BaseAgent
from efi_pilot.prompts.linguist import build_linguist_prompt, LINGUIST_SYSTEM
from efi_pilot.utils.naming_bridge import efi_to_internal
from efi_pilot.config import QWEN_MODEL


class LinguistAgent(BaseAgent):

    def run(self, group: GroupState) -> GroupState:
        chosen = self._select_real(group)
        self._log(
            f"   Linguist 选定文本: {chosen.text_key} "
            f"(hd_label={chosen.hd_label}, selection_reason 见日志)",
            group.group_index,
        )

        efi_label = self._judge_factuality(chosen.text, group.event, group.group_index)

        group.linguist_output = LinguistOutput(
            chosen_text_key=chosen.text_key,
            chosen_text=chosen.text,
            selection_reason=self._make_reason(chosen, group),
            efi_label=efi_label,
        )
        return group

    # ── Step A: 选 Real ──────────────────────────────────────────────────

    def _select_real(self, group: GroupState) -> AgentState:
        real_docs = [s for s in group.documents if s.hd_label == "Real"]

        if len(real_docs) == 1:
            return real_docs[0]

        # Phase2+ 临时兜底：HD 去掉 force-select 后，此处处理 0 或 ≥2 Real
        # TODO Phase3: 替换为 pairwise_compare_and_select(group)
        if len(real_docs) == 0:
            # 0 个 Real：从有 investigator_output 的文本里选 V_fac 最高的
            candidates = [s for s in group.documents if s.investigator_output is not None]
            reason_prefix = "0-Real fallback"
        else:
            # ≥2 个 Real：取 V_fac 最高的
            candidates = real_docs
            reason_prefix = f"{len(real_docs)}-Real V_fac select"

        if not candidates:
            self._log(
                f"   ⚠️ Linguist: 无法确定 Real 候选，默认取第一篇",
                group.group_index,
            )
            return group.documents[0]

        chosen = max(candidates, key=lambda s: s.investigator_output.confidence if s.investigator_output else 0.0)
        self._log(
            f"   Linguist [{reason_prefix}]: 选 {chosen.text_key} "
            f"(V_fac={chosen.investigator_output.confidence:.1f}%)",
            group.group_index,
        )
        return chosen

    def _make_reason(self, chosen: AgentState, group: GroupState) -> str:
        real_count = sum(1 for s in group.documents if s.hd_label == "Real")
        if real_count == 1:
            return f"唯一 Real 文本 (hd_label=Real)"
        if real_count == 0:
            conf = chosen.investigator_output.confidence if chosen.investigator_output else "N/A"
            return f"0-Real fallback: V_fac 最高 ({conf:.1f}%)" if isinstance(conf, float) else f"0-Real fallback"
        conf = chosen.investigator_output.confidence if chosen.investigator_output else "N/A"
        return f"multi-Real: V_fac 最高 ({conf:.1f}%)" if isinstance(conf, float) else "multi-Real fallback"

    # ── Step B: 事实性判断 ───────────────────────────────────────────────

    def _judge_factuality(self, text: str, event: str, group_index: int) -> str:
        prompt = build_linguist_prompt(text, event)
        try:
            response = self.clients["qwen"].chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": LINGUIST_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
            )
            result = response.choices[0].message.content.strip()
            self._log(f"   Linguist EFI 原始返回: {result}", group_index, print_console=False)

            for label in ["CT+", "CT-", "PS+", "PS-", "UU"]:
                if label in result:
                    internal = efi_to_internal(label)
                    self._log(f"   Linguist EFI 判断: {internal}", group_index)
                    return internal

            self._log(f"   ⚠️ Linguist: 无有效标签，默认 Uu", group_index)
            return "Uu"

        except Exception as e:
            self._log(f"   ⚠️ Linguist EFI 调用失败: {str(e)}", group_index)
            return "Uu"
