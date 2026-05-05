"""ArbiterAgent —— Phase2: 四分类三角辩论。
只有 GenuineContradiction → IsFaiHal，其余三类 → IsRW（进 Investigator）。
"""
import re
from efi_pilot.agents.base import AgentState, ArbiterOutput, BaseAgent
from efi_pilot.prompts.arbiter import build_arbiter_prompt, ARBITER_SYSTEM

_LABEL_MAP = {
    "GenuineContradiction": "IsFaiHal",
    "PerspectiveDiversity":  "IsRW",
    "TemporalUpdate":        "IsRW",
    "NoContradiction":       "IsRW",
}


class ArbiterAgent(BaseAgent):

    def run(self, state: AgentState) -> AgentState:
        self._log(f"      Arbiter 检测 {state.text_key}...", state.group_index)
        prompt = build_arbiter_prompt(state.text)

        try:
            response = self.clients["deepseek"].chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": ARBITER_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                stream=False,
            )
            content = response.choices[0].message.content.strip()
            self._log(f"      [DEBUG] Arbiter 原始返回:\n{content}",
                      state.group_index, print_console=False)
            raw_label, confidence, evidence, reasoning = self._parse_response(content)

        except Exception as e:
            self._log(f"      Arbiter 调用失败: {str(e)}", state.group_index)
            raw_label, confidence, evidence, reasoning = "NoContradiction", 50.0, f"检测失败: {str(e)}", None

        mapped = _LABEL_MAP.get(raw_label, "IsRW")
        self._log(
            f"      Arbiter: {raw_label} → {mapped} (置信度: {confidence:.1f}%)",
            state.group_index,
        )
        state.arbiter_output = ArbiterOutput(
            raw_label=raw_label,
            mapped_label=mapped,
            confidence=confidence,
            evidence=evidence,
            reasoning=reasoning,
        )
        return state

    def _parse_response(self, content: str):
        label = "NoContradiction"
        confidence = 50.0
        evidence = ""
        sentence_a = ""
        sentence_b = ""
        confidence_parsed = False

        for line in content.split('\n'):
            line = line.strip()

            if '判断结果' in line:
                for key in _LABEL_MAP:
                    if key in line:
                        label = key
                        break

            elif line.startswith('置信度') and not confidence_parsed:
                try:
                    value_part = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
                    numbers = re.findall(r'\d+\.?\d*', value_part)
                    if numbers:
                        confidence = max(0.0, min(100.0, float(numbers[0])))
                        confidence_parsed = True
                except Exception:
                    pass

            elif '矛盾句A' in line:
                parts = line.split('：', 1)
                if len(parts) == 2:
                    sentence_a = parts[1].strip()

            elif '矛盾句B' in line:
                parts = line.split('：', 1)
                if len(parts) == 2:
                    sentence_b = parts[1].strip()

            elif '为何不能同时为真' in line or '判断依据' in line or '矛盾证据' in line:
                parts = line.split('：', 1)
                if len(parts) == 2:
                    evidence = parts[1].strip()

        # GenuineContradiction 必须有矛盾句对，否则降级
        if label == "GenuineContradiction" and not (sentence_a and sentence_b):
            self._log(
                f"      [Arbiter] GenuineContradiction 但未提供矛盾句对，降级为 NoContradiction",
                print_console=False,
            )
            label = "NoContradiction"

        if sentence_a and sentence_b:
            evidence = f"A:「{sentence_a}」 B:「{sentence_b}」"

        return label, confidence, evidence, None
