"""ArbiterAgent —— 封装原 detect_faithfulness_hallucination。
Phase1: 二元输出（Real / Faithfulness），prompt 和解析逻辑原封不动。
Phase2: 替换为 4 类三角辩论 prompt（修改 prompts/arbiter.py 即可）。
"""
import re
from efi_pilot.agents.base import AgentState, ArbiterOutput, BaseAgent
from efi_pilot.prompts.arbiter import build_arbiter_prompt, ARBITER_SYSTEM


class ArbiterAgent(BaseAgent):

    def run(self, state: AgentState) -> AgentState:
        self._log(f"      🔍 Arbiter 检测 {state.text_key}...", state.group_index)
        prompt = build_arbiter_prompt(state.text)

        try:
            response = self.clients["deepseek"].chat.completions.create(
                model="deepseek-chat",
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

            raw_label, confidence, evidence = self._parse_response(content)

        except Exception as e:
            self._log(f"      ⚠️ Arbiter 调用失败: {str(e)}", state.group_index)
            raw_label, confidence, evidence = "Real", 50.0, f"检测失败: {str(e)}"

        # Phase1 映射：Faithfulness → IsFaiHal，Real → IsRW
        mapped = "IsFaiHal" if raw_label == "Faithfulness" else "IsRW"
        self._log(
            f"      Faithfulness分析: {raw_label} (置信度: {confidence:.1f}%)",
            state.group_index,
        )
        state.arbiter_output = ArbiterOutput(
            raw_label=raw_label,
            mapped_label=mapped,
            confidence=confidence,
            evidence=evidence,
        )
        return state

    def _parse_response(self, content: str):
        label = "Real"
        confidence = 50.0
        evidence = "无矛盾"
        confidence_parsed = False

        for line in content.split('\n'):
            line = line.strip()

            if '判断结果' in line:
                if 'Faithfulness' in line:
                    label = "Faithfulness"
                elif 'Real' in line:
                    label = "Real"

            elif line.startswith('置信度') and not confidence_parsed:
                try:
                    value_part = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
                    numbers = re.findall(r'\d+\.?\d*', value_part)
                    if numbers:
                        confidence = max(0.0, min(100.0, float(numbers[0])))
                        confidence_parsed = True
                except Exception:
                    pass

            elif '矛盾证据' in line or '问题说明' in line:
                parts = line.split('：', 1)
                if len(parts) == 2:
                    evidence = parts[1].strip()
                else:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        evidence = parts[1].strip()

        return label, confidence, evidence
