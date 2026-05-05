"""InvestigatorAgent —— Phase2: 迭代双路径验证。

Phase1: 单轮，串行调用。
Phase2: 第1轮结束后若 final_score 落在不确定区间 [45, 65]，
        用 reformulate_query 换角度搜索，最多跑 MAX_ITERATIONS 轮，
        取置信度最高的一轮作为最终结果。
"""
import re
from sentence_transformers import util
from efi_pilot.agents.base import AgentState, InvestigatorOutput, BaseAgent
from efi_pilot.prompts.investigator import (
    build_semantic_query_prompt,
    build_reformulate_prompt,
    build_qwen_verify_prompt,
    QWEN_VERIFY_SYSTEM,
)
from efi_pilot.utils.api_clients import bocha_search

MAX_ITERATIONS = 2
UNCERTAIN_LOW  = 45.0
UNCERTAIN_HIGH = 65.0


class InvestigatorAgent(BaseAgent):

    def __init__(self, api_clients: dict, logger, embedder):
        super().__init__(api_clients, logger)
        self.embedder = embedder

    def run(self, state: AgentState) -> AgentState:
        self._log(f"      Investigator 检测 {state.text_key}...", state.group_index)

        best_result = None
        prev_query  = ""

        for iteration in range(1, MAX_ITERATIONS + 1):
            self._log(f"      === 第{iteration}轮双路径验证 ===", state.group_index)

            if iteration == 1:
                semantic_score, semantic_evidence, search_query = \
                    self._semantic_similarity_check(state.text, state.group_index)
            else:
                semantic_score, semantic_evidence, search_query = \
                    self._semantic_similarity_check_reformulate(
                        state.text, state.group_index, prev_query, best_result["final_score"]
                    )

            qwen_label, qwen_conf, qwen_evidence = \
                self._qwen_search_verify(state.text, state.group_index)

            qwen_score = qwen_conf if qwen_label == "Real" else (100 - qwen_conf)

            if qwen_conf >= 85:
                alpha, beta = 0.35, 0.65
            elif qwen_conf >= 70:
                alpha, beta = 0.40, 0.60
            elif qwen_conf >= 55:
                alpha, beta = 0.45, 0.55
            else:
                alpha, beta = 0.60, 0.40

            final_score = max(0.0, min(100.0, alpha * semantic_score + beta * qwen_score))

            if final_score >= 55:
                evidence = f"综合验证: 事实基本准确 | 语义:{semantic_score:.1f}% Qwen:{qwen_conf:.1f}%"
            else:
                evidence = f"综合验证: 存在事实性问题 | 语义:{semantic_score:.1f}% Qwen:{qwen_conf:.1f}% | {qwen_evidence}"

            self._log(f"      路径1（语义）: {semantic_score:.1f}%", state.group_index)
            self._log(
                f"      路径2（Qwen）: {qwen_label} {qwen_conf:.1f}% → 真实性分数 {qwen_score:.1f}%",
                state.group_index,
            )
            self._log(
                f"      融合: {alpha}×{semantic_score:.1f} + {beta}×{qwen_score:.1f} = {final_score:.1f}%",
                state.group_index,
            )

            current = dict(
                final_score=final_score, label="IsRW" if final_score >= 55 else "IsFacHal",
                evidence=evidence, semantic_score=semantic_score,
                qwen_label=qwen_label, qwen_conf=qwen_conf, iteration=iteration,
            )

            if best_result is None or abs(final_score - 50) > abs(best_result["final_score"] - 50):
                best_result = current

            prev_query = search_query

            in_uncertain = UNCERTAIN_LOW <= final_score <= UNCERTAIN_HIGH
            if iteration < MAX_ITERATIONS and in_uncertain:
                self._log(
                    f"      分数 {final_score:.1f}% 在不确定区间，启动第{iteration+1}轮改写搜索",
                    state.group_index,
                )
            else:
                break

        self._log(
            f"      最终判断: {best_result['label']} (V_fac={best_result['final_score']:.1f}%, 轮数={best_result['iteration']})",
            state.group_index,
        )

        state.investigator_output = InvestigatorOutput(
            label=best_result["label"],
            confidence=best_result["final_score"],
            evidence=best_result["evidence"],
            semantic_score=best_result["semantic_score"],
            qwen_label=best_result["qwen_label"],
            qwen_confidence=best_result["qwen_conf"],
            iterations=best_result["iteration"],
        )
        return state

    # ── 路径1：语义相似度（第1轮）────────────────────────────────────────

    def _semantic_similarity_check(self, text: str, group_index: int):
        prompt = build_semantic_query_prompt(text)
        return self._run_semantic(prompt, text, group_index)

    def _semantic_similarity_check_reformulate(
        self, text: str, group_index: int, prev_query: str, prev_score: float
    ):
        prompt = build_reformulate_prompt(text, prev_query, prev_score)
        return self._run_semantic(prompt, text, group_index)

    def _run_semantic(self, prompt: str, text: str, group_index: int):
        self._log(f"      路径1: 语义相似度验证...", group_index)
        try:
            response = self.clients["deepseek"].chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=False,
            )
            content = response.choices[0].message.content.strip()
            search_query, key_facts = self._parse_semantic_query_response(content)
        except Exception as e:
            self._log(f"         LLM处理失败: {str(e)}", group_index)
            search_query = text[:50]
            key_facts = [text[:100]]

        self._log(f"         搜索查询: {search_query}", group_index)
        search_results = bocha_search(
            self.clients["bocha_key"], search_query, num_results=3,
            logger=self.logger, group_index=group_index,
        )

        if not search_results:
            self._log(f"         无搜索结果", group_index)
            return 30.0, "无法获取搜索结果", search_query

        context_texts = [f"{r['title']} {r['snippet']}" for r in search_results]
        fact_emb = self.embedder.encode(key_facts or [text[:100]], convert_to_tensor=True)
        ctx_emb  = self.embedder.encode(context_texts, convert_to_tensor=True)
        score = util.cos_sim(fact_emb, ctx_emb).max(dim=1).values.mean().item() * 100
        self._log(f"         语义相似度: {score:.1f}%", group_index)
        return score, "语义验证完成", search_query

    def _parse_semantic_query_response(self, content: str):
        search_query, key_facts = "", []
        for line in content.split('\n'):
            line = line.strip()
            if '搜索查询' in line:
                search_query = line.split('：', 1)[-1].split(':', 1)[-1].strip()
            elif '关键事实' in line:
                fact = line.split('：', 1)[-1].split(':', 1)[-1].strip()
                if fact:
                    key_facts.append(fact)
        return search_query, key_facts

    # ── 路径2：Qwen 联网搜索 ─────────────────────────────────────────────

    def _qwen_search_verify(self, text: str, group_index: int):
        self._log(f"      路径2: Qwen联网搜索验证...", group_index)
        prompt = build_qwen_verify_prompt(text)
        try:
            response = self.clients["qwen"].chat.completions.create(
                model="qwen3.6-flash",
                messages=[
                    {"role": "system", "content": QWEN_VERIFY_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                extra_body={"enable_search": True},
            )
            content = response.choices[0].message.content.strip()
            self._log(f"         [DEBUG] Qwen返回:\n{content}", group_index, print_console=False)
            label, confidence, evidence = self._parse_qwen_response(content)
        except Exception as e:
            self._log(f"         Qwen验证失败: {str(e)}", group_index)
            import traceback
            self._log(traceback.format_exc(), group_index, print_console=False)
            label, confidence, evidence = "Real", 30.0, f"验证失败: {str(e)}"

        self._log(f"         Qwen判断: {label} (置信度: {confidence:.1f}%)", group_index)
        return label, confidence, evidence

    def _parse_qwen_response(self, content: str):
        label, confidence, evidence = "Real", 50.0, "事实基本准确"
        confidence_parsed = False
        for line in content.split('\n'):
            line = line.strip()
            if '判断结果' in line:
                if 'Factuality' in line:
                    label = "Factuality"
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
            elif '问题说明' in line:
                parts = line.split('：', 1)
                evidence = parts[1].strip() if len(parts) == 2 else line.split(':', 1)[-1].strip()
        return label, confidence, evidence
