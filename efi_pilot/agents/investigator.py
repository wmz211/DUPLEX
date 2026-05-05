"""InvestigatorAgent —— 封装原 detect_factuality_hallucination（双路径串行+加权融合）。
Phase1: 单轮，串行调用，prompt 原封不动。
Phase2: 改为迭代循环 + reformulate_query + 双路径并行（同时记录串行 vs 并行性能）。
"""
import re
from sentence_transformers import util
from efi_pilot.agents.base import AgentState, InvestigatorOutput, BaseAgent
from efi_pilot.prompts.investigator import (
    build_semantic_query_prompt, build_qwen_verify_prompt, QWEN_VERIFY_SYSTEM
)
from efi_pilot.utils.api_clients import bocha_search


class InvestigatorAgent(BaseAgent):

    def __init__(self, api_clients: dict, logger, embedder):
        super().__init__(api_clients, logger)
        self.embedder = embedder

    def run(self, state: AgentState) -> AgentState:
        self._log(f"      🔍 Investigator 检测 {state.text_key}...", state.group_index)
        self._log(f"      ═══ 开始双路径串行验证 ═══", state.group_index)

        semantic_score, semantic_evidence = self._semantic_similarity_check(state.text, state.group_index)
        qwen_label, qwen_conf, qwen_evidence = self._qwen_search_verify(state.text, state.group_index)

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

        label = "IsRW" if final_score >= 55 else "IsFacHal"
        if final_score >= 55:
            evidence = f"综合验证: 事实基本准确 | 语义:{semantic_score:.1f}% Qwen:{qwen_conf:.1f}%"
        else:
            evidence = f"综合验证: 存在事实性问题 | 语义:{semantic_score:.1f}% Qwen:{qwen_conf:.1f}% | {qwen_evidence}"

        self._log(f"      ─────────────────────────", state.group_index)
        self._log(f"      路径1（语义）: {semantic_score:.1f}%", state.group_index)
        self._log(
            f"      路径2（Qwen）: {qwen_label} 置信度{qwen_conf:.1f}% → 真实性分数 {qwen_score:.1f}%",
            state.group_index,
        )
        self._log(
            f"      融合公式: {alpha}×{semantic_score:.1f} + {beta}×{qwen_score:.1f} = {final_score:.1f}%",
            state.group_index,
        )
        self._log(
            f"      最终判断: {label} (综合真实性分数: {final_score:.1f}%)",
            state.group_index,
        )
        self._log(f"      ═══════════════════════════", state.group_index)

        state.investigator_output = InvestigatorOutput(
            label=label,
            confidence=final_score,
            evidence=evidence,
            semantic_score=semantic_score,
            qwen_label=qwen_label,
            qwen_confidence=qwen_conf,
        )
        return state

    # ── 路径1：语义相似度 ────────────────────────────────────────────────

    def _semantic_similarity_check(self, text: str, group_index: int):
        self._log(f"      🔍 路径1: 语义相似度验证...", group_index)
        prompt = build_semantic_query_prompt(text)

        try:
            response = self.clients["deepseek"].chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=False,
            )
            content = response.choices[0].message.content.strip()
            search_query, key_facts = self._parse_semantic_query_response(content)
        except Exception as e:
            self._log(f"         ⚠️ LLM处理失败: {str(e)}", group_index)
            search_query = text[:50]
            key_facts = [text[:100]]

        self._log(f"         搜索查询: {search_query}", group_index)
        search_results = bocha_search(
            self.clients["bocha_key"], search_query, num_results=3,
            logger=self.logger, group_index=group_index,
        )

        if not search_results:
            self._log(f"         ⚠️ 无搜索结果", group_index)
            return 30.0, "无法获取搜索结果"

        self._log(f"         找到 {len(search_results)} 个结果", group_index)
        self._log(f"         提取 {len(key_facts)} 个关键事实", group_index)

        context_texts = [f"{r['title']} {r['snippet']}" for r in search_results]
        fact_emb = self.embedder.encode(key_facts, convert_to_tensor=True)
        ctx_emb  = self.embedder.encode(context_texts, convert_to_tensor=True)

        sims = util.cos_sim(fact_emb, ctx_emb)
        avg_sim = sims.max(dim=1).values.mean().item()
        score = avg_sim * 100

        self._log(f"         语义相似度得分: {score:.1f}%", group_index)
        return score, "语义验证完成"

    def _parse_semantic_query_response(self, content: str):
        search_query = ""
        key_facts = []
        for line in content.split('\n'):
            line = line.strip()
            if '搜索查询' in line:
                search_query = line.split('：', 1)[-1].split(':', 1)[-1].strip()
            elif '关键事实' in line:
                fact = line.split('：', 1)[-1].split(':', 1)[-1].strip()
                if fact:
                    key_facts.append(fact)
        if not search_query:
            search_query = ""
        if not key_facts:
            key_facts = []
        return search_query, key_facts

    # ── 路径2：Qwen 联网搜索 ─────────────────────────────────────────────

    def _qwen_search_verify(self, text: str, group_index: int):
        self._log(f"      🔍 路径2: Qwen联网搜索验证...", group_index)
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
            self._log(f"         ⚠️ Qwen验证失败: {str(e)}", group_index)
            import traceback
            self._log(f"         详细错误: {traceback.format_exc()}", group_index, print_console=False)
            label, confidence, evidence = "Real", 30.0, f"验证失败: {str(e)}"

        self._log(f"         Qwen判断: {label} (置信度: {confidence:.1f}%)", group_index)
        return label, confidence, evidence

    def _parse_qwen_response(self, content: str):
        label = "Real"
        confidence = 50.0
        evidence = "事实基本准确"
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
                if len(parts) == 2:
                    evidence = parts[1].strip()
                else:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        evidence = parts[1].strip()

        return label, confidence, evidence
