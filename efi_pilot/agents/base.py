"""数据结构定义和 BaseAgent 抽象类。

层次说明：
  AgentState : HD 阶段使用，per-document（每篇文本一个实例）
  GroupState : EFI 阶段使用，per-group（聚合 6 个 AgentState）
  Orchestrator 在 HD→EFI 过渡处做 List[AgentState] → GroupState 的聚合。

命名约定（内部）：
  hd_label  : "Real" | "FaiHal" | "FacHal"
  efi_label : "CT+" | "CT-" | "PS+" | "PS-" | "Uu"
  输入/输出边界由 utils/naming_bridge.py 负责转换，agents 内部只见内部命名。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ArbiterOutput:
    # Phase1: raw_label ∈ {"Real", "Faithfulness"}（保持原 prompt 输出词）
    # Phase2: raw_label ∈ {"NoContradiction","GenuineContradiction",
    #                       "PerspectiveDiversity","TemporalUpdate"}
    raw_label: str
    mapped_label: str      # "IsRW" | "IsFaiHal"
    confidence: float      # 0-100
    evidence: str
    reasoning: Optional[str] = None   # Phase2 三角辩论推理过程


@dataclass
class InvestigatorOutput:
    label: str             # "IsRW" | "IsFacHal"
    confidence: float      # 0-100（V_fac 融合分，也是 §五 的选 Real 依据）
    evidence: str
    semantic_score: float
    qwen_label: str
    qwen_confidence: float
    iterations: int = 1    # Phase2 迭代次数（Phase1 固定为 1）


@dataclass
class AgentState:
    """HD 阶段单篇文本的完整状态。"""
    doc_id: str
    group_index: int       # 对应 ThreadSafeLogger/ResultCollector 的 doc_index
    text_key: str          # "text1" ~ "text6"
    text: str
    event: str             # 所属组的 event1 描述
    claim_type: Optional[str] = None
    arbiter_output: Optional[ArbiterOutput] = None
    investigator_output: Optional[InvestigatorOutput] = None
    hd_label: Optional[str] = None    # "Real" | "FaiHal" | "FacHal"


@dataclass
class LinguistOutput:
    chosen_text_key: str   # EFI 内部选出的文本（不回写 HD 标签）
    chosen_text: str
    selection_reason: str  # 写日志，便于 case study
    efi_label: str         # "CT+" | "CT-" | "PS+" | "PS-" | "Uu"
    modal_analysis: Optional[str] = None   # Phase3 模态链推理过程


@dataclass
class GroupState:
    """EFI 阶段 6 篇文本一组的完整状态。"""
    doc_id: str
    group_index: int
    event: str
    ground_truth_factuality: str           # events.factuality1，评测用
    documents: List[AgentState] = field(default_factory=list)
    linguist_output: Optional[LinguistOutput] = None


class BaseAgent(ABC):
    def __init__(self, api_clients: dict, logger):
        self.clients = api_clients
        self.logger = logger

    def _log(self, msg: str, group_index: int = None, print_console: bool = True):
        self.logger.log(msg, doc_index=group_index, print_console=print_console)

    @abstractmethod
    def run(self, state):
        """接收状态，填充并返回。"""
        ...
