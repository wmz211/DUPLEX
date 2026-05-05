# 命名映射：输入边界(legacy→internal)和输出边界(internal→legacy)
# 内部代码(agents/*)只能见到 internal 命名
# 转换只发生在 orchestrator 的输入/输出边界处

_TO_INTERNAL = {
    "Faithfulness": "FaiHal",
    "Factuality":   "FacHal",
    "Real":         "Real",
    "UU":           "Uu",
    "Uu":           "Uu",
}

_TO_LEGACY = {
    "FaiHal": "Faithfulness",
    "FacHal": "Factuality",
    "Real":   "Real",
    "Uu":     "UU",
}

# EFI 标签的 internal→legacy（CT+/CT-/PS+/PS- 不变，Uu→UU）
_EFI_TO_LEGACY = {
    "CT+": "CT+",
    "CT-": "CT-",
    "PS+": "PS+",
    "PS-": "PS-",
    "Uu":  "UU",
}

_EFI_TO_INTERNAL = {
    "CT+": "CT+",
    "CT-": "CT-",
    "PS+": "PS+",
    "PS-": "PS-",
    "UU":  "Uu",
    "Uu":  "Uu",
}


def to_internal(label: str) -> str:
    """输入边界：legacy HD 标签 → internal"""
    return _TO_INTERNAL.get(label, label)


def to_legacy(label: str) -> str:
    """输出边界：internal HD 标签 → legacy（用于写 Excel）"""
    return _TO_LEGACY.get(label, label)


def efi_to_legacy(label: str) -> str:
    """输出边界：internal EFI 标签 → legacy（用于写 Excel）"""
    return _EFI_TO_LEGACY.get(label, label)


def efi_to_internal(label: str) -> str:
    """输入边界：legacy EFI 标签 → internal"""
    return _EFI_TO_INTERNAL.get(label, label)
