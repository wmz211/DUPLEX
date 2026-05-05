"""evaluator.py —— 三个指标的计算。

指标 1: HD per-document F1（三分类：Real / FaiHal / FacHal）
指标 2: EFI Real Selection Accuracy（gold 永远是 text1）
指标 3: EFI per-event F1（五分类：CT+ / CT- / PS+ / PS- / Uu）

用法：
  python -m efi_pilot.main --mode eval \
      --hd-output res/hd_results.xlsx \
      --efi-output res/efi_results.xlsx \
      --json-cn hallu/extr-hallu-final.json \
      --json-en hallu/en_experiment_documents2.json
"""
import pandas as pd
from sklearn.metrics import classification_report
from efi_pilot.utils.naming_bridge import to_internal, efi_to_internal
from efi_pilot.utils.data_io import load_json, get_text_from_doc as _get_text_from_doc


# ── Ground truth 加载 ────────────────────────────────────────────────────

_HD_GOLD_MAP = {
    'text1': 'Real',
    'text2': 'FaiHal', 'text3': 'FaiHal',
    'text4': 'FacHal', 'text5': 'FacHal', 'text6': 'FacHal',
}


def build_doc_index(documents: list) -> dict:
    return {d['id']: d for d in documents}


# ── 指标 1：HD per-document F1 ───────────────────────────────────────────

def evaluate_hd(hd_excel: str, json_cn: str, json_en: str) -> dict:
    df = pd.read_excel(hd_excel, engine='openpyxl')
    text_cols = [c for c in df.columns if c.startswith('text') and c[4:].isdigit()]

    docs_cn = build_doc_index(load_json(json_cn))
    docs_en = build_doc_index(load_json(json_en))

    y_true, y_pred = [], []
    total_texts, evaluated_texts = 0, 0

    for _, row in df.iterrows():
        doc_id = str(row['id'])
        doc_index = docs_en if doc_id.startswith('ED') else docs_cn
        doc = doc_index.get(doc_id)

        for col in text_cols:
            gold_label = _HD_GOLD_MAP.get(col)
            if gold_label is None:
                continue
            total_texts += 1

            if pd.isna(row[col]) or str(row[col]).strip() == '':
                continue  # 缺失项不计入，不补零

            pred_label = to_internal(str(row[col]).strip())

            # 验证 gold：文档存在且对应文本列存在
            if doc is None:
                continue
            text = _get_text_from_doc(doc, col)
            if not text:
                continue  # 数据集中该列不存在，不计入

            y_true.append(gold_label)
            y_pred.append(pred_label)
            evaluated_texts += 1

    print(f"\n=== HD per-document F1 ===")
    print(f"参与评测: {evaluated_texts} / {total_texts} 篇文本")
    if not y_true:
        print("无有效数据")
        return {}

    labels = ['Real', 'FaiHal', 'FacHal']
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)
    return {"evaluated": evaluated_texts, "total": total_texts}


# ── 指标 2：EFI Real Selection Accuracy ─────────────────────────────────

def evaluate_real_selection(efi_excel: str) -> dict:
    df = pd.read_excel(efi_excel, engine='openpyxl')
    if 'text_used' not in df.columns:
        print("EFI Excel 缺少 'text_used' 列")
        return {}

    gold = 'text1'
    correct = (df['text_used'] == gold).sum()
    total   = df['text_used'].notna().sum()

    print(f"\n=== EFI Real Selection Accuracy ===")
    print(f"选中 text1 的比例: {correct}/{total} ({correct/total*100:.1f}%)")
    return {"correct": int(correct), "total": int(total),
            "accuracy": float(correct / total) if total > 0 else 0.0}


# ── 指标 3：EFI per-event F1 ─────────────────────────────────────────────

def evaluate_efi(efi_excel: str) -> dict:
    df = pd.read_excel(efi_excel, engine='openpyxl')

    required = {'event1_factuality', 'predict_factuality1'}
    if not required.issubset(df.columns):
        print(f"EFI Excel 缺少必要列: {required - set(df.columns)}")
        return {}

    rows = df[df['event1_factuality'].notna() & df['predict_factuality1'].notna()]
    y_true = [efi_to_internal(str(v).strip()) for v in rows['event1_factuality']]
    y_pred = [efi_to_internal(str(v).strip()) for v in rows['predict_factuality1']]

    total = len(df)
    evaluated = len(rows)

    print(f"\n=== EFI per-event F1 ===")
    print(f"参与评测: {evaluated} / {total} 个事件")
    if not y_true:
        print("无有效数据")
        return {}

    labels = ['CT+', 'CT-', 'PS+', 'PS-', 'Uu']
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)
    return {"evaluated": evaluated, "total": total}


# ── 汇总入口 ─────────────────────────────────────────────────────────────

def run_eval(hd_excel: str, efi_excel: str, json_cn: str, json_en: str):
    print("=" * 60)
    print("EFI-Pilot 评测报告")
    print("=" * 60)

    hd_stats  = evaluate_hd(hd_excel, json_cn, json_en)
    sel_stats = evaluate_real_selection(efi_excel)
    efi_stats = evaluate_efi(efi_excel)

    print("\n" + "=" * 60)
    print("摘要")
    print("=" * 60)
    if hd_stats:
        print(f"HD 评测覆盖率: {hd_stats['evaluated']}/{hd_stats['total']}")
    if sel_stats:
        print(f"Real 选择准确率: {sel_stats['accuracy']*100:.1f}%")
    if efi_stats:
        print(f"EFI 评测覆盖率: {efi_stats['evaluated']}/{efi_stats['total']}")


# _get_text_from_doc 已从 utils.data_io 导入
