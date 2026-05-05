"""data_io.py —— 数据加载与 IO 工具函数（orchestrator 和 evaluator 共用）。"""
import os
import json
import pandas as pd
from efi_pilot.utils.naming_bridge import to_internal


def load_json(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def get_text_from_doc(doc: dict, text_key: str) -> str:
    """按 text_key 从 JSON 文档中取文本内容。"""
    if text_key == 'text1':
        return doc.get('text1', '')
    if text_key in ('text2', 'text3'):
        return doc.get('hallu_type', {}).get('faithfulness', {}).get(text_key, '')
    if text_key in ('text4', 'text5', 'text6'):
        return doc.get('hallu_type', {}).get('factuality', {}).get(text_key, '')
    return ''


def gather_text_items(doc: dict) -> list:
    """按 text1~text6 顺序收集文档中存在的所有文本项。"""
    items = []
    if doc.get('text1'):
        items.append(('text1', doc['text1']))
    faith = doc.get('hallu_type', {}).get('faithfulness', {})
    for k in ('text2', 'text3'):
        if faith.get(k):
            items.append((k, faith[k]))
    fact = doc.get('hallu_type', {}).get('factuality', {})
    for k in ('text4', 'text5', 'text6'):
        if fact.get(k):
            items.append((k, fact[k]))
    return items


def filter_docs(documents: list, start_id=None, end_id=None) -> list:
    """按 start_id / end_id 过滤文档列表，返回 [(orig_index, doc), ...]。"""
    start_processing = (start_id is None)
    result = []
    should_stop = False
    for idx, doc in enumerate(documents):
        doc_id = doc.get('id', 'Unknown')
        if end_id and doc_id == end_id:
            should_stop = True
        if not start_processing:
            if doc_id == start_id:
                start_processing = True
            else:
                continue
        result.append((idx, doc))
        if should_stop:
            break
    return result


def load_hd_results(excel_file: str) -> dict:
    """读 HD 输出 Excel，返回 {doc_id: {text_key: internal_label}}。
    输入边界：legacy 标签 → internal（via naming_bridge.to_internal）。
    """
    df = pd.read_excel(excel_file, engine='openpyxl')
    text_cols = [c for c in df.columns if c.startswith('text') and c[4:].isdigit()]
    results = {}
    for _, row in df.iterrows():
        doc_id = str(row['id'])
        labels = {}
        for col in text_cols:
            if pd.notna(row[col]) and str(row[col]).strip():
                labels[col] = to_internal(str(row[col]).strip())
        results[doc_id] = labels
    return results


def load_hd_results_for_efi(excel_file: str) -> tuple[dict, dict]:
    results = load_hd_results(excel_file)
    df = pd.read_excel(excel_file, engine='openpyxl')
    efi_texts = {}
    if 'efi_text' not in df.columns:
        return results, efi_texts

    for _, row in df.iterrows():
        doc_id = str(row['id'])
        if pd.notna(row['efi_text']) and str(row['efi_text']).strip():
            efi_texts[doc_id] = str(row['efi_text']).strip()
    return results, efi_texts


def append_to_excel(output_file: str, result: dict, logger=None):
    """将单条结果追加到 Excel（read-then-write 模式）。"""
    try:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        if os.path.exists(output_file):
            df = pd.concat(
                [pd.read_excel(output_file, engine='openpyxl'), pd.DataFrame([result])],
                ignore_index=True,
            )
        else:
            df = pd.DataFrame([result])
        df.to_excel(output_file, index=False, engine='openpyxl')
    except Exception as e:
        msg = f"保存 Excel 失败: {e}"
        if logger:
            logger.error(msg)
        else:
            print(f"❌ {msg}")
