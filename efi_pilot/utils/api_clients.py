"""API 客户端初始化。每次调用 make_api_clients 创建新实例（per-thread 使用）。"""
import json
import requests
from openai import OpenAI

def make_qwen_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def make_api_clients(bocha_key: str, qwen_key: str) -> dict:
    """创建一组 API 客户端。per-thread 调用，不共享。"""
    return {
        "bocha_key": bocha_key,
        "qwen": make_qwen_client(qwen_key),
    }


def bocha_search(bocha_key: str, query: str, num_results: int = 3,
                 logger=None, group_index=None) -> list:
    """博查 API 搜索。返回结果列表，失败时返回空列表。"""
    url = "https://api.bochaai.com/v1/web-search"
    payload = json.dumps({"query": query, "summary": True, "count": num_results})
    headers = {
        'Authorization': f'Bearer {bocha_key}',
        'Content-Type': 'application/json',
    }
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = []
        if isinstance(data, dict) and 'data' in data:
            web_pages = data['data'].get('webPages', {}).get('value', [])
            for item in web_pages:
                results.append({
                    'title':       item.get('name', ''),
                    'snippet':     item.get('snippet', ''),
                    'link':        item.get('url', ''),
                    'displayLink': item.get('siteName', ''),
                })
        return results
    except Exception as e:
        if logger:
            logger.log(f"      ⚠️ 博查搜索失败: {str(e)}", doc_index=group_index)
        return []
