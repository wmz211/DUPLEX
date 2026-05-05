"""共享 SentenceTransformer 单例（线程安全）。"""
import threading

_embedder = None
_lock = threading.Lock()


def get_shared_embedder(logger=None):
    """返回全局共享的 embedder 实例，第一次调用时加载（线程安全）。"""
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                try:
                    _embedder = SentenceTransformer('sentence-transformers/all-MPNet-base-v2')
                    if logger:
                        logger.log("OK 成功加载语义模型 all-MPNet-base-v2", doc_index=None)
                except Exception:
                    _embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                    if logger:
                        logger.log("OK 加载备用语义模型 all-MiniLM-L6-v2", doc_index=None)
    return _embedder
