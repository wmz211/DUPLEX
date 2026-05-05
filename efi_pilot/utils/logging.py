"""ThreadSafeLogger 和 ResultCollector —— 原封不动从 qwen_search.py 迁移。"""
import os
import threading
from queue import PriorityQueue
from datetime import datetime
import pandas as pd


class ThreadSafeLogger:
    """线程安全的日志记录器 - 支持按文档顺序输出或实时输出"""

    def __init__(self, log_file: str, realtime_mode: bool = False):
        self.log_file = log_file
        self.lock = threading.Lock()
        self.log_buffers = {}  # {doc_index: [log_messages]}
        self.next_doc_index = 0  # 下一个应该输出的文档索引
        self.realtime_mode = realtime_mode

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message: str, doc_index: int = None, print_console: bool = True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        if doc_index is None or self.realtime_mode:
            with self.lock:
                if print_console:
                    print(message)
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_message + '\n')
        else:
            with self.lock:
                if doc_index not in self.log_buffers:
                    self.log_buffers[doc_index] = []
                self.log_buffers[doc_index].append((log_message, message, print_console))

    def flush_logs(self, doc_index: int):
        with self.lock:
            while self.next_doc_index in self.log_buffers:
                logs = self.log_buffers.pop(self.next_doc_index)
                for log_message, display_message, print_console in logs:
                    if print_console:
                        print(display_message)
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(log_message + '\n')
                self.next_doc_index += 1


class ResultCollector:
    """结果收集器 - 保证按文档顺序输出"""

    def __init__(self, output_file: str, logger: ThreadSafeLogger):
        self.output_file = output_file
        self.logger = logger
        self.lock = threading.Lock()
        self.results_queue = PriorityQueue()
        self.next_index = 0
        self.completed_docs = {}

    def add_result(self, doc_index: int, result: dict):
        with self.lock:
            self.completed_docs[doc_index] = result
            self.logger.log(f"   [DEBUG] 文档#{doc_index} 结果已接收，尝试flush...",
                            doc_index=None, print_console=False)
            self._try_flush()

    def _try_flush(self):
        flushed_count = 0
        while self.next_index in self.completed_docs:
            result = self.completed_docs.pop(self.next_index)
            self.logger.log(
                f"   [DEBUG] 正在保存文档#{self.next_index} (ID: {result.get('id', 'Unknown')})...",
                doc_index=None, print_console=False
            )
            self._append_to_excel(result)
            flushed_count += 1
            self.next_index += 1

        if flushed_count > 0:
            self.logger.log(f"   ✅ 批量保存了 {flushed_count} 个文档到Excel", doc_index=None)

    def _append_to_excel(self, result: dict):
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

            if os.path.exists(self.output_file):
                try:
                    df_existing = pd.read_excel(self.output_file, engine='openpyxl')
                    df_new = pd.DataFrame([result])
                    df = pd.concat([df_existing, df_new], ignore_index=True)
                    self.logger.log(f"   📝 追加到现有文件 (ID: {result.get('id', 'Unknown')})",
                                    doc_index=None)
                except Exception as e:
                    self.logger.log(f"   ⚠️ 现有文件损坏，将重新创建: {str(e)}", doc_index=None)
                    backup = self.output_file.replace('.xlsx', '_backup_corrupted.xlsx')
                    try:
                        os.rename(self.output_file, backup)
                        self.logger.log(f"   💾 已备份损坏文件至: {backup}", doc_index=None)
                    except Exception:
                        os.remove(self.output_file)
                        self.logger.log(f"   🗑️ 已删除损坏文件", doc_index=None)
                    df = pd.DataFrame([result])
            else:
                df = pd.DataFrame([result])
                self.logger.log(f"   ✨ 创建新文件", doc_index=None)

            df.to_excel(self.output_file, index=False, engine='openpyxl')
            self.logger.log(f"   💾 文档 {result.get('id', 'Unknown')} 结果已保存\n",
                            doc_index=None)

        except Exception as e:
            self.logger.log(f"   ❌ 保存Excel失败: {str(e)}", doc_index=None)
            try:
                csv_file = self.output_file.replace('.xlsx', '.csv')
                df = pd.DataFrame([result])
                df.to_csv(csv_file, index=False, mode='a',
                          header=not os.path.exists(csv_file), encoding='utf-8-sig')
                self.logger.log(f"   💾 已改为保存到CSV: {csv_file}", doc_index=None)
            except Exception as csv_error:
                self.logger.log(f"   ❌ CSV保存也失败: {str(csv_error)}", doc_index=None)
