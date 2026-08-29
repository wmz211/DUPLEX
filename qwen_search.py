import requests
import json
import random
from openai import OpenAI
from typing import Dict, List, Tuple
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import openpyxl
import threading
from queue import Queue, PriorityQueue
from efi_pilot.utils.environment import require_api_keys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class ThreadSafeLogger:
    """线程安全的日志记录器 - 支持按文档顺序输出或实时输出"""

    def __init__(self, log_file: str, realtime_mode: bool = False):
        self.log_file = log_file
        self.lock = threading.Lock()
        self.log_buffers = {}  # {doc_index: [log_messages]}
        self.next_doc_index = 0  # 下一个应该输出的文档索引
        self.realtime_mode = realtime_mode  # 是否实时输出

        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message: str, doc_index: int = None, print_console: bool = True):
        """线程安全的日志记录

        Args:
            message: 日志消息
            doc_index: 文档索引（如果提供，则缓存日志；否则立即输出）
            print_console: 是否打印到控制台
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        thread_id = threading.current_thread().name
        log_message = f"[{timestamp}] {message}"

        if doc_index is None or self.realtime_mode:
            # 全局日志或实时模式，立即输出
            with self.lock:
                if print_console:
                    print(message)
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_message + '\n')
        else:
            # 文档级别日志且非实时模式，先缓存
            with self.lock:
                if doc_index not in self.log_buffers:
                    self.log_buffers[doc_index] = []
                self.log_buffers[doc_index].append((log_message, message, print_console))

    def flush_logs(self, doc_index: int):
        """按顺序刷新日志到文件和控制台

        只有当doc_index是下一个应该输出的文档时才输出
        """
        with self.lock:
            # 尝试输出所有连续完成的文档日志
            while self.next_doc_index in self.log_buffers:
                logs = self.log_buffers.pop(self.next_doc_index)

                # 输出到控制台和文件
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
        self.results_queue = PriorityQueue()  # 优先级队列，按doc_index排序
        self.next_index = 0  # 下一个应该输出的文档索引
        self.completed_docs = {}  # 已完成但未输出的文档

    def add_result(self, doc_index: int, result: Dict):
        """添加结果（线程安全）"""
        with self.lock:
            self.completed_docs[doc_index] = result
            # 调试日志
            self.logger.log(f"   [DEBUG] 文档#{doc_index} 结果已接收，尝试flush...", doc_index=None, print_console=False)
            self._try_flush()

    def _try_flush(self):
        """尝试输出按顺序完成的结果"""
        flushed_count = 0
        while self.next_index in self.completed_docs:
            result = self.completed_docs.pop(self.next_index)
            self.logger.log(f"   [DEBUG] 正在保存文档#{self.next_index} (ID: {result.get('id', 'Unknown')})...",
                            doc_index=None, print_console=False)
            self._append_to_excel(result)
            flushed_count += 1
            self.next_index += 1

        if flushed_count > 0:
            self.logger.log(f"   ✅ 批量保存了 {flushed_count} 个文档到Excel", doc_index=None)

    def _append_to_excel(self, result: Dict):
        """将结果追加到Excel文件（已加锁）"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

            # 读取现有数据
            if os.path.exists(self.output_file):
                try:
                    df_existing = pd.read_excel(self.output_file, engine='openpyxl')
                    df_new = pd.DataFrame([result])
                    df = pd.concat([df_existing, df_new], ignore_index=True)
                    self.logger.log(f"   📝 追加到现有文件 (ID: {result.get('id', 'Unknown')})", doc_index=None)
                except Exception as e:
                    self.logger.log(f"   ⚠️ 现有文件损坏，将重新创建: {str(e)}", doc_index=None)
                    backup_file = self.output_file.replace('.xlsx', '_backup_corrupted.xlsx')
                    if os.path.exists(self.output_file):
                        try:
                            os.rename(self.output_file, backup_file)
                            self.logger.log(f"   💾 已备份损坏文件至: {backup_file}", doc_index=None)
                        except:
                            os.remove(self.output_file)
                            self.logger.log(f"   🗑️ 已删除损坏文件", doc_index=None)
                    df = pd.DataFrame([result])
            else:
                df = pd.DataFrame([result])
                self.logger.log(f"   ✨ 创建新文件", doc_index=None)

            # 保存
            df.to_excel(self.output_file, index=False, engine='openpyxl')
            self.logger.log(f"   💾 文档 {result.get('id', 'Unknown')} 结果已保存\n", doc_index=None)

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


class NewsHallucinationDetector:
    def __init__(self, deepseek_api_key: str, bocha_api_key: str, qwen_api_key: str,
                 logger: ThreadSafeLogger, shared_embedder=None):
        self.deepseek_client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        )

        self.bocha_api_key = bocha_api_key

        self.qwen_client = OpenAI(
            api_key=qwen_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.logger = logger
        self.doc_index = None  # 当前处理的文档索引

        # 使用共享的embedder或创建新的
        if shared_embedder is not None:
            self._embedder = shared_embedder
        else:
            self._embedder = None
            self.embedder_lock = threading.Lock()

    def set_doc_index(self, doc_index: int):
        """设置当前处理的文档索引"""
        self.doc_index = doc_index

    @property
    def embedder(self):
        """延迟加载并缓存SentenceTransformer模型（线程安全）"""
        if self._embedder is None:
            # 只在没有共享embedder时才加载
            if not hasattr(self, 'embedder_lock'):
                self.embedder_lock = threading.Lock()
            with self.embedder_lock:
                if self._embedder is None:
                    try:
                        self._embedder = SentenceTransformer('sentence-transformers/all-MPNet-base-v2')
                        self.logger.log("✓ 成功加载语义模型 all-MPNet-base-v2", doc_index=None)
                    except:
                        self._embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                        self.logger.log("✓ 加载备用语义模型 all-MiniLM-L6-v2", doc_index=None)
        return self._embedder

    def log(self, message: str, print_console: bool = True):
        """委托给logger，自动传递doc_index"""
        self.logger.log(message, doc_index=self.doc_index, print_console=print_console)

    def detect_faithfulness_hallucination(self, text: str, text_key: str = None) -> Tuple[str, float, str]:
        """检测忠实性幻觉（内部矛盾）- 不需要事件信息"""
        if text_key:
            self.log(f"      🔍 检测 {text_key}...")
        prompt = f"""你是专业的逻辑一致性分析专家。请分析以下新闻文本是否存在内部矛盾。

新闻文本：
{text}

**分析要求：**
请在内部完成以下步骤（不需要输出中间过程）：
1. 将文本分解为独立的句子
2. 在句子之间寻找矛盾的陈述
3. 确定是否存在逻辑不一致

**判断标准示例：**

例1（有矛盾）：
"俄罗斯JSC全地形车厂再次生产苏联时期的螺旋桨汽车...这款汽车能在沼泽、沙滩、浅海等多种地形行驶...不过由于技术限制，这款汽车实际上只能在平坦路面上行驶，无法适应复杂地形。"
→ 判断：Faithfulness（"能在多种地形行驶"与"只能在平坦路面行驶"矛盾）

例2（有矛盾）：
"利比亚卫生部门称塞卜拉泰武装冲突已导致26人死亡、170人受伤...值得注意的是，此次冲突并未造成任何人员伤亡。"
→ 判断：Faithfulness（"26人死亡、170人受伤"与"未造成任何人员伤亡"矛盾）

**重要说明：**
- 只有在矛盾非常明显且无法用合理方式解释时，才判定为Faithfulness
- 如果陈述之间只是补充关系而非矛盾，应判定为Real
- 如果矛盾可以用"后续报道更新"或"不同角度描述"解释，应判定为Real

**置信度定义：表示你对判断结果的确定程度（0-100）**

如果判断为 Faithfulness（有内部矛盾）：
- 矛盾非常明显、证据充分 → 置信度 85-95
- 矛盾较为明显 → 置信度 70-84
- 可能有矛盾但不太确定 → 置信度 55-69

如果判断为 Real（逻辑一致）：
- 非常确定文本逻辑一致、无任何矛盾 → 置信度 85-95
- 比较确定逻辑一致 → 置信度 70-84
- 基本一致但有些模糊 → 置信度 55-69

请严格按以下格式回答（每项占一行）：
判断结果：[Real/Faithfulness]
置信度：[0-100的数字，不要加百分号]
矛盾证据：[如果是Faithfulness，列出具体矛盾的句子对；如果是Real，写"无矛盾"]"""

        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system",
                     "content": "你是逻辑一致性分析专家，擅长发现文本中的内部矛盾。置信度表示你对判断结果的确定程度。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                stream=False
            )
            content = response.choices[0].message.content.strip()

            self.log(f"      [DEBUG] LLM原始返回:\n{content}", print_console=False)

            # 解析结果
            label = "Real"
            confidence = 50.0
            evidence = "无矛盾"
            confidence_parsed = False

            lines = content.split('\n')
            for line in lines:
                line = line.strip()

                if '判断结果' in line:
                    if 'Faithfulness' in line:
                        label = "Faithfulness"
                    elif 'Real' in line:
                        label = "Real"

                elif line.startswith('置信度') and not confidence_parsed:
                    try:
                        import re
                        if '：' in line:
                            value_part = line.split('：', 1)[1]
                        elif ':' in line:
                            value_part = line.split(':', 1)[1]
                        else:
                            continue

                        numbers = re.findall(r'\d+\.?\d*', value_part)
                        if numbers:
                            confidence = float(numbers[0])
                            confidence = max(0.0, min(100.0, confidence))
                            confidence_parsed = True
                    except Exception as e:
                        self.log(f"      [DEBUG] 置信度解析失败: {line}, 错误: {e}", print_console=False)

                elif '矛盾证据' in line or '问题说明' in line:
                    parts = line.split('：', 1)
                    if len(parts) == 2:
                        evidence = parts[1].strip()
                    else:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            evidence = parts[1].strip()

            self.log(f"      Faithfulness分析: {label} (置信度: {confidence:.1f}%)")
            return label, confidence, evidence

        except Exception as e:
            self.log(f"      ⚠️ Faithfulness检测失败: {str(e)}")
            return "Real", 50.0, f"检测失败: {str(e)}"

    def bocha_search(self, query: str, num_results: int = 3) -> List[Dict]:
        """博查API搜索（搜索3个结果）"""
        url = "https://api.bochaai.com/v1/web-search"
        payload = json.dumps({"query": query, "summary": True, "count": num_results})
        headers = {
            'Authorization': f'Bearer {self.bocha_api_key}',
            'Content-Type': 'application/json'
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
                        'title': item.get('name', ''),
                        'snippet': item.get('snippet', ''),
                        'link': item.get('url', ''),
                        'displayLink': item.get('siteName', '')
                    })
            return results
        except Exception as e:
            self.log(f"      ⚠️ 博查搜索失败: {str(e)}")
            return []

    def semantic_similarity_check(self, text: str) -> Tuple[float, str]:
        """路径1: 博查搜索 + 语义相似度验证（优化：合并LLM调用）"""
        self.log(f"      🔍 路径1: 语义相似度验证...")

        combined_prompt = f"""请完成两个任务：

新闻文本：
{text}

任务1：用一句话（不超过30字）总结核心内容，适合搜索引擎查询
任务2：提取3个最关键的可验证事实（时间、人物、地点、数字、事件等）

请按以下格式回答：
搜索查询：[一句话总结]
关键事实1：[第一个事实]
关键事实2：[第二个事实]
关键事实3：[第三个事实]"""

        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": combined_prompt}],
                temperature=0.3,
                stream=False
            )
            content = response.choices[0].message.content.strip()

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
                search_query = text[:50]
            if not key_facts:
                key_facts = [text[:100]]

        except Exception as e:
            self.log(f"         ⚠️ LLM处理失败: {str(e)}")
            search_query = text[:50]
            key_facts = [text[:100]]

        self.log(f"         搜索查询: {search_query}")

        search_results = self.bocha_search(search_query, num_results=3)

        if not search_results:
            self.log(f"         ⚠️ 无搜索结果")
            return 30.0, "无法获取搜索结果"

        self.log(f"         找到 {len(search_results)} 个结果")
        self.log(f"         提取 {len(key_facts)} 个关键事实")

        context_texts = [f"{r['title']} {r['snippet']}" for r in search_results]
        fact_embeddings = self.embedder.encode(key_facts, convert_to_tensor=True)
        context_embeddings = self.embedder.encode(context_texts, convert_to_tensor=True)

        similarities = util.cos_sim(fact_embeddings, context_embeddings)
        max_similarities = similarities.max(dim=1).values
        avg_similarity = max_similarities.mean().item()
        semantic_score = avg_similarity * 100

        self.log(f"         语义相似度得分: {semantic_score:.1f}%")
        return semantic_score, "语义验证完成"

    def qwen_search_verify(self, text: str) -> Tuple[str, float, str]:
        """路径2: Qwen联网搜索验证"""
        self.log(f"      🔍 路径2: Qwen联网搜索验证...")

        verify_prompt = f"""你是专业的事实核查专家。请对以下新闻文本进行事实性验证。

新闻文本：
{text}

请按以下步骤操作：
1. 提取新闻中的关键事实（人物、时间、地点、数字、具体事件等）
2. 使用联网搜索功能查找相关信息进行验证
3. 对比新闻内容与搜索到的权威信息源,注意细节错误
4. 判断新闻事实是否准确

判断标准：
- 如果核心事实（人物、时间、地点、数字、关键事件）与权威信息源有明显出入或捏造，判定为 Factuality（事实性幻觉）
- 注意是否存在捏造因果关系或背景，无中生有，张冠李戴，扭曲关系或比例等事实性错误
eg:**捏造因果关系或背景**:为真实事件编造一个不存在的动机、阴谋或背后故事
   **无中生有**:在真实事件中混入完全虚构的、戏剧性的情节
   **张冠李戴**:混淆来源,将其他事件的元素嫁接到此事件上
   **扭曲关系或比例**:夸大或缩小某个事实的重要性,或扭曲人物/事件之间的关系
- 如果新闻事实与搜索结果基本一致，或搜索结果无法验证但也没有明显矛盾，判定为 Real（真实）
- 查询是应注意年份信息，如果是历史事件，你需要根据历史事件的时间进行查询检索，不能因为查询到的是近期的事件就对历史事件判错，以此为例：
    示例：新闻报道"2017年10月加州山火导致23人死亡"
    ❌ 错误思路：我只搜到2025年的山火，所以2017年的是假的

**置信度定义：表示你对判断结果的确定程度（0-100）**

如果判断为 Factuality（事实性错误）：
- 核心事实明显错误、有充分证据证明造假 → 置信度 85-95（非常确定它有错误）
- 多个事实与权威信息不符 → 置信度 70-84（比较确定它有错误）
- 部分事实可疑但不完全确定 → 置信度 55-69（倾向认为有错误）
- 信息可疑且检索不到相关证据 → 置信度 50-54 （不清楚是否有错）

如果判断为 Real（事实准确）：
- 所有关键事实都得到权威信息源证实 → 置信度 85-95（非常确定它准确）
- 大部分事实准确、与搜索结果一致 → 置信度 70-84（比较确定它准确）
- 基本准确或无法完全验证但无明显矛盾 → 置信度 55-69（倾向认为准确）
- 信息较准确但检索不到相关证据 → 置信度 50-54 （不清楚是否正确）

**重要：**
- 优先参考权威媒体、官方网站、学术资料等可信来源
- 置信度表示你对判断结果的确定程度
- 如果搜索到的信息与新闻内容矛盾，详细说明矛盾之处

请严格按以下格式回答（每项占一行）：
判断结果：[Real/Factuality]
置信度：[0-100的数字]
问题说明：[如果是Factuality，详细说明哪些事实有误；如果是Real，写"事实基本准确"或"信息不足但无明显矛盾"]

以此为例：
 "以色列总理内塔尼亚胡涉嫌贪腐案于2019年1月8日正式开庭审理，成为该国历史上首例在职总理接受刑事审判的案件。据称，此次审判是在美国联邦调查局（FBI）介入并提供关键证据后被迫启动的，美方指控内塔尼亚胡通过其子与一家瑞士离岸媒体公司秘密合作，操纵国际舆论以掩盖资产转移行为。该案涉及他被控非法收受价值超过28万美元的奢侈品礼物，其中包括定制钻石袖扣和限量版劳力士手表，这些物品据称藏匿于特拉维夫一处军事情报基地内。此外，检方披露，内塔尼亚胡曾与某中东国家的情报机构达成非正式协议，以换取国内媒体对其政府的正面报道。若罪名成立，他可能面临最高11年监禁，并被永久剥夺担任公职资格。截至2023年11月底，案件已进入司法程序，但尚未作出判决。期间，法院曾因安全局势和程序原因多次调整听证会安排，其中一次庭审因疑似网络攻击导致证据系统瘫痪而被迫中止。"
经过联网搜索从权威网站中获取信息得出这样的分析：
捏造因果关系或背景: 添加FBI介入并迫使审判启动的情节; 编造内塔尼亚胡通过其子与瑞士离岸公司合作操纵舆论的阴谋; 虚构与某中东国家情报机构达成协议的背景;  
无中生有: 声称奢侈品藏匿于特拉维夫军事情报基地; 添加庭审因网络攻击导致证据系统瘫痪而中止的情节;  
张冠李戴: 将尚未发生的2025年案件提前至2019年开庭，嫁接为"首例在职总理受审"事件;  
扭曲关系或比例: 将收受礼物行为夸大为掩盖资产转移的国际操纵行为; 将媒体利益交换扭曲为涉及外国情报机构的秘密协议; 刑期从10年增至11年并添加永久剥夺公职资格以增强严重性;  
基本信息错误: 开庭时间由2025年9月17日错误改为2019年1月8日; 截止时间由2025年9月下旬改为2023年11月底; 金额由26万美元改为28万美元; 刑期由最高10年改为11年;"



"""

        try:
            response = self.qwen_client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是专业的事实核查专家，擅长通过联网搜索验证新闻真实性。"},
                    {"role": "user", "content": verify_prompt}
                ],
                temperature=0.3,
                extra_body={
                    "enable_search": True
                }
            )

            content = response.choices[0].message.content.strip()
            self.log(f"         [DEBUG] Qwen返回:\n{content}", print_console=False)

            label = "Real"
            confidence = 50.0
            evidence = "事实基本准确"
            confidence_parsed = False

            import re
            for line in content.split('\n'):
                line = line.strip()

                if '判断结果' in line:
                    if 'Factuality' in line:
                        label = "Factuality"
                    elif 'Real' in line:
                        label = "Real"

                elif line.startswith('置信度') and not confidence_parsed:
                    try:
                        if '：' in line:
                            value_part = line.split('：', 1)[1]
                        elif ':' in line:
                            value_part = line.split(':', 1)[1]
                        else:
                            continue

                        numbers = re.findall(r'\d+\.?\d*', value_part)
                        if numbers:
                            confidence = float(numbers[0])
                            confidence = max(0.0, min(100.0, confidence))
                            confidence_parsed = True
                    except:
                        pass

                elif '问题说明' in line:
                    parts = line.split('：', 1)
                    if len(parts) == 2:
                        evidence = parts[1].strip()
                    else:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            evidence = parts[1].strip()

            self.log(f"         Qwen判断: {label} (置信度: {confidence:.1f}%)")
            return label, confidence, evidence

        except Exception as e:
            self.log(f"         ⚠️ Qwen验证失败: {str(e)}")
            import traceback
            self.log(f"         详细错误: {traceback.format_exc()}", print_console=False)
            return "Real", 30.0, f"验证失败: {str(e)}"

    def detect_factuality_hallucination(self, text: str, text_key: str = None) -> Tuple[str, float, str]:
        """方案A: 并行双验证 + 加权融合"""
        if text_key:
            self.log(f"      🔍 检测 {text_key}...")

        self.log(f"      ═══ 开始双路径并行验证 ═══")

        semantic_score, semantic_evidence = self.semantic_similarity_check(text)
        qwen_label, qwen_conf, qwen_evidence = self.qwen_search_verify(text)

        if qwen_label == "Real":
            qwen_score = qwen_conf
        else:
            qwen_score = 100 - qwen_conf

        if qwen_conf >= 85:
            alpha, beta = 0.35, 0.65
        elif qwen_conf >= 70:
            alpha, beta = 0.40, 0.60
        elif qwen_conf >= 55:
            alpha, beta = 0.45, 0.55
        else:
            alpha, beta = 0.60, 0.40
        final_score = alpha * semantic_score + beta * qwen_score

        final_score = max(0, min(100, final_score))

        threshold = 55
        if final_score >= threshold:
            final_label = "Real"
            final_evidence = f"综合验证: 事实基本准确 | 语义:{semantic_score:.1f}% Qwen:{qwen_conf:.1f}%"
        else:
            final_label = "Factuality"
            final_evidence = f"综合验证: 存在事实性问题 | 语义:{semantic_score:.1f}% Qwen:{qwen_conf:.1f}% | {qwen_evidence}"

        self.log(f"      ─────────────────────────")
        self.log(f"      路径1（语义）: {semantic_score:.1f}%")
        self.log(f"      路径2（Qwen）: {qwen_label} 置信度{qwen_conf:.1f}% → 转为真实性分数 {qwen_score:.1f}%")
        self.log(f"      融合公式: {alpha}×{semantic_score:.1f} + {beta}×{qwen_score:.1f} = {final_score:.1f}%")
        self.log(f"      最终判断: {final_label} (综合真实性分数: {final_score:.1f}%)")
        self.log(f"      ═══════════════════════════")

        return final_label, final_score, final_evidence


def process_document(doc_index: int, doc: Dict, detector: NewsHallucinationDetector,
                     result_collector: ResultCollector) -> Dict:
    """处理单个文档的所有文本"""
    # 设置当前文档索引（用于日志缓存）
    detector.set_doc_index(doc_index)

    doc_id = doc.get('id', 'Unknown')
    event = doc['events'].get('event1', '')

    detector.log(f"\n{'=' * 60}")
    detector.log(f"📄 处理文档 #{doc_index}: {doc_id}")
    detector.log(f"📌 事件: {event}")

    results = {'id': doc_id}

    # 收集所有文本及其键名
    text_items = []

    if 'text1' in doc and doc['text1']:
        text_items.append(('text1', doc['text1']))

    if 'hallu_type' in doc:
        hallu_type = doc['hallu_type']

        if 'faithfulness' in hallu_type:
            faith_data = hallu_type['faithfulness']
            for key in ['text2', 'text3']:
                if key in faith_data and faith_data[key]:
                    text_items.append((key, faith_data[key]))

        if 'factuality' in hallu_type:
            fact_data = hallu_type['factuality']
            for key in ['text4', 'text5', 'text6']:
                if key in fact_data and fact_data[key]:
                    text_items.append((key, fact_data[key]))

    if not text_items:
        detector.log(f"   ⚠️ 文档无有效文本")
        result_collector.add_result(doc_index, results)
        # 刷新日志
        detector.logger.flush_logs(doc_index)
        return results

    detector.log(f"   找到 {len(text_items)} 个文本")

    random.shuffle(text_items)
    detector.log(f"   检测顺序（随机）: {', '.join([k for k, _ in text_items])}")

    # 阶段1: 忠实性幻觉检测
    detector.log(f"\n{'─' * 50}")
    detector.log("📍 阶段1: 忠实性幻觉检测")
    detector.log(f"{'─' * 50}")

    faithfulness_results = []
    remaining_texts = []

    for text_key, text in text_items:
        faith_label, faith_conf, faith_evidence = detector.detect_faithfulness_hallucination(text, text_key=text_key)

        faithfulness_results.append({
            'key': text_key,
            'label': faith_label,
            'confidence': faith_conf,
            'evidence': faith_evidence
        })

        if faith_label == "Faithfulness":
            results[text_key] = 'Faithfulness'
            detector.log(f"   ❌ {text_key}: Faithfulness (置信度: {faith_conf:.1f}%)")
        else:
            remaining_texts.append((text_key, text))
            detector.log(f"   ✓ {text_key}: 通过忠实性检测 (置信度: {faith_conf:.1f}%)")

    faith_count = len([r for r in faithfulness_results if r['label'] == 'Faithfulness'])
    detector.log(f"\n   📊 忠实性检测结果: {faith_count}个Faithfulness, {len(remaining_texts)}个待验证")

    # 阶段2: 事实性幻觉检测
    if not remaining_texts:
        detector.log(f"\n   ⚠️ 所有文本都是忠实性幻觉，无法继续事实性检测")
        result_collector.add_result(doc_index, results)
        # 刷新日志
        detector.logger.flush_logs(doc_index)
        return results

    detector.log(f"\n{'─' * 50}")
    detector.log("📍 阶段2: 事实性幻觉检测（使用Qwen联网搜索）")
    detector.log(f"{'─' * 50}")

    factuality_results = []

    for text_key, text in remaining_texts:
        fact_label, fact_conf, fact_evidence = detector.detect_factuality_hallucination(text, text_key=text_key)

        factuality_results.append({
            'key': text_key,
            'label': fact_label,
            'confidence': fact_conf,
            'evidence': fact_evidence
        })

        detector.log(f"   分析 {text_key}: {fact_label} (置信度: {fact_conf:.1f}%)")

    # 寻找唯一的真实文本
    real_candidates = [r for r in factuality_results if r['label'] == 'Real']

    if not real_candidates:
        detector.log(f"\n   ⚠️ 未找到Real文本，强制选择置信度最高的文本作为Real")
        factuality_results.sort(key=lambda x: x['confidence'], reverse=True)
        best_candidate = factuality_results[0]
        best_candidate['label'] = 'Real'
        detector.log(f"   🔄 将 {best_candidate['key']} 标记为Real (置信度: {best_candidate['confidence']:.1f}%)")
        real_candidates = [best_candidate]

    real_candidates.sort(key=lambda x: x['confidence'], reverse=True)
    true_real = real_candidates[0]

    detector.log(f"\n   🏆 唯一真实文本: {true_real['key']} (置信度: {true_real['confidence']:.1f}%)")

    for r in factuality_results:
        if r['key'] == true_real['key']:
            results[r['key']] = 'Real'
        else:
            results[r['key']] = 'Factuality'
            if r['label'] == 'Real':
                detector.log(f"      {r['key']}: Real改为Factuality（置信度低于 {true_real['key']}）")

    # 最终统计
    final_faith_count = sum(1 for k, v in results.items() if k.startswith('text') and v == 'Faithfulness')
    final_fact_count = sum(1 for k, v in results.items() if k.startswith('text') and v == 'Factuality')
    final_real_count = sum(1 for k, v in results.items() if k.startswith('text') and v == 'Real')

    detector.log(f"\n✅ 文档 {doc_id} 处理完成")
    detector.log(
        f"   最终结果: {final_real_count}个Real, {final_faith_count}个Faithfulness, {final_fact_count}个Factuality")

    # 通过ResultCollector保证顺序输出
    result_collector.add_result(doc_index, results)

    # 刷新日志（按顺序输出）
    detector.logger.flush_logs(doc_index)

    return results


def main():
    # 检查必要的库
    required_packages = {
        'openpyxl': 'openpyxl',
        'sentence_transformers': 'sentence-transformers',
        'openai': 'openai',
        'pandas': 'pandas',
        'requests': 'requests'
    }

    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)

    if missing_packages:
        print("❌ 缺少必要的库，请运行以下命令安装：")
        print(f"pip install {' '.join(missing_packages)}")
        return

    # API配置只从环境变量读取，避免密钥进入源码或命令历史。
    api_keys = require_api_keys("deepseek", "bocha", "qwen")

    # 文件配置
    input_file = "hallu/extr-hallu-final.json"
    output_file = "res/hallucination_detection_results3.xlsx"
    log_file = "log/detection 0-1000.txt"

    # 线程数配置
    MAX_WORKERS = int(input("请输入并发线程数（建议3-5）: ") or "3")

    # 日志模式选择
    log_mode = input("日志输出模式 - 实时(r)/按序(o)，默认实时: ").strip().lower()
    realtime_mode = log_mode != 'o'  # 默认实时模式

    if realtime_mode:
        print("✓ 选择实时输出模式（立即看到处理日志，但可能乱序）")
    else:
        print("✓ 选择按序输出模式（等待前面文档完成后才输出，保证顺序）")

    # 初始化共享组件
    logger = ThreadSafeLogger(log_file, realtime_mode=realtime_mode)
    result_collector = ResultCollector(output_file, logger)

    logger.log("=" * 60, doc_index=None)
    logger.log(f"🚀 新闻幻觉检测系统启动（多线程版本 - {MAX_WORKERS}线程）", doc_index=None)
    logger.log("   优化: 搜索结果3个 | DeepSeek调用合并 | 线程安全日志 | 按序输出", doc_index=None)
    logger.log("=" * 60, doc_index=None)

    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        if not isinstance(documents, list):
            documents = [documents]
        logger.log(f"📂 读取到 {len(documents)} 个文档", doc_index=None)
    except Exception as e:
        logger.log(f"❌ 读取文件失败: {str(e)}", doc_index=None)
        return

    # 询问起始和结束位置
    start_id = input("请输入起始文档ID（留空则从头开始）: ").strip()
    end_id = input("请输入结束文档ID（留空则处理到最后）: ").strip()

    # 过滤文档
    start_processing = not start_id
    docs_to_process = []
    should_stop = False

    for idx, doc in enumerate(documents):
        doc_id = doc.get('id', 'Unknown')

        # 检查是否应该停止（在添加到处理列表之前）
        if end_id and doc_id == end_id:
            should_stop = True

        # 检查是否开始处理
        if not start_processing:
            if doc_id == start_id:
                start_processing = True
            else:
                continue

        # 添加到处理列表
        docs_to_process.append((idx, doc))

        # 如果是结束文档，添加后停止
        if should_stop:
            logger.log(f"🛑 已添加结束文档 {doc_id}，将在此停止", doc_index=None)
            break

    if start_id and not any(d.get('id') == start_id for _, d in docs_to_process):
        logger.log(f"⚠️  警告: 未找到起始文档ID '{start_id}'", doc_index=None)

    if end_id and not should_stop:
        logger.log(f"⚠️  警告: 未找到结束文档ID '{end_id}'", doc_index=None)

    logger.log(f"\n🎯 准备处理 {len(docs_to_process)} 个文档", doc_index=None)
    if docs_to_process:
        first_id = docs_to_process[0][1].get('id', 'Unknown')
        last_id = docs_to_process[-1][1].get('id', 'Unknown')
        logger.log(f"📍 处理范围: {first_id} → {last_id}", doc_index=None)
    logger.log(f"⚡ 使用 {MAX_WORKERS} 个线程并行处理", doc_index=None)

    # 预加载共享的embedder模型（只加载一次）
    logger.log("📦 正在加载语义模型（所有线程共享）...", doc_index=None)
    try:
        shared_embedder = SentenceTransformer('sentence-transformers/all-MPNet-base-v2')
        logger.log("✓ 成功加载语义模型 all-MPNet-base-v2\n", doc_index=None)
    except:
        shared_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.log("✓ 加载备用语义模型 all-MiniLM-L6-v2\n", doc_index=None)

    start_time = time.time()

    # 使用线程池处理文档
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="Worker") as executor:
        # 提交所有任务
        future_to_doc = {}
        for new_index, (orig_index, doc) in enumerate(docs_to_process):  # 重新编号从0开始
            # 为每个任务创建独立的detector实例，但共享embedder模型
            detector = NewsHallucinationDetector(
                api_keys["deepseek"], api_keys["bocha"], api_keys["qwen"], logger,
                shared_embedder=shared_embedder  # 传递共享的模型
            )
            future = executor.submit(
                process_document,
                new_index,  # 使用重新编号的索引（从0开始）
                doc,
                detector,
                result_collector
            )
            future_to_doc[future] = (new_index, doc.get('id', 'Unknown'))

        # 等待所有任务完成
        completed = 0
        total = len(docs_to_process)

        for future in as_completed(future_to_doc):
            doc_index, doc_id = future_to_doc[future]
            try:
                result = future.result()
                completed += 1
                # 实时显示进度（不写入日志文件）
                progress_msg = f"⏳ 进度: {completed}/{total} ({completed / total * 100:.1f}%) - 刚完成: {doc_id}"
                print(progress_msg)  # 直接打印，不通过logger
            except Exception as e:
                logger.log(f"❌ 处理文档 {doc_id} (#{doc_index}) 失败: {str(e)}", doc_index=None)
                import traceback
                logger.log(f"详细错误: {traceback.format_exc()}", doc_index=None)

    elapsed_time = time.time() - start_time

    logger.log("\n" + "=" * 60, doc_index=None)
    logger.log("✅ 所有文档处理完成", doc_index=None)
    logger.log(f"⏱️  总耗时: {elapsed_time:.2f} 秒", doc_index=None)
    logger.log(f"📊 平均每个文档: {elapsed_time / len(docs_to_process):.2f} 秒", doc_index=None)
    logger.log(f"📁 结果已保存至: {output_file}", doc_index=None)
    logger.log(f"📝 日志已保存至: {log_file}", doc_index=None)
    logger.log("=" * 60, doc_index=None)


if __name__ == "__main__":
    main()
