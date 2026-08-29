import pandas as pd
import numpy as np
import json
import os
import logging
from openai import OpenAI
from datetime import datetime
import argparse
import time
from efi_pilot.utils.environment import require_api_keys
from efi_pilot.config import QWEN_MODEL


class EventFactualityAnalyzer:
    def __init__(self, model=QWEN_MODEL):
        """
        Initialize Event Factuality Analyzer

        Args:
            model: Model name to use
        """
        self.api_key = require_api_keys("qwen")["qwen"]
        self.model = model

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # Setup logging
        self.logger = logging.getLogger('EventFactualityAnalyzer')

    def setup_logging(self, log_file):
        """Setup logging configuration"""
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def judge_factuality(self, text, event_description):
        """
        Judge the factuality of an event based on text

        Args:
            text: The news text
            event_description: The event description

        Returns:
            str: Factuality label (CT+, CT-, PS+, PS-, UU)
        """
        prompt = f"""你是一位专业的事实核查专家。请根据新闻文本分析给定事件的事实性。

新闻文本：
{text}

事件描述：
{event_description}

事实性定义：
- **CT+ (明确肯定 Certain True)**：新闻文本明确肯定该事件发生, 或对事件表示肯定，与原文描述一致
- **CT- (明确否定 Certain False)**：新闻文本明确否认该事件发生, 或与原文描述相反
- **PS+ (猜测可能 Possible)**：新闻文本暗示该事件可能发生或将要发生（使用"可能"、"或许"、"据报道"等词汇）。或者对事件表示猜测
- **PS- (猜测不可能 Possible False)**：新闻文本暗示（预测）该事件可能没有或可能不会发生。
- **Uu (无法判断 Unable to judge)**：无法从文本中判断该事件是否发生

要求：
1. 仔细阅读新闻文本和事件描述
2. 判断事件与文本之间的事实性关系
3. 特别注意情态动词、模糊表达和否定词
4. 只输出以下标签之一：CT+, CT-, PS+, PS-, Uu
5. 不要输出任何解释，只输出标签

示例：
eg1：
"event1": "印度军方一架直升机在藏南地区坠毁导致7名军人丧生原因是机群老化"，
"text1": "外媒关注印军直升机在藏南坠毁：或因机群 老化 。  原标题：外媒称印军直升机藏南坠毁致7人丧生：或因机群 老化. 参考消息网10月7日报道外媒称，印度军方表示，一架印度军用直升机6日在中国藏南地区失事，机上7名军人全部丧生。 一名发言人称，5名印度空军机组成员和2名陆军人员因直升机失事而死亡。报道称，米17V5直升机主要用于军事运输，是由隶属俄罗斯直升机公司的喀山直升机厂生产的。 报道称，由于机群 老化 ，印度空军失事率很高。 过去30年已有170多名飞行员死亡。 大多数事故都与苏联时代的米格飞机有关，这种飞机因而赢得了"飞行棺材"的绰号。 "
分析如下：事件提到直升机坠毁原因是机群老化，而text1中则是对该原因表示猜测并未肯定。
输出：PS+.
eg2:
"event1": "溶菌酶晶体在受压时可产生压电效应并已实现商业化能量收集装置"，
"text1": "2017年，爱尔兰利莫瑞克大学的研究团队首次证实人类眼泪中的溶菌酶晶体在受压时可产生压电效应，相关成果发表于《应用物理快报》。该发现表明，溶菌酶作为一种天然蛋白质，具备将机械能转化为电能的潜力，有望应用于生物相容性能源设备和医疗植入器件。据2025年最新行业分析报告显示，全球溶菌酶市场规模预计达1.38亿美元，其应用已从传统食品防腐、医药抗菌扩展至生物材料与功能性医疗器械领域。尽管基于溶菌酶的"眼泪发电"技术仍处于实验室探索阶段，尚未实现商业化能量收集装置，但其在生物电子接口、自供能生物传感器等前沿方向展现出潜在价值，相关基础研究仍在持续推进。",
分析如下：原文中明确提到尚未实现商业化能量收集装置，而事件描述已实现商业化能量收集装置，与原文描述不一致。
输出:CT-.
eg3:
"event1": "沙特国王首次访问俄罗斯并与普京讨论石油等问题".
"text1": "沙特国王首次访俄将与普京讨论石油等问题。 2017年10月05日13:12。 中国新闻网。 原标题：沙特阿拉伯国王萨勒曼首次出访俄罗斯将 会见 普京-EOP-. 中新网10月5日电据外媒报道，沙特阿拉伯国王萨勒曼抵达莫斯科访问，成为第一位访问俄罗斯的沙特国王。 据报道，萨勒曼在4天的访问期间计划 会见 俄罗斯总统普京，讨论包括石油以及叙利亚在内的若干问题。 报道指出，萨勒曼的访问显示两国关系好转，俄罗斯和沙特阿拉伯的石油产量在全世界名列前茅。 据悉，沙特阿拉伯和俄罗斯将在萨勒曼访问期间，签署核能发电技术和扩大食品出口等一系列协议。",
分析如下: 原文显示该事件将要发生，只是在计划当中，对事件的进展表示预测
输出:PS+.
eg4:
"event1": "美国务卿蒂勒森被特朗普解除职务并由迈克·蓬佩奥接任".
"text1": "美国务卿蒂勒森于2018年3月13日被总统特朗普解除职务，由中央情报局局长迈克·蓬佩奥接任。特朗普在当天通过推特宣布了这一决定。蒂勒森本人随后确认已接到通知，并于3月31日正式离职。此次人事变动发生在美朝领导人会晤前夕，被视为特朗普对外交团队的重大调整。蓬佩奥上任后主导了与朝鲜的外交接触及后续谈判。蒂勒森离任后未再担任美国政府要职。",
分析如下：原文与事件描述完全一致
输出：CT+
eg5:
"event1":"某市将在本月底前完成所有老旧小区的电梯安装工程。"
"text1":"近期，关于我市老旧小区电梯安装工程的进展引发社会关注。据部分施工方透露，由于近期连续降雨和材料运输延迟，部分楼栋的电梯安装可能无法按原计划在本月底前完成。"
分析：原文显示电梯的安装可能无法完成安装，对应的event可能无法完成。
输出：PS-
eg6:
"event1": "俄罗斯向叙利亚提供防空导弹系统"
"text1": "以色列告知美国，称俄罗斯即将向叙利亚政府出售先进的地对空导弹,但俄罗斯专家否认对叙供应导弹报道.据悉，俄罗斯一些专家否认了国外媒体有关俄罗斯近期要向叙利亚 供应 数套S-300地空导弹系统的报道，称这是针对俄罗斯信息战的一部分。"
分析：文本明确引述俄罗斯专家的回应："俄罗斯一些专家否认了国外媒体有关俄罗斯近期要向叙利亚供应数套S-300地空导弹系统的报道"。信息状态为未证实的传闻：关于"提供导弹"的信息来源是"以色列告知美国"的报道，属于单方面传闻。
     "传闻 vs. 否认"的冲突格局，强烈暗示了该事件在近期可能并未发生或报道不实，即符合"猜测不可能"的定义。
eg7:
"event1": "美国政府重新考虑并决定留在《巴黎协定》内"
"text1": "新华社北京9月18日电美国国务卿蒂勒森17日说，在合适条件下，美国可能留在《巴黎协定》内。 距美国总统特朗普高调宣布退出《巴黎协定》才逾百日，这一表态引发媒体关注。分析人士认为，这场文字游戏背后，美国"退约"立场其实没有改变。 【新闻事实】-EOP-. 蒂勒森17日接受美国哥伦比亚广播公司采访时说，如果能制定出对美方"公正和平衡"的条款，特朗普愿与《巴黎协定》各伙伴国合作。 合适条件下，美国仍可能 留在 《巴黎协定》内。 同一天，特朗普的国家安全事务助理麦克马斯特在另一个电视采访中，直接否认美国气候变化立场发生改变，称所谓美国不会退出《巴黎协定》的说法是"虚假报道"。 事实上，在这两段自相矛盾的表态之前，就有消息说特朗普政府要重新考虑退出《巴黎协定》问题。白宫发言人萨拉•桑德斯16日晚些时候作出回应，称美国对待《巴黎协定》的立场没有变化。 中国社科院美国问题专家刁大明认为，美国仍 留在 《巴黎协定》内可能性不大；即使特朗普真的宣布返回《巴黎协定》，一定是美国以巨大利益交换为前提。 刁大明分析，特朗普政府发出不同声音有两种可能性。 一是特朗普在气候变化议题上的立场出现摇摆，政府内部人员贸然向媒体放风；二是白宫团队内部的确出现分歧，有些极力推动回到《巴黎协定》的成员提前发声，意图制造媒体意义上的既成事实。 刁大明说，特朗普是共和党籍总统，必须考虑到共和党背后美国能源产业巨大的利益驱动。 他彻底摆脱共和党传统路线、回到《巴黎协定》的可能性不大。 "
分析如下：文本并未断言"美国绝对、在任何情况下都不会重新考虑或留下"（因此不是CT-）。但通过呈现权威的否认和严苛的条件，将"决定留在"的可能性降得很低，符合"猜测不可能"的定义。并且专家分析也是"可能性很低"，符合ps-的定义。

你的判断（只输出一个标签）："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的事实核查专家。你擅长分析新闻文本与事件之间的事实性关系。你总是输出简洁的判断结果，不输出任何解释。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1  # Low temperature for more consistent judgments
            )

            result = response.choices[0].message.content.strip()

            # Validate the result
            valid_labels = ['CT+', 'CT-', 'PS+', 'PS-', 'UU']
            for label in valid_labels:
                if label in result:
                    return label

            # If no valid label found, return UU
            self.logger.warning(f"Invalid factuality result: {result}, defaulting to UU")
            return "UU"

        except Exception as e:
            self.logger.error(f"Error judging factuality: {e}")
            return "UU"

    def load_real_doc_ids(self, excel_file):
        """
        Load document IDs and their Real text column mapping from Excel

        Args:
            excel_file: Path to Excel file

        Returns:
            dict: Dictionary mapping document ID to text column name (e.g., {'CD0': 'text1', 'CD1': 'text2'})
        """
        try:
            df = pd.read_excel(excel_file, engine='openpyxl')

            # Check if id column exists
            if 'id' not in df.columns:
                self.logger.error("Excel file does not have 'id' column")
                return {}

            # Find text columns (text1, text2, text3, etc.)
            text_columns = [col for col in df.columns if col.startswith('text') and col[4:].isdigit()]

            if not text_columns:
                self.logger.error("Excel file does not have any text columns (text1, text2, etc.)")
                return {}

            real_doc_mapping = {}

            # For each row, find which text column has 'Real'
            for idx, row in df.iterrows():
                doc_id = row['id']

                for text_col in text_columns:
                    if pd.notna(row[text_col]) and str(row[text_col]).strip() == 'Real':
                        real_doc_mapping[doc_id] = text_col
                        break  # Found the Real text for this document

            self.logger.info(f"Found {len(real_doc_mapping)} documents with Real text")

            # Log distribution of Real texts
            text_distribution = {}
            for text_col in real_doc_mapping.values():
                text_distribution[text_col] = text_distribution.get(text_col, 0) + 1

            for text_col, count in sorted(text_distribution.items()):
                self.logger.info(f"  {text_col}: {count} documents")

            return real_doc_mapping

        except Exception as e:
            self.logger.error(f"Error loading Excel file: {e}")
            return {}

    def load_json_data(self, json_file):
        """
        Load JSON data

        Args:
            json_file: Path to JSON file

        Returns:
            list: List of documents
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                data = [data]

            self.logger.info(f"Loaded {len(data)} documents from {json_file}")
            return data

        except Exception as e:
            self.logger.error(f"Error loading JSON file {json_file}: {e}")
            return []

    def find_document_by_id(self, documents, doc_id):
        """
        Find document by ID

        Args:
            documents: List of documents
            doc_id: Document ID to find

        Returns:
            dict: Document or None
        """
        for doc in documents:
            if doc.get('id') == doc_id:
                return doc
        return None

    def get_text_from_document(self, document, text_col):
        """
        Get text from document based on text column name

        Args:
            document: Document data
            text_col: Text column name (e.g., 'text1', 'text2', 'text3', etc.)

        Returns:
            str: Text content or empty string
        """
        if text_col == 'text1':
            return document.get('text1', '')

        # For text2 and text3 (faithfulness hallucinations)
        if text_col in ['text2', 'text3']:
            hallu_type = document.get('hallu_type', {})
            faithfulness = hallu_type.get('faithfulness', {})
            return faithfulness.get(text_col, '')

        # For text4, text5, text6 (factuality hallucinations)
        if text_col in ['text4', 'text5', 'text6']:
            hallu_type = document.get('hallu_type', {})
            factuality = hallu_type.get('factuality', {})
            return factuality.get(text_col, '')

        return ''

    def process_document(self, doc_id, document, text_col):
        """
        Process a single document

        Args:
            doc_id: Document ID
            document: Document data
            text_col: Text column name to use (e.g., 'text1', 'text2', etc.)

        Returns:
            dict: Processing result
        """
        self.logger.info(f"Processing document {doc_id} with {text_col}...")

        # Extract text based on text_col
        text = self.get_text_from_document(document, text_col)
        events = document.get('events', {})

        if not text:
            self.logger.warning(f"Document {doc_id} has no {text_col}")
            return None

        result = {
            'id': doc_id,
            'text_used': text_col  # Record which text was used
        }

        # Process event1
        if 'event1' in events:
            event1_desc = events['event1']
            # Note: the key is 'factuality1' not 'event1_factuality'
            factuality1 = events.get('factuality1', 'N/A')

            self.logger.info(f"  Analyzing event1: {event1_desc[:50]}...")
            predict_factuality1 = self.judge_factuality(text, event1_desc)

            result['event1_factuality'] = factuality1
            result['predict_factuality1'] = predict_factuality1

            self.logger.info(f"  Event1: Ground truth={factuality1}, Predicted={predict_factuality1}")
        else:
            self.logger.warning(f"  Document {doc_id} has no event1")
            result['event1_factuality'] = None
            result['predict_factuality1'] = None

        # Process event2 if exists
        if 'event2' in events:
            event2_desc = events['event2']
            # Note: the key is 'factuality2' not 'event2_factuality'
            factuality2 = events.get('factuality2', 'N/A')

            self.logger.info(f"  Analyzing event2: {event2_desc[:50]}...")
            predict_factuality2 = self.judge_factuality(text, event2_desc)

            result['event2_factuality'] = factuality2
            result['predict_factuality2'] = predict_factuality2

            self.logger.info(f"  Event2: Ground truth={factuality2}, Predicted={predict_factuality2}")
        else:
            # No event2, don't add these columns
            pass

        return result

    def append_result_to_excel(self, output_file, result):
        """
        Append result to Excel file

        Args:
            output_file: Output Excel file path
            result: Result dictionary
        """
        try:
            # Check if file exists
            if os.path.exists(output_file):
                df_existing = pd.read_excel(output_file, engine='openpyxl')
                df_new = pd.DataFrame([result])
                df = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df = pd.DataFrame([result])

            # Save to Excel
            df.to_excel(output_file, index=False, engine='openpyxl')
            self.logger.info(f"  ✓ Result appended to {output_file}")

        except Exception as e:
            self.logger.error(f"  ✗ Error saving to Excel: {e}")

    def run(self, excel_input, json_cn, json_en, output_file, log_file, start_id=None, end_id=None):
        """
        Run the analysis

        Args:
            excel_input: Input Excel file path
            json_cn: Chinese JSON file path
            json_en: English JSON file path
            output_file: Output Excel file path
            log_file: Log file path
            start_id: Starting document ID (optional)
            end_id: Ending document ID (optional)
        """
        # Setup logging
        self.setup_logging(log_file)

        self.logger.info("=" * 60)
        self.logger.info("Event Factuality Analysis Started")
        self.logger.info("=" * 60)

        # Load Real document ID and text column mapping from Excel
        self.logger.info(f"Loading Real document mapping from {excel_input}...")
        real_doc_mapping = self.load_real_doc_ids(excel_input)

        if not real_doc_mapping:
            self.logger.error("No Real documents found. Exiting.")
            return

        # Load JSON data for both CN and EN
        self.logger.info(f"Loading Chinese documents from {json_cn}...")
        documents_cn = self.load_json_data(json_cn)

        self.logger.info(f"Loading English documents from {json_en}...")
        documents_en = self.load_json_data(json_en)

        if not documents_cn and not documents_en:
            self.logger.error("No documents loaded from JSON files. Exiting.")
            return

        # Convert mapping to ordered list for filtering
        doc_ids_list = list(real_doc_mapping.keys())

        # Filter by start_id and end_id
        if start_id or end_id:
            original_count = len(doc_ids_list)

            if start_id:
                try:
                    start_index = doc_ids_list.index(start_id)
                    doc_ids_list = doc_ids_list[start_index:]
                    self.logger.info(f"Starting from ID {start_id}")
                except ValueError:
                    self.logger.warning(f"Start ID {start_id} not found, processing all")

            if end_id:
                try:
                    end_index = doc_ids_list.index(end_id)
                    doc_ids_list = doc_ids_list[:end_index + 1]
                    self.logger.info(f"Ending at ID {end_id}")
                except ValueError:
                    self.logger.warning(f"End ID {end_id} not found, processing to end")

            self.logger.info(f"Processing {len(doc_ids_list)} documents (out of {original_count})")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Process each document
        total = len(doc_ids_list)
        success_count = 0
        fail_count = 0

        for i, doc_id in enumerate(doc_ids_list):
            text_col = real_doc_mapping[doc_id]
            self.logger.info(f"\n[{i + 1}/{total}] Processing document {doc_id} using {text_col}")

            try:
                # Determine which JSON to use based on document ID prefix
                if doc_id.startswith('ED'):
                    documents = documents_en
                    self.logger.info(f"  Using English JSON for {doc_id}")
                elif doc_id.startswith('CD'):
                    documents = documents_cn
                    self.logger.info(f"  Using Chinese JSON for {doc_id}")
                else:
                    self.logger.warning(f"  Unknown document ID prefix: {doc_id}")
                    fail_count += 1
                    continue

                # Find document in the appropriate JSON
                document = self.find_document_by_id(documents, doc_id)

                if not document:
                    self.logger.warning(f"  Document {doc_id} not found in JSON")
                    fail_count += 1
                    continue

                # Process document with the specified text column
                result = self.process_document(doc_id, document, text_col)

                if result:
                    # Append to Excel
                    self.append_result_to_excel(output_file, result)
                    success_count += 1
                    self.logger.info(f"  ✓ Document {doc_id} processed successfully")
                else:
                    fail_count += 1
                    self.logger.warning(f"  ✗ Document {doc_id} processing failed")

                # Add delay to avoid API rate limits
                time.sleep(0.5)

            except Exception as e:
                fail_count += 1
                self.logger.error(f"  ✗ Error processing document {doc_id}: {e}")

        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Processing Complete!")
        self.logger.info(f"Total documents: {total}")
        self.logger.info(f"Successfully processed: {success_count}")
        self.logger.info(f"Failed: {fail_count}")
        self.logger.info(f"Results saved to: {output_file}")
        self.logger.info(f"Log saved to: {log_file}")
        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Event Factuality Analysis Tool')
    parser.add_argument('--excel-input', default="ablation_res/ablation_without_bocha.xlsx",
                        help='Input Excel file path (containing text columns with Real labels)')
    parser.add_argument('--json-cn', default='hallu/extr-hallu-final.json',
                        help='Chinese JSON file path (for CD documents)')
    parser.add_argument('--json-en', default='hallu/en_experiment_documents2.json',
                        help='English JSON file path (for ED documents)')
    parser.add_argument('--output', default='ablation_res/without_bocha_factuality.xlsx',
                        help='Output Excel file path')
    parser.add_argument('--log', default='log/factuality.txt',
                        help='Log file path (default: log/factuality.txt)')
    parser.add_argument('--start-id', type=str, default='CD300',
                        help='Starting document ID (optional)')
    parser.add_argument('--end-id', type=str, default=None,
                        help='Ending document ID (optional)')
    parser.add_argument('--model', default=QWEN_MODEL,
                        help=f'Model name to use (default: {QWEN_MODEL})')

    args = parser.parse_args()

    # Create analyzer and run
    analyzer = EventFactualityAnalyzer(model=args.model)
    analyzer.run(
        excel_input=args.excel_input,
        json_cn=args.json_cn,
        json_en=args.json_en,
        output_file=args.output,
        log_file=args.log,
        start_id=args.start_id,
        end_id=args.end_id
    )


if __name__ == "__main__":
    main()
