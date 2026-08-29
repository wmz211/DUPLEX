# 全量迁移至 Qwen 3.8 Flash 设计

## 目标

将仓库中所有大语言模型调用统一迁移到 Qwen 兼容接口，并固定使用 `qwen3.8-flash`。删除 DeepSeek 客户端及其环境变量依赖。Bocha 继续作为独立搜索服务使用。

任何真实 API Key 都不得写入源码、示例配置、测试数据或 Git 提交。用户提供的 Qwen Key 只允许存在于本地环境配置中。

## 范围

- `DispatcherAgent`、`ArbiterAgent`、`InvestigatorAgent` 和 `LinguistAgent` 全部使用 Qwen 客户端。
- `InvestigatorAgent` 的查询生成与联网事实核查使用同一 Qwen 客户端。
- 同步迁移独立入口 `qwen_search.py` 和 `predict_factuality.py`，避免旧入口继续依赖 DeepSeek 或旧 Qwen 模型。
- 将模型名集中定义为单一常量 `qwen3.8-flash`，各调用点引用该常量。
- HD 模式只要求 `QWEN_API_KEY` 与 `BOCHA_API_KEY`；EFI 模式只要求 `QWEN_API_KEY`。
- 删除 `DEEPSEEK_API_KEY` 示例、读取逻辑、客户端构造和配置传递。

## 架构与数据流

`efi_pilot.utils.api_clients` 提供 Qwen 客户端、Bocha 搜索函数和统一模型常量。HD 编排器为每个工作线程创建 Qwen 客户端，并以 `qwen` 键传给各 Agent。Dispatcher、Arbiter、Investigator 和 Linguist 不再访问 `deepseek` 键。

命令行入口从环境变量读取必要密钥：HD 读取 Qwen 与 Bocha，EFI 只读取 Qwen。密钥值只在内存中的配置和客户端之间传递，日志与异常信息不得输出密钥。

旧版 `qwen_search.py` 删除 DeepSeek 客户端参数和成员，原先由 DeepSeek 完成的查询生成与文本分析改用已有 Qwen 客户端。`predict_factuality.py` 的默认模型同步改为统一常量。

## 本地密钥配置

仓库继续只通过环境变量读取真实凭据，不在版本控制中保存带值的 `.env`。`.env.example` 只保留：

```text
BOCHA_API_KEY=
QWEN_API_KEY=
```

本次实现可将用户提供的 Qwen Key 写入 Windows 当前用户级 `QWEN_API_KEY`，但执行过程和最终报告不得回显完整值。新启动的终端和进程读取该用户环境变量；当前已运行进程不保证自动刷新。

## 错误处理

- 缺少必要环境变量时，在创建客户端之前报出缺少的变量名，不包含变量值。
- Qwen 或 Bocha 的既有网络异常处理保持不变。
- 不再接受或回退到 DeepSeek Key；迁移完成后出现 DeepSeek 配置应被测试或扫描识别为遗留项。

## 验证

1. 扫描 Python、配置和文档中的模型字面量，确认运行时代码只使用 `qwen3.8-flash`。
2. 扫描 `deepseek` 客户端引用和 `DEEPSEEK_API_KEY`，确认运行时代码与配置示例不存在遗留依赖。
3. 更新环境变量测试、Agent 调用测试及客户端构造测试。
4. 运行全部单元测试和 Python 编译检查。
5. 对工作区、Git 索引和所有可达提交执行脱敏扫描；报告命中位置但不输出疑似密钥内容。
6. 确认真实密钥文件未被跟踪，提交只包含源码、测试、示例配置和设计/计划文档。

## Git 安全边界

扫描规则覆盖常见 API Key 前缀和已知环境变量赋值，但输出只包含提交、路径和行号。若 Git 历史存在真实密钥，必须先撤销该密钥，再使用精确替换重写历史并强制更新远端；不能仅靠删除当前文件消除历史泄露。

本次提交不得包含用户提供的 Qwen Key。由于密钥已通过对话明文传递，建议完成配置后在服务控制台轮换，最终只使用新生成且从未进入仓库或对话记录的凭据。
