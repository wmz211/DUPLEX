# 全量迁移至 Qwen 3.8 Flash 实施计划

## 目标

将所有运行时 LLM 调用统一到 Qwen 兼容接口和 `qwen3.8-flash`，移除 DeepSeek 配置依赖，安全配置本地 Qwen 环境变量，并验证当前工作区、Git 索引和历史中不存在真实 API Key。

## 任务一：统一客户端和模型配置

涉及文件：

- `efi_pilot/utils/api_clients.py`
- `efi_pilot/config.py`
- `efi_pilot/utils/environment.py`
- `.env.example`
- `tests/test_environment.py`

步骤：

1. 在无第三方依赖的配置模块中新增统一模型常量 `QWEN_MODEL = "qwen3.8-flash"`。
2. 删除 DeepSeek 客户端构造和 `DEEPSEEK_API_KEY` 映射。
3. 将线程客户端工厂改为只接收 Bocha Key 与 Qwen Key。
4. 更新环境变量示例和测试，确认只要求实际使用的变量。

## 任务二：迁移模块化 Agent 和编排入口

涉及文件：

- `efi_pilot/agents/dispatcher.py`
- `efi_pilot/agents/arbiter.py`
- `efi_pilot/agents/investigator.py`
- `efi_pilot/agents/linguist.py`
- `efi_pilot/orchestrator.py`
- `efi_pilot/main.py`
- `tests/test_investigator_loop.py`

步骤：

1. 所有 Agent 从统一位置导入模型常量。
2. Dispatcher、Arbiter 和 Investigator 查询生成改用 `clients["qwen"]`。
3. 所有模型参数改为统一常量。
4. HD 配置删除 `deepseek_key`，只传递 Bocha 与 Qwen。
5. 更新相关 Mock 和调用断言。

## 任务三：迁移独立旧入口

涉及文件：

- `qwen_search.py`
- `predict_factuality.py`

步骤：

1. 删除旧 HD 类的 DeepSeek 客户端和构造参数。
2. 将原 DeepSeek 调用改用 Qwen 客户端及统一模型常量。
3. 更新启动时环境变量读取与实例化参数。
4. 将事件事实性脚本的默认模型和 CLI 默认值改为统一常量。

## 任务四：本地凭据与安全扫描

步骤：

1. 将用户提供的 Qwen Key 写入 Windows 当前用户级 `QWEN_API_KEY`，命令与输出不得打印变量值。
2. 只检查 `QWEN_API_KEY` 是否存在及长度是否非零，不回显内容。
3. 使用脱敏扫描检查工作树、Git 索引与所有可达提交中的疑似真实密钥，只记录路径、提交和行号。
4. 确认 `.env` 与常见本地凭据文件未被 Git 跟踪。
5. 如历史扫描命中真实密钥，停止普通提交并报告需要轮换和历史清理。

## 任务五：验证与提交

步骤：

1. 扫描运行时代码中所有 `model=`，确认统一为模型常量或 `qwen3.8-flash`。
2. 确认运行时代码不存在 DeepSeek 客户端、模型或环境变量依赖。
3. 运行 `unittest discover`。
4. 运行 `compileall` 和 `git diff --check`。
5. 仅提交源码、测试、示例配置和实施计划；不提交任何密钥或本地生成文件。
