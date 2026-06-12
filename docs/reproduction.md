# Tscope 复现说明

Tscope 用来检测自然语言测试用例中的冗余。核心思想不是直接比较整段文本，而是先抽取测试相关的实体和关系，把每条测试用例解剖成原子测试 tuple，再用 tuple covering rule 判断冗余。

本仓库原 README 说明论文数据不能公开。因此这里复现的是方法链路的可运行版本：使用公开样例数据，默认调用 `deepseek-v4-flash` 做实体/关系抽取，后续 tuple 构造和冗余检测在本地确定性执行。

## 1. 方法理解

Tscope 的流程分为四步：

1. Data Pre-processing：清洗、分句、tokenize 测试用例文本。
2. Entity and Relation Extraction：抽取五类实体和四类关系。
3. Test Case Dissection：以 Component 为核心，把相关 Behavior、Prerequisite、Manner、Constraint 组合为 tuple。
4. Redundancy Detection：比较两个测试用例的 tuple，若一个测试用例的 tuple 能覆盖另一个测试用例的所有 tuple，则后者被判为冗余。

五类实体：

- `Component`：被测试的功能组件或对象。
- `Behavior`：作用在组件上的行为。
- `Prerequisite`：执行测试前的前置条件。
- `Manner`：测试执行方式、工具或交互方式。
- `Constraint`：需要满足的约束。

四类关系：

- `Act`：Component 与 Behavior。
- `Require`：Component 与 Prerequisite。
- `Use`：Component 与 Manner。
- `Satisfy`：Component 与 Constraint。

## 2. 文件说明

新增复现文件：

```text
reproduce_tscope.py
model_client.py
data/reproduction/sample_testcases.json
requirements_reproduction.txt
docs/reproduction.md
```

默认样例数据包含 7 条测试用例，覆盖以下场景：

- 行为和组件映射不同，因此不冗余。
- 执行方式不同，因此不冗余。
- 约束不同，因此不冗余。
- `browse visit history` 与 `view visit history` 语义等价，因此冗余。

## 3. 环境准备

当前脚本只依赖 Python 标准库和 `requests`。

```bash
cd /home/public/minzhi/testcase_detection
python3 -m pip install -r requirements_reproduction.txt
```

当前机器系统 Python 已经有 `requests`，所以已直接跑通。

## 4. API key 和模型配置

推荐使用环境变量：

```bash
export ANTCHAT_API_KEY="YOUR_REAL_KEY"
```

如果没有设置环境变量，脚本会按当前机器已有方式读取：

```text
/home/public/minzhi/code/llm_judge_pressure_risk_sessions.py 里的 API_KEY
```

默认大模型配置：

- API base URL：`https://antchat.alipay.com`
- Model：`deepseek-v4-flash`
- 调用封装：`model_client.py`

如果后续要换成其他官方 OpenAI-compatible endpoint，可以这样传参：

```bash
python3 reproduce_tscope.py \
  --api-base-url https://api.deepseek.com \
  --model deepseek-chat
```

## 5. 一键运行

默认使用 DeepSeek v4 Flash 做实体/关系抽取：

```bash
cd /home/public/minzhi/testcase_detection
NO_PROXY=antchat.alipay.com,antchat-gray.alipay.com \
python3 reproduce_tscope.py \
  --extractor llm \
  --model deepseek-v4-flash \
  --output-dir outputs/reproduction_tscope_llm \
  --threshold 0.65
```

如需离线 smoke test，可以改用规则抽取：

```bash
python3 reproduce_tscope.py \
  --extractor rule \
  --output-dir outputs/reproduction_tscope_rule
```

## 6. 输出文件

本次真实运行输出目录：

```text
/home/public/minzhi/testcase_detection/outputs/reproduction_tscope_llm
```

主要文件：

- `extractions.json`：DeepSeek 抽取出的实体和关系。
- `tuples.json`：每条测试用例被解剖后的 tuple。
- `redundancy_results.json`：tuple covering 冗余检测结果。

## 7. 本次真实运行结果

已在当前机器调用 `deepseek-v4-flash` 跑通一次。

关键结果：

```text
test_case_count: 7
redundant_pair_count: 2
redundant pair:
  TC-525-like <-> TC-525-dup
```

典型 tuple：

```json
{
  "Component": "visit history",
  "Behavior": "browse",
  "Prerequisite": "after opening the resource center",
  "Manner": "mouse",
  "Constraint": "NULL"
}
```

带约束样例也正确进入 tuple：

```json
{
  "Component": "preset applications",
  "Behavior": "verify",
  "Prerequisite": "after the system installation",
  "Manner": "NULL",
  "Constraint": "including ftp application"
}
```

这说明复现链路已经真实跑通：DeepSeek v4 Flash 完成实体/关系抽取，本地代码完成 tuple 构造和冗余判定。
