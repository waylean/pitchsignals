# PitchSignals

> 透明、证据驱动的足球赛前预测研究工具。

[English README](./README.en.md)

PitchSignals 是一个面向足球比赛预测研究的本地应用。它可以获取赛程和 1X2 赔率，收集公开信息，并通过多维约束分析比赛胜、平、负的倾向概率、置信度和权重关系。

## 项目定位

PitchSignals 的核心理念是：

> 不隐藏不确定性，不夸大预测能力，把足球预测变成可以审计、可以复盘、可以逐步改进的研究流程。

PitchSignals 不是投注平台，不是投注技巧工具，也不是商业预测 API。它是一个用于研究足球预测链路、数据层质量、权重关系和模型校准的实验项目。

## 界面预览

打开项目后，用户先看到简洁的赛程入口。

![PitchSignals 初始界面](./assets/pitchsignals-01-empty.png)

点击“获取赛程”后，可以看到今明两天可预测的比赛和 1X2 赔率。

![PitchSignals 赛程界面](./assets/pitchsignals-02-schedule.png)

选择一场比赛并预测后，系统会展示胜平负概率、综合置信度和参考数据权重。

![PitchSignals 预测结果界面](./assets/pitchsignals-03-prediction.png)

## 致谢

PitchSignals 的足球相关信息搜索流程参考并使用了 [Agent-Reach](https://github.com/Panniantong/Agent-Reach) 的思路与能力。Agent-Reach 是一个让 AI Agent 获得互联网搜索和信息读取能力的开源项目，它提供了面向多平台公开信息搜索的工具化能力。

感谢 Agent-Reach 项目让 AI Agent 能更方便地获取公开网络信息，也让本项目可以围绕足球新闻、赛前预览、阵容传闻、战术讨论等信息做进一步研究。

## 重要声明

PitchSignals 仅用于研究、学习和模型实验。

严禁将本项目用于：

1. 博彩行业
2. 赌博、投注平台、体育投注推荐
3. 商业预测服务
4. 自动化下注
5. 任何以盈利为目的的体育博彩相关用途

本项目输出的概率和结论不构成投注建议、投资建议或任何形式的商业决策建议。足球比赛存在高度不确定性，任何预测结果都可能错误。

许可证选择：本项目采用非商业用途许可证，见 [LICENSE](./LICENSE)。

## 项目能做什么

PitchSignals 当前支持：

1. 获取今明两天的足球赛程
2. 获取比赛 1X2 市场赔率
3. 将赔率转换为市场隐含概率
4. 收集球队实力、阵容伤停、战术对位、裁判环境、舆论叙事等信息
5. 使用 LLM 将难结构化的公开文本转成低置信结构化信号
6. 通过足球预测权重模型输出胜、平、负概率
7. 给出明确预测结果、综合置信度和参考数据权重
8. 在切换比赛时保留已经预测过的结果
9. 支持中文/英文界面，中文界面会显示中文国家名

## 需要准备什么

基础环境：

1. Python 3.11+
2. Node.js 18+
3. 一个兼容 OpenAI API 协议的 LLM API key
4. 一个免费的 The Odds API key

The Odds API 官网：[https://the-odds-api.com/](https://the-odds-api.com/)

LLM 接入方式：

PitchSignals 默认使用 OpenAI-compatible Chat Completions 接口。只要你的模型服务提供兼容 `/v1/chat/completions` 的接口，一般都可以通过以下三个变量接入：

```env
LLM_BASE_URL=https://your-openai-compatible-base-url/v1
LLM_MODEL=your-model-name
LLM_API_KEY=your-llm-api-key
```

这类方式通常可以兼容 OpenAI、OpenRouter、DeepSeek、Mimo、Together、SiliconFlow 等提供 OpenAI-compatible endpoint 的服务。若某个模型只提供原生 Anthropic 或其他非 OpenAI 协议，需要先使用兼容代理或网关。

后端环境变量示例：

```env
LLM_ENABLED=true
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://your-openai-compatible-base-url/v1
LLM_MODEL=your-model-name
LLM_API_KEY=your-llm-api-key

THE_ODDS_API_KEY=your-the-odds-api-key
THE_ODDS_API_REGIONS=us,uk,eu
THE_ODDS_API_SCHEDULE_SPORT_KEYS=soccer_fifa_world_cup,soccer_uefa_champs_league,soccer_epl,soccer_spain_la_liga,soccer_italy_serie_a,soccer_germany_bundesliga,soccer_france_ligue_one,soccer_usa_mls
```

## Agent 快速部署

如果你使用 Codex、Cursor、Claude Code、OpenHands 或其他编程 Agent，可以直接使用 [AGENT_DEPLOY.md](./AGENT_DEPLOY.md) 里的部署指令。

最短方式：

1. 打开 [AGENT_DEPLOY.md](./AGENT_DEPLOY.md)
2. 替换 `YOUR_LLM_API_KEY`、`YOUR_LLM_BASE_URL`、`YOUR_LLM_MODEL`、`YOUR_THE_ODDS_API_KEY`
3. 把整段指令发给 Agent
4. Agent 会安装依赖、写入本地 `.env`、启动后端和前端，并返回可访问地址

## 如果拿不到赔率怎么办

赔率是非常重要的方向性数据，因为市场价格通常聚合了大量公开和半公开信息。

如果 The Odds API 拿不到数据：

1. 赛程页面可能无法正常显示比赛
2. 市场赔率因子会缺失
3. 预测稳定性会下降
4. 系统会更多依赖球队实力、公开文本、阵容、战术、裁判环境等其他因子
5. 最终置信度应当降低

换句话说：没有赔率时，系统仍然可以保留研究价值，但预测结果不应该被视为同等稳定。

## 快速启动

启动后端：

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e .
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

启动前端：

```bash
cd frontend
npm install
npm run dev -- --port 5173
```

打开：

```text
http://127.0.0.1:5173
```

## 使用流程

1. 点击“获取赛程”
2. 选择一场比赛
3. 点击“预测这场比赛”
4. 查看预测结果：
   - AI 给出的预测结果
   - 胜、平、负概率
   - 综合置信度
   - 资料覆盖、信息时效、预测稳定性
   - 预测参考数据权重

## 架构说明

PitchSignals 的核心架构分为四层：

```text
数据层
  -> 预测权重层
    -> 预测层
      -> 反馈优化层
```

### 1. 数据层

数据层是项目的核心。

它负责收集、验证、筛选和结构化足球比赛相关信息。当前覆盖：

1. 赛程
2. 1X2 赔率
3. 市场隐含概率
4. 世界足球 Elo / 球队实力
5. 阵容与伤停
6. 预测首发与可用性
7. 战术对位
8. 裁判与环境
9. 天气、地点、时间等上下文
10. 新闻、赛前预览、非官方消息
11. 舆论与叙事

数据层会尽量区分：

1. 官方信息
2. 半官方信息
3. 媒体报道
4. 预测类内容
5. 非官方传闻

非官方信息不会直接高权重影响预测。它会先通过 LLM 转成低置信结构化信号，再进入模型。

### 2. 预测权重层

预测权重层用于决定不同数据因子对最终预测的影响。

当前权重模型来源于长期足球比赛记录、历史赛果、赔率数据和回测实验。它不是手写拍脑袋规则，而是基于历史比赛中的可复盘数据逐步训练和校准出来的权重框架。

当前主要因子包括：

1. 市场赔率
2. 球队实力
3. 阵容与伤停
4. 战术对位
5. 裁判与环境
6. 舆论与叙事

如果某个因子缺失，系统不应该假装数据完整，而是降低对应因子的可信度，并降低最终综合置信度。

### 3. 预测层

预测层会综合：

1. 结构化数据
2. 市场赔率
3. 因子权重
4. 因子可信度
5. 模型一致性

最终输出：

1. 胜、平、负概率
2. 推荐倾向
3. 综合置信度
4. 预测参考数据

预测层不是让 LLM 直接“猜结果”。LLM 主要负责信息结构化和证据审阅，最终概率由可审计的权重模型生成。

### 4. 反馈优化层

比赛结束后，可以将真实结果回填。

反馈优化层的目标是：

1. 记录预测是否正确
2. 对比预测概率和真实结果
3. 分析错误来自哪个因子
4. 判断是否需要调整权重
5. 逐步改进模型校准

长期来看，反馈数据会形成 Model Leaderboard、Evaluation Report 和版本化 ensemble weights。

## 数据层的广泛性

PitchSignals 不只看球队实力或赔率。

一个足球比赛结果可能受到很多因素影响，例如：

1. 球队长期实力
2. 近期状态
3. 伤停情况
4. 预计首发
5. 教练战术
6. 比赛地点
7. 天气
8. 裁判尺度
9. 赛程密度
10. 市场赔率变化
11. 新闻舆论
12. 非官方消息

本项目的价值在于把这些信息放进同一个可解释框架里，并让每个信息源都有权重、可信度和缺失状态。

## 当前价值与边界

PitchSignals 适合作为：

1. 足球预测研究工具
2. 数据收集与特征工程实验
3. 赔率和模型差异分析工具
4. LLM 结构化公开信息的实验项目
5. 可解释预测系统原型

PitchSignals 不适合作为：

1. 博彩工具
2. 自动投注平台
3. 商业预测 API
4. 高风险决策系统

## 运行检查

后端：

```bash
python -m compileall backend/app
```

前端：

```bash
cd frontend
npm run build
```

健康检查：

```text
GET http://127.0.0.1:8000/health
```

赛程接口：

```text
GET http://127.0.0.1:8000/football/schedule/today?days=2
```
