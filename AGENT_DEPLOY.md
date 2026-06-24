# Agent Deploy Prompt

Copy the prompt below, replace the four values, and send it to your coding agent.

Required replacements:

1. `YOUR_LLM_BASE_URL`
2. `YOUR_LLM_MODEL`
3. `YOUR_LLM_API_KEY`
4. `YOUR_THE_ODDS_API_KEY`

Optional replacement:

1. `YOUR_ZHIHU_ACCESS_SECRET`

```text
你是一个本地部署 Agent。请在当前机器上部署 PitchSignals，并保证我可以通过浏览器访问。

项目要求：
1. 后端运行在 http://127.0.0.1:8000
2. 前端运行在 http://127.0.0.1:5173
3. 密钥只写入本地 `backend/.env`，回复中只确认配置完成，不展示完整密钥内容
4. 如果端口被占用，请换一个可用端口，并告诉我最终访问地址
5. 部署完成后，请检查 /health、赛程接口和前端页面

请创建或更新 backend/.env，内容如下：

LLM_ENABLED=true
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=YOUR_LLM_BASE_URL
LLM_MODEL=YOUR_LLM_MODEL
LLM_API_KEY=YOUR_LLM_API_KEY
LLM_TIMEOUT_SECONDS=120
LLM_MAX_TOKENS=900

THE_ODDS_API_KEY=YOUR_THE_ODDS_API_KEY
THE_ODDS_API_REGIONS=us,uk,eu
THE_ODDS_API_DEFAULT_SPORT_KEY=soccer_fifa_world_cup
THE_ODDS_API_SCHEDULE_SPORT_KEYS=soccer_fifa_world_cup,soccer_uefa_champs_league,soccer_epl,soccer_spain_la_liga,soccer_italy_serie_a,soccer_germany_bundesliga,soccer_france_ligue_one,soccer_usa_mls

AGENT_REACH_ENABLED=true
ZHIHU_SEARCH_ENABLED=false
ZHIHU_ACCESS_SECRET=
ZHIHU_SEARCH_MODE=zhihu
ZHIHU_SEARCH_MAX_QUERIES=4
ZHIHU_SEARCH_RESULTS_PER_QUERY=3
STRICT_WORKFLOW=true
PREDICTION_LEDGER_PATH=work/prediction_ledger.jsonl

如果我提供了知乎开放平台 Access Secret，请将：
ZHIHU_SEARCH_ENABLED=true
ZHIHU_ACCESS_SECRET=YOUR_ZHIHU_ACCESS_SECRET

部署步骤：
1. 进入 backend
2. 创建 Python 3.11+ 虚拟环境
3. 安装后端依赖：pip install -e .
4. 启动 FastAPI：uvicorn app.main:app --host 127.0.0.1 --port 8000
5. 进入 frontend
6. 安装前端依赖：npm install
7. 启动前端：npm run dev -- --port 5173
8. 检查 http://127.0.0.1:8000/health
9. 检查 http://127.0.0.1:8000/football/schedule/today?days=2
10. 检查 http://127.0.0.1:5173

如果 The Odds API 返回空赛程，请说明这是数据源/赛事可用性问题，而不是前端故障。
如果 LLM API 不兼容，请说明需要 OpenAI-compatible /v1/chat/completions 接口，或者需要换成兼容网关。

完成后，请告诉我：
1. 后端地址
2. 前端地址
3. health 检查结果
4. 赛程接口是否拿到比赛
5. 前端是否可以打开
```

## LLM Compatibility Notes

PitchSignals currently calls an OpenAI-compatible Chat Completions endpoint.

Most providers can work if they expose:

```text
POST /v1/chat/completions
Authorization: Bearer <api-key>
```

If your provider only supports native Anthropic, Gemini, or another protocol, use an OpenAI-compatible gateway or proxy.
