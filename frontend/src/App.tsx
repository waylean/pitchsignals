import { useMemo, useState } from "react";
import {
  Activity,
  CalendarDays,
  CheckCircle2,
  CloudSun,
  Gauge,
  Loader2,
  SearchCheck,
  ShieldAlert,
  Sparkles,
  Trophy,
} from "lucide-react";

type Language = "zh" | "en";

type ScheduleMatch = {
  match_id: string;
  sport_key: string;
  league?: string | null;
  commence_time: string;
  home_team: string;
  away_team: string;
  odds?: {
    available?: boolean;
    home_odds?: number;
    draw_odds?: number;
    away_odds?: number;
    home_probability?: number;
    draw_probability?: number;
    away_probability?: number;
    bookmaker_count?: number;
    bookmakers?: string[];
  };
  source: string;
};

type ScheduleResponse = {
  status: string;
  date: string;
  timezone: string;
  matches: ScheduleMatch[];
  notes?: string[];
};

type FactorScore = {
  key: string;
  label: string;
  value: number;
  weight: number;
  confidence: number;
  evidence_count: number;
  rationale: string;
};

type PredictionResponse = {
  task_id: string;
  normalized_question: string;
  outcomes: Record<string, number>;
  pick?: string | null;
  pick_probability?: number | null;
  confidence: number;
  data_completeness: number;
  freshness: number;
  model_agreement: number;
  evidence_gate_status: string;
  factors: FactorScore[];
  uncertainties: string[];
  workflow_trace: string[];
  research_review?: {
    status?: string;
    summary?: string;
    route_check?: Record<string, { status?: string; reason?: string }>;
    missing_data?: string[];
    risk_flags?: string[];
    _llm?: { model?: string };
  } | null;
};

const API_BASE = "http://127.0.0.1:8000";

export function App() {
  const [language, setLanguage] = useState<Language>("zh");
  const [schedule, setSchedule] = useState<ScheduleResponse | null>(null);
  const [selectedMatchId, setSelectedMatchId] = useState<string | null>(null);
  const [predictionsByMatch, setPredictionsByMatch] = useState<Record<string, PredictionResponse>>({});
  const [isLoadingSchedule, setIsLoadingSchedule] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictingMatchId, setPredictingMatchId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectedMatch = useMemo(
    () => schedule?.matches.find((match) => match.match_id === selectedMatchId) ?? null,
    [schedule, selectedMatchId]
  );
  const currentPrediction = selectedMatchId ? predictionsByMatch[selectedMatchId] ?? null : null;

  const fetchTodaySchedule = async () => {
    setIsLoadingSchedule(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/football/schedule/today?days=2`);
      if (!response.ok) throw new Error(`schedule ${response.status}`);
      const data = (await response.json()) as ScheduleResponse;
      setSchedule(data);
      setSelectedMatchId(data.matches[0]?.match_id ?? null);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : t(language, "scheduleFailed"));
    } finally {
      setIsLoadingSchedule(false);
    }
  };

  const predictSelectedMatch = async () => {
    if (!selectedMatch) return;
    const matchId = selectedMatch.match_id;
    setIsPredicting(true);
    setPredictingMatchId(matchId);
    setError(null);
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 210000);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          question: `${selectedMatch.home_team} vs ${selectedMatch.away_team}, who will win?`,
          domain: "football",
          outcome_type: "three_way",
          outcomes: ["home_win", "draw", "away_win"],
          event_time: selectedMatch.commence_time,
          context: {
            skip_ai_decomposition: true,
            competitors: [selectedMatch.home_team, selectedMatch.away_team],
            match_id: selectedMatch.match_id,
            the_odds_api_sport_key: selectedMatch.sport_key,
            match_source: "football_schedule",
            league: selectedMatch.league,
            schedule_odds: selectedMatch.odds,
            search_query: `${selectedMatch.home_team} ${selectedMatch.away_team} ${selectedMatch.league ?? ""} preview prediction predicted lineup tactical analysis`,
          },
        }),
      });
      if (!response.ok) throw new Error(`prediction ${response.status}`);
      const nextPrediction = (await response.json()) as PredictionResponse;
      setPredictionsByMatch((previous) => ({
        ...previous,
        [matchId]: nextPrediction,
      }));
    } catch (exc) {
      const aborted = exc instanceof DOMException && exc.name === "AbortError";
      setError(aborted ? t(language, "predictionTimeout") : exc instanceof Error ? exc.message : t(language, "predictionFailed"));
    } finally {
      window.clearTimeout(timeout);
      setIsPredicting(false);
      setPredictingMatchId(null);
    }
  };

  return (
    <main className="footballShell">
      <section className="footballTopbar">
        <div className="brandMark">
          <Trophy size={24} />
        </div>
        <div>
          <h1>{t(language, "appTitle")}</h1>
          <p>{t(language, "appSubtitle")}</p>
        </div>
        <div className="topbarActions">
          <button className={language === "zh" ? "active" : ""} type="button" onClick={() => setLanguage("zh")}>
            中
          </button>
          <button className={language === "en" ? "active" : ""} type="button" onClick={() => setLanguage("en")}>
            EN
          </button>
        </div>
      </section>

      <section className="footballWorkspace">
        <aside className="fixturePanel">
          <div className="panelHeader">
            <div>
              <span>{t(language, "today")}</span>
              <strong className="scheduleDate">{formatScheduleDateRange(schedule?.date, language)}</strong>
            </div>
            <button className="primaryButton" type="button" onClick={fetchTodaySchedule} disabled={isLoadingSchedule}>
              {isLoadingSchedule ? <Loader2 className="spin" size={17} /> : <CalendarDays size={17} />}
              {t(language, "fetchSchedule")}
            </button>
          </div>

          {error && (
            <div className="errorBanner">
              <ShieldAlert size={17} />
              <span>{error}</span>
            </div>
          )}

          <div className="fixtureList">
            {!schedule && <EmptyState icon={CalendarDays} text={t(language, "emptySchedule")} />}
            {schedule && schedule.matches.length === 0 && (
              <EmptyState icon={CloudSun} text={schedule.notes?.[0] ?? t(language, "noMatches")} />
            )}
            {schedule?.matches.map((match) => (
              <button
                className={`fixtureCard ${selectedMatchId === match.match_id ? "selected" : ""}`}
                key={`${match.sport_key}-${match.match_id}`}
                type="button"
                onClick={() => {
                  setSelectedMatchId(match.match_id);
                }}
              >
                <span>{formatTime(match.commence_time)}</span>
                <strong>
                  {displayTeamName(match.home_team, language)} <small>vs</small> {displayTeamName(match.away_team, language)}
                </strong>
                <em>{match.league ?? match.sport_key}</em>
                {match.odds?.available && (
                  <div className="miniOdds">
                    <b>{match.odds.home_odds?.toFixed(2)}</b>
                    <b>{match.odds.draw_odds?.toFixed(2)}</b>
                    <b>{match.odds.away_odds?.toFixed(2)}</b>
                  </div>
                )}
              </button>
            ))}
          </div>
        </aside>

        <section className="predictionPanel">
          <div className="selectedMatchHeader">
            {selectedMatch ? (
              <>
                <div>
                  <span>{selectedMatch.league ?? selectedMatch.sport_key}</span>
                  <h2>
                    {displayTeamName(selectedMatch.home_team, language)} <small>vs</small> {displayTeamName(selectedMatch.away_team, language)}
                  </h2>
                  <p>{formatDateTime(selectedMatch.commence_time)}</p>
                </div>
                <button className="predictButton" type="button" onClick={predictSelectedMatch} disabled={isPredicting}>
                  {isPredicting && predictingMatchId === selectedMatch.match_id ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
                  {t(language, "predict")}
                </button>
              </>
            ) : (
              <EmptyState icon={SearchCheck} text={t(language, "selectMatch")} />
            )}
          </div>

          {selectedMatch?.odds?.available && <OddsStrip match={selectedMatch} language={language} />}

          {isPredicting && selectedMatch?.match_id === predictingMatchId && (
            <div className="loadingResearch">
              <Loader2 className="spin" size={24} />
              <div>
                <strong>{t(language, "researching")}</strong>
                <span>{t(language, "researchingDetail")}</span>
              </div>
            </div>
          )}

          {currentPrediction && selectedMatch && <PredictionReport prediction={currentPrediction} match={selectedMatch} language={language} />}
        </section>
      </section>
    </main>
  );
}

function PredictionReport({ prediction, match, language }: { prediction: PredictionResponse; match: ScheduleMatch; language: Language }) {
  const teamLabels = {
    home_win: displayTeamName(match.home_team, language),
    draw: t(language, "draw"),
    away_win: displayTeamName(match.away_team, language),
  };
  const pick = prediction.pick ?? topOutcome(prediction.outcomes);
  return (
    <div className="reportBlock">
      <div className="verdictCard">
        <span>{t(language, "aiPick")}</span>
        <strong>{teamLabels[pick as keyof typeof teamLabels] ?? pick}</strong>
        <p>
          {t(language, "leanProbability")} {percent(prediction.pick_probability ?? prediction.outcomes[pick] ?? 0)}
          {" · "}
          {t(language, "confidence")} {percent(prediction.confidence)}
        </p>
      </div>

      <div className="metricGrid">
        <Metric
          icon={Gauge}
          label={t(language, "evidenceCoverage")}
          value={scoreLabel(prediction.data_completeness, language, "coverage")}
          detail={`${t(language, "technicalScore")} ${percent(prediction.data_completeness)}`}
        />
        <Metric
          icon={Activity}
          label={t(language, "infoTimeliness")}
          value={scoreLabel(prediction.freshness, language, "freshness")}
          detail={`${t(language, "technicalScore")} ${percent(prediction.freshness)}`}
        />
        <Metric
          icon={CheckCircle2}
          label={t(language, "predictionStability")}
          value={scoreLabel(prediction.model_agreement, language, "agreement")}
          detail={`${t(language, "technicalScore")} ${percent(prediction.model_agreement)}`}
        />
      </div>

      <div className="probabilityRows">
        {Object.entries(prediction.outcomes).map(([outcome, probability]) => (
          <div className={outcome === pick ? "probabilityRow picked" : "probabilityRow"} key={outcome}>
            <span>{teamLabels[outcome as keyof typeof teamLabels] ?? outcome}</span>
            <i>
              <b style={{ width: `${Math.max(probability * 100, 2)}%` }} />
            </i>
            <strong>{percent(probability)}</strong>
          </div>
        ))}
      </div>

      <section className="factorSection">
        <h3>{t(language, "factorTitle")}</h3>
        <div className="factorCards">
          {prediction.factors.map((factor) => (
            <article className="factorCard compact" key={factor.key}>
              <div>
                <strong>{factorLabel(factor.key, factor.label, language)}</strong>
              </div>
              <aside>
                <b>{percent(factor.weight)}</b>
                <span>
                  {factor.evidence_count} {t(language, "evidence")} / {percent(factor.confidence)}
                </span>
              </aside>
            </article>
          ))}
        </div>
      </section>

    </div>
  );
}

function OddsStrip({ match, language }: { match: ScheduleMatch; language: Language }) {
  const odds = match.odds;
  if (!odds?.available) return null;
  return (
    <div className="oddsStrip">
      <span>{t(language, "marketOdds")}</span>
      <b>{displayTeamName(match.home_team, language)}: {odds.home_odds?.toFixed(2)}</b>
      <b>{t(language, "draw")}: {odds.draw_odds?.toFixed(2)}</b>
      <b>{displayTeamName(match.away_team, language)}: {odds.away_odds?.toFixed(2)}</b>
      <em>{odds.bookmaker_count ?? 0} {t(language, "books")}</em>
    </div>
  );
}

function Metric({ icon: Icon, label, value, detail }: { icon: typeof Gauge; label: string; value: string; detail?: string }) {
  return (
    <div className="metricCard" title={detail}>
      <Icon size={18} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyState({ icon: Icon, text }: { icon: typeof CalendarDays; text: string }) {
  return (
    <div className="emptyState">
      <Icon size={22} />
      <span>{text}</span>
    </div>
  );
}

function topOutcome(outcomes: Record<string, number>) {
  return Object.entries(outcomes).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "draw";
}

function percent(value: number) {
  return `${Math.round((value || 0) * 100)}%`;
}

function displayTeamName(name: string, language: Language) {
  if (language === "en") return name;
  const normalized = name.toLowerCase().replace(/\s+/g, " ").trim();
  const directNames: Record<string, string> = {
    england: "英格兰",
    scotland: "苏格兰",
    wales: "威尔士",
    "northern ireland": "北爱尔兰",
    kosovo: "科索沃",
    "czech republic": "捷克",
    "south korea": "韩国",
    "north korea": "朝鲜",
    iran: "伊朗",
    russia: "俄罗斯",
    "bosnia & herzegovina": "波黑",
    "bosnia and herzegovina": "波黑",
    "cape verde": "佛得角",
    "ivory coast": "科特迪瓦",
    "dr congo": "刚果（金）",
    "congo dr": "刚果（金）",
    "congo": "刚果（布）",
    "usa": "美国",
    "united states": "美国",
    "uae": "阿联酋",
    "united arab emirates": "阿联酋",
  };
  if (directNames[normalized]) return directNames[normalized];

  const code = REGION_CODE_BY_TEAM[normalized];
  if (code) {
    try {
      return new Intl.DisplayNames(["zh-CN"], { type: "region" }).of(code) ?? name;
    } catch {
      return name;
    }
  }
  return name;
}

const REGION_CODE_BY_TEAM: Record<string, string> = {
  afghanistan: "AF",
  albania: "AL",
  algeria: "DZ",
  andorra: "AD",
  angola: "AO",
  argentina: "AR",
  armenia: "AM",
  australia: "AU",
  austria: "AT",
  azerbaijan: "AZ",
  bahrain: "BH",
  bangladesh: "BD",
  belarus: "BY",
  belgium: "BE",
  bolivia: "BO",
  botswana: "BW",
  brazil: "BR",
  bulgaria: "BG",
  cameroon: "CM",
  canada: "CA",
  chile: "CL",
  china: "CN",
  colombia: "CO",
  "costa rica": "CR",
  croatia: "HR",
  cyprus: "CY",
  denmark: "DK",
  ecuador: "EC",
  egypt: "EG",
  estonia: "EE",
  finland: "FI",
  france: "FR",
  georgia: "GE",
  germany: "DE",
  ghana: "GH",
  greece: "GR",
  honduras: "HN",
  hungary: "HU",
  iceland: "IS",
  india: "IN",
  indonesia: "ID",
  iraq: "IQ",
  ireland: "IE",
  israel: "IL",
  italy: "IT",
  jamaica: "JM",
  japan: "JP",
  jordan: "JO",
  kazakhstan: "KZ",
  kuwait: "KW",
  latvia: "LV",
  lebanon: "LB",
  liechtenstein: "LI",
  lithuania: "LT",
  luxembourg: "LU",
  malaysia: "MY",
  malta: "MT",
  mexico: "MX",
  moldova: "MD",
  montenegro: "ME",
  morocco: "MA",
  netherlands: "NL",
  "new zealand": "NZ",
  nigeria: "NG",
  norway: "NO",
  oman: "OM",
  panama: "PA",
  paraguay: "PY",
  peru: "PE",
  poland: "PL",
  portugal: "PT",
  qatar: "QA",
  romania: "RO",
  "saudi arabia": "SA",
  serbia: "RS",
  singapore: "SG",
  slovakia: "SK",
  slovenia: "SI",
  "south africa": "ZA",
  spain: "ES",
  sweden: "SE",
  switzerland: "CH",
  syria: "SY",
  thailand: "TH",
  tunisia: "TN",
  turkey: "TR",
  ukraine: "UA",
  uruguay: "UY",
  uzbekistan: "UZ",
  venezuela: "VE",
  vietnam: "VN",
};

function scoreLabel(value: number, language: Language, type: "coverage" | "freshness" | "agreement") {
  const score = value || 0;
  if (language === "en") {
    if (type === "coverage") return score >= 0.72 ? "Well supported" : score >= 0.48 ? "Usable" : "Limited";
    if (type === "freshness") return score >= 0.7 ? "Fresh" : score >= 0.45 ? "Recent enough" : "Stale";
    return score >= 0.72 ? "Very stable" : score >= 0.48 ? "Fairly stable" : "Not stable enough";
  }
  if (type === "coverage") return score >= 0.72 ? "资料充分" : score >= 0.48 ? "资料可用" : "资料偏少";
  if (type === "freshness") return score >= 0.7 ? "信息很新" : score >= 0.45 ? "时效可用" : "信息偏旧";
  return score >= 0.72 ? "非常稳定" : score >= 0.48 ? "比较稳定" : "不够稳定";
}

function formatTime(value: string) {
  return new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function formatScheduleDateRange(value: string | undefined, language: Language) {
  if (!value) {
    return new Intl.DateTimeFormat(language === "zh" ? "zh-CN" : "en", {
      month: "short",
      day: "numeric",
    }).format(new Date());
  }
  const [start, end] = value.split("..");
  const startLabel = formatScheduleDatePart(start, language);
  const endLabel = end ? formatScheduleDatePart(end, language) : "";
  return endLabel && endLabel !== startLabel ? `${startLabel} - ${endLabel}` : startLabel;
}

function formatScheduleDatePart(value: string, language: Language) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value.trim());
  if (!match) return value;
  const month = Number(match[2]);
  const day = Number(match[3]);
  if (language === "zh") return `${month}月${day}日`;
  const date = new Date(Number(match[1]), month - 1, day);
  return new Intl.DateTimeFormat("en", { month: "short", day: "numeric" }).format(date);
}

function formatDateTime(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function t(language: Language, key: string) {
  const zh: Record<string, string> = {
    appTitle: "足球比赛预测",
    appSubtitle: "获取今明两天赛程，选择比赛后自动采集赔率、实力、阵容、战术和非官方候选信息。",
    today: "今明赛程",
    fetchSchedule: "获取赛程",
    emptySchedule: "点击获取今明两天可预测的足球比赛。",
    noMatches: "今明两天没有获取到可预测赛程。",
    scheduleFailed: "赛程获取失败。",
    predictionFailed: "预测失败。",
    predictionTimeout: "预测等待超时，后端可能仍在采集信息或等待 AI 结构化。",
    selectMatch: "请选择一场比赛开始预测。",
    predict: "预测这场比赛",
    researching: "正在研究这场比赛",
    researchingDetail: "采集赔率、公开信息，并让 AI 把非官方文本转成结构化信号。",
    marketOdds: "市场赔率",
    draw: "平局",
    books: "家机构",
    aiPick: "AI 给出的预测结果",
    leanProbability: "倾向概率",
    confidence: "综合置信度",
    evidenceCoverage: "资料覆盖",
    infoTimeliness: "信息时效",
    predictionStability: "预测稳定性",
    technicalScore: "技术分",
    factorTitle: "预测参考数据",
    evidence: "证据",
    aiReview: "AI 审阅",
    noReview: "AI 审阅暂无摘要。",
  };
  const en: Record<string, string> = {
    appTitle: "Football Match Predictor",
    appSubtitle: "Fetch today and tomorrow's fixtures, select a match, and run odds, strength, lineup, tactical, and unofficial-signal analysis.",
    today: "Today + Tomorrow",
    fetchSchedule: "Fetch fixtures",
    emptySchedule: "Fetch today and tomorrow's football matches to start.",
    noMatches: "No predictable fixtures were found for today or tomorrow.",
    scheduleFailed: "Schedule fetch failed.",
    predictionFailed: "Prediction failed.",
    predictionTimeout: "Prediction timed out while collecting data or structuring AI signals.",
    selectMatch: "Select a match to predict.",
    predict: "Predict match",
    researching: "Researching this match",
    researchingDetail: "Collecting odds, public information, and structuring unofficial text into signals.",
    marketOdds: "Market odds",
    draw: "Draw",
    books: "books",
    aiPick: "AI prediction",
    leanProbability: "Lean probability",
    confidence: "Confidence",
    evidenceCoverage: "Evidence coverage",
    infoTimeliness: "Information timing",
    predictionStability: "Prediction stability",
    technicalScore: "Technical score",
    factorTitle: "Prediction Reference Data",
    evidence: "evidence",
    aiReview: "AI review",
    noReview: "No AI review summary.",
  };
  return (language === "zh" ? zh : en)[key] ?? key;
}

function factorLabel(key: string, fallback: string, language: Language) {
  if (language === "en") return fallback;
  const labels: Record<string, string> = {
    market_odds: "市场赔率",
    team_strength: "球队实力",
    lineup_availability: "阵容与伤停",
    tactical_matchup: "战术对位",
    referee_environment: "裁判与环境",
    sentiment_narrative: "舆论与叙事",
  };
  return labels[key] ?? fallback;
}

function routeLabel(key: string, language: Language) {
  if (language === "en") return key.replace(/_/g, " ");
  const labels: Record<string, string> = {
    data_layer: "数据层",
    market_odds: "市场赔率",
    team_strength: "球队实力",
    lineup_availability: "阵容与伤停",
    tactical_matchup: "战术对位",
    referee_environment: "裁判与环境",
    sentiment_narrative: "舆论与叙事",
    unofficial_signal_aggregation: "非官方消息聚合",
    prediction_weights: "预测权重",
    prediction: "预测结论",
    feedback_readiness: "反馈准备",
  };
  return labels[key] ?? key;
}

function statusLabel(status: string, language: Language) {
  if (language === "en") return status.replace(/_/g, " ");
  const labels: Record<string, string> = {
    complete: "完整",
    partial: "部分完成",
    missing: "缺失",
    ok: "正常",
    error: "错误",
  };
  return labels[status] ?? status;
}

function localizeText(value: string, language: Language) {
  if (language === "en") return value;
  return value
    .replace(/Selected fixture market gives ([a-zA-Z\s.'&-]+) ([0-9.]+)%, draw ([0-9.]+)%, ([a-zA-Z\s.'&-]+) ([0-9.]+)% after overround adjustment\./g, "赛程赔率显示：$1 胜率 $2%，平局 $3%，$4 胜率 $5%（已校正庄家水位）。")
    .replace(/World Football Elo gives ([a-zA-Z\s.'&-]+) (\d+) and ([a-zA-Z\s.'&-]+) (\d+), a ([+-]?\d+) rating edge\./g, "世界足球 Elo 显示：$1 为 $2 分，$3 为 $4 分，评分差为 $5。")
    .replace(/Market evidence collected, but no match-specific structured odds were extracted\./g, "已收集市场赔率相关信息，但没有提取到本场比赛的结构化 1X2 赔率。")
    .replace(/No confirmed lineup or injury signal was structured, so a conservative squad-depth and availability proxy weakly favors ([a-zA-Z\s.'&-]+); factor confidence is (\d+)%\./g, "没有提取到确认首发或伤停信号，因此使用阵容深度与可用性代理，略微倾向 $1；该因素可信度为 $2%。")
    .replace(/No explicit tactical matchup signal was structured, so a conservative market-and-strength tactical proxy weakly favors ([a-zA-Z\s.'&-]+); factor confidence is (\d+)%\./g, "没有提取到明确战术对位信号，因此使用市场与实力代理，略微倾向 $1；该因素可信度为 $2%。")
    .replace(/No confirmed lineup or injury signal was structured/g, "没有提取到确认首发或伤停信号")
    .replace(/No explicit tactical matchup signal was structured/g, "没有提取到明确战术对位信号")
    .replace(/Overall this weakly favors ([a-zA-Z\s.'-]+); factor confidence is (\d+)%\./g, "整体上略微倾向 $1；该因素可信度为 $2%。")
    .replace(/No factor-specific evidence collected yet\./g, "暂未收集到该因素的专门证据。")
    .replace(/AI-structured unofficial candidate evidence/g, "AI 结构化的非官方候选证据")
    .replace(/team_strength/g, "球队实力")
    .replace(/market_odds/g, "市场赔率");
}
