from app.schemas import PredictionRequest


TEAM_ALIASES = {
    "\u632a\u5a01": "norway",
    "\u4f0a\u62c9\u514b": "iraq",
    "\u963f\u6839\u5ef7": "argentina",
    "\u6cd5\u56fd": "france",
    "\u82f1\u683c\u5170": "england",
    "\u514b\u7f57\u5730\u4e9a": "croatia",
    "\u5df4\u897f": "brazil",
    "\u5fb7\u56fd": "germany",
    "\u897f\u73ed\u7259": "spain",
    "\u8461\u8404\u7259": "portugal",
    "\u8377\u5170": "netherlands",
    "\u610f\u5927\u5229": "italy",
    "\u6bd4\u5229\u65f6": "belgium",
    "\u4e39\u9ea6": "denmark",
    "\u745e\u5178": "sweden",
    "\u745e\u58eb": "switzerland",
    "\u5965\u5730\u5229": "austria",
    "\u6ce2\u5170": "poland",
    "\u6377\u514b": "czechia",
    "\u571f\u8033\u5176": "turkey",
    "\u58a8\u897f\u54e5": "mexico",
    "\u7f8e\u56fd": "usa",
    "\u52a0\u62ff\u5927": "canada",
    "\u65e5\u672c": "japan",
    "\u97e9\u56fd": "south korea",
    "\u6fb3\u5927\u5229\u4e9a": "australia",
    "\u6469\u6d1b\u54e5": "morocco",
    "\u57c3\u53ca": "egypt",
    "\u4e4c\u62c9\u572d": "uruguay",
    "\u54e5\u4f26\u6bd4\u4e9a": "colombia",
    "\u667a\u5229": "chile",
    "\u79d8\u9c81": "peru",
    "\u5384\u74dc\u591a\u5c14": "ecuador",
    "\u5df4\u62c9\u572d": "paraguay",
}

KNOWN_TEAMS = [
    *TEAM_ALIASES.values(),
    "south korea",
    "bosnia and herzegovina",
    "bosnia & herzegovina",
    "south africa",
    "ivory coast",
    "czech republic",
    "united states",
    "usa",
    "ghana",
    "panama",
    "qatar",
    "scotland",
    "haiti",
]


def infer_competitors(request: PredictionRequest) -> tuple[str | None, str | None]:
    context_competitors = request.context.get("competitors")
    if isinstance(context_competitors, list) and len(context_competitors) >= 2:
        return _canonical(str(context_competitors[0])), _canonical(str(context_competitors[1]))

    text = request.question.lower()
    for source, target in TEAM_ALIASES.items():
        text = text.replace(source, f" {target} ")

    separators = [" vs ", " v ", " versus ", "\u8e22", "\u5bf9\u9635", "\u5bf9", " against "]
    for separator in separators:
        if separator in text:
            left, right = text.split(separator, 1)
            return _entity_from_side(left, prefer_last=True), _entity_from_side(right, prefer_last=False)
    mentioned = []
    for target in dict.fromkeys(TEAM_ALIASES.values()):
        if f" {target} " in f" {text} ":
            mentioned.append(target)
    if len(mentioned) >= 2:
        return mentioned[0], mentioned[1]
    return None, None


def _canonical(text: str) -> str:
    lowered = text.lower().strip()
    for source, target in TEAM_ALIASES.items():
        lowered = lowered.replace(source, target)
    lowered = lowered.replace("&", "and")
    for char in "?,.!:;()[]\u3002\uff0c\uff1f\uff01":
        lowered = lowered.replace(char, " ")
    lowered = " ".join(lowered.split())
    replacements = {
        "bosnia herzegovina": "bosnia and herzegovina",
        "bosnia & herzegovina": "bosnia and herzegovina",
        "czech republic": "czechia",
        "united states": "usa",
    }
    return replacements.get(lowered, lowered)


def _entity_from_side(text: str, prefer_last: bool) -> str | None:
    normalized = _canonical(text)
    candidates = []
    for team in sorted({_canonical(team) for team in KNOWN_TEAMS}, key=len, reverse=True):
        pattern = f" {team} "
        haystack = f" {normalized} "
        index = haystack.rfind(pattern) if prefer_last else haystack.find(pattern)
        if index >= 0:
            candidates.append((index, team))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=prefer_last)
        return candidates[0][1]
    return _last_entity(normalized) if prefer_last else _first_entity(normalized)


def _last_entity(text: str) -> str | None:
    tokens = [token.strip(" ?,.!?:;()[]\u3002\uff0c\uff1f\uff01") for token in text.split()]
    tokens = [token for token in tokens if token]
    return tokens[-1] if tokens else None


def _first_entity(text: str) -> str | None:
    tokens = [token.strip(" ?,.!?:;()[]\u3002\uff0c\uff1f\uff01") for token in text.split()]
    tokens = [token for token in tokens if token]
    return tokens[0] if tokens else None
