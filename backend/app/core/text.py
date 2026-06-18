_QUERY_ALIASES = {
    "\u632a\u5a01": "Norway",
    "\u4f0a\u62c9\u514b": "Iraq",
    "\u4e16\u754c\u676f": "World Cup",
    "\u4eca\u5e74": "2026",
    "\u8c01\u8d62": "who will win",
    "\u8db3\u7403": "football",
    "\u7bee\u7403": "basketball",
    "\u91d1\u878d": "finance",
    "\u82f1\u683c\u5170": "England",
    "\u514b\u7f57\u5730\u4e9a": "Croatia",
    "\u58a8\u897f\u54e5": "Mexico",
    "\u97e9\u56fd": "South Korea",
}


def expand_query(text: str) -> str:
    additions = []
    for source, target in _QUERY_ALIASES.items():
        if source in text:
            additions.append(target)
    if len(additions) >= 2:
        return " ".join(dict.fromkeys(additions))
    if additions:
        return f"{' '.join(dict.fromkeys(additions))} {text}"
    return text
