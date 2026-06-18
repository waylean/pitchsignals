from app.domain_packs.base import DomainPack
from app.domain_packs.basketball import BASKETBALL_PACK
from app.domain_packs.elections import ELECTIONS_PACK
from app.domain_packs.entertainment import ENTERTAINMENT_PACK
from app.domain_packs.finance import FINANCE_PACK
from app.domain_packs.football import FOOTBALL_PACK

PACKS: dict[str, DomainPack] = {
    pack.key: pack
    for pack in [
        FOOTBALL_PACK,
        BASKETBALL_PACK,
        FINANCE_PACK,
        ELECTIONS_PACK,
        ENTERTAINMENT_PACK,
    ]
}


def get_domain_pack(key: str) -> DomainPack:
    return PACKS.get(key, FOOTBALL_PACK)


def list_domain_packs() -> list[dict]:
    return [pack.model_dump() for pack in PACKS.values()]
