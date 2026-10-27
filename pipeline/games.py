from amadeus import paths

# Short name -> Steam install folder for the games we pull data from
GAMES = {
    "sg": "STEINS;GATE",
    "sg0": "STEINS;GATE 0",
    "phenogram": "SG_Phenogram",
    "darling": "SG_My Darling's Embrace",
}


def archive_path(game, archive):
    """Path to an .mpk in a game's USRDIR, e.g. archive_path("sg0", "chara")."""
    if game not in GAMES:
        raise KeyError(f"unknown game {game!r}, expected one of {', '.join(GAMES)}")
    name = archive if archive.endswith(".mpk") else f"{archive}.mpk"
    return paths.STEAM_COMMON / GAMES[game] / "USRDIR" / name
