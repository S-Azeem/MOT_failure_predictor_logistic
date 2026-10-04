"""Helpers shared by more than one pipeline script."""
import sqlite3

from config import DB_PATH


def connect_db() -> sqlite3.Connection:
    """Open the MOT database READ-ONLY. Fails clearly if the file is missing,
    instead of sqlite3's default of silently creating an empty database."""
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)


def generation(reg_year: int) -> str:
    """BMW 3 Series generation from year of first use (boundaries approximate)."""
    if reg_year < 1991:
        return "E30 and older"
    if reg_year <= 1998:
        return "E36"
    if reg_year <= 2005:
        return "E46"
    if reg_year <= 2012:
        return "E90"
    if reg_year <= 2019:
        return "F30"
    return "G20"
