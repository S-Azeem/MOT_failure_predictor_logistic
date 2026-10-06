"""
Which vehicles the all-vehicle model covers, and how makes are tidied.

The same rules must run in two places:
  - the training build (registered as SQLite functions in build_training_all.py)
  - the Cloudflare Worker (exported into the model JSON, re-implemented in JavaScript)
so they live here, in one place, as plain lists and two small functions.

In scope: cars and light vans. Out of scope: motorcycles, mopeds and scooters.
The DVSA data has no vehicle-class field, so bikes are identified by make and model:
  - pure motorcycle makes are excluded outright
  - Honda and Suzuki make both: keep only known CAR models (by the model's first word)
  - BMW makes both: exclude known BIKE model patterns
  - Triumph is excluded entirely (bikes, plus pre-1985 classic cars now mostly MOT-exempt)
"""
import re

# Spelling variants merged into one make
MAKE_MERGES = {
    "MERCEDES": "MERCEDES-BENZ",
    "MERCEDES BENZ": "MERCEDES-BENZ",
    "SMART (MCC)": "SMART",
    "CHRYSLER-JEEP": "JEEP",
    "HARLEY DAVIDSON": "HARLEY-DAVIDSON",
    "LANDROVER": "LAND ROVER",
    "LAND-ROVER": "LAND ROVER",
    "VW": "VOLKSWAGEN",
    "ROLLS ROYCE": "ROLLS-ROYCE",
    "ALFA-ROMEO": "ALFA ROMEO",
}

# Makes that only build motorcycles, mopeds or scooters (plus Triumph, see above)
EXCLUDED_MAKES = {
    "YAMAHA", "KAWASAKI", "PIAGGIO", "APRILIA", "KTM", "DUCATI", "HARLEY-DAVIDSON",
    "LEXMOTO", "PUCH", "ROYAL ENFIELD", "VESPA", "SINNIS", "MOTO GUZZI", "MV AGUSTA",
    "BENELLI", "KYMCO", "SYM", "HUSQVARNA", "HUSABERG", "BSA", "NORTON", "ZONTES",
    "BUELL", "CAGIVA", "GILERA", "INDIAN", "LAMBRETTA", "MUTT", "BRIXTON", "CCM",
    "HYOSUNG", "DAELIM", "JINLUN", "PIONEER", "SKYJET", "SUKIDA", "ZONGSHEN", "LONCIN",
    "KEEWAY", "WK", "AJS", "MASH", "SUPER SOCO", "NIU", "HERALD", "DERBI", "RIEJU",
    "SHERCO", "BETA", "GAS GAS", "TGB", "PEUGEOT MOTOCYCLES", "MOTORINI", "QUADRO",
    "TRIUMPH",
}

# Honda and Suzuki: keep a vehicle only if the model's first word is a known car
CAR_MODELS = {
    "HONDA": {
        "CIVIC", "JAZZ", "CR-V", "ACCORD", "HR-V", "PRELUDE", "FR-V", "CONCERTO",
        "S2000", "INSIGHT", "CR-Z", "LEGEND", "STREAM", "SHUTTLE", "LOGO", "CRX",
        "CR-X", "INTEGRA", "NSX", "E", "E:NY1", "ZR-V", "CITY", "BALLADE", "AERODECK",
        "ODYSSEY", "QUINTET",
    },
    "SUZUKI": {
        "SWIFT", "VITARA", "GRAND", "ALTO", "SX4", "IGNIS", "JIMNY", "WAGON-R+",
        "WAGON-R", "WAGON", "CELERIO", "SPLASH", "BALENO", "CARRY", "SUPER", "LIANA",
        "S-CROSS", "KIZASHI", "SAMURAI", "SJ410", "SJ413", "SJ", "X-90", "ACROSS",
        "SWACE", "CAPPUCCINO", "ESCUDO", "CULTUS",
    },
}

# BMW: these model patterns are motorcycles (R1200, R 1250 GS, K1600, S 1000 RR,
# F 800 GS, G 310 R, C 400 X ...), as are the R and K "SERIES" labels
BMW_BIKE = re.compile(r"^([RKFGCS] ?\d{2,4}|R SERIES|K SERIES|R NINE|HP\d|CE ?0)")
UNKNOWN_MODELS = {"", "UNKNOWN"}


def clean_make(make) -> str:
    """Upper-case, trim and merge spelling variants. Blank becomes UNKNOWN."""
    m = (make or "").strip().upper()
    return MAKE_MERGES.get(m, m) or "UNKNOWN"


def in_scope(make, model) -> int:
    """1 if the vehicle is a car or light van the model covers, else 0."""
    mk = clean_make(make)
    md = (model or "").strip().upper()
    if mk in EXCLUDED_MAKES:
        return 0
    if mk in CAR_MODELS:
        first = md.split(" ")[0] if md else ""
        return int(first in CAR_MODELS[mk])
    if mk == "BMW":
        return int(md not in UNKNOWN_MODELS and not BMW_BIKE.match(md))
    return 1


def rules_for_export() -> dict:
    """The rules as plain data, for the model JSON the Worker reads."""
    return {
        "make_merges": MAKE_MERGES,
        "excluded_makes": sorted(EXCLUDED_MAKES),
        "car_models": {k: sorted(v) for k, v in CAR_MODELS.items()},
        "bmw_bike_regex": BMW_BIKE.pattern,
        "unknown_models": sorted(UNKNOWN_MODELS),
    }