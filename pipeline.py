"""
Bright - content pipeline prototype
====================================

Flow
----
  fetch       pull recent items from a handful of RSS sources
  classify    ask Claude to score uplift, extract country/tag/one-liner
  override    apply manual country reassignments from country_overrides.json
  dedupe      drop near-duplicates (cosine on embeddings)
  bucket      group by country (incl. WORLD bucket for unrouted stories)
  cover       attach a polymorphic cover field to each story
  emit        write stories.json + country_overrides.template.json

Cover field schema
------------------
Every emitted story carries a `cover` object describing how to render its
visual. The schema is intentionally polymorphic so we can swap to other
cover types (AI-generated images, custom illustrations, etc.) without
changing the surrounding API.

Today (v1):
    "cover": {
        "type": "template",
        "template_id": "orbit",
        "color_scheme": "sky"
    }

Future (image variant - no schema migration needed, client dispatches on type):
    "cover": {
        "type": "image",
        "url": "https://cdn.bright.app/covers/abc123.png",
        "alt": "Hippos returning to Lake Naivasha"
    }

The mobile/web client renders a <Cover> component that switches on
`cover.type`. Both variants can coexist mid-migration.

Run
---
    pip install feedparser anthropic numpy
    pip install voyageai            # optional, real embeddings in live mode

    python pipeline.py --demo
    python pipeline.py --live --max-per-source 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

SOURCES = [
    {"name": "Good News Network",     "url": "https://www.goodnewsnetwork.org/feed/"},
    {"name": "Positive News",         "url": "https://www.positive.news/feed/"},
    {"name": "Reasons to be Cheerful","url": "https://reasonstobecheerful.world/feed/"},
    # Mongabay: environmental / conservation news. CC-licensed, fully free
    # RSS, daily cadence. Strong solutions-leaning hit rate.
    {"name": "Mongabay",              "url": "https://news.mongabay.com/feed/"},
    # The Guardian "Upside" — their dedicated solutions journalism section.
    # Section feed only (not the parent paper's main feed), so signal stays
    # very high.
    {"name": "Guardian Upside",       "url": "https://www.theguardian.com/world/series/the-upside/rss"},
    # Fix the News (Substack) — free weekly editions in full via RSS. Note:
    # each item is a *roundup* of 15-20 progress stories rather than a
    # single article, so the classifier sees one combined post per week.
    # Expect these to land in the WORLD bucket more often than not, since
    # the body spans many countries; we accept that trade-off because the
    # roundups themselves are high-quality "Bright"-fit reading.
    {"name": "Fix the News",          "url": "https://www.fixthenews.com/feed"},
]

MIN_SCORE = 7
TOP_N_PER_COUNTRY = 5
DEDUPE_THRESHOLD = 0.85
# Bumped from 0.70: live data showed many false positives at the lower
# threshold — short headlines share enough common bigrams that unrelated
# stories were getting collapsed.
DEDUPE_THRESHOLD_FALLBACK = 0.85

WORLD_CODE = "WORLD"
WORLD_META = {"name": "World", "flag": "\U0001F30D", "lat": None, "lng": None}

OVERRIDES_PATH = Path("country_overrides.json")
OVERRIDES_TEMPLATE_PATH = Path("country_overrides.template.json")

# ----------------------------------------------------------------------------
# COVER LIBRARY — polymorphic schema, v1 uses "template" type
# ----------------------------------------------------------------------------

# Tag -> list of candidate template_ids. Pipeline picks one per story
# deterministically by hashing the title. Add new templates here as the
# library grows; the rest of the pipeline doesn't need to change.
COVER_TEMPLATES_BY_TAG = {
    "environment": ["leaves", "wave", "mountain"],
    "science":     ["orbit", "constellation"],
    "community":   ["circles", "sun"],
    "health":      ["heart", "sun"],
}

# Tag -> default color scheme. Client interprets these names; canonical
# values are the keys of CSS variable groups in the app.
COLOR_SCHEME_BY_TAG = {
    "environment": "moss",
    "science":     "sky",
    "community":   "warm",
    "health":      "gold",
}

DEFAULT_TEMPLATE = "sun"
DEFAULT_COLOR_SCHEME = "ink"


def select_cover(tag: str, title: str) -> dict:
    """Pick a cover for a story. Returns the polymorphic cover dict.

    Deterministic by (tag, title) so re-running the pipeline doesn't
    shuffle cover assignments and cause visual flicker for the user.
    """
    candidates = COVER_TEMPLATES_BY_TAG.get(tag, [DEFAULT_TEMPLATE])
    color_scheme = COLOR_SCHEME_BY_TAG.get(tag, DEFAULT_COLOR_SCHEME)
    h = int.from_bytes(hashlib.md5(title.encode()).digest()[:4], "big")
    template_id = candidates[h % len(candidates)]
    return {
        "type": "template",
        "template_id": template_id,
        "color_scheme": color_scheme,
    }

# ----------------------------------------------------------------------------
# COUNTRY METADATA
# ----------------------------------------------------------------------------

COUNTRY_META = {
    # Generated via iso2_to_flag() at module load — see below.
}

# Comprehensive ISO 3166-1 alpha-2 list. Adding a new country is one entry:
# code, name, lat, lng. Flag emoji is derived from the code at module load.
# Capital city coordinates are used as the marker position; for federations
# without a single capital we use the de facto seat of government.
# Genuinely uninhabited territories (Bouvet, South Georgia, Heard Island,
# British Indian Ocean Territory, French Southern Territories) are omitted
# since stories never reference them.
_COUNTRIES = [
    ("AD", "Andorra",                    42.51,    1.52),
    ("AE", "United Arab Emirates",       24.47,   54.37),
    ("AF", "Afghanistan",                34.53,   69.17),
    ("AG", "Antigua and Barbuda",        17.12,  -61.85),
    ("AI", "Anguilla",                   18.22,  -63.07),
    ("AL", "Albania",                    41.33,   19.82),
    ("AM", "Armenia",                    40.18,   44.51),
    ("AO", "Angola",                     -8.84,   13.23),
    ("AR", "Argentina",                 -34.61,  -58.38),
    ("AS", "American Samoa",            -14.28, -170.70),
    ("AT", "Austria",                    48.21,   16.37),
    ("AU", "Australia",                 -35.28,  149.13),
    ("AW", "Aruba",                      12.52,  -70.04),
    ("AX", "Aland Islands",              60.10,   19.93),
    ("AZ", "Azerbaijan",                 40.41,   49.87),
    ("BA", "Bosnia and Herzegovina",     43.86,   18.41),
    ("BB", "Barbados",                   13.10,  -59.62),
    ("BD", "Bangladesh",                 23.81,   90.41),
    ("BE", "Belgium",                    50.85,    4.35),
    ("BF", "Burkina Faso",               12.37,   -1.52),
    ("BG", "Bulgaria",                   42.70,   23.32),
    ("BH", "Bahrain",                    26.23,   50.59),
    ("BI", "Burundi",                    -3.43,   29.93),
    ("BJ", "Benin",                       6.50,    2.60),
    ("BL", "Saint Barthelemy",           17.90,  -62.85),
    ("BM", "Bermuda",                    32.30,  -64.78),
    ("BN", "Brunei",                      4.89,  114.94),
    ("BO", "Bolivia",                   -16.49,  -68.15),
    ("BQ", "Caribbean Netherlands",      12.15,  -68.27),
    ("BR", "Brazil",                    -15.78,  -47.93),
    ("BS", "Bahamas",                    25.07,  -77.34),
    ("BT", "Bhutan",                     27.47,   89.64),
    ("BW", "Botswana",                  -24.66,   25.93),
    ("BY", "Belarus",                    53.90,   27.57),
    ("BZ", "Belize",                     17.25,  -88.77),
    ("CA", "Canada",                     45.42,  -75.69),
    ("CC", "Cocos Islands",             -12.13,   96.84),
    ("CD", "DR Congo",                   -4.32,   15.32),
    ("CF", "Central African Republic",    4.36,   18.55),
    ("CG", "Republic of the Congo",      -4.27,   15.28),
    ("CH", "Switzerland",                46.95,    7.45),
    ("CI", "Cote d'Ivoire",               6.83,   -5.27),
    ("CK", "Cook Islands",              -21.22, -159.78),
    ("CL", "Chile",                     -33.45,  -70.67),
    ("CM", "Cameroon",                    3.85,   11.50),
    ("CN", "China",                      39.90,  116.40),
    ("CO", "Colombia",                    4.71,  -74.07),
    ("CR", "Costa Rica",                  9.93,  -84.08),
    ("CU", "Cuba",                       23.13,  -82.38),
    ("CV", "Cape Verde",                 14.93,  -23.51),
    ("CW", "Curacao",                    12.12,  -68.93),
    ("CX", "Christmas Island",          -10.50,  105.67),
    ("CY", "Cyprus",                     35.17,   33.36),
    ("CZ", "Czech Republic",             50.08,   14.44),
    ("DE", "Germany",                    52.52,   13.40),
    ("DJ", "Djibouti",                   11.60,   43.15),
    ("DK", "Denmark",                    55.68,   12.57),
    ("DM", "Dominica",                   15.31,  -61.39),
    ("DO", "Dominican Republic",         18.47,  -69.91),
    ("DZ", "Algeria",                    36.75,    3.04),
    ("EC", "Ecuador",                    -0.18,  -78.47),
    ("EE", "Estonia",                    59.44,   24.75),
    ("EG", "Egypt",                      30.04,   31.24),
    ("EH", "Western Sahara",             27.15,  -13.20),
    ("ER", "Eritrea",                    15.34,   38.93),
    ("ES", "Spain",                      40.42,   -3.70),
    ("ET", "Ethiopia",                    9.03,   38.74),
    ("FI", "Finland",                    60.17,   24.94),
    ("FJ", "Fiji",                      -18.14,  178.44),
    ("FK", "Falkland Islands",          -51.70,  -57.85),
    ("FM", "Micronesia",                  6.92,  158.16),
    ("FO", "Faroe Islands",              62.01,   -6.78),
    ("FR", "France",                     48.86,    2.35),
    ("GA", "Gabon",                       0.42,    9.47),
    ("GB", "United Kingdom",             51.51,   -0.13),
    ("GD", "Grenada",                    12.06,  -61.75),
    ("GE", "Georgia",                    41.72,   44.78),
    ("GF", "French Guiana",               4.93,  -52.33),
    ("GG", "Guernsey",                   49.46,   -2.59),
    ("GH", "Ghana",                       5.60,   -0.19),
    ("GI", "Gibraltar",                  36.14,   -5.35),
    ("GL", "Greenland",                  64.18,  -51.74),
    ("GM", "Gambia",                     13.45,  -16.58),
    ("GN", "Guinea",                      9.64,  -13.58),
    ("GP", "Guadeloupe",                 16.27,  -61.55),
    ("GQ", "Equatorial Guinea",           3.75,    8.78),
    ("GR", "Greece",                     37.98,   23.73),
    ("GT", "Guatemala",                  14.63,  -90.51),
    ("GU", "Guam",                       13.47,  144.75),
    ("GW", "Guinea-Bissau",              11.85,  -15.59),
    ("GY", "Guyana",                      6.80,  -58.16),
    ("HK", "Hong Kong",                  22.30,  114.17),
    ("HN", "Honduras",                   14.07,  -87.19),
    ("HR", "Croatia",                    45.81,   15.98),
    ("HT", "Haiti",                      18.59,  -72.31),
    ("HU", "Hungary",                    47.50,   19.04),
    ("ID", "Indonesia",                  -6.21,  106.85),
    ("IE", "Ireland",                    53.35,   -6.26),
    ("IL", "Israel",                     31.78,   35.22),
    ("IM", "Isle of Man",                54.15,   -4.48),
    ("IN", "India",                      28.61,   77.21),
    ("IQ", "Iraq",                       33.31,   44.36),
    ("IR", "Iran",                       35.69,   51.39),
    ("IS", "Iceland",                    64.13,  -21.95),
    ("IT", "Italy",                      41.90,   12.50),
    ("JE", "Jersey",                     49.19,   -2.10),
    ("JM", "Jamaica",                    17.97,  -76.79),
    ("JO", "Jordan",                     31.95,   35.93),
    ("JP", "Japan",                      35.68,  139.65),
    ("KE", "Kenya",                      -1.29,   36.82),
    ("KG", "Kyrgyzstan",                 42.87,   74.59),
    ("KH", "Cambodia",                   11.55,  104.92),
    ("KI", "Kiribati",                    1.32,  172.97),
    ("KM", "Comoros",                   -11.71,   43.24),
    ("KN", "Saint Kitts and Nevis",      17.30,  -62.72),
    ("KP", "North Korea",                39.02,  125.75),
    ("KR", "South Korea",                37.57,  126.98),
    ("KW", "Kuwait",                     29.38,   47.99),
    ("KY", "Cayman Islands",             19.29,  -81.38),
    ("KZ", "Kazakhstan",                 51.16,   71.47),
    ("LA", "Laos",                       17.97,  102.61),
    ("LB", "Lebanon",                    33.89,   35.50),
    ("LC", "Saint Lucia",                14.01,  -60.99),
    ("LI", "Liechtenstein",              47.14,    9.52),
    ("LK", "Sri Lanka",                   6.93,   79.86),
    ("LR", "Liberia",                     6.30,  -10.80),
    ("LS", "Lesotho",                   -29.31,   27.48),
    ("LT", "Lithuania",                  54.69,   25.28),
    ("LU", "Luxembourg",                 49.61,    6.13),
    ("LV", "Latvia",                     56.95,   24.11),
    ("LY", "Libya",                      32.89,   13.19),
    ("MA", "Morocco",                    34.02,   -6.83),
    ("MC", "Monaco",                     43.74,    7.42),
    ("MD", "Moldova",                    47.01,   28.86),
    ("ME", "Montenegro",                 42.44,   19.26),
    ("MF", "Saint Martin",               18.07,  -63.05),
    ("MG", "Madagascar",                -18.88,   47.51),
    ("MH", "Marshall Islands",            7.09,  171.38),
    ("MK", "North Macedonia",            41.99,   21.43),
    ("ML", "Mali",                       12.65,   -8.00),
    ("MM", "Myanmar",                    19.76,   96.08),
    ("MN", "Mongolia",                   47.92,  106.92),
    ("MO", "Macao",                      22.20,  113.55),
    ("MP", "Northern Mariana Islands",   15.18,  145.75),
    ("MQ", "Martinique",                 14.61,  -61.08),
    ("MR", "Mauritania",                 18.08,  -15.98),
    ("MS", "Montserrat",                 16.74,  -62.19),
    ("MT", "Malta",                      35.90,   14.51),
    ("MU", "Mauritius",                 -20.16,   57.50),
    ("MV", "Maldives",                    4.18,   73.51),
    ("MW", "Malawi",                    -13.97,   33.79),
    ("MX", "Mexico",                     19.43,  -99.13),
    ("MY", "Malaysia",                    3.14,  101.69),
    ("MZ", "Mozambique",                -25.97,   32.58),
    ("NA", "Namibia",                   -22.56,   17.08),
    ("NC", "New Caledonia",             -22.27,  166.45),
    ("NE", "Niger",                      13.51,    2.10),
    ("NF", "Norfolk Island",            -29.04,  167.95),
    ("NG", "Nigeria",                     9.08,    7.40),
    ("NI", "Nicaragua",                  12.11,  -86.24),
    ("NL", "Netherlands",                52.37,    4.90),
    ("NO", "Norway",                     59.91,   10.75),
    ("NP", "Nepal",                      27.71,   85.32),
    ("NR", "Nauru",                      -0.55,  166.92),
    ("NU", "Niue",                      -19.05, -169.92),
    ("NZ", "New Zealand",               -41.29,  174.78),
    ("OM", "Oman",                       23.59,   58.41),
    ("PA", "Panama",                      8.97,  -79.53),
    ("PE", "Peru",                      -12.05,  -77.04),
    ("PF", "French Polynesia",          -17.54, -149.57),
    ("PG", "Papua New Guinea",           -9.44,  147.18),
    ("PH", "Philippines",                14.60,  120.98),
    ("PK", "Pakistan",                   33.69,   73.06),
    ("PL", "Poland",                     52.23,   21.01),
    ("PM", "Saint Pierre and Miquelon",  46.78,  -56.18),
    ("PN", "Pitcairn",                  -25.07, -130.10),
    ("PR", "Puerto Rico",                18.47,  -66.11),
    ("PS", "Palestine",                  31.90,   35.20),
    ("PT", "Portugal",                   38.72,   -9.14),
    ("PW", "Palau",                       7.50,  134.62),
    ("PY", "Paraguay",                  -25.26,  -57.58),
    ("QA", "Qatar",                      25.29,   51.53),
    ("RE", "Reunion",                   -20.88,   55.45),
    ("RO", "Romania",                    44.43,   26.10),
    ("RS", "Serbia",                     44.82,   20.46),
    ("RU", "Russia",                     55.75,   37.62),
    ("RW", "Rwanda",                     -1.95,   30.06),
    ("SA", "Saudi Arabia",               24.71,   46.68),
    ("SB", "Solomon Islands",            -9.43,  159.96),
    ("SC", "Seychelles",                 -4.62,   55.45),
    ("SD", "Sudan",                      15.50,   32.56),
    ("SE", "Sweden",                     59.33,   18.07),
    ("SG", "Singapore",                   1.35,  103.82),
    ("SH", "Saint Helena",              -15.93,   -5.72),
    ("SI", "Slovenia",                   46.06,   14.51),
    ("SJ", "Svalbard",                   78.22,   15.65),
    ("SK", "Slovakia",                   48.15,   17.11),
    ("SL", "Sierra Leone",                8.49,  -13.23),
    ("SM", "San Marino",                 43.94,   12.45),
    ("SN", "Senegal",                    14.69,  -17.45),
    ("SO", "Somalia",                     2.05,   45.32),
    ("SR", "Suriname",                    5.85,  -55.20),
    ("SS", "South Sudan",                 4.86,   31.60),
    ("ST", "Sao Tome and Principe",       0.34,    6.73),
    ("SV", "El Salvador",                13.69,  -89.22),
    ("SX", "Sint Maarten",               18.04,  -63.07),
    ("SY", "Syria",                      33.51,   36.30),
    ("SZ", "Eswatini",                  -26.30,   31.13),
    ("TC", "Turks and Caicos",           21.46,  -71.14),
    ("TD", "Chad",                       12.13,   15.06),
    ("TG", "Togo",                        6.13,    1.21),
    ("TH", "Thailand",                   13.76,  100.50),
    ("TJ", "Tajikistan",                 38.56,   68.78),
    ("TK", "Tokelau",                    -8.97, -171.85),
    ("TL", "Timor-Leste",                -8.55,  125.58),
    ("TM", "Turkmenistan",               37.96,   58.32),
    ("TN", "Tunisia",                    36.81,   10.18),
    ("TO", "Tonga",                     -21.13, -175.20),
    ("TR", "Turkey",                     39.93,   32.86),
    ("TT", "Trinidad and Tobago",        10.66,  -61.51),
    ("TV", "Tuvalu",                     -8.52,  179.20),
    ("TW", "Taiwan",                     25.03,  121.57),
    ("TZ", "Tanzania",                   -6.16,   35.75),
    ("UA", "Ukraine",                    50.45,   30.52),
    ("UG", "Uganda",                      0.32,   32.58),
    ("US", "United States",              38.90,  -77.04),
    ("UY", "Uruguay",                   -34.90,  -56.16),
    ("UZ", "Uzbekistan",                 41.30,   69.27),
    ("VA", "Vatican City",               41.90,   12.45),
    ("VC", "Saint Vincent",              13.16,  -61.22),
    ("VE", "Venezuela",                  10.49,  -66.88),
    ("VG", "British Virgin Islands",     18.43,  -64.62),
    ("VI", "US Virgin Islands",          18.34,  -64.93),
    ("VN", "Vietnam",                    21.03,  105.85),
    ("VU", "Vanuatu",                   -17.74,  168.31),
    ("WF", "Wallis and Futuna",         -13.78, -177.16),
    ("WS", "Samoa",                     -13.84, -171.76),
    ("XK", "Kosovo",                     42.67,   21.17),
    ("YE", "Yemen",                      15.35,   44.21),
    ("YT", "Mayotte",                   -12.78,   45.23),
    ("ZA", "South Africa",              -25.75,   28.19),
    ("ZM", "Zambia",                    -15.42,   28.28),
    ("ZW", "Zimbabwe",                  -17.83,   31.05),
]

for _code, _name, _lat, _lng in _COUNTRIES:
    _c = _code.upper()
    COUNTRY_META[_code] = {
        "name": _name,
        "flag": (
            chr(0x1F1E6 + ord(_c[0]) - ord("A"))
            + chr(0x1F1E6 + ord(_c[1]) - ord("A"))
        ),
        "lat": _lat,
        "lng": _lng,
    }


def iso2_to_flag(code: str) -> str:
    code = (code or "").upper()
    if len(code) != 2 or not code.isalpha():
        return ""
    return chr(0x1F1E6 + ord(code[0]) - ord("A")) + chr(0x1F1E6 + ord(code[1]) - ord("A"))


CLASSIFY_PROMPT = """\
You are an editor for "Bright", an app that surfaces solutions journalism -
not feel-good fluff, not corporate PR dressed as good news, but real,
structural positive change.

Score this article 0-10 on the GENUINE UPLIFT rubric:

 10  Verified structural change. A policy that worked, a public-health
     metric improved, a peer-reviewed breakthrough, a community problem
     measurably solved. Includes numbers or named outcomes.
  7  Solid solutions story. A working program with evidence, even if local.
  5  Heartwarming human interest. Pleasant but not solutions journalism
     (e.g., a stranger pays for someone's groceries, an animal is rescued).
  3  Vague positivity, listicle, "you won't believe" framing.
  0  Fake-positive: CEO celebrates record profits; brand "raises awareness";
     performative goodwill; achievement that mostly benefits the announcer.

Then extract:
  - country: ISO 3166-1 alpha-2 (uppercase), or null if the story is global,
             multi-country, or unclear. Do NOT guess. Better to return null.
  - tag: exactly one of [environment, science, community, health]
  - one_line: <= 22 words, neutral, no hype, no emoji, no rhetorical questions
  - extended: 3-4 sentences, 70-120 words, in your own neutral voice. Add
             context: who's involved, key numbers, why it matters. Do NOT
             copy phrases verbatim from the article body (we don't republish).
             Do NOT editorialize, moralize, or use rhetorical questions.

Article title: {title}
Article body (may be truncated): {body}
Source: {source}

Return ONLY a JSON object on a single line, no prose, no code fence:
{{"score": <int 0-10>, "country": "<ISO2 or null>", "tag": "<one tag>", "one_line": "<summary>", "extended": "<paragraph>"}}
"""

# ----------------------------------------------------------------------------
# DATA TYPES
# ----------------------------------------------------------------------------

@dataclass
class RawItem:
    title: str
    body: str
    link: str
    source: str
    published: str

@dataclass
class ClassifiedItem:
    title: str
    link: str
    source: str
    published: str
    score: int
    country: str | None
    tag: str
    one_line: str
    extended_summary: str = ""
    # ISO timestamp for when this story first entered the Bright pipeline.
    # Used for the rolling-window prune (different from `published`, which
    # is when the source itself published the article).
    first_seen_at: str = ""

@dataclass
class CountryBucket:
    code: str
    name: str
    flag: str
    lat: float | None
    lng: float | None
    stories: list[dict] = field(default_factory=list)

# ----------------------------------------------------------------------------
# FETCH
# ----------------------------------------------------------------------------

def fetch_rss(sources, max_per_source=20) -> list[RawItem]:
    import feedparser
    items: list[RawItem] = []
    for src in sources:
        print(f"  fetching {src['name']}...", file=sys.stderr)
        try:
            feed = feedparser.parse(src["url"])
        except Exception as e:
            print(f"    skipped ({e})", file=sys.stderr)
            continue
        for entry in feed.entries[:max_per_source]:
            body = entry.get("summary") or entry.get("description") or ""
            body = re.sub(r"<[^>]+>", " ", body)
            body = re.sub(r"\s+", " ", body).strip()[:1500]
            items.append(RawItem(
                title=entry.get("title", "").strip(),
                body=body,
                link=entry.get("link", ""),
                source=src["name"],
                published=entry.get("published", ""),
            ))
    print(f"  fetched {len(items)} raw items", file=sys.stderr)
    return items

# ----------------------------------------------------------------------------
# CLASSIFY (LLM)
# ----------------------------------------------------------------------------

def classify_live(items: list[RawItem]) -> list[ClassifiedItem]:
    from anthropic import Anthropic
    client = Anthropic()
    out: list[ClassifiedItem] = []
    for i, it in enumerate(items, 1):
        prompt = CLASSIFY_PROMPT.format(title=it.title, body=it.body, source=it.source)
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            parsed = json.loads(raw)
        except Exception as e:
            print(f"    [{i}/{len(items)}] classify failed: {e}", file=sys.stderr)
            continue
        if parsed.get("country") in (None, "null", ""):
            country = None
        else:
            country = str(parsed["country"]).upper()
        out.append(ClassifiedItem(
            title=it.title, link=it.link, source=it.source, published=it.published,
            score=int(parsed.get("score", 0)),
            country=country,
            tag=str(parsed.get("tag", "community")),
            one_line=str(parsed.get("one_line", "")),
            extended_summary=str(parsed.get("extended", "")),
            first_seen_at=datetime.now(timezone.utc).isoformat(),
        ))
        print(f"    [{i}/{len(items)}] score={parsed.get('score')} "
              f"country={country} tag={parsed.get('tag')}", file=sys.stderr)
    return out


def load_existing_stories(path: Path) -> list[ClassifiedItem]:
    """Read a previous stories.json and reconstruct ClassifiedItems.
    Returns [] if the file is missing or malformed. Used to maintain the
    rolling-window history across pipeline runs.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  could not parse existing {path} ({e}); starting fresh",
              file=sys.stderr)
        return []

    out: list[ClassifiedItem] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for country in data.get("countries", []):
        country_code = country.get("code")
        # Skip the WORLD bucket — those go through normal flow if relevant.
        if country_code == WORLD_CODE:
            country_for_item = None
        else:
            country_for_item = country_code
        for s in country.get("stories", []):
            out.append(ClassifiedItem(
                title=s.get("title", ""),
                link=s.get("link", ""),
                source=s.get("source", ""),
                published="",  # not preserved across runs
                # Score isn't stored in stories.json (we only emit stories
                # that already passed MIN_SCORE), so assume MIN_SCORE.
                score=MIN_SCORE,
                country=country_for_item,
                tag=s.get("tag", "community"),
                one_line=s.get("summary", ""),
                extended_summary=s.get("extended_summary", ""),
                # If an existing entry has no first_seen_at (legacy data),
                # treat it as seen now so it lives the full window from
                # this point forward.
                first_seen_at=s.get("first_seen_at") or now_iso,
            ))
    return out


def classify_demo(items: list[RawItem]) -> list[ClassifiedItem]:
    out: list[ClassifiedItem] = []
    for it in items:
        canned = DEMO_RESPONSES.get(it.title) or {
            "score": 3, "country": None, "tag": "community",
            "one_line": it.title[:120], "extended": "",
        }
        out.append(ClassifiedItem(
            title=it.title, link=it.link, source=it.source, published=it.published,
            score=canned["score"], country=canned["country"],
            tag=canned["tag"], one_line=canned["one_line"],
            extended_summary=canned.get("extended", ""),
            first_seen_at=datetime.now(timezone.utc).isoformat(),
        ))
    return out

# ----------------------------------------------------------------------------
# EMBED + DEDUPE
# ----------------------------------------------------------------------------

def _hashed_bigram_embed(text: str, dim: int = 256) -> np.ndarray:
    text = text.lower()
    vec = np.zeros(dim, dtype=np.float32)
    if len(text) < 2:
        return vec
    for i in range(len(text) - 1):
        bg = text[i:i + 2]
        h = int.from_bytes(hashlib.md5(bg.encode()).digest()[:4], "big") % dim
        vec[h] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def embed(texts, allow_voyage=True):
    if allow_voyage and os.environ.get("VOYAGE_API_KEY"):
        try:
            import voyageai
            client = voyageai.Client()
            result = client.embed(texts, model="voyage-3-lite", input_type="document")
            arr = np.array(result.embeddings, dtype=np.float32)
            return arr, "voyage-3-lite"
        except Exception as e:
            print(f"  voyage embedding failed ({e}); using hashed fallback",
                  file=sys.stderr)
    arr = np.stack([_hashed_bigram_embed(t) for t in texts])
    return arr, "hashed-bigram"


def dedupe(items, threshold=None, allow_voyage=True):
    if len(items) <= 1:
        return list(items)
    items = sorted(items, key=lambda x: (x.country is not None, x.score), reverse=True)
    texts = [f"{it.title}. {it.one_line}" for it in items]
    embs, backend = embed(texts, allow_voyage=allow_voyage)
    print(f"  embeddings via {backend} (dim={embs.shape[1]})", file=sys.stderr)
    if threshold is None:
        threshold = (DEDUPE_THRESHOLD if backend == "voyage-3-lite"
                     else DEDUPE_THRESHOLD_FALLBACK)
    kept_idx: list[int] = []
    for i in range(len(items)):
        is_dup = False
        for j in kept_idx:
            cos = float(embs[i] @ embs[j])
            if cos >= threshold:
                is_dup = True
                print(
                    f"    dropped dup (cos={cos:.2f}): "
                    f"{items[i].title[:60]!r}\n"
                    f"      ~ kept: {items[j].title[:60]!r}",
                    file=sys.stderr,
                )
                break
        if not is_dup:
            kept_idx.append(i)
    return [items[i] for i in kept_idx]


def dedupe_new_against_all(new_items, cached_items, allow_voyage=True):
    """Filter `new_items` to those that aren't near-duplicates of (a) other
    new items or (b) any cached item. Cached items are never modified or
    removed — this protects the rolling-window history from being eroded
    by false-positive dedupes over time.
    """
    if not new_items:
        return []

    # Sort new items so that, when two new items are duplicates of each
    # other, we keep the higher-scoring one.
    new_items = sorted(
        new_items,
        key=lambda x: (x.country is not None, x.score),
        reverse=True,
    )

    all_items = list(cached_items) + list(new_items)
    texts = [f"{it.title}. {it.one_line}" for it in all_items]
    embs, backend = embed(texts, allow_voyage=allow_voyage)
    print(f"  embeddings via {backend} (dim={embs.shape[1]})", file=sys.stderr)

    threshold = (DEDUPE_THRESHOLD if backend == "voyage-3-lite"
                 else DEDUPE_THRESHOLD_FALLBACK)

    n_cached = len(cached_items)
    # Cached items are auto-kept and protected.
    kept_indices = list(range(n_cached))
    kept_new = []

    for i, new_it in enumerate(new_items):
        idx = n_cached + i
        is_dup = False
        for j in kept_indices:
            cos = float(embs[idx] @ embs[j])
            if cos >= threshold:
                is_dup = True
                if j < n_cached:
                    other = cached_items[j]
                    print(
                        f"    skipped new (dup of cached, cos={cos:.2f}): "
                        f"{new_it.title[:60]!r}\n"
                        f"      ~ cached: {other.title[:60]!r}",
                        file=sys.stderr,
                    )
                else:
                    other = new_items[j - n_cached]
                    print(
                        f"    skipped new (dup of new, cos={cos:.2f}): "
                        f"{new_it.title[:60]!r}\n"
                        f"      ~ kept: {other.title[:60]!r}",
                        file=sys.stderr,
                    )
                break
        if not is_dup:
            kept_indices.append(idx)
            kept_new.append(new_it)

    return kept_new

# ----------------------------------------------------------------------------
# OVERRIDES
# ----------------------------------------------------------------------------

def load_overrides(path=OVERRIDES_PATH):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  could not parse {path} ({e}); ignoring", file=sys.stderr)
        return {}
    out: dict[str, str] = {}
    for k, v in data.items():
        if k.startswith(("_", "$")) or not v:
            continue
        out[k] = str(v).upper()
    return out


def apply_overrides(items, overrides):
    if not overrides:
        return items
    n = 0
    for it in items:
        if it.link in overrides:
            it.country = overrides[it.link]
            n += 1
    print(f"  applied {n} manual override(s) from {OVERRIDES_PATH}", file=sys.stderr)
    return items


def emit_overrides_template(world_stories, path=OVERRIDES_TEMPLATE_PATH):
    payload: dict = {
        "_help": (
            "This file lists stories the pipeline couldn't confidently route "
            "to a country. To assign one, copy this file to "
            "country_overrides.json (drop .template), fill in an ISO-2 code "
            "on the right side of each link, and re-run the pipeline. Keys "
            "starting with _ are ignored."
        ),
        "_iso2_examples": "KE JP BR IS IN NZ PT US GB DE CA AU MX CO VN NG",
        "_unrouted_count": len(world_stories),
    }
    for s in world_stories:
        if s.get("link"):
            payload[s["link"]] = ""
    payload["_candidates"] = [
        {"link": s.get("link", ""), "title": s.get("title", ""),
         "tag": s.get("tag", ""), "source": s.get("source", "")}
        for s in world_stories
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"  wrote {path} ({len(world_stories)} unrouted candidate(s))",
          file=sys.stderr)

# ----------------------------------------------------------------------------
# BUCKET + EMIT
# ----------------------------------------------------------------------------

def _time_ago(published: str) -> str:
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(published)
        delta = datetime.now(dt.tzinfo) - dt
        hours = int(delta.total_seconds() // 3600)
        if hours < 1:
            return "just now"
        if hours < 24:
            return f"{hours}h ago"
        return f"{hours // 24}d ago"
    except Exception:
        return ""


def _time_ago_iso(iso_ts: str) -> str:
    """Same as _time_ago but for ISO 8601 timestamps. Used for first_seen_at,
    which is what we want to display on stories that have rolled over from
    earlier pipeline runs.
    """
    if not iso_ts:
        return ""
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        hours = int(delta.total_seconds() // 3600)
        if hours < 1:
            return "just now"
        if hours < 24:
            return f"{hours}h ago"
        return f"{hours // 24}d ago"
    except Exception:
        return ""


def bucket(items):
    by_country: dict[str, list] = {}
    for it in items:
        if it.score < MIN_SCORE:
            continue
        if it.country is None:
            by_country.setdefault(WORLD_CODE, []).append(it)
            continue
        if it.country not in COUNTRY_META:
            print(f"  unknown country {it.country!r} -> WORLD "
                  f"({it.title[:40]}...)", file=sys.stderr)
            by_country.setdefault(WORLD_CODE, []).append(it)
            continue
        by_country.setdefault(it.country, []).append(it)

    buckets = []
    for code, lst in by_country.items():
        if code == WORLD_CODE:
            lst = sorted(lst, key=lambda x: x.score, reverse=True)
            meta = WORLD_META
        else:
            lst = sorted(lst, key=lambda x: x.score, reverse=True)[:TOP_N_PER_COUNTRY]
            meta = COUNTRY_META[code]
        b = CountryBucket(code=code, name=meta["name"], flag=meta["flag"],
                          lat=meta["lat"], lng=meta["lng"])
        for it in lst:
            b.stories.append({
                "tag": it.tag,
                "title": it.title,
                "summary": it.one_line,
                "extended_summary": it.extended_summary,
                "source": it.source,
                # `time` is recomputed each run from first_seen_at so the
                # relative string stays accurate as stories age.
                "time": _time_ago_iso(it.first_seen_at),
                "first_seen_at": it.first_seen_at,
                "link": it.link,
                # Polymorphic cover field — see top of file for schema notes.
                "cover": select_cover(it.tag, it.title),
            })
        buckets.append(b)
    buckets.sort(key=lambda b: (b.code == WORLD_CODE, -len(b.stories)))
    return buckets


def emit(buckets, path: Path):
    payload = {
        "schema_version": "1.2",  # bump: extended_summary field added
        "generated_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "story_count": sum(len(b.stories) for b in buckets),
        "country_count": len(buckets),
        "countries": [
            {"code": b.code, "name": b.name, "flag": b.flag,
             "lat": b.lat, "lng": b.lng, "count": len(b.stories),
             "stories": b.stories}
            for b in buckets
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"  wrote {path} ({payload['story_count']} stories across "
          f"{payload['country_count']} countries)", file=sys.stderr)

# ----------------------------------------------------------------------------
# DEMO DATA
# ----------------------------------------------------------------------------

DEMO_ITEMS = [
    RawItem("Lake Naivasha hippos return after decade-long absence",
            "Conservation efforts and reduced agricultural runoff have brought wildlife back to the Kenyan lake; rangers counted 340 hippos this month, up from 89 in 2015.",
            "https://example.org/a1", "Daily Nation", "Mon, 27 Apr 2026 09:00:00 GMT"),
    RawItem("Kamikatsu, Japan, becomes country's first zero-waste town",
            "After 20 years of resident-led sorting into 45 categories, the village of 1,400 sends nothing to landfill; 81% recycled, 19% composted.",
            "https://example.org/a2", "Positive News", "Mon, 27 Apr 2026 12:30:00 GMT"),
    RawItem("Portugal hits 95% renewable electricity for the quarter",
            "Wind, solar and hydro combined to nearly eliminate fossil fuel generation from January through March, according to grid operator REN.",
            "https://example.org/a3", "Reasons to be Cheerful", "Tue, 28 Apr 2026 06:00:00 GMT"),
    RawItem("Reykjavik ends street homelessness through Housing First",
            "A decade-long Icelandic program has provided permanent housing to 100% of long-term unhoused residents.",
            "https://example.org/a4", "Good News Network", "Tue, 28 Apr 2026 14:00:00 GMT"),
    RawItem("Kerala village becomes fully literate after 40-year campaign",
            "Last remaining adult learner, age 84, completed reading certification this week.",
            "https://example.org/a5", "Positive News", "Wed, 29 Apr 2026 02:00:00 GMT"),
    RawItem("Kakapo population reaches 300 for first time on record",
            "The flightless New Zealand parrot, once down to 51 birds, has steadily recovered through intensive conservation on predator-free islands.",
            "https://example.org/a6", "Reasons to be Cheerful", "Wed, 29 Apr 2026 04:00:00 GMT"),
    RawItem("Direct air capture facility goes carbon-negative",
            "The Hellisheidi plant in Iceland now removes more CO2 from the atmosphere than its operations release, an industry first verified by independent audit.",
            "https://example.org/a7", "Positive News", "Wed, 29 Apr 2026 05:30:00 GMT"),
    RawItem("CEO of EnerCorp celebrates record Q1 profits, donates 0.1% to charity",
            "The energy giant's chief executive announced record-breaking earnings alongside a small charitable contribution.",
            "https://example.org/b1", "Good News Network", "Tue, 28 Apr 2026 16:00:00 GMT"),
    RawItem("You won't believe what this golden retriever did at the airport",
            "A heartwarming moment captured on TikTok shows a dog greeting its owner.",
            "https://example.org/b2", "Good News Network", "Wed, 29 Apr 2026 01:00:00 GMT"),
    RawItem("Stranger pays for grandmother's groceries at checkout",
            "A small act of kindness in a Texas supermarket went viral this week.",
            "https://example.org/b3", "Good News Network", "Wed, 29 Apr 2026 03:00:00 GMT"),
    RawItem("Iceland direct air capture plant becomes carbon-negative",
            "Hellisheidi facility achieves milestone removing more carbon than it emits.",
            "https://example.org/c1", "Good News Network", "Wed, 29 Apr 2026 06:00:00 GMT"),
    RawItem("Global cervical cancer mortality drops 30% in a decade, WHO reports",
            "A coordinated push on HPV vaccination and screening across 110 countries has driven the steepest decline in cervical cancer deaths on record.",
            "https://example.org/d1", "Reasons to be Cheerful", "Wed, 29 Apr 2026 11:00:00 GMT"),
    RawItem("Marine protected areas now cover 8.4% of world's oceans",
            "A multilateral assessment shows coordinated treaties have tripled protected ocean since 2010, putting the 30-by-30 target within reach.",
            "https://example.org/d2", "Positive News", "Wed, 29 Apr 2026 13:00:00 GMT"),
]

DEMO_RESPONSES = {
    "Lake Naivasha hippos return after decade-long absence":
        {"score": 8, "country": "KE", "tag": "environment",
         "one_line": "Conservation and reduced runoff brought hippo numbers back to 340 at Kenya's Lake Naivasha.",
         "extended": "Once down to 89 individuals in 2015, the hippo population at Kenya's Lake Naivasha has climbed back to 340 this month. Park rangers credit a coordinated push between the county government, surrounding flower farms, and conservation NGOs to redirect agricultural runoff and stabilize water levels. The recovery has been steady rather than sudden, which scientists say is a sign that the underlying habitat improvements are taking hold."},
    "Kamikatsu, Japan, becomes country's first zero-waste town":
        {"score": 9, "country": "JP", "tag": "environment",
         "one_line": "After 20 years of resident sorting, a Japanese village of 1,400 sends nothing to landfill.",
         "extended": "After two decades of resident-led sorting into 45 separate waste categories, the Japanese village of Kamikatsu has become the country's first town to send nothing to landfill. Today, 81% of household waste is recycled and 19% is composted on-site. The program began as a cost-saving measure for the 1,400-person village in Shikoku, and has since become a reference model for municipal zero-waste systems across Asia."},
    "Portugal hits 95% renewable electricity for the quarter":
        {"score": 10, "country": "PT", "tag": "environment",
         "one_line": "Wind, solar and hydro met 95% of Portuguese electricity demand in Q1, per grid operator REN.",
         "extended": "Wind, solar, and hydroelectric sources combined to meet 95% of Portugal's electricity demand from January through March, according to grid operator REN. Fossil fuel generation, almost entirely natural gas, accounted for the remaining 5% during low-wind periods. Portugal's renewable buildout has accelerated since 2020, with rooftop solar installations roughly tripling year-over-year and offshore wind projects coming online along the Atlantic coast."},
    "Reykjavik ends street homelessness through Housing First":
        {"score": 10, "country": "IS", "tag": "community",
         "one_line": "Iceland's decade-long Housing First program has rehoused 100% of long-term unhoused residents.",
         "extended": "A decade-long Housing First program in Reykjavik has provided permanent housing to every long-term unhoused resident in the Icelandic capital. The model gives people apartments first and offers wraparound services afterward, rather than requiring sobriety or treatment as a precondition. City data shows the approach has cost less per person than the traditional shelter system it replaced, and tenants stay housed at significantly higher rates."},
    "Kerala village becomes fully literate after 40-year campaign":
        {"score": 9, "country": "IN", "tag": "community",
         "one_line": "An 84-year-old completed reading certification, finishing Kerala's 40-year rolling literacy drive.",
         "extended": "An 84-year-old grandmother completed reading certification this week, becoming the last remaining adult learner from a 40-year rolling literacy campaign in Kerala that began in 1986. The southern Indian state has long pursued universal literacy as a continuous public-health program rather than a one-time effort. State officials say the next phase of the campaign will focus on digital literacy among elderly residents, with a particular emphasis on basic mobile-banking and telehealth skills."},
    "Kakapo population reaches 300 for first time on record":
        {"score": 8, "country": "NZ", "tag": "environment",
         "one_line": "New Zealand's flightless kakapo, once down to 51 birds, recovered to 300 via predator-free islands.",
         "extended": "New Zealand's flightless kakapo parrot has reached a population of 300 for the first time on record, up from a low of 51 birds in 1995. Recovery has been driven by intensive conservation work on predator-free offshore islands, where every breeding bird is individually tracked and monitored. Department of Conservation rangers say the next challenge is genetic diversity, since the entire current population descends from a small founding group rescued in the 1990s."},
    "Direct air capture facility goes carbon-negative":
        {"score": 9, "country": "IS", "tag": "science",
         "one_line": "Iceland's Hellisheidi plant now removes more CO2 than it emits, verified by independent audit.",
         "extended": "The Hellisheidi geothermal facility in Iceland has become the first direct air capture plant audited as net carbon-negative, meaning it removes more carbon dioxide from the atmosphere than its operations release. Captured CO2 is mineralized into basalt rock, locking it away on geological timescales. The plant currently processes around 4,000 tonnes of CO2 per year, with a planned expansion that would lift annual capture to 36,000 tonnes by 2027."},
    "Iceland direct air capture plant becomes carbon-negative":
        {"score": 7, "country": "IS", "tag": "science",
         "one_line": "Hellisheidi plant achieves carbon-negative milestone, removing more carbon than it emits.",
         "extended": "Iceland's Hellisheidi facility has crossed a carbon-negative threshold, with audited operations now removing more CO2 than they emit."},
    "CEO of EnerCorp celebrates record Q1 profits, donates 0.1% to charity":
        {"score": 1, "country": None, "tag": "community",
         "one_line": "Energy CEO announced record profits with a token charitable contribution.",
         "extended": "Energy giant EnerCorp announced record-breaking Q1 earnings, alongside a small charitable contribution by the CEO."},
    "You won't believe what this golden retriever did at the airport":
        {"score": 2, "country": None, "tag": "community",
         "one_line": "A viral video shows a dog greeting its owner at the airport.",
         "extended": "A short video posted to social media shows a golden retriever enthusiastically greeting its owner at airport arrivals. The clip has been viewed several million times."},
    "Stranger pays for grandmother's groceries at checkout":
        {"score": 4, "country": "US", "tag": "community",
         "one_line": "A shopper covered another customer's groceries at a Texas supermarket.",
         "extended": "A shopper at a Texas supermarket covered the bill for an elderly customer's groceries at checkout. The exchange was filmed by another shopper and shared widely on social media."},
    "Global cervical cancer mortality drops 30% in a decade, WHO reports":
        {"score": 9, "country": None, "tag": "health",
         "one_line": "Coordinated HPV vaccination and screening across 110 countries drove a 30% decline in cervical cancer deaths.",
         "extended": "Cervical cancer mortality has dropped 30% over the past decade across the 110 countries with active HPV vaccination and screening programs, the World Health Organization reported. The decline is the steepest on record relative to the cancer's incidence rate. WHO attributes most of the gain to school-based HPV vaccination of adolescent girls, with adult cervical screening playing an important secondary role in catching pre-cancerous lesions early."},
    "Marine protected areas now cover 8.4% of world's oceans":
        {"score": 8, "country": None, "tag": "environment",
         "one_line": "Multilateral treaties tripled marine protected area since 2010, putting the 30-by-30 target within reach.",
         "extended": "Coordinated multilateral treaties have tripled the share of the world's oceans under formal protection since 2010, bringing the total to 8.4%. The pace of new designations puts the international 30-by-30 target — protecting 30% of oceans by the year 2030 — within plausible reach. Most of the recent additions are remote high-seas zones around the Pacific and Antarctic, where commercial pressure has historically been lower."},
}

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="canned RSS + canned LLM responses, no API keys")
    ap.add_argument("--live", action="store_true",
                    help="real RSS + real Claude calls")
    ap.add_argument("--max-per-source", type=int, default=20)
    ap.add_argument("--out", default="stories.json")
    ap.add_argument("--history-days", type=int, default=21,
                    help="rolling window: stories first seen more than this "
                         "many days ago are pruned. Default 21 (3 weeks).")
    ap.add_argument("--debug", action="store_true",
                    help="also write classifications-debug.json with every "
                         "story Claude rated, regardless of MIN_SCORE filter, "
                         "and print a sorted summary to stderr")
    args = ap.parse_args()

    if not (args.demo or args.live):
        ap.error("pick --demo or --live")

    print("==> fetch", file=sys.stderr)
    if args.demo:
        raw_items = list(DEMO_ITEMS)
        print(f"  demo mode: {len(raw_items)} canned items", file=sys.stderr)
    else:
        raw_items = fetch_rss(SOURCES, max_per_source=args.max_per_source)

    print("==> load history", file=sys.stderr)
    out_path = Path(args.out)
    existing = load_existing_stories(out_path)
    existing_links = {it.link for it in existing if it.link}
    print(f"  loaded {len(existing)} existing stories from {out_path}",
          file=sys.stderr)

    print("==> classify (skipping already-known)", file=sys.stderr)
    new_raw = [r for r in raw_items if r.link not in existing_links]
    print(f"  {len(new_raw)} new items to classify "
          f"({len(raw_items) - len(new_raw)} already cached)", file=sys.stderr)
    if args.demo:
        new_items = classify_demo(new_raw)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY not set; use --demo or export the key.")
        new_items = classify_live(new_raw)

    # Dedupe new items against (a) other new items and (b) existing cache.
    # Cached items are protected — never removed by dedupe — so the rolling
    # history isn't eroded by false-positive dedupes over time.
    if new_items:
        print(f"==> dedupe new items ({len(new_items)} -> ...)", file=sys.stderr)
        new_items = dedupe_new_against_all(
            new_items, existing, allow_voyage=not args.demo,
        )
        print(f"  {len(new_items)} new items kept", file=sys.stderr)

    # Merge new with existing, then prune anything older than the window.
    items = existing + new_items
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.history_days)
    pruned = []
    for it in items:
        try:
            seen = datetime.fromisoformat(
                it.first_seen_at.replace("Z", "+00:00")
            )
            if seen < cutoff:
                continue
        except Exception:
            # No / malformed first_seen_at — keep it; it gets a fresh stamp
            # via the load_existing_stories path so the next run will see it
            # consistently.
            pass
        pruned.append(it)
    dropped = len(items) - len(pruned)
    if dropped > 0:
        print(f"  pruned {dropped} stories older than "
              f"{args.history_days} days", file=sys.stderr)
    items = pruned

    if args.debug:
        debug_path = Path("classifications-debug.json")
        debug_path.write_text(
            json.dumps(
                [
                    {
                        "score": it.score,
                        "country": it.country,
                        "tag": it.tag,
                        "title": it.title,
                        "source": it.source,
                        "one_line": it.one_line,
                        "extended_summary": it.extended_summary,
                        "link": it.link,
                    }
                    for it in sorted(items, key=lambda x: x.score, reverse=True)
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {debug_path} ({len(items)} classifications)",
              file=sys.stderr)
        print("\n=== All classifications, sorted by score (desc) ===",
              file=sys.stderr)
        for it in sorted(items, key=lambda x: x.score, reverse=True):
            country = it.country or "??"
            print(f"  [{it.score:>2}] {country:<3} {it.tag:<12} "
                  f"{it.title[:60]}", file=sys.stderr)
        print("", file=sys.stderr)

    print("==> overrides", file=sys.stderr)
    overrides = load_overrides()
    items = apply_overrides(items, overrides)

    print("==> bucket + emit (with covers)", file=sys.stderr)
    buckets = bucket(items)
    emit(buckets, out_path)

    world_stories: list[dict] = []
    for b in buckets:
        if b.code == WORLD_CODE:
            world_stories = b.stories
            break
    emit_overrides_template(world_stories)

    print("\nSummary:", file=sys.stderr)
    for b in buckets:
        print(f"  {b.flag} {b.name:<20} {len(b.stories)} stories", file=sys.stderr)
    if world_stories:
        print(f"\n{len(world_stories)} unrouted in WORLD bucket. To reassign: "
              f"edit {OVERRIDES_TEMPLATE_PATH}, save as {OVERRIDES_PATH}, "
              f"re-run.", file=sys.stderr)


if __name__ == "__main__":
    main()
