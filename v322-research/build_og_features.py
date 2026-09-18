from pathlib import Path
import json
import re
import statistics
from datetime import datetime
from bisect import bisect_left
from collections import Counter

FILES = [
    Path("/media/user/ECX-SSD/Predict-Edge-Archive/backups/Predict_Edge_v3.21.2_Working_Dataset_20260916-114848.json"),
    Path("/media/user/ECX-SSD/Predict-Edge-Archive/backups/Predict_Edge_FULL_ARCHIVE_PreTrim_20260916-084835.json"),
    Path("/media/user/ECX-SSD/Predict-Edge-Archive/datasets/frozen/Predict_Edge_Dataset_Backup_20260917-082924.json"),
    Path("/media/user/ECX-SSD/Predict-Edge-Archive/datasets/frozen/Predict_Edge_Dataset_Backup_20260917-100710.json"),
    Path("/media/user/ECX-SSD/Predict-Edge-Archive/datasets/frozen/Predict_Edge_Dataset_Backup_20260917-190716.json"),
]

HORIZONS = [5, 10, 15, 30]
TOLERANCE = 1.5
LOOKBACKS = [1, 5, 15, 30]

def timestamp(x):
    if not x:
        return None

    try:
        return datetime.fromisoformat(
            str(x).replace("Z", "+00:00")
        ).timestamp()
    except Exception:
        return None

def fingerprint(r):
    return (
        str(r.get("symbol", "")),
        str(r.get("capturedAt", "")),
        str(r.get("contractTarget", "")),
        str(r.get("probability", "")),
    )

def strike(r):
    for k in (
        "strike",
        "strikePrice",
        "contractStrike",
        "targetPrice",
        "threshold",
    ):
        try:
            v = r.get(k)

            if v not in (None, ""):
                return float(
                    str(v)
                    .replace("$", "")
                    .replace(",", "")
                )
        except Exception:
            pass

    text = " ".join(
        str(r.get(k, "") or "")
        for k in (
            "contractTargetText",
            "contract",
            "event",
        )
    )

    nums = re.findall(
        r'\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)',
        text
    )

    vals = []

    for x in nums:
        try:
            v = float(x.replace(",", ""))

            if 500 <= v <= 10000:
                vals.append(v)
        except Exception:
            pass

    return vals[-1] if vals else None

# ============================================================
# LOAD DEDUPLICATED OG LIVE HISTORY + ETH REFERENCE HISTORY
# ============================================================

rows = {}
eth = {}

for path in FILES:
    root = json.loads(
        path.read_text(errors="ignore")
    )

    # ----------------------------
    # OG Live History
    # ----------------------------

    for r in root.get("liveHistory", []):
        if not isinstance(r, dict):
            continue

        fp = fingerprint(r)

        if not fp[0] or not fp[1]:
            continue

        old = rows.get(fp)

        # Keep richest duplicate.
        if old is None or len(r) > len(old):
            rows[fp] = r

    # ----------------------------
    # ETH Reference History
    # ----------------------------

    for r in root.get("ethReferenceHistory", []):
        if not isinstance(r, dict):
            continue

        t = timestamp(
            r.get("fetchedAt")
            or r.get("capturedAt")
            or r.get("apiTime")
            or r.get("timestamp")
        )

        try:
            price = float(
                r.get("price")
                or r.get("ethReferencePrice")
                or r.get("mid")
            )
        except Exception:
            continue

        if t is not None and price > 0:
            eth[t] = price


# ============================================================
# CREATE ORDERED ETH SERIES
# ============================================================

series = sorted(eth.items())

times = [
    item[0]
    for item in series
]


def price_before(target, max_age=120):
    """
    Return the latest ETH price at or before target,
    provided that observation is no more than max_age
    seconds older than target.
    """

    i = bisect_left(times, target) - 1

    if i < 0:
        return None

    t, price = series[i]

    if target - t > max_age:
        return None

    return price


print("========== v3.22 OG DATA LOAD ==========")
print()
print("DEDUPLICATED OG:", len(rows))
print("ETH REFERENCES:", len(series))

if series:
    print(
        "ETH FIRST:",
        datetime.fromtimestamp(series[0][0]).isoformat()
    )

    print(
        "ETH LAST:",
        datetime.fromtimestamp(series[-1][0]).isoformat()
    )

print()
print("Data load complete.")

# ============================================================
# BUILD v3.22 CONTROLLED FEATURE ROWS
# ============================================================

features = []

for r in rows.values():

    # Development data must be clean PRE-target data.
    if r.get("phaseCode") != "PRE":
        continue

    if r.get("modelEligible") is not True:
        continue

    try:
        delta = abs(float(r.get("phaseDeltaMinutes")))
    except Exception:
        continue

    horizon = min(
        HORIZONS,
        key=lambda h: abs(delta - h)
    )

    if abs(delta - horizon) > TOLERANCE:
        continue

    s = strike(r)

    if s is None:
        continue

    try:
        current_eth = float(r.get("ethReferencePrice"))
        market_probability = float(r.get("probability"))
    except Exception:
        continue

    captured = timestamp(r.get("capturedAt"))

    if captured is None:
        continue

    # ----------------------------------------
    # ETH MOMENTUM
    # ----------------------------------------

    moves = {}
    complete = True

    for minutes in LOOKBACKS:

        old_price = price_before(
            captured - minutes * 60
        )

        if old_price is None or old_price <= 0:
            complete = False
            break

        moves[minutes] = (
            (current_eth / old_price) - 1
        ) * 100

    if not complete:
        continue

    # ----------------------------------------
    # 15-MINUTE REALIZED VOLATILITY
    # ----------------------------------------

    minute_prices = []

    for minute in range(15, -1, -1):

        p = price_before(
            captured - minute * 60
        )

        if p is not None:
            minute_prices.append(p)

    minute_returns = []

    for a, b in zip(
        minute_prices,
        minute_prices[1:]
    ):
        if a > 0:
            minute_returns.append(
                ((b / a) - 1) * 100
            )

    volatility_15m = None

    if len(minute_returns) >= 5:
        volatility_15m = statistics.pstdev(
            minute_returns
        )

    # ----------------------------------------
    # CONTRACT MARKET INFORMATION
    # ----------------------------------------

    def number(name):
        try:
            v = r.get(name)

            if v in (None, ""):
                return None

            return float(v)
        except Exception:
            return None

    distance_dollars = current_eth - s
    distance_percent = (
        distance_dollars / s
    ) * 100

    features.append({
        "symbol": r.get("symbol"),
        "capturedAt": r.get("capturedAt"),
        "contractTarget": r.get("contractTarget"),

        "horizonMinutes": horizon,

        "strike": s,
        "ethPrice": current_eth,

        "distanceDollars": distance_dollars,
        "distancePercent": distance_percent,

        "marketProbability": market_probability,

        "bid": number("apiBid"),
        "ask": number("apiAsk"),
        "spread": number("apiSpread"),

        "momentum1m": moves[1],
        "momentum5m": moves[5],
        "momentum15m": moves[15],
        "momentum30m": moves[30],

        "volatility15m": volatility_15m,
    })


# ============================================================
# REPORT
# ============================================================

print()
print("========== v3.22 FEATURE DATASET ==========")

print("FEATURE ROWS:", len(features))

targets = {
    x["contractTarget"]
    for x in features
}

symbols = {
    x["symbol"]
    for x in features
}

print("UNIQUE TARGETS:", len(targets))
print("UNIQUE SYMBOLS:", len(symbols))

print()
print("===== HORIZONS =====")

hc = Counter(
    x["horizonMinutes"]
    for x in features
)

for h in HORIZONS:
    print(f"T-{h:2}: {hc[h]}")

print()
print("===== FEATURE COMPLETENESS =====")

fields = [
    "distanceDollars",
    "distancePercent",
    "marketProbability",
    "bid",
    "ask",
    "spread",
    "momentum1m",
    "momentum5m",
    "momentum15m",
    "momentum30m",
    "volatility15m",
]

for field in fields:

    n = sum(
        x.get(field) is not None
        for x in features
    )

    pct = (
        100 * n / len(features)
        if features else 0
    )

    print(
        f"{field:22} "
        f"{n:4}/{len(features):4} "
        f"{pct:6.1f}%"
    )

print()
print("===== SAMPLE =====")

for x in features[:3]:
    print(json.dumps(x, indent=2))

print()
print("[PASS] Research feature construction complete.")
print("[PASS] PAPER_TRADE_V322 was NOT used.")
print("[PASS] Production was NOT modified.")
