from pathlib import Path
import json
import re
import math
import statistics
from datetime import datetime
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)

# ============================================================
# FROZEN v3.22 PARAMETERS
# ============================================================

HORIZONS = [5, 10, 15, 30]
TOLERANCE = 1.5
LOOKBACKS = [1, 5, 15, 30]

VOL_FLOOR = 0.020

MODEL_FEATURES = [
    "normalizedDistance",
    "momentum1m",
    "momentum5m",
    "momentum15m",
    "momentum30m",
]

OG_FILES = [
    Path("/media/user/ECX-SSD/Predict-Edge-Archive/backups/Predict_Edge_v3.21.2_Working_Dataset_20260916-114848.json"),
    Path("/media/user/ECX-SSD/Predict-Edge-Archive/backups/Predict_Edge_FULL_ARCHIVE_PreTrim_20260916-084835.json"),
    Path("/media/user/ECX-SSD/Predict-Edge-Archive/datasets/frozen/Predict_Edge_Dataset_Backup_20260917-082924.json"),
    Path("/media/user/ECX-SSD/Predict-Edge-Archive/datasets/frozen/Predict_Edge_Dataset_Backup_20260917-100710.json"),
    Path("/media/user/ECX-SSD/Predict-Edge-Archive/datasets/frozen/Predict_Edge_Dataset_Backup_20260917-190716.json"),
]

LIVE_FILE = Path(
    "/media/user/ECX-SSD/Predict-Edge-Archive/datasets/live/"
    "predict-edge-live.jsonl"
)


def timestamp(x):
    if not x:
        return None

    try:
        return datetime.fromisoformat(
            str(x).replace("Z", "+00:00")
        ).timestamp()
    except Exception:
        return None


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


print("========== v3.22 PROSPECTIVE VALIDATOR ==========")
print()
print("Frozen model features:")
for f in MODEL_FEATURES:
    print(" ", f)

print()
print("VOL_FLOOR:", VOL_FLOOR)
print("OG FILES:", len(OG_FILES))
print("LIVE FILE:", LIVE_FILE)
print("LIVE EXISTS:", LIVE_FILE.exists())

print()
print("[PASS] Validator framework created.")
print("[PASS] No model executed yet.")
print("[PASS] No archive modified.")

# ============================================================
# LOAD HISTORICAL OG TRAINING DATA ONLY
# ============================================================

def fingerprint(r):
    return (
        str(r.get("symbol", "")),
        str(r.get("capturedAt", "")),
        str(r.get("contractTarget", "")),
        str(r.get("probability", "")),
    )


rows = {}
eth = {}

for path in OG_FILES:

    root = json.loads(
        path.read_text(errors="ignore")
    )

    for r in root.get("liveHistory", []):

        if not isinstance(r, dict):
            continue

        fp = fingerprint(r)

        if not fp[0] or not fp[1]:
            continue

        old = rows.get(fp)

        if old is None or len(r) > len(old):
            rows[fp] = r

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


series = sorted(eth.items())
times = [x[0] for x in series]


def price_before(target, max_age=120):

    i = bisect_left(times, target) - 1

    if i < 0:
        return None

    t, price = series[i]

    if target - t > max_age:
        return None

    return price


# ============================================================
# REBUILD FROZEN HISTORICAL FEATURES
# ============================================================

historical = []

for r in rows.values():

    if r.get("phaseCode") != "PRE":
        continue

    if r.get("modelEligible") is not True:
        continue

    try:
        delta = abs(
            float(r.get("phaseDeltaMinutes"))
        )
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
        current_eth = float(
            r.get("ethReferencePrice")
        )

        market_probability = float(
            r.get("probability")
        )
    except Exception:
        continue

    captured = timestamp(
        r.get("capturedAt")
    )

    target = timestamp(
        r.get("contractTarget")
    )

    if captured is None or target is None:
        continue

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

    minute_prices = []

    for minute in range(15, -1, -1):

        px = price_before(
            captured - minute * 60
        )

        if px is not None:
            minute_prices.append(px)

    minute_returns = []

    for a, b in zip(
        minute_prices,
        minute_prices[1:]
    ):

        if a > 0:
            minute_returns.append(
                ((b / a) - 1) * 100
            )

    if len(minute_returns) < 5:
        continue

    volatility = statistics.pstdev(
        minute_returns
    )

    distance_percent = (
        (current_eth - s) / s
    ) * 100

    effective_vol = max(
        volatility,
        VOL_FLOOR
    )

    expected_move = (
        effective_vol *
        math.sqrt(horizon)
    )

    normalized_distance = math.tanh(
        distance_percent / expected_move
    )

    # ----------------------------------------
    # ROBUST HISTORICAL OUTCOME
    # ±120 seconds around contract target
    # ----------------------------------------

    lo = bisect_left(
        times,
        target - 120
    )

    hi = bisect_right(
        times,
        target + 120
    )

    target_prices = [
        price
        for t, price in series[lo:hi]
        if abs(t - target) <= 120
    ]

    if not target_prices:
        continue

    if all(px > s for px in target_prices):
        outcome = 1

    elif all(px < s for px in target_prices):
        outcome = 0

    else:
        # Ambiguous contract: reference prices
        # crossed the strike near settlement.
        continue

    historical.append({
        "symbol": r.get("symbol"),
        "capturedAt": r.get("capturedAt"),
        "target": r.get("contractTarget"),
        "horizon": horizon,
        "strike": s,
        "marketProbability":
            market_probability / 100.0,
        "normalizedDistance":
            normalized_distance,
        "momentum1m": moves[1],
        "momentum5m": moves[5],
        "momentum15m": moves[15],
        "momentum30m": moves[30],
        "outcome": outcome,
        "timingError": abs(delta - horizon),
    })


print()
print("========== HISTORICAL FEATURE REBUILD ==========")
print("DEDUPLICATED OG:", len(rows))
print("ETH REFERENCES:", len(series))
print("ROBUST FEATURE ROWS:", len(historical))
print(
    "TARGETS:",
    len({
        r["target"]
        for r in historical
    })
)
print(
    "SYMBOLS:",
    len({
        r["symbol"]
        for r in historical
    })
)


# ============================================================
# CANONICALIZE:
# ONE OBSERVATION CLOSEST TO EXACT HORIZON
# PER TARGET / SYMBOL / STRIKE / HORIZON
# ============================================================

canonical = {}

for r in historical:

    key = (
        r["target"],
        r["symbol"],
        r["strike"],
        r["horizon"],
    )

    old = canonical.get(key)

    if (
        old is None
        or r["timingError"] < old["timingError"]
    ):
        canonical[key] = r


canonical = list(canonical.values())


print()
print("========== CANONICAL HISTORICAL TRAINING ==========")
print("CANONICAL ROWS:", len(canonical))
print(
    "TARGETS:",
    len({
        r["target"]
        for r in canonical
    })
)
print(
    "CONTRACTS:",
    len({
        (
            r["target"],
            r["symbol"],
            r["strike"],
        )
        for r in canonical
    })
)

print()
print("===== CANONICAL HORIZONS =====")

for h in HORIZONS:

    subset = [
        r
        for r in canonical
        if r["horizon"] == h
    ]

    print(
        f"T-{h:<2}",
        f"N={len(subset):3}",
        "MAX_TIMING_ERROR=",
        f"{max((r['timingError'] for r in subset), default=0)*60:.1f}s"
    )


print()
print("===== TRAINING LABELS =====")

labels = Counter(
    r["outcome"]
    for r in canonical
)

print("NO :", labels[0])
print("YES:", labels[1])


print()
print("[PASS] Historical data only.")
print("[PASS] Canonicalization complete.")
print("[PASS] No model fitted.")
print("[PASS] Prospective records NOT scored.")
print("[PASS] Archive NOT modified.")

# ============================================================
# LOAD UNTOUCHED PROSPECTIVE DATA + LIVE ETH REFERENCES
# ============================================================

paper = []
live_eth = {}

with LIVE_FILE.open("r", encoding="utf-8") as f:

    for line in f:

        try:
            wrapper = json.loads(line)
        except Exception:
            continue

        payload = wrapper.get("payload")

        if not isinstance(payload, dict):
            continue

        rtype = payload.get("recordType")
        record = payload.get("record")

        if not isinstance(record, dict):
            continue

        if rtype == "PAPER_TRADE_V322":

            # Integrity gate.
            if record.get("trainingEligible") is True:
                raise RuntimeError(
                    "Prospective trainingEligible=True"
                )

            if record.get("fittedV322Probability") is not None:
                raise RuntimeError(
                    "Prospective fitted probability already populated"
                )

            paper.append(record)

        elif rtype == "ETH_REFERENCE":

            if record.get("valid") is not True:
                continue

            t = timestamp(
                record.get("fetchedAt")
                or record.get("apiTime")
            )

            try:
                price = float(
                    record.get("price")
                )
            except Exception:
                continue

            if t is not None and price > 0:
                live_eth[t] = price


live_series = sorted(live_eth.items())
live_times = [x[0] for x in live_series]


def live_price_before(target, max_age=120):

    i = bisect_left(
        live_times,
        target
    ) - 1

    if i < 0:
        return None

    t, price = live_series[i]

    if target - t > max_age:
        return None

    return price


print()
print("========== PROSPECTIVE SOURCE LOAD ==========")
print("PAPER RECORDS:", len(paper))
print("LIVE ETH REFERENCES:", len(live_series))


# ============================================================
# BUILD PROSPECTIVE FEATURES
# EXACT SAME FROZEN FEATURE DEFINITIONS
# ============================================================

prospective = []
rejected = Counter()

for r in paper:

    s = strike(r)

    if s is None:
        rejected["strike"] += 1
        continue

    try:
        horizon = int(
            r.get("horizonMinutes")
        )

        current_eth = float(
            r.get("ethReferencePrice")
        )

        market_probability = float(
            r.get("marketProbability")
        )
    except Exception:
        rejected["basic_fields"] += 1
        continue

    if horizon not in HORIZONS:
        rejected["horizon"] += 1
        continue

    captured = timestamp(
        r.get("capturedAt")
    )

    target = timestamp(
        r.get("contractTarget")
    )

    if captured is None or target is None:
        rejected["timestamp"] += 1
        continue

    # ----------------------------------------
    # MOMENTUM: past data only
    # ----------------------------------------

    moves = {}
    complete = True

    for minutes in LOOKBACKS:

        old_price = live_price_before(
            captured - minutes * 60
        )

        if old_price is None or old_price <= 0:
            complete = False
            break

        moves[minutes] = (
            (current_eth / old_price) - 1
        ) * 100

    if not complete:
        rejected["momentum"] += 1
        continue

    # ----------------------------------------
    # 15m realized volatility: past data only
    # ----------------------------------------

    minute_prices = []

    for minute in range(15, -1, -1):

        px = live_price_before(
            captured - minute * 60
        )

        if px is not None:
            minute_prices.append(px)

    minute_returns = []

    for a, b in zip(
        minute_prices,
        minute_prices[1:]
    ):

        if a > 0:
            minute_returns.append(
                ((b / a) - 1) * 100
            )

    if len(minute_returns) < 5:
        rejected["volatility"] += 1
        continue

    volatility = statistics.pstdev(
        minute_returns
    )

    distance_percent = (
        (current_eth - s) / s
    ) * 100

    effective_vol = max(
        volatility,
        VOL_FLOOR
    )

    expected_move = (
        effective_vol *
        math.sqrt(horizon)
    )

    normalized_distance = math.tanh(
        distance_percent / expected_move
    )

    # ----------------------------------------
    # OUTCOME: target-time data only
    # Used only AFTER features are constructed.
    # ----------------------------------------

    lo = bisect_left(
        live_times,
        target - 120
    )

    hi = bisect_right(
        live_times,
        target + 120
    )

    target_prices = [
        price
        for t, price in live_series[lo:hi]
        if abs(t - target) <= 120
    ]

    if not target_prices:
        rejected["outcome_no_data"] += 1
        continue

    if all(px > s for px in target_prices):
        outcome = 1

    elif all(px < s for px in target_prices):
        outcome = 0

    else:
        rejected["outcome_ambiguous"] += 1
        continue

    prospective.append({
        "symbol": r.get("symbol"),
        "target": r.get("contractTarget"),
        "capturedAt": r.get("capturedAt"),
        "horizon": horizon,
        "strike": s,

        "marketProbability":
            market_probability / 100.0,

        "normalizedDistance":
            normalized_distance,

        "momentum1m": moves[1],
        "momentum5m": moves[5],
        "momentum15m": moves[15],
        "momentum30m": moves[30],

        "outcome": outcome,
    })


print()
print("========== PROSPECTIVE FEATURE BUILD ==========")
print("FEATUREABLE ROBUST ROWS:", len(prospective))
print(
    "TARGETS:",
    len({
        r["target"]
        for r in prospective
    })
)
print(
    "CONTRACTS:",
    len({
        (
            r["target"],
            r["symbol"],
            r["strike"],
        )
        for r in prospective
    })
)

print()
print("REJECTIONS:")

if rejected:
    for name, n in sorted(rejected.items()):
        print(f"  {name:<24} {n}")
else:
    print("  NONE")


# ============================================================
# FIT FROZEN MODEL
# HISTORICAL CANONICAL DATA ONLY
# ============================================================

X_train = np.array([
    [
        r[name]
        for name in MODEL_FEATURES
    ]
    for r in canonical
])

y_train = np.array([
    r["outcome"]
    for r in canonical
])


model = Pipeline([
    (
        "scale",
        StandardScaler()
    ),
    (
        "logistic",
        LogisticRegression(
            C=1,
            max_iter=5000,
            random_state=322,
        )
    ),
])


model.fit(
    X_train,
    y_train
)


print()
print("========== MODEL FIT ==========")
print("TRAINING ROWS:", len(X_train))
print("TRAINING TARGETS:", len({
    r["target"]
    for r in canonical
}))
print("PROSPECTIVE TRAINING ROWS: 0")
print("[PASS] Model trained on historical canonical OG only.")


# ============================================================
# SCORE PROSPECTIVE HOLDOUT IN MEMORY
# ============================================================

X_test = np.array([
    [
        r[name]
        for name in MODEL_FEATURES
    ]
    for r in prospective
])

y = np.array([
    r["outcome"]
    for r in prospective
])

market = np.array([
    r["marketProbability"]
    for r in prospective
])

v322 = model.predict_proba(
    X_test
)[:, 1]


for r, probability in zip(
    prospective,
    v322
):
    r["v322Probability"] = float(
        probability
    )


def metrics(prob, truth):

    pred = (
        prob >= 0.5
    ).astype(int)

    return {
        "n": len(truth),
        "acc": accuracy_score(
            truth,
            pred
        ),
        "brier": brier_score_loss(
            truth,
            prob
        ),
        "logloss": log_loss(
            truth,
            prob,
            labels=[0, 1],
        ),
    }


m_market = metrics(
    market,
    y
)

m_v322 = metrics(
    v322,
    y
)


print()
print("==============================================")
print("      v3.22 UNTOUCHED PROSPECTIVE RESULT")
print("==============================================")

print()
print("ROWS:", len(prospective))
print(
    "TARGETS:",
    len({
        r["target"]
        for r in prospective
    })
)

print()
print("MARKET")
print(
    f"  ACC:     {m_market['acc']*100:.3f}%"
)
print(
    f"  BRIER:   {m_market['brier']:.5f}"
)
print(
    f"  LOGLOSS: {m_market['logloss']:.5f}"
)

print()
print("v3.22")
print(
    f"  ACC:     {m_v322['acc']*100:.3f}%"
)
print(
    f"  BRIER:   {m_v322['brier']:.5f}"
)
print(
    f"  LOGLOSS: {m_v322['logloss']:.5f}"
)

print()
print(
    "ACCURACY DELTA:",
    f"{(m_v322['acc']-m_market['acc'])*100:+.3f} pp"
)

print(
    "BRIER IMPROVEMENT:",
    f"{m_market['brier']-m_v322['brier']:+.5f}"
)


# ============================================================
# BY HORIZON
# ============================================================

print()
print("===== BY HORIZON =====")

for h in HORIZONS:

    idx = np.array([
        i
        for i, r in enumerate(prospective)
        if r["horizon"] == h
    ])

    if len(idx) == 0:
        continue

    mm = metrics(
        market[idx],
        y[idx]
    )

    vv = metrics(
        v322[idx],
        y[idx]
    )

    print(
        f"T-{h:<2} "
        f"N={len(idx):3} "
        f"MARKET_ACC={mm['acc']*100:6.2f}% "
        f"V322_ACC={vv['acc']*100:6.2f}% "
        f"MARKET_BRIER={mm['brier']:.5f} "
        f"V322_BRIER={vv['brier']:.5f} "
        f"DELTA={mm['brier']-vv['brier']:+.5f}"
    )


# ============================================================
# TARGET-BY-TARGET BRIER
# ============================================================

print()
print("===== TARGET RESULTS =====")

groups = defaultdict(list)

for i, r in enumerate(prospective):
    groups[r["target"]].append(i)


wins_v322 = 0
wins_market = 0
ties = 0

for target in sorted(groups):

    idx = np.array(
        groups[target]
    )

    mm = metrics(
        market[idx],
        y[idx]
    )

    vv = metrics(
        v322[idx],
        y[idx]
    )

    delta = (
        mm["brier"] -
        vv["brier"]
    )

    if delta > 1e-12:
        winner = "V322"
        wins_v322 += 1

    elif delta < -1e-12:
        winner = "MARKET"
        wins_market += 1

    else:
        winner = "TIE"
        ties += 1

    print(
        target,
        f"N={len(idx):3}",
        f"MARKET={mm['brier']:.5f}",
        f"V322={vv['brier']:.5f}",
        f"DELTA={delta:+.5f}",
        winner
    )


print()
print("===== TARGET WIN SUMMARY =====")
print("V322:", wins_v322)
print("MARKET:", wins_market)
print("TIES:", ties)


print()
print("==============================================")
print("[PASS] Prospective scoring completed in memory.")
print("[PASS] PAPER_TRADE_V322 archive records unchanged.")
print("[PASS] No prospective observation used for training.")
print("[PASS] Production index.html unchanged by validator.")
print("==============================================")

# ============================================================
# TARGET-BALANCED PROSPECTIVE AUDIT
#
# Each expiration target receives equal weight regardless
# of how many contracts were observed for that target.
# No fitting. No tuning. No archive writes.
# ============================================================

print()
print("==============================================")
print("       TARGET-BALANCED PROSPECTIVE AUDIT")
print("==============================================")

target_audit = []

for target in sorted(groups):

    idx = np.array(groups[target])

    truth_t = y[idx]
    market_t = market[idx]
    v322_t = v322[idx]

    market_metrics = metrics(
        market_t,
        truth_t
    )

    v322_metrics = metrics(
        v322_t,
        truth_t
    )

    target_audit.append({
        "target": target,
        "n": len(idx),

        "market_acc":
            market_metrics["acc"],

        "v322_acc":
            v322_metrics["acc"],

        "market_brier":
            market_metrics["brier"],

        "v322_brier":
            v322_metrics["brier"],

        "market_logloss":
            market_metrics["logloss"],

        "v322_logloss":
            v322_metrics["logloss"],
    })


# ------------------------------------------------------------
# Equal-weight means:
# every target contributes exactly 1/number_of_targets.
# ------------------------------------------------------------

balanced_market_acc = statistics.mean(
    x["market_acc"]
    for x in target_audit
)

balanced_v322_acc = statistics.mean(
    x["v322_acc"]
    for x in target_audit
)

balanced_market_brier = statistics.mean(
    x["market_brier"]
    for x in target_audit
)

balanced_v322_brier = statistics.mean(
    x["v322_brier"]
    for x in target_audit
)

balanced_market_logloss = statistics.mean(
    x["market_logloss"]
    for x in target_audit
)

balanced_v322_logloss = statistics.mean(
    x["v322_logloss"]
    for x in target_audit
)


print()
print("TARGETS:", len(target_audit))
print("ROWS:", len(prospective))

print()
print("===== EQUAL-WEIGHT TARGET METRICS =====")

print()
print("MARKET")
print(
    f"  ACC:     "
    f"{balanced_market_acc*100:.3f}%"
)
print(
    f"  BRIER:   "
    f"{balanced_market_brier:.5f}"
)
print(
    f"  LOGLOSS: "
    f"{balanced_market_logloss:.5f}"
)

print()
print("v3.22")
print(
    f"  ACC:     "
    f"{balanced_v322_acc*100:.3f}%"
)
print(
    f"  BRIER:   "
    f"{balanced_v322_brier:.5f}"
)
print(
    f"  LOGLOSS: "
    f"{balanced_v322_logloss:.5f}"
)

print()
print(
    "BALANCED ACCURACY DELTA:",
    f"{(balanced_v322_acc-balanced_market_acc)*100:+.3f} pp"
)

print(
    "BALANCED BRIER IMPROVEMENT:",
    f"{balanced_market_brier-balanced_v322_brier:+.5f}"
)

print(
    "BALANCED LOGLOSS IMPROVEMENT:",
    f"{balanced_market_logloss-balanced_v322_logloss:+.5f}"
)


# ------------------------------------------------------------
# Target-by-target audit
# ------------------------------------------------------------

print()
print("===== INDIVIDUAL TARGETS =====")

v322_brier_wins = 0
market_brier_wins = 0
brier_ties = 0

v322_acc_wins = 0
market_acc_wins = 0
acc_ties = 0

for x in target_audit:

    brier_delta = (
        x["market_brier"]
        -
        x["v322_brier"]
    )

    acc_delta = (
        x["v322_acc"]
        -
        x["market_acc"]
    )

    if brier_delta > 1e-12:
        bwinner = "V322"
        v322_brier_wins += 1

    elif brier_delta < -1e-12:
        bwinner = "MARKET"
        market_brier_wins += 1

    else:
        bwinner = "TIE"
        brier_ties += 1

    if acc_delta > 1e-12:
        awinner = "V322"
        v322_acc_wins += 1

    elif acc_delta < -1e-12:
        awinner = "MARKET"
        market_acc_wins += 1

    else:
        awinner = "TIE"
        acc_ties += 1

    print(
        x["target"],
        f"N={x['n']:3}",
        f"M_ACC={x['market_acc']*100:6.2f}%",
        f"V_ACC={x['v322_acc']*100:6.2f}%",
        f"M_BR={x['market_brier']:.5f}",
        f"V_BR={x['v322_brier']:.5f}",
        f"BR_DELTA={brier_delta:+.5f}",
        bwinner
    )


print()
print("===== TARGET WIN COUNTS =====")

print(
    "BRIER:",
    f"V322={v322_brier_wins}",
    f"MARKET={market_brier_wins}",
    f"TIES={brier_ties}"
)

print(
    "ACCURACY:",
    f"V322={v322_acc_wins}",
    f"MARKET={market_acc_wins}",
    f"TIES={acc_ties}"
)


# ------------------------------------------------------------
# Median target performance
# Useful because a single extreme target cannot dominate median.
# ------------------------------------------------------------

median_market_brier = statistics.median(
    x["market_brier"]
    for x in target_audit
)

median_v322_brier = statistics.median(
    x["v322_brier"]
    for x in target_audit
)

median_market_acc = statistics.median(
    x["market_acc"]
    for x in target_audit
)

median_v322_acc = statistics.median(
    x["v322_acc"]
    for x in target_audit
)

print()
print("===== MEDIAN TARGET PERFORMANCE =====")

print(
    "MARKET MEDIAN ACC:",
    f"{median_market_acc*100:.3f}%"
)

print(
    "V322 MEDIAN ACC:",
    f"{median_v322_acc*100:.3f}%"
)

print(
    "MARKET MEDIAN BRIER:",
    f"{median_market_brier:.5f}"
)

print(
    "V322 MEDIAN BRIER:",
    f"{median_v322_brier:.5f}"
)


print()
print("==============================================")
print("[PASS] Target-balanced audit complete.")
print("[PASS] Every target received equal weight.")
print("[PASS] Frozen v3.22 model was NOT refitted.")
print("[PASS] No threshold or parameter was tuned.")
print("[PASS] Prospective archive was NOT modified.")
print("==============================================")


# ============================================================
# v3.22 ETH REFERENCE SENSITIVITY
#
# RESEARCH ONLY
#
# Isolates ONLY current ETH reference displacement.
# Momentum, volatility, training data, scaler and model remain
# exactly those used by the frozen validator above.
#
# Observed independent-consensus vs Crypto.com absolute deltas:
#
# median = 0.01556%
# P95    = 0.03563%
# max    = 0.04127%
#
# Both +/- directions are tested because the short live sample
# must not be treated as establishing a permanent directional
# bias between feeds.
# ============================================================

print()
print(
    "========== ETH REFERENCE SENSITIVITY =========="
)

REFERENCE_SHIFTS = [
    ("MEDIAN", 0.01556),
    ("P95",    0.03563),
    ("MAX",    0.04127),
]


def shifted_normalized_distance(
    row,
    shift_percent,
):

    # Recover distance ratio represented by the frozen
    # normalizedDistance:
    #
    # normalized = tanh(distancePercent / expectedMove)
    #
    # We cannot reconstruct current ETH and strike uniquely
    # from normalizedDistance alone. Fortunately each
    # prospective row retains strike but not current ETH /
    # expectedMove.
    #
    # Therefore this section intentionally rebuilds the
    # required current-time quantities from the same live
    # ETH series used by the validator.

    captured = timestamp(
        row["capturedAt"]
    )

    current_eth = live_price_before(
        captured
    )

    if (
        current_eth is None
        or current_eth <= 0
    ):
        return None

    strike = float(
        row["strike"]
    )

    horizon = float(
        row["horizon"]
    )

    # Rebuild frozen 15-minute volatility exactly.
    minute_prices = []

    for minute in range(
        15,
        -1,
        -1
    ):

        px = live_price_before(
            captured - minute * 60
        )

        if px is not None:
            minute_prices.append(px)

    minute_returns = []

    for a, b in zip(
        minute_prices,
        minute_prices[1:]
    ):

        if a > 0:
            minute_returns.append(
                ((b / a) - 1) * 100
            )

    if len(minute_returns) < 5:
        return None

    volatility = statistics.pstdev(
        minute_returns
    )

    effective_vol = max(
        volatility,
        VOL_FLOOR
    )

    expected_move = (
        effective_vol
        * math.sqrt(horizon)
    )

    # Change ONLY current ETH reference.
    shifted_eth = (
        current_eth
        * (1 + shift_percent / 100.0)
    )

    shifted_distance_percent = (
        (shifted_eth - strike)
        / strike
    ) * 100

    return math.tanh(
        shifted_distance_percent
        / expected_move
    )


baseline_probabilities = np.array([
    r["v322Probability"]
    for r in prospective
])


print(
    "ROWS:",
    len(prospective)
)

print(
    "TARGETS:",
    len({
        r["target"]
        for r in prospective
    })
)

print()
print(
    "Shift represents ETH-reference displacement,"
)
print(
    "not model retraining or outcome-based tuning."
)


for label, magnitude in REFERENCE_SHIFTS:

    for sign in (-1, 1):

        shift = (
            magnitude * sign
        )

        shifted_features = []
        baseline_subset = []

        for index, row in enumerate(
            prospective
        ):

            nd = shifted_normalized_distance(
                row,
                shift,
            )

            if nd is None:
                continue

            features = []

            for name in MODEL_FEATURES:

                if name == "normalizedDistance":
                    features.append(nd)
                else:
                    features.append(
                        row[name]
                    )

            shifted_features.append(
                features
            )

            baseline_subset.append(
                baseline_probabilities[index]
            )

        if not shifted_features:
            print(
                label,
                f"{shift:+.5f}%",
                "NO FEATUREABLE ROWS"
            )
            continue

        shifted_probabilities = (
            model.predict_proba(
                np.array(
                    shifted_features
                )
            )[:, 1]
        )

        baseline_subset = np.array(
            baseline_subset
        )

        delta = (
            shifted_probabilities
            - baseline_subset
        )

        abs_delta = np.abs(
            delta
        )

        direction_flips = int(
            np.sum(
                (
                    baseline_subset >= 0.5
                )
                !=
                (
                    shifted_probabilities >= 0.5
                )
            )
        )

        print()
        print(
            f"===== {label} ETH SHIFT "
            f"{shift:+.5f}% ====="
        )

        print(
            "ROWS:",
            len(abs_delta)
        )

        print(
            "MEDIAN PROBABILITY CHANGE:",
            f"{np.median(abs_delta) * 100:.4f} pp"
        )

        print(
            "P90 PROBABILITY CHANGE:",
            f"{np.percentile(abs_delta, 90) * 100:.4f} pp"
        )

        print(
            "P95 PROBABILITY CHANGE:",
            f"{np.percentile(abs_delta, 95) * 100:.4f} pp"
        )

        print(
            "MAX PROBABILITY CHANGE:",
            f"{np.max(abs_delta) * 100:.4f} pp"
        )

        print(
            "MEAN SIGNED CHANGE:",
            f"{np.mean(delta) * 100:+.4f} pp"
        )

        print(
            "DIRECTION FLIPS:",
            direction_flips
        )


print()
print(
    "========== SENSITIVITY TEST COMPLETE =========="
)

print(
    "[PASS] Historical training data unchanged."
)
print(
    "[PASS] Momentum features unchanged."
)
print(
    "[PASS] Volatility feature construction unchanged."
)
print(
    "[PASS] Only current ETH reference displaced."
)
print(
    "[PASS] No prospective outcomes used for tuning."
)
print(
    "[PASS] No archive records modified."
)
print(
    "[PASS] No production files modified."
)
