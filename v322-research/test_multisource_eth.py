#!/usr/bin/env python3

import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone


TIMEOUT = 5
SAMPLES = 5
INTERVAL = 5


def fetch_json(url):

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "Predict-Edge-v3.22-Research/1.0"
        }
    )

    start = time.time()

    with urllib.request.urlopen(
        req,
        timeout=TIMEOUT
    ) as response:

        data = json.loads(
            response.read().decode("utf-8")
        )

    latency_ms = (
        time.time() - start
    ) * 1000

    return data, latency_ms


def coinbase():

    data, latency = fetch_json(
        "https://api.coinbase.com/v2/prices/ETH-USD/spot"
    )

    price = float(
        data["data"]["amount"]
    )

    return price, latency


def kraken():

    data, latency = fetch_json(
        "https://api.kraken.com/0/public/Ticker?pair=ETHUSD"
    )

    result = next(
        iter(data["result"].values())
    )

    bid = float(result["b"][0])
    ask = float(result["a"][0])

    midpoint = (
        bid + ask
    ) / 2

    return midpoint, latency


def gemini():

    data, latency = fetch_json(
        "https://api.gemini.com/v1/pubticker/ethusd"
    )

    bid = float(data["bid"])
    ask = float(data["ask"])

    midpoint = (
        bid + ask
    ) / 2

    return midpoint, latency


FETCHERS = {
    "Coinbase": coinbase,
    "Kraken": kraken,
    "Gemini": gemini,
}


print(
    "========== v3.22 MULTI-SOURCE ETH TEST =========="
)

print(
    f"SAMPLES={SAMPLES}  INTERVAL={INTERVAL}s"
)

print()


all_disagreements = []
success_counts = {
    name: 0
    for name in FETCHERS
}


for sample in range(
    1,
    SAMPLES + 1
):

    print(
        f"===== SAMPLE {sample}/{SAMPLES} ====="
    )

    print(
        "TIME:",
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    prices = {}

    for name, fetcher in FETCHERS.items():

        try:

            price, latency = fetcher()

            prices[name] = price

            success_counts[name] += 1

            print(
                f"{name:<10} "
                f"${price:,.4f}  "
                f"{latency:7.1f} ms"
            )

        except Exception as e:

            print(
                f"{name:<10} FAIL "
                f"{type(e).__name__}: {e}"
            )

    if len(prices) >= 2:

        values = list(
            prices.values()
        )

        consensus = statistics.median(
            values
        )

        low = min(values)
        high = max(values)

        disagreement_dollars = (
            high - low
        )

        disagreement_percent = (
            disagreement_dollars
            / consensus
        ) * 100

        all_disagreements.append(
            disagreement_percent
        )

        print()
        print(
            "CONSENSUS:",
            f"${consensus:,.4f}"
        )

        print(
            "LOW/HIGH:",
            f"${low:,.4f}",
            "/",
            f"${high:,.4f}"
        )

        print(
            "SOURCE RANGE:",
            f"${disagreement_dollars:.4f}",
            f"({disagreement_percent:.5f}%)"
        )

        print(
            "SOURCE COUNT:",
            len(prices)
        )

    else:

        print()
        print(
            "CONSENSUS: UNAVAILABLE"
        )

    print()

    if sample < SAMPLES:
        time.sleep(INTERVAL)


print(
    "========== SOURCE HEALTH =========="
)

for name in FETCHERS:

    print(
        f"{name:<10} "
        f"{success_counts[name]}/{SAMPLES}"
    )


if all_disagreements:

    print()
    print(
        "MEDIAN SOURCE DISAGREEMENT:",
        f"{statistics.median(all_disagreements):.5f}%"
    )

    print(
        "MAX SOURCE DISAGREEMENT:",
        f"{max(all_disagreements):.5f}%"
    )


print()
print(
    "[PASS] Research-only multi-source test complete."
)
print(
    "[PASS] No archive records modified."
)
print(
    "[PASS] No production files modified."
)
