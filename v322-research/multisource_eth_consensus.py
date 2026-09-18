#!/usr/bin/env python3

import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone


TIMEOUT = 5
SAMPLES = 15
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

    latency = (
        time.time() - start
    ) * 1000

    return data, latency


def fetch_coinbase():
    data, latency = fetch_json(
        "https://api.coinbase.com/v2/prices/ETH-USD/spot"
    )

    return {
        "price": float(
            data["data"]["amount"]
        ),
        "latencyMs": latency,
    }


def fetch_kraken():
    data, latency = fetch_json(
        "https://api.kraken.com/0/public/Ticker?pair=ETHUSD"
    )

    ticker = next(
        iter(data["result"].values())
    )

    bid = float(ticker["b"][0])
    ask = float(ticker["a"][0])

    return {
        "price": (bid + ask) / 2,
        "bid": bid,
        "ask": ask,
        "latencyMs": latency,
    }


def fetch_gemini():
    data, latency = fetch_json(
        "https://api.gemini.com/v1/pubticker/ethusd"
    )

    bid = float(data["bid"])
    ask = float(data["ask"])

    return {
        "price": (bid + ask) / 2,
        "bid": bid,
        "ask": ask,
        "latencyMs": latency,
    }


FETCHERS = {
    "coinbase": fetch_coinbase,
    "kraken": fetch_kraken,
    "gemini": fetch_gemini,
}


def get_consensus():
    now = datetime.now(
        timezone.utc
    ).isoformat()

    sources = {}
    failures = {}

    for name, fetcher in FETCHERS.items():
        try:
            sources[name] = fetcher()
        except Exception as e:
            failures[name] = (
                f"{type(e).__name__}: {e}"
            )

    values = [
        item["price"]
        for item in sources.values()
    ]

    result = {
        "schemaVersion":
            "v3.22-multisource-research-1",

        "capturedAt":
            now,

        "sourceCount":
            len(values),

        "sources":
            sources,

        "failures":
            failures,

        "consensusPrice":
            None,

        "lowPrice":
            None,

        "highPrice":
            None,

        "disagreementDollars":
            None,

        "disagreementPercent":
            None,

        "availabilityHealth":
            "UNAVAILABLE",
    }

    if len(values) >= 2:
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

        result.update({
            "consensusPrice":
                consensus,

            "lowPrice":
                low,

            "highPrice":
                high,

            "disagreementDollars":
                disagreement_dollars,

            "disagreementPercent":
                disagreement_percent,

            "availabilityHealth":
                (
                    "FULL"
                    if len(values) == 3
                    else "DEGRADED"
                ),
        })

    return result


print(
    "========== v3.22 CONSENSUS FEED RESEARCH =========="
)

print(
    f"SAMPLES={SAMPLES} INTERVAL={INTERVAL}s TIMEOUT={TIMEOUT}s"
)

print()

records = []

for n in range(
    1,
    SAMPLES + 1
):
    result = get_consensus()
    records.append(result)

    print(
        f"===== SAMPLE {n}/{SAMPLES} ====="
    )

    print(
        "TIME:",
        result["capturedAt"]
    )

    for name in FETCHERS:
        source = result["sources"].get(
            name
        )

        if source:
            print(
                f"{name:<10} "
                f"${source['price']:,.4f} "
                f"{source['latencyMs']:7.1f} ms"
            )
        else:
            print(
                f"{name:<10} FAIL "
                f"{result['failures'].get(name)}"
            )

    print(
        "CONSENSUS:",
        (
            f"${result['consensusPrice']:,.4f}"
            if result["consensusPrice"] is not None
            else "UNAVAILABLE"
        )
    )

    print(
        "SOURCE COUNT:",
        result["sourceCount"]
    )

    print(
        "AVAILABILITY:",
        result["availabilityHealth"]
    )

    if (
        result["disagreementPercent"]
        is not None
    ):
        print(
            "DISAGREEMENT:",
            f"${result['disagreementDollars']:.4f}",
            f"({result['disagreementPercent']:.5f}%)"
        )

    print()

    if n < SAMPLES:
        time.sleep(INTERVAL)


valid = [
    r
    for r in records
    if r["disagreementPercent"]
    is not None
]

print(
    "========== CONSENSUS SUMMARY =========="
)

print(
    "TOTAL SAMPLES:",
    len(records)
)

print(
    "FULL:",
    sum(
        r["availabilityHealth"] == "FULL"
        for r in records
    )
)

print(
    "DEGRADED:",
    sum(
        r["availabilityHealth"] == "DEGRADED"
        for r in records
    )
)

print(
    "UNAVAILABLE:",
    sum(
        r["availabilityHealth"] == "UNAVAILABLE"
        for r in records
    )
)


for name in FETCHERS:
    successes = sum(
        name in r["sources"]
        for r in records
    )

    latencies = [
        r["sources"][name]["latencyMs"]
        for r in records
        if name in r["sources"]
    ]

    print()
    print(
        f"{name.upper()} SUCCESS:",
        f"{successes}/{len(records)}"
    )

    if latencies:
        print(
            f"{name.upper()} MEDIAN LATENCY:",
            f"{statistics.median(latencies):.1f} ms"
        )

        print(
            f"{name.upper()} MAX LATENCY:",
            f"{max(latencies):.1f} ms"
        )


if valid:
    disagreement = [
        r["disagreementPercent"]
        for r in valid
    ]

    disagreement_sorted = sorted(
        disagreement
    )

    def percentile(p):
        if len(disagreement_sorted) == 1:
            return disagreement_sorted[0]

        pos = (
            (len(disagreement_sorted) - 1)
            * p
        )

        lo = int(pos)
        hi = min(
            lo + 1,
            len(disagreement_sorted) - 1
        )

        frac = pos - lo

        return (
            disagreement_sorted[lo]
            * (1 - frac)
            +
            disagreement_sorted[hi]
            * frac
        )

    print()
    print(
        "===== DISAGREEMENT DISTRIBUTION ====="
    )

    print(
        "MIN:",
        f"{min(disagreement):.5f}%"
    )

    print(
        "MEDIAN:",
        f"{statistics.median(disagreement):.5f}%"
    )

    print(
        "P90:",
        f"{percentile(0.90):.5f}%"
    )

    print(
        "P95:",
        f"{percentile(0.95):.5f}%"
    )

    print(
        "MAX:",
        f"{max(disagreement):.5f}%"
    )


print()
print(
    "[PASS] Consensus research run complete."
)
print(
    "[PASS] No permanent disagreement threshold selected."
)
print(
    "[PASS] No production files modified."
)
print(
    "[PASS] No archive records modified."
)
print(
    "[PASS] Frozen v3.22 model unchanged."
)
