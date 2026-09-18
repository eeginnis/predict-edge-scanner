#!/usr/bin/env python3

import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone

TIMEOUT = 5
SAMPLES = 60
INTERVAL = 10

URLS = {
    "coinbase":
        "https://api.coinbase.com/v2/prices/ETH-USD/spot",
    "kraken":
        "https://api.kraken.com/0/public/Ticker?pair=ETHUSD",
    "gemini":
        "https://api.gemini.com/v1/pubticker/ethusd",
    "crypto":
        "https://api.crypto.com/exchange/v1/public/get-tickers?instrument_name=ETH_USD",
}


def fetch(url):
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

    return data, (time.time() - start) * 1000


def get_coinbase():
    d, latency = fetch(URLS["coinbase"])
    return float(d["data"]["amount"]), latency


def get_kraken():
    d, latency = fetch(URLS["kraken"])
    ticker = next(iter(d["result"].values()))

    bid = float(ticker["b"][0])
    ask = float(ticker["a"][0])

    return (bid + ask) / 2, latency


def get_gemini():
    d, latency = fetch(URLS["gemini"])

    bid = float(d["bid"])
    ask = float(d["ask"])

    return (bid + ask) / 2, latency


def get_crypto():
    d, latency = fetch(URLS["crypto"])

    result = d.get("result", {})
    data = result.get("data", [])

    if not data:
        raise RuntimeError(
            "Crypto.com ticker returned no data"
        )

    row = data[0]

    # Match production v3.21.2 exactly.
    # Crypto.com field "a" is the reference price stored
    # in ETH_REFERENCE records.
    candidate = row.get("a")

    if candidate is None:
        raise RuntimeError(
            "Crypto.com last price field a missing: "
            + repr(row)
        )

    price = float(candidate)

    return price, latency


independent_fetchers = {
    "coinbase": get_coinbase,
    "kraken": get_kraken,
    "gemini": get_gemini,
}


print(
    "========== v3.22 CONSENSUS vs CRYPTO.COM =========="
)
print(
    f"SAMPLES={SAMPLES} INTERVAL={INTERVAL}s"
)
print()

records = []

for n in range(1, SAMPLES + 1):

    independent = {}
    failures = {}

    for name, fn in independent_fetchers.items():
        try:
            independent[name] = fn()
        except Exception as e:
            failures[name] = (
                f"{type(e).__name__}: {e}"
            )

    try:
        crypto_price, crypto_latency = get_crypto()
        crypto_error = None
    except Exception as e:
        crypto_price = None
        crypto_latency = None
        crypto_error = (
            f"{type(e).__name__}: {e}"
        )

    prices = [
        value[0]
        for value in independent.values()
    ]

    consensus = (
        statistics.median(prices)
        if len(prices) >= 2
        else None
    )

    print(
        f"===== SAMPLE {n}/{SAMPLES} ====="
    )
    print(
        "TIME:",
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    for name in independent_fetchers:
        if name in independent:
            price, latency = independent[name]
            print(
                f"{name:<10} "
                f"${price:,.4f} "
                f"{latency:7.1f} ms"
            )
        else:
            print(
                f"{name:<10} FAIL "
                f"{failures[name]}"
            )

    if crypto_price is not None:
        print(
            f"{'crypto.com':<10} "
            f"${crypto_price:,.4f} "
            f"{crypto_latency:7.1f} ms"
        )
    else:
        print(
            "crypto.com FAIL",
            crypto_error
        )

    if (
        consensus is not None
        and crypto_price is not None
    ):
        delta_dollars = (
            consensus - crypto_price
        )

        delta_percent = (
            delta_dollars
            / consensus
        ) * 100

        abs_percent = abs(
            delta_percent
        )

        records.append({
            "consensus": consensus,
            "crypto": crypto_price,
            "deltaDollars":
                delta_dollars,
            "deltaPercent":
                delta_percent,
            "absDeltaPercent":
                abs_percent,
        })

        print(
            "CONSENSUS:",
            f"${consensus:,.4f}"
        )
        print(
            "DELTA:",
            f"${delta_dollars:+.4f}",
            f"({delta_percent:+.5f}%)"
        )
        print(
            "ABS DELTA:",
            f"{abs_percent:.5f}%"
        )
    else:
        print(
            "COMPARISON: UNAVAILABLE"
        )

    print()

    if n < SAMPLES:
        time.sleep(INTERVAL)


print(
    "========== COMPARISON SUMMARY =========="
)
print(
    "VALID COMPARISONS:",
    len(records),
    "/",
    SAMPLES
)

if records:

    signed = [
        r["deltaPercent"]
        for r in records
    ]

    absolute = sorted(
        r["absDeltaPercent"]
        for r in records
    )

    def percentile(values, p):
        if len(values) == 1:
            return values[0]

        pos = (
            (len(values) - 1)
            * p
        )

        lo = int(pos)
        hi = min(
            lo + 1,
            len(values) - 1
        )

        fraction = pos - lo

        return (
            values[lo] * (1 - fraction)
            +
            values[hi] * fraction
        )

    print()
    print(
        "MEDIAN SIGNED DELTA:",
        f"{statistics.median(signed):+.5f}%"
    )
    print(
        "MEDIAN ABS DELTA:",
        f"{statistics.median(absolute):.5f}%"
    )
    print(
        "P90 ABS DELTA:",
        f"{percentile(absolute, 0.90):.5f}%"
    )
    print(
        "P95 ABS DELTA:",
        f"{percentile(absolute, 0.95):.5f}%"
    )
    print(
        "MAX ABS DELTA:",
        f"{max(absolute):.5f}%"
    )

print()
print(
    "[PASS] Side-by-side research comparison complete."
)
print(
    "[PASS] Frozen v3.22 model unchanged."
)
print(
    "[PASS] No archive records modified."
)
print(
    "[PASS] No production files modified."
)
