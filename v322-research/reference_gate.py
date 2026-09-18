#!/usr/bin/env python3

import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone


# ============================================================
# v3.22 INDEPENDENT ETH REFERENCE GATE
#
# MODEL INPUT:
#   Crypto.com ETH_USD remains the frozen model reference.
#
# INDEPENDENT VALIDATION:
#   Coinbase + Kraken + Gemini median consensus.
#
# This module DOES NOT substitute consensus into the model.
# ============================================================

TIMEOUT = 5

SOURCE_DISAGREEMENT_LIMIT = 0.10
CRYPTO_CONSENSUS_LIMIT = 0.10


def now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


def fetch_json(url):
    started = time.perf_counter()

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "Predict-Edge-v3.22-Reference-Gate"
        },
    )

    with urllib.request.urlopen(
        req,
        timeout=TIMEOUT
    ) as response:

        body = json.loads(
            response.read().decode("utf-8")
        )

    latency_ms = (
        time.perf_counter() - started
    ) * 1000

    return body, latency_ms


def fetch_coinbase():
    body, latency = fetch_json(
        "https://api.coinbase.com/"
        "v2/prices/ETH-USD/spot"
    )

    price = float(
        body["data"]["amount"]
    )

    return {
        "source": "coinbase",
        "price": price,
        "latencyMs": latency,
    }


def fetch_kraken():
    body, latency = fetch_json(
        "https://api.kraken.com/"
        "0/public/Ticker?pair=ETHUSD"
    )

    if body.get("error"):
        raise RuntimeError(
            f"Kraken error: {body['error']}"
        )

    ticker = next(
        iter(body["result"].values())
    )

    bid = float(ticker["b"][0])
    ask = float(ticker["a"][0])

    return {
        "source": "kraken",
        "price": (bid + ask) / 2.0,
        "bid": bid,
        "ask": ask,
        "latencyMs": latency,
    }


def fetch_gemini():
    body, latency = fetch_json(
        "https://api.gemini.com/"
        "v1/pubticker/ethusd"
    )

    bid = float(body["bid"])
    ask = float(body["ask"])

    return {
        "source": "gemini",
        "price": (bid + ask) / 2.0,
        "bid": bid,
        "ask": ask,
        "latencyMs": latency,
    }


def fetch_crypto():
    body, latency = fetch_json(
        "https://api.crypto.com/"
        "exchange/v1/public/get-tickers"
        "?instrument_name=ETH_USD"
    )

    if body.get("code") != 0:
        raise RuntimeError(
            f"Crypto.com code: {body.get('code')}"
        )

    rows = (
        body.get("result", {})
        .get("data", [])
    )

    if not rows:
        raise RuntimeError(
            "Crypto.com returned no ticker."
        )

    row = rows[0]

    if row.get("i") != "ETH_USD":
        raise RuntimeError(
            "Unexpected Crypto.com instrument."
        )

    # EXACT production semantics:
    # d.a / row["a"] is the current model reference.
    price = float(row["a"])

    return {
        "source": "crypto.com",
        "price": price,
        "bid":
            float(row["b"])
            if row.get("b") is not None
            else None,
        "ask":
            float(row["k"])
            if row.get("k") is not None
            else None,
        "apiTimestamp": row.get("t"),
        "latencyMs": latency,
    }


def pct_difference(a, b):
    if not a or not b:
        return None

    return (
        abs(a - b)
        / ((a + b) / 2.0)
    ) * 100.0


def build_reference_gate():

    independent = []
    failures = {}

    fetchers = (
        ("coinbase", fetch_coinbase),
        ("kraken", fetch_kraken),
        ("gemini", fetch_gemini),
    )

    for name, fn in fetchers:
        try:
            independent.append(
                fn()
            )
        except Exception as exc:
            failures[name] = str(exc)

    crypto = None

    try:
        crypto = fetch_crypto()
    except Exception as exc:
        failures["crypto.com"] = str(exc)

    prices = [
        r["price"]
        for r in independent
        if r.get("price") is not None
    ]

    source_count = len(prices)

    consensus = (
        statistics.median(prices)
        if prices
        else None
    )

    source_disagreement = None

    if len(prices) >= 2:
        lo = min(prices)
        hi = max(prices)

        source_disagreement = (
            ((hi - lo) / consensus) * 100
            if consensus
            else None
        )

    crypto_deviation = None

    if (
        crypto is not None
        and consensus is not None
    ):
        crypto_deviation = pct_difference(
            crypto["price"],
            consensus,
        )

    # ----------------------------------------
    # Independent-source availability
    # ----------------------------------------

    if source_count < 2:
        independent_status = "UNAVAILABLE"

    elif source_count == 2:
        independent_status = "DEGRADED"

    elif (
        source_disagreement is not None
        and
        source_disagreement >
        SOURCE_DISAGREEMENT_LIMIT
    ):
        independent_status = "DIVERGENT"

    else:
        independent_status = "HEALTHY"

    # ----------------------------------------
    # Final reference status
    # ----------------------------------------

    if independent_status == "UNAVAILABLE":
        status = "UNAVAILABLE"

    elif independent_status == "DIVERGENT":
        status = "DIVERGENT"

    elif crypto is None:
        status = "CRYPTO_UNAVAILABLE"

    elif (
        crypto_deviation is not None
        and
        crypto_deviation >
        CRYPTO_CONSENSUS_LIMIT
    ):
        status = "REFERENCE_DIVERGENCE"

    elif independent_status == "DEGRADED":
        status = "DEGRADED"

    else:
        status = "HEALTHY"

    return {
        "schemaVersion":
            "v3.22-reference-gate-1",

        "capturedAt":
            now_iso(),

        "status":
            status,

        "independentStatus":
            independent_status,

        "sourceCount":
            source_count,

        "consensusPrice":
            consensus,

        "sourceDisagreementPercent":
            source_disagreement,

        "cryptoPrice":
            (
                crypto["price"]
                if crypto
                else None
            ),

        "cryptoConsensusDeviationPercent":
            crypto_deviation,

        "limits": {
            "sourceDisagreementPercent":
                SOURCE_DISAGREEMENT_LIMIT,
            "cryptoConsensusDeviationPercent":
                CRYPTO_CONSENSUS_LIMIT,
        },

        "independentSources":
            independent,

        "crypto":
            crypto,

        "failures":
            failures,

        "modelReferencePolicy":
            "CRYPTO_COM_UNCHANGED",

        "consensusPolicy":
            "INDEPENDENT_VALIDATION_ONLY",
    }


if __name__ == "__main__":

    result = build_reference_gate()

    print(json.dumps(
        result,
        indent=2,
        sort_keys=True,
    ))
