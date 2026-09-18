Predict Edge Scanner v3.22
==========================

Overview
--------
Predict Edge Scanner is a read-only research and analysis application for
crypto prediction-market contracts.

Version 3.22 adds an independently trained strike-aware probability model
alongside the market-implied probability. The model does not use market
probability as an input.

v3.22 Model
-----------
The frozen v3.22 model uses:

- Normalized strike distance
- 1-minute ETH momentum
- 5-minute ETH momentum
- 15-minute ETH momentum
- 30-minute ETH momentum

Normalized strike distance incorporates the remaining contract horizon and
15-minute realized volatility, with a 0.020% volatility floor.

The model is evaluated only within controlled T-5, T-10, T-15 and T-30
horizons.

Reference Health
----------------
Crypto.com ETH remains the frozen model reference input.

Coinbase, Kraken and Gemini are used independently to construct a reference
consensus for health checking. The independent consensus does not replace
the Crypto.com model input.

The scanner reports reference status including HEALTHY, DEGRADED,
DIVERGENT, REFERENCE_DIVERGENCE or UNAVAILABLE as applicable.

Validation
----------
The frozen model was trained only on the historical canonical dataset.

Historical expanding walk-forward validation:
- 171 forward observations across 16 targets
- v3.22 Brier score: 0.06350
- Market comparator Brier score: 0.09223

Prospective validation remains separate from training.

Current prospective target-balanced results:
- v3.22 Brier score: 0.11671
- Market comparator Brier score: 0.13157
- v3.22 log loss: 0.37660
- Market comparator log loss: 0.40876

Current evidence supports improved probability estimation versus the market
comparator. Target-balanced directional accuracy is effectively even on the
current small target sample.

These results are research observations, not guarantees of future
performance.

Selected Trade Analysis
-----------------------
For contracts inside a controlled horizon, the application displays:

- Market Probability
- v3.22 Probability
- Edge in percentage points
- ETH reference
- Contract strike
- Independent Reference Health

Edge is simply:

    v3.22 probability - market-implied probability

It is not an automatic BUY or SELL signal.

Important
---------
- Read-only.
- No automatic trading or order execution.
- No login credentials or private keys are required.
- Market probability is a comparator, not a v3.22 model feature.
- Independent ETH consensus is a validation layer, not a substitute model input.
- Historical and prospective results do not guarantee future performance.

Files
-----
- index.html
- manifest.webmanifest
- sw.js
- icon.svg
- v322-research/

Local Use
---------
From the project directory:

    python3 -m http.server 8787 --bind 0.0.0.0

Then open the scanner in a browser using the local server address.

Installation
------------
When hosted over HTTPS, the application can be installed as a web app from
supported browsers.
