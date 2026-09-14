Predict Edge Scanner v3

What changed in v3
- API rate-limit visibility using Crypto.com response headers.
- Expiration countdowns.
- Automatic handoff from Scanner to Analyzer.
- Break-even probability after estimated fees.
- Modeled YES/NO edge in probability points.
- Optional capped fractional-Kelly position sizing.
- Scenario sensitivity table for +/-0.50% index moves and +/-25% volatility.
- Journal statistics (entries, wins/closed, win rate, recorded P/L).
- Improved watchlist countdown.
- Better filter controls and maximum events-per-scan control.
- 10-point bands retained exactly as requested.

Important
- Read-only. No trading, no login credentials, no private keys.
- Crypto.com Predictions public API is anonymous/public but rate limited.
- Nadex currently identifies the underlying as the Nadex BTC Index / Nadex ETH Index for supported crypto event contracts.
- This app does not claim to have a direct Nadex real-time index API feed. Enter or integrate a verified benchmark before relying on the independent estimate.
- The model is a simplified volatility model, not a guarantee.

iPhone installation
1. Host all files on an HTTPS static host.
2. Open the hosted URL in Safari.
3. Share -> Add to Home Screen.
4. Launch PredictScanner.

Files
- index.html
- manifest.webmanifest
- sw.js
- icon.svg
