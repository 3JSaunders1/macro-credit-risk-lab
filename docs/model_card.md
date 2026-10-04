# Model Card: Macro-Driven Credit Risk Lab

## Intended use
Research and portfolio demonstration of macroeconomic forecasting, macro-to-credit
loss modeling, and model-based stress testing. Not intended for real reserving or
capital decisions.

## Components
- **Macro forecasting:** VAR, SVAR (Cholesky), SVAR (sign restrictions), BVAR
  (Minnesota prior), and local projections on quarterly unemployment and inflation
- **Credit loss model:** dynamic fractional logit on the consumer charge-off rate
  (persistence, 4-quarter change in unemployment, inflation, event indicators)
- **Scenario engine:** structural shocks propagated through the BVAR, feeding the
  loss model; 9-quarter cumulative losses

## Data
FRED, 1985 Q1 to 2026 Q2 (166 quarters): unemployment (UNRATE), CPI inflation
(CPIAUCSL, year over year), consumer charge-off rate (CORCACBS). October 2025,
missing during the government shutdown, is linearly interpolated.

## Performance
- **Loss model, one step ahead (2007 onward):** RMSE 0.27 pp vs. 0.30 for naive persistence
- **Loss model, conditional 2008 backtest (trained through 2006):** 67% of actual cumulative losses
- **Forecasts (2000 onward, excluding COVID):** VAR 18% better than a random walk for
  one-quarter unemployment; BVAR robust through COVID; no gains statistically significant
- **Interval coverage:** 86-89% at one quarter, 73-83% at four quarters (target 90%)
- **Scenarios:** Severely Adverse cumulative 9-quarter loss of 9.5%, vs. 10.7% actual in 2008-2010

## Known limitations
- The loss target is a charge-off rate (roughly PD x LGD), not a pure default probability
- Unemployment and inflation alone understate housing- and credit-driven crises;
  the Severely Adverse scenario produces about 12% less cumulative loss than 2008
- Four-quarter forecast intervals are too narrow (parameter uncertainty and breaks ignored)
- Scenarios assume no pandemic-style policy support
- The static `/score` mapping uses calibrated priors and is illustrative only

## Monitoring plan
- Re-estimate quarterly as new FRED data arrives
- Track one-step loss errors and interval coverage; review if coverage falls below 80%
- Compare scenario losses with historical episodes at each update
