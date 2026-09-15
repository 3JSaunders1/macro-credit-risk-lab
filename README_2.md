# 📊 Macro Credit Risk Modeling System

A macro-financial risk simulation system that connects **macroeconomic variables (unemployment, inflation)** to **credit default risk** using a combination of:

- Vector Autoregression (VAR) forecasting
- Logistic Probability of Default (PD) model
- Stress testing scenarios
- REST API (FastAPI)
- Interactive dashboard (Streamlit)

The project is designed as a **lightweight, production-style econometrics + credit risk engine**.

---

# 🧠 What this project does

This system models the relationship:

> Macroeconomic conditions → Future macro forecasts → Credit default risk

It answers:

- What happens to credit risk if unemployment rises?
- How do inflation shocks affect default probability?
- What happens under recession-style stress scenarios?

---

# ⚙️ System Components

This project is composed of four core parts:

---

## 1. Data Layer

Loads macroeconomic time series:

- unemployment rate
- inflation rate
- time index (date)

Used as input for forecasting.

---

## 2. VAR Forecasting Model

The system uses a Vector Autoregression (VAR) model to:

- model joint dynamics between unemployment and inflation
- capture feedback loops between macro variables
- forecast next-step macro conditions

It also performs:

- **ADF stationarity tests**
- **VAR stability checks (root condition)**

Output:
- 1-step-ahead forecasts for unemployment and inflation
- confidence intervals for forecasts

---

## 3. Credit Risk Model (Logistic PD)

The credit model maps macro variables to probability of default:

```text
log(PD / (1 - PD)) = β₀ + β₁·unemployment + β₂·inflation