"""
main.py
-------
CLI entry point
"""

from services.pipeline_service import run_fresh_pipeline


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)


def main():
    result = run_fresh_pipeline()

    # -------------------------
    # FORECAST
    # -------------------------
    print_section("Macro Forecast")

    fc = result["forecast"]
    ci = result["forecast_ci"]

    print(f"Unemployment : {fc['unemployment']:.2f}%")
    print(f"  CI         : [{ci['unemployment']['lower']:.2f}, {ci['unemployment']['upper']:.2f}]")

    print(f"Inflation    : {fc['inflation']:.2f}%")
    print(f"  CI         : [{ci['inflation']['lower']:.2f}, {ci['inflation']['upper']:.2f}]")

    # -------------------------
    # CREDIT
    # -------------------------
    print_section("Credit Risk")

    print(f"PD      : {result['predicted_pd']:.4%}")
    print(f"Rating  : {result['rating']}")

    # -------------------------
    # SENSITIVITY
    # -------------------------
    print_section("Sensitivity")

    sens = result["sensitivity"]

    print(f"+1pp U → ΔPD : {sens['unemployment_shock_delta']:+.4%}")
    print(f"+1pp π → ΔPD : {sens['inflation_shock_delta']:+.4%}")

    # -------------------------
    # STRESS
    # -------------------------
    print_section("Stress Test")

    print(result["stress_test"].round(2).to_string(index=False))

    # -------------------------
    # DIAGNOSTICS (SAFE)
    # -------------------------
    print_section("Diagnostics")

    print(f"VAR Lags Used : {result.get('var_lags_used')}")
    print(f"VAR Stable    : {result.get('var_stability')}")


if __name__ == "__main__":
    main()