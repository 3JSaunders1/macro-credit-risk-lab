import numpy as np


def compare_irfs(irf_a, irf_b):
    return irf_a.irfs - irf_b.irfs


def compare_forecasts(run_a, run_b):
    return {
        "pd_diff": run_a["predicted_pd"] - run_b["predicted_pd"],
        "u_diff": run_a["forecast"]["unemployment"] - run_b["forecast"]["unemployment"],
        "pi_diff": run_a["forecast"]["inflation"] - run_b["forecast"]["inflation"],
    }


def compare_feved(fevd_a, fevd_b):
    return fevd_a.decomp - fevd_b.decomp