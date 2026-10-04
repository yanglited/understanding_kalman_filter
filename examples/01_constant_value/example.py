#!/usr/bin/env -S uv run
"""Example one: estimate a constant hidden value from noisy measurements.

Run:   uv run examples/01_constant_value/example.py              (opens an interactive plot)
       uv run examples/01_constant_value/example.py --html examples/01_constant_value/plot.html
"""
import argparse

import numpy as np

from kf.filter import update
from kf.plot import show_or_save, tracking_figure


def run(steps=150, true_value=32.0, measurement_sigma=10.0, estimate=50.0, estimate_sigma=20.0, seed=0):
    rng = np.random.default_rng(None if seed < 0 else seed)
    truth = np.full(steps, true_value)
    measurements = truth + rng.normal(0, measurement_sigma, steps)
    measurement_var = measurement_sigma ** 2
    estimate_var = estimate_sigma ** 2

    estimates, variances = [], []
    for n in range(steps):
        estimate, estimate_var = update(estimate, estimate_var, measurements[n], measurement_var)
        estimates.append(estimate)
        variances.append(estimate_var)
        print(f'Run n: {n}, Measurement: {measurements[n]:3.2f}, estimate: {estimate:3.2f}, '
              f'estimate var: {estimate_var:3.2f}, abs error: {np.abs(true_value - estimate):3.2f} ')
    return truth, measurements, np.array(estimates), np.array(variances)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--true-value", type=float, default=32.0)
    p.add_argument("--measurement-sigma", type=float, default=10.0)
    p.add_argument("--estimate", type=float, default=50.0, help="initial guess")
    p.add_argument("--estimate-sigma", type=float, default=20.0, help="std-dev of the initial guess")
    p.add_argument("--seed", type=int, default=0, help="-1 for a fresh random draw")
    p.add_argument("--html", metavar="PATH", help="write the interactive plot here instead of opening a window")
    a = p.parse_args()

    print("Running example one")
    t, m, est, var = run(a.steps, a.true_value, a.measurement_sigma, a.estimate, a.estimate_sigma, a.seed)
    print(f"final estimate {est[-1]:.2f} ± {np.sqrt(var[-1]):.2f}  (true {a.true_value}, "
          f"running average {m.mean():.2f})")
    show_or_save(tracking_figure(t, m, est, var, "Kalman filter, example one: a constant value seen through noise"), a.html)


if __name__ == "__main__":
    main()
