#!/usr/bin/env -S uv run
"""Example two: track a value moving in a straight line at a known speed.

Each step is now predict (move the estimate by the known motion, grow its
uncertainty) followed by update (fold in the noisy measurement).

Run:   uv run examples/02_straight_line/example.py              (opens an interactive plot)
       uv run examples/02_straight_line/example.py --html examples/02_straight_line/plot.html --beliefs-html examples/02_straight_line/beliefs.html
"""
import argparse

import numpy as np

from kf.filter import predict, update
from kf.plot import belief_animation, show_or_save, tracking_figure


def run(steps=100, start=0.0, velocity=2.0, measurement_sigma=15.0, motion_sigma=1.0,
        estimate=30.0, estimate_sigma=30.0, seed=0):
    rng = np.random.default_rng(None if seed < 0 else seed)
    truth = start + velocity * np.arange(steps)
    measurements = truth + rng.normal(0, measurement_sigma, steps)
    measurement_var = measurement_sigma ** 2
    motion_var = motion_sigma ** 2
    estimate_var = estimate_sigma ** 2

    estimates, variances, prior_means, prior_vars = [], [], [], []
    for n in range(steps):
        if n > 0:
            estimate, estimate_var = predict(estimate, estimate_var, velocity, motion_var)
        prior_means.append(estimate)
        prior_vars.append(estimate_var)
        estimate, estimate_var = update(estimate, estimate_var, measurements[n], measurement_var)
        estimates.append(estimate)
        variances.append(estimate_var)
        print(f'Run n: {n}, Measurement: {measurements[n]:3.2f}, estimate: {estimate:3.2f}, '
              f'estimate var: {estimate_var:3.2f}, abs error: {np.abs(truth[n] - estimate):3.2f} ')
    return dict(truth=truth, measurements=measurements, measurement_var=measurement_var,
                prior_means=np.array(prior_means), prior_vars=np.array(prior_vars),
                post_means=np.array(estimates), post_vars=np.array(variances))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--start", type=float, default=0.0, help="true position at step 0")
    p.add_argument("--velocity", type=float, default=2.0, help="known speed, units per step")
    p.add_argument("--measurement-sigma", type=float, default=15.0)
    p.add_argument("--motion-sigma", type=float, default=1.0, help="std-dev of the per-step motion (process noise)")
    p.add_argument("--estimate", type=float, default=30.0, help="initial guess")
    p.add_argument("--estimate-sigma", type=float, default=30.0, help="std-dev of the initial guess")
    p.add_argument("--seed", type=int, default=0, help="-1 for a fresh random draw")
    p.add_argument("--html", metavar="PATH", help="write the tracking plot here instead of opening a window")
    p.add_argument("--beliefs-html", metavar="PATH", help="write the step-by-step belief animation here")
    a = p.parse_args()

    print("Running example two")
    r = run(a.steps, a.start, a.velocity, a.measurement_sigma, a.motion_sigma, a.estimate, a.estimate_sigma, a.seed)
    print(f"final estimate {r['post_means'][-1]:.2f} ± {np.sqrt(r['post_vars'][-1]):.2f}  (true {r['truth'][-1]:.2f}, "
          f"running average {r['measurements'].mean():.2f})")
    title = "Kalman filter, example two: straight-line motion at a known speed"
    show_or_save(tracking_figure(r["truth"], r["measurements"], r["post_means"], r["post_vars"], title), a.html)
    show_or_save(belief_animation(**r, title="Example two: predict widens the belief, the measurement narrows it",
                                  show_previous=True), a.beliefs_html)


if __name__ == "__main__":
    main()
