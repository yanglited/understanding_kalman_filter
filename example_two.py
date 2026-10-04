#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "plotly"]
# ///
"""Example two: track a value moving in a straight line at a known speed.

Each step is now predict (move the estimate by the known motion, grow its
uncertainty) followed by update (fold in the noisy measurement).

Run:   uv run example_two.py              (opens an interactive plot)
       uv run example_two.py --html docs/example_two.html
"""
import argparse

import numpy as np

import kalman_filter


def run(steps=100, start=0.0, velocity=2.0, measurement_sigma=15.0, motion_sigma=1.0,
        estimate=30.0, estimate_sigma=30.0, seed=0):
    rng = np.random.default_rng(None if seed < 0 else seed)
    truth = start + velocity * np.arange(steps)
    measurements = truth + rng.normal(0, measurement_sigma, steps)
    measurement_var = measurement_sigma ** 2
    motion_var = motion_sigma ** 2
    estimate_var = estimate_sigma ** 2

    estimates, variances = [], []
    for n in range(steps):
        if n > 0:
            estimate, estimate_var = kalman_filter.predict(estimate, estimate_var, velocity, motion_var)
        estimate, estimate_var = kalman_filter.update(estimate, estimate_var, measurements[n], measurement_var)
        estimates.append(estimate)
        variances.append(estimate_var)
        print(f'Run n: {n}, Measurement: {measurements[n]:3.2f}, estimate: {estimate:3.2f}, '
              f'estimate var: {estimate_var:3.2f}, abs error: {np.abs(truth[n] - estimate):3.2f} ')
    return truth, measurements, np.array(estimates), np.array(variances)


def plot(truth, measurements, estimates, variances):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = np.arange(1, len(measurements) + 1)
    naive_mean = np.cumsum(measurements) / n
    std = np.sqrt(variances)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08,
                        subplot_titles=("Estimate vs. measurements", "Estimate uncertainty (std-dev)"))
    fig.add_trace(go.Scatter(x=np.r_[n, n[::-1]], y=np.r_[estimates + std, (estimates - std)[::-1]],
                             fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                             name="estimate ± 1σ", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=measurements, mode="markers", name="measurements",
                             marker=dict(color="#9a9a94", size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=truth, mode="lines", name="true position",
                             line=dict(color="#52514e", dash="dash", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=naive_mean, mode="lines", name="running average (no predict step)",
                             line=dict(color="#eb6834", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=estimates, mode="lines", name="Kalman estimate",
                             line=dict(color="#2a78d6", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=std, mode="lines", name="estimate std-dev",
                             line=dict(color="#2a78d6", width=2)), row=2, col=1)
    fig.update_layout(title="Kalman filter, example two: straight-line motion at a known speed",
                      template="plotly_white", hovermode="x unified", height=650,
                      legend=dict(orientation="h", y=-0.1))
    fig.update_xaxes(title_text="time step", row=2, col=1)
    return fig


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
    p.add_argument("--html", metavar="PATH", help="write the interactive plot here instead of opening a window")
    a = p.parse_args()

    print("Running example two")
    t, m, est, var = run(a.steps, a.start, a.velocity, a.measurement_sigma, a.motion_sigma,
                         a.estimate, a.estimate_sigma, a.seed)
    print(f"final estimate {est[-1]:.2f} ± {np.sqrt(var[-1]):.2f}  (true {t[-1]:.2f}, "
          f"running average {m.mean():.2f})")

    fig = plot(t, m, est, var)
    if a.html:
        fig.write_html(a.html, include_plotlyjs="cdn")
        print("wrote", a.html)
    else:
        fig.show()


if __name__ == "__main__":
    main()
