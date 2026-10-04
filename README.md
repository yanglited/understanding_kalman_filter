# Understanding the Kalman Filter

[![Live demo](https://img.shields.io/badge/demo-interactive%20plots-2a78d6)](https://yanglited.github.io/understanding_kalman_filter/examples/02_straight_line/plot.html)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

A step-by-step introduction to Kalman filtering. Each example adds one idea, with the math
worked out in full and a runnable NumPy implementation you can plot and experiment with.
Each example folder holds its script, its write-up and its plots. The filter itself and the
shared plotting code live in [`kf/`](kf/), so every example reads as a few lines on top of the same
two functions.

| # | Example | New idea | Read | Run | Look |
|---|---------|----------|------|-----|------|
| 1 | [Constant value seen through noise](examples/01_constant_value/) | the **update** step | [math](examples/01_constant_value/README.md#the-math-this-example-uses) · [code](examples/01_constant_value/example.py) | `uv run examples/01_constant_value/example.py` | [interactive plot](https://yanglited.github.io/understanding_kalman_filter/examples/01_constant_value/plot.html) |
| 2 | [Value moving in a straight line](examples/02_straight_line/) | the **predict** step | [math](examples/02_straight_line/README.md#the-math-this-example-uses) · [code](examples/02_straight_line/example.py) | `uv run examples/02_straight_line/example.py` | [interactive plot](https://yanglited.github.io/understanding_kalman_filter/examples/02_straight_line/plot.html) |

The step after that, unknown speed and the matrix form, is outlined at the end of example two.

![Example two: the Kalman estimate tracks the moving value while a running average lags behind](examples/02_straight_line/plot.png)

## Run

```bash
uv run examples/01_constant_value/example.py    # zero setup: uv fetches numpy + plotly
# or
pip install -e . && python examples/01_constant_value/example.py
```

Every script takes `--help`, `--seed -1` for a fresh random draw, and `--html PATH` to save the
interactive plot instead of opening a window.

## Foundation math

[docs/gaussians.md](docs/gaussians.md) derives the two facts everything rests on: the product of
two Gaussians is a Gaussian (the update step) and the sum of two independent Gaussians is a
Gaussian (the predict step).

## License

MIT
