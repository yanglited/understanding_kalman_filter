# Example two: a value moving in a straight line

[code](example.py) · [filter](../../kf/filter.py) · [interactive plot](https://yanglited.github.io/understanding_kalman_filter/examples/02_straight_line/plot.html) · [foundation math](../../docs/gaussians.md)

$X$ now moves in a straight line: $X_n = X_{n-1} + v + W_n$, where the speed $v$ is known and
$W_n \sim \mathcal{N}(0, \sigma_w^2)$ is a small random disturbance to the motion. We observe
$Y_n = X_n + Z_n$ as before.

## The math this example uses

One new idea: the **predict** step. Moving the state by a known amount with its own Gaussian
uncertainty is the sum of two independent Gaussians, so the means add and the variances add
([derivation](../../docs/gaussians.md#the-sum-of-two-gaussians-the-predict-step)). Each step is a
predict followed by the update from example one:

$$
\text{predict:}\quad \mu \leftarrow \mu + v,\quad \sigma^2 \leftarrow \sigma^2 + \sigma_w^2
\qquad\qquad
\text{update:}\quad \mu \leftarrow \frac{\mu \sigma_Z^2 + y_n \sigma^2}{\sigma^2 + \sigma_Z^2},\quad
\sigma^2 \leftarrow \frac{\sigma^2 \sigma_Z^2}{\sigma^2 + \sigma_Z^2}
$$

In code these are the `predict` and `update` functions in [`kf/filter.py`](../../kf/filter.py).

## Run it

```bash
uv run examples/02_straight_line/example.py             # from the repo root
uv run examples/02_straight_line/example.py --motion-sigma 4 --measurement-sigma 5 --seed -1
uv run examples/02_straight_line/example.py --html plot.html
```

![Example two: the Kalman estimate tracks the moving value while a running average lags behind](plot.png)

## What to notice

1. A plain running average has no predict step, so it lags further behind the moving value at
   every step. The filter stays on the line because it moves its estimate by $v$ before looking
   at the measurement.
2. The uncertainty no longer shrinks to zero. Every predict adds $\sigma_w^2$ and every update
   takes some away, and the two balance at a steady state $\sigma_\infty^2$ that solves

$$
\sigma_\infty^2 = \frac{(\sigma_\infty^2 + \sigma_w^2)\thinspace \sigma_Z^2}{\sigma_\infty^2 + \sigma_w^2 + \sigma_Z^2}
\quad\Longrightarrow\quad
\sigma_\infty^4 + \sigma_w^2\thinspace \sigma_\infty^2 - \sigma_w^2\thinspace \sigma_Z^2 = 0 .
$$

With the defaults $\sigma_w = 1$ and $\sigma_Z = 15$ this gives $\sigma_\infty^2 \approx 14.5$,
i.e. $\sigma_\infty \approx 3.8$, which is the floor the lower panel settles on.

Next step: when the speed is *not* known, the state becomes the pair (position, velocity) and the
same two steps are written with matrices. That is the Kalman filter as it is usually presented.
