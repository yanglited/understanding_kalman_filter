
## Background information on Gaussian PDFs with Bayes Rule:

We denote $Y$ the random variable of the observation of another random variable $X$. The observation can be modeled as follows.

$$
Y = X + Z,
$$

where $Z \sim \mathcal{N}(0, \sigma_{Z})$ is the observation noise. 

We denote $f_{X|Y}(x,y)$ the conditional probability density function of random variable $X$ given the observation $Y$. Using Bayes rule, we may write

$$
f_{X|Y}(x,y) = \frac{f_{X,Y}(x,y)}{f_Y(y)} = \frac{f_{Y|X}(y,x) f_{X}(x)}{f_Y(y)},
$$

where the conditional probability density function (PDF) of the observation given $X$ can be written as

$$
f_{Y|X}(y,x) = \frac{1}{\sqrt{2 \pi \sigma_Z^2}} \exp\left(-\frac{(y-x)^2}{2\sigma_Z^2}\right).
$$

The prior PDF of $X$ can be written as

$$
f_{X}(x) = \frac{1}{\sqrt{2 \pi \sigma_p^2}} \exp\left(-\frac{(x-\mu_p)^2}{2\sigma_p^2}\right).
$$

The PDF of $Y$ is unknown but its value when $Y=y_0$ can be found when normalizing the conditional PDF of $f_{X|Y}(x,y_0)$.

We know that the multiplication of two Gaussian PDFs is still Gaussian, but to really understand it let's derive it here.

$$
\begin{aligned}
f_{Y|X}(y,x) f_{X}(x) &= \frac{1}{\sqrt{2 \pi \sigma_{Z}^2}} \exp\left(-\frac{(y-x)^2}{2\sigma_Z^2}\right) \frac{1}{\sqrt{2 \pi \sigma_p^2}} \exp\left(-\frac{(x-\mu_p)^2}{2\sigma_p^2}\right) \\
&= \frac{1}{\sqrt{2 \pi \sigma_{Z}^2}} \frac{1}{\sqrt{2 \pi \sigma_p^2}} \exp\left(-\frac{\sigma_p^2(x-y)^2 + \sigma_Z^2(x-\mu_p)^2}{2\sigma_Z^2\sigma_p^2}\right) \\
&= \frac{1}{\sqrt{2 \pi}}\frac{1}{\sqrt{2 \pi \sigma_Z^2 \sigma_p^2}} \exp\left(-\frac{\sigma_p^2(x^2- 2xy + y^2) + \sigma_Z^2(x^2- 2x\mu_p + \mu_p^2)}{2\sigma_Z^2\sigma_p^2}\right) \\
&= \frac{1}{\sqrt{2 \pi}}\frac{1}{\sqrt{2 \pi \sigma_Z^2 \sigma_p^2}} \exp\left(-\frac{(\sigma_p^2 + \sigma_Z^2)x^2 -2(y\sigma_p^2 + \mu_p\sigma_Z^2)x+ \sigma_p^2y^2 + \sigma_Z^2\mu_p^2 }{2\sigma_Z^2\sigma_p^2}\right) \\
&= \frac{1}{\sqrt{2 \pi}}\frac{1}{\sqrt{2 \pi \sigma_Z^2 \sigma_p^2}} \exp\left(-\frac{x^2 -2\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)x}{\sigma_p^2 + \sigma_Z^2} + \frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2}}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right) \\
&= \frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}}\frac{1}{\sqrt{2 \pi \frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}} \exp\left(-\frac{x^2 -2\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)x}{\sigma_p^2 + \sigma_Z^2} + \frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2}}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right) \\
&= \frac{1}{\sqrt{2 \pi \frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}} \exp\left(-\frac{x^2 -2\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)x}{\sigma_p^2 + \sigma_Z^2} + \frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2}}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right) \frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}} \\
&= \frac{1}{\sqrt{2 \pi \frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}} \exp\left(-\frac{x^2 -2\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)x}{\sigma_p^2 + \sigma_Z^2} + \left(\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2 + \frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2} - \left(\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right) \frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}} \\
&= \frac{1}{\sqrt{2 \pi \frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}} \exp\left(-\frac{x^2 -2\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)x}{\sigma_p^2 + \sigma_Z^2} + \left(\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right) \exp\left(-\frac{\frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2} - \left(\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right) \frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}} \\
&= \left[\frac{1}{\sqrt{2 \pi \frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}} \exp\left(-\frac{\left(x-\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right)\right] \left[\frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}} \exp\left(-\frac{\frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2} - \left(\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right)\right]
\end{aligned}
$$

The above first part is the Gaussian pdf with mean $\mu_{new} = \frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}$, and variance $\sigma_{new}^2 = \frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}$. The second part of the above is not a function of $x$ and it can be evaluated to a constant when $y=y_0$. 

Due to normalization, we know

$$
f_Y(y) = \frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}} \exp\left(-\frac{\frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2} - \left(\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right),
$$

such that

$$
f_{X|Y}(x,y) = \frac{f_{X,Y}(x,y)}{f_Y(y)} = \frac{f_{Y|X}(y,x) f_{X}(x)}{f_Y(y)} = \frac{1}{\sqrt{2 \pi \sigma_{new}^2}} \exp\left(-\frac{\left(x-\mu_{new}\right)^2}{2\sigma_{new}^2}\right).
$$

<!-- TODO As a result, we know the update rule for the posterior PDF of X is  -->


## The sum of two Gaussians (the predict step)

Look again at the normalizing constant $f_Y(y)$ above. By construction it is

$$
f_Y(y) = \int f_{Y|X}(y,x) f_X(x) \thinspace dx = \int f_Z(y-x) f_X(x) \thinspace dx,
$$

the convolution of the PDFs of $X$ and $Z$, which is the PDF of the sum $Y = X + Z$. We did not
need to do the integral: the product in the previous section is a Gaussian in $x$ (which
integrates to one) times $f_Y(y)$, so $f_Y(y)$ is whatever was left over. Simplifying its
exponent, the two terms combine as

$$
\frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2} - \left(\frac{y\sigma_p^2 + \mu_p\sigma_Z^2}{\sigma_p^2 + \sigma_Z^2}\right)^2
= \frac{\sigma_p^2\sigma_Z^2 (y-\mu_p)^2}{(\sigma_p^2 + \sigma_Z^2)^2},
$$

so that

$$
f_Y(y) = \frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}} \exp\left(-\frac{(y-\mu_p)^2}{2(\sigma_p^2 + \sigma_Z^2)}\right).
$$

**The sum of two independent Gaussians is Gaussian: the means add and the variances add.**
This is the whole predict step. If the state moves by a known amount $v$ plus Gaussian motion
noise of variance $\sigma_w^2$, then

$$
\mu \leftarrow \mu + v, \qquad \sigma^2 \leftarrow \sigma^2 + \sigma_w^2 .
$$

Together with the update rule from the previous section, this is the complete one-dimensional
Kalman filter: **predict** (means add, variances add), then **update** (the posterior from Bayes'
rule). Example two below puts the two steps together.


## Example One:

Random variable $X$ takes one value and does not move. We keep making observations on $X$ with observation noise. Goal is to estimate the true hidden value of $X$.

Run it with `uv run example_one.py` from the repo root, or open the
[interactive plot](https://yanglited.github.io/understanding_kalman_filter/example_one.html).

![Example one](example_one.png)

The code lives in [`kalman_filter.py`](../kalman_filter.py) (the update rule) and
[`example_one.py`](../example_one.py) (simulation and plot).

References:
1. [SciPy Cookbook: Kalman filtering](https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html)
2. [Kalman filters: a step-by-step implementation guide in Python](https://medium.com/analytics-vidhya/kalman-filters-a-step-by-step-implementation-guide-in-python-91e7e123b968)
3. [Garima13a/Kalman-Filters](https://github.com/Garima13a/Kalman-Filters)
4. [Product of two Gaussian PDFs is a Gaussian PDF (Math.SE)](https://math.stackexchange.com/questions/1112866/product-of-two-gaussian-pdfs-is-a-gaussian-pdf-but-product-of-two-gaussian-vari)


## Example Two:

$X$ now moves in a straight line: $X_n = X_{n-1} + v + W_n$, where the speed $v$ is known and
$W_n \sim \mathcal{N}(0, \sigma_w^2)$ is a small random disturbance to the motion. We observe
$Y_n = X_n + Z_n$ as before. Each step is a predict followed by an update:

$$
\text{predict:}\quad \mu \leftarrow \mu + v,\quad \sigma^2 \leftarrow \sigma^2 + \sigma_w^2
\qquad\qquad
\text{update:}\quad \mu \leftarrow \frac{\mu \sigma_Z^2 + y_n \sigma^2}{\sigma^2 + \sigma_Z^2},\quad
\sigma^2 \leftarrow \frac{\sigma^2 \sigma_Z^2}{\sigma^2 + \sigma_Z^2}
$$

Run it with `uv run example_two.py`, or open the
[interactive plot](https://yanglited.github.io/understanding_kalman_filter/example_two.html).

![Example two](example_two.png)

Two things to notice:

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
