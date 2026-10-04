# Gaussians: product and sum

The two facts every example in this repo is built on. The **product** of two Gaussians gives the
Kalman *update* step; the **sum** of two independent Gaussians gives the *predict* step.
Used by [example one](../examples/01_constant_value/) (update) and
[example two](../examples/02_straight_line/) (predict and update).

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

the convolution of the PDFs of $X$ and $Z$, which is the PDF of the sum $Y = X + Z$ because $X$
and $Z$ are independent. We did not
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
noise of variance $\sigma_w^2$, independent of the current estimate, then

$$
\mu \leftarrow \mu + v, \qquad \sigma^2 \leftarrow \sigma^2 + \sigma_w^2 .
$$

Together with the update rule from the previous section, this is the complete one-dimensional
Kalman filter: **predict** (means add, variances add), then **update** (the posterior from Bayes'
rule). [Example two](../examples/02_straight_line/) puts the two steps together.
