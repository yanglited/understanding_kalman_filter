# understanding_kalman_filter
Experiments to understand Kalman filter


## How to use the docker experiment environment:
```bash
$ cd path/to/project/base/dir
$ cd docker
$ docker compose build --progress=plain
$ docker compose run kalman-filter-experiments
```

Once you are inside the docker container, do:
```bash
(base) root@hostname:~# conda activate test_env
(test_env) root@hostname:~# ./run_first_example.py 
```

## First example:

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

The PDF of $Y$ is unknown but it's value when $Y=y_0$ can be found when normalizing the conditional PDF of $f_{X|Y}(x,y_0)$.

We know that the multiplication of two Gaussian PDFs is still Gaussian, but to really understand it let's derive it here.

$$
f_{Y|X}(y,x) f_{X}(x)
= \frac{1}{\sqrt{2 \pi \sigma_{Z}^2}} \exp\left(-\frac{(y-x)^2}{2\sigma_Z^2}\right) \frac{1}{\sqrt{2 \pi \sigma_p^2}} \exp\left(-\frac{(x-\mu_p)^2}{2\sigma_p^2}\right)
$$

$$
= \frac{1}{\sqrt{2 \pi \sigma_{Z}^2}} \frac{1}{\sqrt{2 \pi \sigma_p^2}} \exp\left(-\frac{\sigma_p^2(x-y)^2 + \sigma_Z^2(x-\mu_p)^2}{2\sigma_Z^2\sigma_p^2}\right)
$$

$$
= \frac{1}{\sqrt{2 \pi}}\frac{1}{\sqrt{2 \pi \sigma_Z^2 \sigma_p^2}} \exp\left(-\frac{\sigma_p^2(x^2- 2xy + y^2) + \sigma_Z^2(x^2- 2x\mu_p + \mu_p^2)}{2\sigma_Z^2\sigma_p^2}\right)
$$


$$
= \frac{1}{\sqrt{2 \pi}}\frac{1}{\sqrt{2 \pi \sigma_Z^2 \sigma_p^2}} \exp\left(-\frac{(\sigma_p^2 + \sigma_Z^2)x^2 -2(y\sigma_p^2 + \mu_p\sigma_Z^2)x+ \sigma_p^2y^2 + \sigma_Z^2\mu_p^2 }{2\sigma_Z^2\sigma_p^2}\right)
$$

$$
= \frac{1}{\sqrt{2 \pi}}\frac{1}{\sqrt{2 \pi \sigma_Z^2 \sigma_p^2}} \exp\left(-\frac{x^2 -2\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)x}{\sigma_p^2 + \sigma_Z^2} + \frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2}}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right)
$$

$$
= \frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}}\frac{1}{\sqrt{2 \pi \frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}} \exp\left(-\frac{x^2 -2\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)x}{\sigma_p^2 + \sigma_Z^2} + \frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2}}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right)
$$

$$
= \frac{1}{\sqrt{2 \pi \frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}} \exp\left(-\frac{x^2 -2\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)x}{\sigma_p^2 + \sigma_Z^2} + \frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2}}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right) \frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}}
$$

$$
= \frac{1}{\sqrt{2 \pi \frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}} \exp\left(-\frac{x^2 -2\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)x}{\sigma_p^2 + \sigma_Z^2} + \left(\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2 + \frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2} - \left(\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right) \frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}}
$$

$$
= \frac{1}{\sqrt{2 \pi \frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}} \exp\left(-\frac{x^2 -2\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)x}{\sigma_p^2 + \sigma_Z^2} + \left(\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right) \exp\left(-\frac{\frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2} - \left(\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right) \frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}}
$$

$$
= \left[\frac{1}{\sqrt{2 \pi \frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}} \exp\left(-\frac{\left(x-\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right)\right] \left[\frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}} \exp\left(-\frac{\frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2} - \left(\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right)\right]
$$

The above first part is the Gaussian pdf with mean $\mu_{new} = \frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}$, and variance $\sigma_{new}^2 = \frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}$. The second part of the above is not a function of $x$ and it can be evaluated to a constant when $y=y_0$. Due to normalization, we know

$$
f_Y(y) = \frac{1}{\sqrt{2 \pi(\sigma_p^2 + \sigma_Z^2)}} \exp\left(-\frac{\frac{\sigma_p^2y^2 + \sigma_Z^2\mu_p^2}{\sigma_p^2 + \sigma_Z^2} - \left(\frac{(y\sigma_p^2 + \mu_p\sigma_Z^2)}{\sigma_p^2 + \sigma_Z^2}\right)^2}{2\frac{\sigma_Z^2\sigma_p^2}{\sigma_p^2 + \sigma_Z^2}}\right),
$$

such that

$$
f_{X|Y}(x,y) = \frac{f_{X,Y}(x,y)}{f_Y(y)} = \frac{f_{Y|X}(y,x) f_{X}(x)}{f_Y(y)} = \frac{1}{\sqrt{2 \pi \sigma_{new}^2}} \exp\left(-\frac{\left(x-\mu_{new}\right)^2}{2\sigma_{new}^2}\right).
$$

References:
1. https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html
2. https://medium.com/analytics-vidhya/kalman-filters-a-step-by-step-implementation-guide-in-python-91e7e123b968
3. https://github.com/Garima13a/Kalman-Filters
4. https://math.stackexchange.com/questions/1112866/product-of-two-gaussian-pdfs-is-a-gaussian-pdf-but-product-of-two-gaussian-vari
