#!/usr/bin/env python

import numpy as np

# fmt: off
from modules import kalman_filter

# fmt: on


if __name__ == "__main__":
    print("Running example one")

    steps = 2000 # number of steps to run 
    true_value = 32 # underlying true value but unknown

    measurement_sigma = 10 # measurement noise standard deviation
    measurement_var = np.square(measurement_sigma)
    measurements = np.random.normal(true_value, measurement_sigma, steps)

    estimate = 50 # initial estimate
    prior_sigma = 20
    prior_var = np.square(prior_sigma)

    for n in range(steps):
        estimate, prior_var = kalman_filter.update(
            estimate, prior_var, measurements[n], measurement_var
        )
        avgy = np.mean(measurements[0:n+1])
        print(
            f"n: {n}, "
            f"Y: {measurements[n]:3.2f}, "
            f"avg(Y): {avgy:3.2f}, "
            f"E{{X|Y}}: {estimate:3.2f}, "
            f"prior var: {prior_var:3.2f}, "
            f"abs error: {np.abs(true_value - estimate):3.2f} "
        )

# TODO: format code
