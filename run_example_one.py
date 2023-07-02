#!/usr/bin/env python

import numpy as np

# fmt: off
from modules import kalman_filter
# fmt: on


if __name__ == "__main__":
    print("Running example one")

    steps = 150
    true_value = 32

    measurement_sigma = 10
    measurement_var = np.square(measurement_sigma)
    measurements = np.random.normal(true_value, measurement_sigma, steps)

    estimate = 50
    estimate_sigma = 20
    estimate_var = np.square(estimate_sigma)

    for n in range(steps):
        estimate, estimate_var = kalman_filter.update(estimate, estimate_var, measurements[n], measurement_var)
        print(f'Run n: {n}, Measurement: {measurements[n]:3.2f}, estimate: {estimate:3.2f}, estimate var: {estimate_var:3.2f}, abs error: {np.abs(true_value - estimate):3.2f} ')

# TODO: Naive avg:{np.mean(measurements[0:n]):3.2f} print out?
# TODO: format code
