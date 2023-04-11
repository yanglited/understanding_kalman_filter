#!/usr/bin/env python

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import numpy as np
import modules.kalman_filter as kalman


if __name__ == "__main__":
    print("Running example")

    steps = 10
    true_value = 32

    measurement_sigma = 3
    measurement_var = np.square(measurement_sigma)
    measurements = np.random.normal(true_value, measurement_sigma, steps)

    estimate = 50
    estimate_sigma = 20
    estimate_var = np.square(estimate_sigma)

    for n in range(steps):
        estimate, estimate_var = kalman.update(
            estimate, estimate_var, measurements[n], measurement_var)
        print(
            f'Run n: {n}, Measurement: {measurements[n]:3.2f}, estimate: {estimate:3.2f}, estimate var: {estimate_var:3.2f}, abs error: {np.abs(true_value - estimate):3.2f} ')

# TODO: Naive avg:{np.mean(measurements[0:n]):3.2f} print out?
# TODO: format code
