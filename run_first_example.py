#!/usr/bin/env python

import numpy as np


def update(prior_mean, estimate_var, measurement_mean, measurement_var):
    '''This function finds the updated mean and variance of the state estimation using prior and measurement information, following the rule of the Bayes on Gaussians.'''
    new_mean = (prior_mean * measurement_var + measurement_mean *
                estimate_var) / (estimate_var + measurement_var)

    new_var = 1 / (1 / estimate_var + 1 / measurement_var)

    return [new_mean, new_var]


if __name__ == "__main__":
    print("Running example")

    steps = 100
    true_value = 32

    measurement_sigma = 3
    measurement_var = np.square(measurement_sigma)
    measurements = np.random.normal(true_value, measurement_sigma, steps)

    estimate = 50
    estimate_sigma = 20
    estimate_var = np.square(estimate_sigma)

    for n in range(steps):
        estimate, estimate_var = update(
            estimate, estimate_var, measurements[n], measurement_var)
        print(
            f'Run n: {n}, Measurement: {measurements[n]:3.2f}, estimate: {estimate:3.2f}, estimate var: {estimate_var:3.2f}, abs error: {np.abs(true_value - estimate):3.2f} ')

# TODO: Naive avg:{np.mean(measurements[0:n]):3.2f} print out?
