def update(prior_mean, prior_var, measurement_mean, measurement_var):
    '''This function finds the updated mean and variance of the state estimation
    using prior and measurement information, following the rule of the Bayes on
    Gaussians.'''
    new_mean = (prior_mean * measurement_var + measurement_mean * prior_var) / (prior_var + measurement_var)

    new_var = 1 / (1 / prior_var + 1 / measurement_var)

    # new mean and new var becomes the new prior mean and var when you have new measurements
    return [new_mean, new_var]
