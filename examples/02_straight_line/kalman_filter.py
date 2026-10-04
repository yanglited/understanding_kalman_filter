def update(prior_mean, prior_var, measurement_mean, measurement_var):
    '''This function finds the updated mean and variance of the state estimation
    using prior and measurement information, following the rule of the Bayes on
    Gaussians.'''
    new_mean = (prior_mean * measurement_var + measurement_mean * prior_var) / (prior_var + measurement_var)

    new_var = 1 / (1 / prior_var + 1 / measurement_var)

    # new mean and new var becomes the new prior mean and var when you have new measurements
    return [new_mean, new_var]


def predict(prior_mean, prior_var, motion_mean, motion_var):
    '''This function moves the state estimate forward by a known motion that has
    its own uncertainty. The state and the motion are independent Gaussians, and
    the sum of two independent Gaussians is a Gaussian whose mean is the sum of
    the means and whose variance is the sum of the variances.'''
    new_mean = prior_mean + motion_mean

    new_var = prior_var + motion_var

    return [new_mean, new_var]
