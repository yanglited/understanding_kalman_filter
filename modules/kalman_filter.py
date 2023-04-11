def update(prior_mean, estimate_var, measurement_mean, measurement_var):
    '''This function finds the updated mean and variance of the state estimation 
    using prior and measurement information, following the rule of the Bayes on 
    Gaussians.'''
    new_mean = (prior_mean * measurement_var + measurement_mean *
                estimate_var) / (estimate_var + measurement_var)

    new_var = 1 / (1 / estimate_var + 1 / measurement_var)

    return [new_mean, new_var]