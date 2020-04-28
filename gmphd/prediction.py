def predict(gaussian_mixture, dynamic_model, process_noise, prob_survival):
    for component in gaussian_mixture:
        component.mean = dynamic_model.dot(component.mean)
        component.covariance = dynamic_model.dot(component.covariance).dot(dynamic_model.transpose()) + process_noise
        component.weight = component.weight * prob_survival

    return gaussian_mixture
