def prune(posterior_intensity, truncation_threshold):
    posterior_intensity = [gaussian for gaussian in posterior_intensity if gaussian.weight >= truncation_threshold]
    return posterior_intensity
