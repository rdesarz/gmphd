from gmphd.update import update_intensity
from gmphd.prediction import predict_intensity
from gmphd.postprocessing import prune, merging


class GaussianMixturePhdFilter:
    def __init__(self, dynamic_model, process_noise, prob_survival, birth_intensity,
                 measurement_model, measurement_noise, prob_detection, clutter_model, pruning_threshold,
                 merging_threshold):
        self.dynamic_model = dynamic_model
        self.process_noise = process_noise
        self.prob_survival = prob_survival
        self.birth_intensity = birth_intensity
        self.measurement_model = measurement_model
        self.measurement_noise = measurement_noise
        self.prob_detection = prob_detection
        self.clutter_model = clutter_model
        self.current_intensity = birth_intensity
        self.pruning_threshold = pruning_threshold
        self.merging_threshold = merging_threshold


def predict(self):
    predict_intensity(self.current_intensity, self.dynamic_model, self.process_noise,
                      self.prob_survival,
                      self.birth_intensity)
    prune(self.current_intensity, self.pruning_threshold)
    merging(self.current_intensity, self.merging_threshold)


def update(self, measurements):
    update_intensity(self.current_intensity, measurements, self.measurement_model, self.measurement_noise,
                     self.prob_detection,
                     self.clutter_model)


def get_intensity(self):
    return self.current_intensity
