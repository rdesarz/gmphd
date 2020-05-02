class Poisson:
    def __init__(self, surveillance_volume, clutter_density):
        self.surveillance_volume = surveillance_volume
        self.clutter_density = clutter_density

    def __call__(self):
        return self.surveillance_volume * self.clutter_density
