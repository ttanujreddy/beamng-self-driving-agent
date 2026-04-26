# Dummy net simulates a finished neural network's calls

import math

def predict(data):
    output = [0.0, 0.5, 0.0]
    steer_normalized = data[1] / (math.pi / 2)
    output[0] = -(steer_normalized * 1.2)
    return output