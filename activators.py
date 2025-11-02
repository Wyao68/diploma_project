import numpy as np


class ReluActivator(object):
    def forward(self, weighted_input):
        #return weighted_input
        return np.maximum(0, weighted_input)

    def backward(self, output):
        derivative = np.ones_like(output)
        derivative[output <= 0] = 0
        return derivative


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1


class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)


class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output