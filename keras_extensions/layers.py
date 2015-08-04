import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from keras.layers.core import Layer

class SampleBernoulli(Layer):
    """
    Layer which samples a Bernoulli distribution whose statistics (means, 'p') are given 
    as inputs to the layer.

    :param mode:    'maximum_likelihood' for maximum likelihood sample, 
                    'random' for random sample.
    """
    def __init__(self, mode='maximum_likelihood'):
        super(SampleBernoulli, self).__init__()
        self.mode = mode
        if self.mode == 'random':
            self.srng = RandomStreams(seed=np.random.randint(10e6))

    def get_output(self, train=False):
        p = self.get_input(train)
        if self.mode == 'maximum_likelihood':
            # draw maximum likelihood sample from Bernoulli distribution
            #    x* = argmax_x p(x) = 1         if p(x=1) >= 0.5
            #                         0         otherwise
            return T.round(p, mode='half_away_from_zero')
        elif self.mode == 'random':
            # draw random sample from Bernoulli distribution
            #    x* = x ~ p(x) = 1              if p(x=1) > uniform(0, 1)
            #                    0              otherwise
            return self.srng.binomial(size=p.shape, n=1, p=p, dtype=theano.config.floatX)
        else:
            raise NotImplementedError('Unknown sample mode!')
