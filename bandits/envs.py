from __future__ import division
import numpy as np


class Environment:

    def __init__(self, n, dist='bernoulli', **kwargs):

        self.arms = n
        self.dist = dist

        if self.dist == 'bernoulli':

            if 'mean' in kwargs:
                assert len(kwargs['mean']) == n
                self.params = kwargs['mean']
            else:
                self.params = np.random.uniform(0, 1, n)

            self.best_arm = np.argmax(self.params)

        if self.dist == 'stable':

            if 'alpha' in kwargs:
                alpha_ = kwargs['alpha']
                if type(alpha_) is float:
                    self.alpha =  np.array([alpha_] * self.arms)
                elif type(alpha_) is list or type(alpha_) is np.ndarray:
                    assert len(alpha_) == self.arms and all(alpha_ > 0)\
                        and all(alpha_ <= 2.01)
                    self.alpha = alpha_
            else:
                self.alpha = np.random.uniform(0, 2, n)

            if 'beta' in kwargs:
                beta_ = kwargs['beta']
                if type(beta_) is float:
                    self.beta == np.array([beta_] * self.arms)
                elif type(beta_) is list or type(beta_) is np.ndarray:
                    assert len(beta_) == self.arms
                    self.beta = beta_
            else:
                self.beta = np.zeros((1, self.arms))

            if 'scale' in kwargs:
                scale_ = kwargs['scale']
                if type(scale_) is float:
                    self.scale == np.array([scale_] * self.arms)
                elif type(scale_) is list or type(scale_) is np.ndarray:
                    assert len(scale_) == self.arms
                    self.scale = scale_
            else:
                self.scale = np.ones((1, self.arms))

            if 'mean' in kwargs:
                assert len(kwargs['mean']) == n
                self.mean = np.array(kwargs['mean'])
            else:
                self.mean = np.random.uniform(0, 1, n)

            if all(self.alpha > 1):
                self.best_arm = np.argmax(self.mean)
            else:
                # No best arm exists since means are infinite
                self.best_arm = None

    def sample(self):

        return self.iter(0)

    def iter(self, arm):

        assert arm < self.arms

        if self.dist == 'bernoulli':

            return np.random.binomial(1, self.params[arm], 1)

        if self.dist == 'stable':
            
            _alpha = self.alpha[arm]
            _beta = self.beta[arm]

            b = np.arctan(_beta * np.tan (0.5 * np.pi *_alpha))
            s = (1 + (_beta * np.tan(0.5 * np.pi * _alpha))**2)**(0.5/_alpha)

            v = np.random.uniform(-0.5*np.pi, 0.5*np.pi)
            w = np.random.exponential(1)

            if _alpha == 1:
                z = 2/np.pi * ((0.5*np.pi + _beta*v)*np.tan(v) - 
                    _beta * np.log((np.pi*0.5*w*np.cos(v))/(0.5*np.pi + _beta*v)))
            else:
                z = s * (np.sin(_alpha*(v + b))/(np.cos(v)**(1/_alpha))) *\
                    (np.cos(v - _alpha*(v + b))/w)**((1-_alpha)/_alpha)
 
            return float(z*self.scale[arm] + self.mean[arm])
