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
                    self.alpha == [alpha_] * self.arms
                elif type(alpha_) is list or type(alpha_) is np.ndarray:
                    assert len(alpha_) == self.arms and all(alpha_ > 0)\
                        and all(alpha_ <= 2)
                    self.alpha = alpha_
            else:
                self.alpha = np.random.uniform(0, 2, n)

            if 'beta' in kwargs:
                beta_ = kwargs['beta']
                if type(beta_) is float:
                    self.beta == [beta_] * self.arms
                elif type(beta_) is list or type(beta_) is np.ndarray:
                    assert len(beta_) == self.arms
                    self.beta = beta_
            else:
                self.beta = np.zeros((1, self.arms))

            if 'scale' in kwargs:
                scale_ = kwargs['scale']
                if type(scale_) is float:
                    self.scale == [scale_] * self.arms
                elif type(scale_) is list or type(scale_) is np.ndarray:
                    assert len(scale_) == self.arms
                    self.scale = scale_
            else:
                self.scale = np.ones((1, self.arms))

            if 'mean' in kwargs:
                assert len(kwargs['mean']) == n
                self.mean = kwargs['mean']
            else:
                self.mean = np.random.uniform(0, 1, n)


            k = 1 - np.abs(1-self.alpha)
            phi_0 = -0.5*self.beta*k/self.alpha

            beta_prime = np.array([beta if alpha == 1 else
                -np.tan(0.5*np.pi*(1-alpha))*np.tan(alpha*phi) for
                alpha, beta, phi in zip(self.alphas, self.betas, phi_0)])

            self.k = k
            self.phi0 = phi_0
            sellf.beta_prime = beta_prime

            if all(self.alpha > 1):
                self.best_arm = np.argmax(self.mean)
            else:
                # No best arm exists since means are infinite
                self.best_arm = None

    def iter(self, arm):

        if self.dist == 'bernoulli':

            assert arm < self.arms
            return np.random.binomial(1, self.params[arm], 1)

        if self.dist == 'stable':

            assert arm < self.arms
            alpha_ = self.alpha[arm]
            phi0_ = self.phi0[arm]

            u = np.random.uniform(0, 1)
            phi = np.pi * u**(-0.5)
            eps = 1 - alpha_
            tau = -eps*np.tan(alpha_* phi0_)
            w = -np.ln(np.random.uniform(0, 1))
            z = (np.cos(eps*phi) -
                np.tan(alpha_*phi0_)*np.sin(eps*phi)/(w * np.cos(phi)))
            d = z**(eps/alpha_)/eps
            s = np.tan(alpha_*phi0_) + z**(eps/alpha_)*(np.sin(alpha_*phi) -
                np.tan(alpha_*phi0_)*np.cos(alpha_*phi))/np.cos(phi)

            return s
