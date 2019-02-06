import numpy as np
from bandits import envs

def cms_alpha(alpha, beta, mu, sigma):
    ''' Generate a random sample from the alpha-stable distribution f
        using the CMS Algorithm, where f : S_alpha(beta, mu, sigma) '''
    k = 1 - np.abs(1-alpha)
    phi_0 = -0.5*beta*k/alpha
    beta_prime = beta if alpha == 1 else\
        -np.tan(0.5*np.pi*(1-alpha))*np.tan(alpha*phi_0)

    u = np.random.uniform(0, 1)
    phi = np.pi * u**(-0.5)
    eps = 1 - alpha
    tau = -eps*np.tan(alpha_* phi_0)
    w = -np.ln(np.random.uniform(0, 1))
    z = (np.cos(eps*phi) -
        np.tan(alpha*phi_0)*np.sin(eps*phi)/(w * np.cos(phi)))
    d = z**(eps/alpha)/eps
    s = np.tan(alpha*phi_0) + z**(eps/alpha)*(np.sin(alpha*phi) -\
        np.tan(alpha*phi_0)*np.cos(alpha*phi))/np.cos(phi)

    return s

def main(n_arms, n_iter, n_sample_iter, n_burn_in, alpha=2, sigma=1, mus=None, priors=None):

    assert n_arms > 1

    if mus:
        assert n_arms = len(mus)
    else:
        mus = np.random.uniform(0, 100, n_arms)

    if priors:
        assert type(priors) is np.ndarray
        assert n_arms = priors.shape[0]

    else:

        p_mu = np.full((n_arms, 1), 50)
        p_sigma = np.full((n_arms, 1), sigma)

        priors = np.concatenate((p_mu, p_sigma), 1)


    for iter in range(n_iter):

        mu, la = sample_mu_lambda(priors, )
