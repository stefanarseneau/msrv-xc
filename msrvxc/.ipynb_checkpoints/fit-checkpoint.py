import lmfit
from multiprocessing import Pool
import emcee
import numpy as np

def log_likelihood_spec(theta, wl, fl, ivar, interpolator):   
    teff, logg, Z, rv = theta
    
    if ivar is not None:
        cost = 1/2*np.sum(((fl - np.interp(wl, np.linspace(3600, 9000, 23074), interpolator(list(theta))[0]))** 2 *ivar**2 ))
        #chisqr = sum( (fl - np.interp(wl, np.linspace(3600, 9000, 23074), interpolator(list(theta))[0]) * ivar)**2 )
    else:
        cost = 1/2*np.sum(((fl - np.interp(wl, np.linspace(3600, 9000, 23074), interpolator(list(theta))[0]))** 2 ))
        #chisqr = sum( (fl - np.interp(wl, np.linspace(3600, 9000, 23074), interpolator(list(theta))[0]))**2 )
        
    return -cost

def log_prior_spec(theta):
    T, logg, Z, rv = theta[:4]
    
    Tmin, Tmax = 3500, 7000
    loggmin, loggmax = 2.5, 5
    Zmin, Zmax = -2.5, 0.5
    rvmin, rvmax = -200, 200
    
    if Tmin < T < Tmax and loggmin < logg < loggmax and Zmin < Z < Zmax and rvmin < rv < rvmax:
        return 0.0
    return -np.inf

def log_probability_spec(theta, wl, fl, ivar, interpolator):
    lp = log_prior_spec(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_spec(theta, wl, fl, ivar, interpolator)


def fit_rv(interpolator, wl, fl, ivar = None, make_plot = False, vary_logg = False, p0 = [5000, 4, 0, 0]):      
    pos = p0 + 1e-4 * np.random.randn(32, 4)
    nwalkers, ndim = pos.shape
        
    #ith Pool(processes=7) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability_spec, args=(wl, fl, ivar, interpolator))
    sampler.run_mcmc(pos, 5000, progress=True);
    
    return sampler, nwalkers, ndim