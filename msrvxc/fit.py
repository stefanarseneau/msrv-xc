import lmfit
from multiprocessing import Pool
import emcee
import numpy as np
from Payne import utils as payne_utils


c_kms = 2.99792458e5 # speed of light in km/s
def simulate_spec(interpolator, interp_wvl, wl, theta):
    teff, logg, Z, rv = theta[0].value, theta[1].value, theta[2].value, theta[3].value
    
    wl = wl * np.sqrt((1 - rv/c_kms)/(1 + rv/c_kms))
    
    return np.interp(wl, interp_wvl, interpolator( (teff, logg, Z, 0) ))

def chisqr(params, interpolator, interp_wvl, wl, fl, ivar):
    theta = (params['teff'], params['logg'], params['z'], params['RV'])
    
    if ivar is not None:
        return ( (fl - simulate_spec(interpolator, interp_wvl,wl,theta))**2 * ivar )
    else:
        return ( (fl - simulate_spec(interpolator,interp_wvl,wl,theta)) )**2
    
def xcorr_rv(params, interpolator, interp_wvl, wl, fl, ivar,
             min_rv = -1500, max_rv = 1500, npoint = 250, quad_window = 100):
    """
    Find best RV via x-correlation on grid and quadratic fitting the peak.

    Parameters
    ----------
    wl : array_like
        wavelengths in Angstroms.
    fl : array_like
        flux array.
    ivar : array_like
        inverse-variance.
    corvmodel : LMFIT Model class
        LMFIT model with normalization instructions.
    params : LMFIT Parameters class
        parameters at which to evaluate corvmodel.
    min_rv : float, optional
        lower end of RV grid. The default is -1500.
    max_rv : float, optional
        upper end of RV grid. The default is 1500.
    npoint : int, optional
        numver of equi-spaced points in RV grid. The default is 250.
    quad_window : float, optional
        window around minimum to fit quadratic model, 
        in km/s. The default is 300.

    Returns
    -------
    rv : float
        best-fit radial velocity.
    rvgrid : array_like
        grid of radial velocities.
    cc : array_like
        chi-square statistic evaluated at each RV.

    """        
    rvgrid = np.arange(min_rv, max_rv, 0.5)
    cc = np.zeros(len(rvgrid))
        
    residual = lambda params: chisqr(params, interpolator, interp_wvl, wl, fl, ivar)
    
    for ii,rv in enumerate(rvgrid):
        params['RV'].set(value = rv)
        chi = np.sum(residual(params))
        cc[ii] = chi
        
    window = int(quad_window / np.diff(rvgrid)[0])
    
    argmin = np.nanargmin(cc)
    print(rvgrid[argmin])
    c1 = argmin - window

    if c1 < 0:
        c1 = 0

    c2 = argmin + window + 1

    #print(c1, c2)

    rvgrid = rvgrid[c1:c2]
    cc = cc[c1:c2]

    try:
        pcoef = np.polyfit(rvgrid, cc, 2)
        rv = - 0.5 * pcoef[1] / pcoef[0]  
    except:
        print('pcoef failed!! returning min of chi function')
        rv = rvgrid[np.nanargmin(cc)]
       
    converged = False
    e_rv = np.nan
    t_cc = cc - cc[argmin]
    for i in range(len(t_cc)):
        if (t_cc[i] < 1):
            e_rv = np.abs(rvgrid[i] - r_cc[i])
            break
        elif (t_cc[-i] < 1):
            e_rv = np.abs(rvgrid[-i] - r_cc[-i])
                break
    #rv = rvgrid[np.nanargmin(cc)]
        
    return rv, e_rv, rvgrid, cc

def fit_rv(interpolator, interp_wvl, wl, fl, ivar = None, full_interpolator = True, p0 = [5000, 3, 0, 0]):
    params = lmfit.Parameters()

    params.add('teff', value = p0[0], min = 3500, max = 7000, vary = True)
    params.add('logg', value = p0[1], min=2.5, max=5, vary=True)
    params.add('z', value = p0[2], min = -2.5, max = 0.5, vary = True)
    params.add('RV', value = p0[3], min = -1500, max = 1500, vary = True)
        
    init = lmfit.minimize(chisqr, params, kws = dict(interpolator = interpolator, interp_wvl = interp_wvl, wl = wl, fl = fl, ivar = ivar), method = 'leastsq')
            
    rv, e_rv, rvgrid, cc = xcorr_rv(init_params, interpolator, interp_wvl, wl, fl, ivar)
    
    return init.params, rv, e_rv, rvgrid, cc


















def log_likelihood_spec(theta, wl, fl, ivar, interpolator):   
    teff, logg, Z, rv = theta
    
    if ivar is not None:
        cost = 1/2*np.sum(((fl - simulate_spec(theta))** 2 *ivar ))
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


def fit_params(interpolator, wl, fl, ivar = None, make_plot = False, vary_logg = False, p0 = [5000, 4, 0, 0]):      
    """
    Lmao this doesn't fucking work
    """
    
    pos = p0 + 1e-4 * np.random.randn(32, 4)
    nwalkers, ndim = pos.shape
        
    #ith Pool(processes=7) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability_spec, args=(wl, fl, ivar, interpolator))
    sampler.run_mcmc(pos, 5000, progress=True);
    
    return sampler, nwalkers, ndim