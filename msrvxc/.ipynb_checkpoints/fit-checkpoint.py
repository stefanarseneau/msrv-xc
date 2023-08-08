import lmfit
from multiprocessing import Pool
import emcee
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#from Payne import utils as payne_utils


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
    
def xcorr_rv(params, interpolator, interp_wvl, wl, fl, ivar, plot = False,
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
    rcc = np.zeros(len(rvgrid))
        
    residual = lambda params: chisqr(params, interpolator, interp_wvl, wl, fl, ivar)
    
    for ii,rv in enumerate(rvgrid):
        params['RV'].set(value = rv)
        resid = residual(params)
        chi = np.nansum(resid)
        redchi = np.nansum(resid) / (len(resid) - 1)
        cc[ii] = chi
        rcc[ii] = redchi
        
    window = int(quad_window / np.diff(rvgrid)[0])

    # plt.plot(rvgrid, cc)
    # plt.show()
    
    argmin = np.nanargmin(cc)
    c1 = argmin - window

    if c1 < 0:
        c1 = 0

    c2 = argmin + window + 1

    #print(c1, c2)

    rvgrid = rvgrid[c1:c2]
    cc = cc[c1:c2]
    rcc = rcc[c1:c2]

    try:
        pcoef = np.polyfit(rvgrid, cc, 2)
        rv = - 0.5 * pcoef[1] / pcoef[0]  
        
        print(pcoef[0])
        print(pcoef[1])
        print(pcoef[2])
        
        t_cc = pcoef[0] * rv**2 + pcoef[1] * rv + pcoef[2]
        
        intersect = ( (-pcoef[1] + np.sqrt(pcoef[1]**2 - 4 * pcoef[0] * (pcoef[2] - t_cc - 1))) / (2 * pcoef[0]), 
                     (-pcoef[1] - np.sqrt(pcoef[1]**2 - 4 * pcoef[0] * (pcoef[2] - t_cc - 1))) / (2 * pcoef[0]) )
        
        e_rv = np.abs(intersect[0] - intersect[1]) / 2
        redchi = np.interp(rv, rvgrid, rcc)
    
        if plot:
            xgrid = np.linspace(min(rvgrid), max(rvgrid), 50)
            
            plt.figure(figsize = (10,5))
            pcoef = np.polyfit(rvgrid, cc, 2)
            plt.plot(rvgrid, cc, label = r'Actual $\chi^2$ curve')
            plt.plot(xgrid, pcoef[0]*xgrid**2 + pcoef[1]*xgrid + pcoef[2], label = r'Fitted $\chi^2$ curve')
            
            plt.axvline(x = rv)
            plt.axvline(x = rv + e_rv, ls = ':')
            plt.axvline(x = rv - e_rv, ls = ':')
            plt.axhline(y = t_cc, label = 'Minimum $\chi^2$')
    except:
        print('pcoef failed!! returning min of chi function & err = 999')
        rv = rvgrid[np.nanargmin(cc)]
        e_rv = 999
    
    #rv = rvgrid[np.nanargmin(cc)]
        
    return rv, e_rv, redchi, rvgrid, cc

def fit_rv(interpolator, interp_wvl, wl, fl, ivar = None, plot = True, full_interpolator = True, p0 = [5000, 3, 0, 0]):
    params = lmfit.Parameters()

    params.add('teff', value = p0[0], min = 3500, max = 7000, vary = True)
    params.add('logg', value = p0[1], min=2.5, max=5, vary=True)
    params.add('z', value = p0[2], min = -2.5, max = 0.5, vary = True)
    params.add('RV', value = p0[3], min = -1500, max = 1500, vary = True)
        
    init = lmfit.minimize(chisqr, params, kws = dict(interpolator = interpolator, interp_wvl = interp_wvl, wl = wl, fl = fl, ivar = ivar), method = 'leastsq')
                
    rv, e_rv, redchi, rvgrid, cc = xcorr_rv(init.params, interpolator, interp_wvl, wl, fl, ivar, plot)
    
    #print(init.params)
    
    #param_grid = []
    #for i in tqdm(range(100)):
    #    temp_params = init.params.copy()
    #    
    #    if init.params['teff'].stderr is not None:
    #        teff = init.params['teff'].value + (init.params['teff'].stderr) * np.random.normal()
    #    else:
    #        teff = init.params['teff'].value + (100) * np.random.normal()
    #        
    #    if init.params['logg'].stderr is not None:
    #        logg = init.params['logg'].value + (init.params['logg'].stderr) * np.random.normal()
    #    else:
    #        logg = init.params['logg'].value + (0.1) * np.random.normal()
    #        
    #    if init.params['z'].stderr is not None:
    #        z = init.params['z'].value + (init.params['z'].stderr) * np.random.normal()
    #    else:
    #        z = init.params['z'].value + (0.1 * init.params['z'].value) * np.random.normal()
    #    
    #    temp_params['teff'].value = teff
    #    temp_params['logg'].value = logg
    #    temp_params['z'].value = z
    #    
    #    trv, te_rv, trvgrid, tcc = xcorr_rv(temp_params, interpolator, interp_wvl, wl, fl, ivar, plot)
    #    
    #    param_grid.append([teff, logg, z, trv, te_rv])
    #
    #print(init.params)
        
    
    return rv, e_rv, redchi, init


















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