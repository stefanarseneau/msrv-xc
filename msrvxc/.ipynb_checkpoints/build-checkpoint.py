import numpy as np
import pickle
import sys
import subprocess

from Payne import utils as payne_utils

from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

from tqdm import tqdm


from . import utils 

# R500 --> 5769
# R1000 --> 
# R2000 --> 23074 | norm --> 4300
# R10000 --> 115367 | norm --> 21498

c_kms = 2.99792458e5 # speed of light in km/s

def build_bosz_grid(teff_grid = [3500, 7100, 250], logg_grid = [2.5, 5.5, 0.5], metal_grid = [-2.5, 1.5, 1], carbon_grid = [0, 1, 1], alpha_grid = [0, 1, 1], 
               wl_range = [3600, 9000], R = 10000, R_target = 10000):
    """
    build_bosz_grid: builds interpolator grids from bosz
    inputs:
        teff_grid          array of [teff_minimum, teff_maximum, step]
        logg_grid          array of [logg_minimum, logg_maximum, step]
        metal_grid         array of [metal_minimum, metal_maximum, step]
        carbon_grid        array of [carbon_minimum, carbon_maximum, step]
        alpha_grid         array of [alpha_minimum, alpha_maximum, step]
        wl_range           two-element array of minimum and maximum wavelengths to calculate the grid
        R                  bosz instrument broadening
        R_target           desired resolution of the grid
    outputs:
        wl_grid            wavelength grid of the interpolator
        raw_bosz           interpolator for entire range of wavelengths across bosz
        interp_bosz        interpolator for specified wavelength range
        interp_bosz_norm   interpolator for continuum normalized spectrum within the specified wavelength range
    """

    teffs = np.arange(teff_grid[0], teff_grid[1], teff_grid[2])
    loggs = np.arange(logg_grid[0], logg_grid[1], logg_grid[2])
    metals = np.arange(metal_grid[0], metal_grid[1], metal_grid[2])
    #carbons = np.arange(carbon_grid[0], carbon_grid[1], carbon_grid[2])
    #alphas = np.arange(alpha_grid[0], alpha_grid[1], alpha_grid[2])
    
    #print(carbons)
    #print(alphas)
        
    basepath = 'https://archive.stsci.edu/missions/hlsp/bosz/fits/insbroad_{:06d}/'.format(R)
    temppath = 'https://archive.stsci.edu/missions/hlsp/bosz/fits/insbroad_{:06d}/metal_+0.00/carbon_+0.00/alpha_+0.00/amp00cp00op00t10000g20v20modrt0b{}rs.fits'.format(R, R)
    
    print(temppath)
    
    spec = fits.open(temppath)
    wvl = spec[1].data['Wavelength']
    wavl_range = (wl_range[0]<wvl)*(wvl<wl_range[1])
    
    wl_grid = wvl[wavl_range]
    npoints = len(wl_grid)
    
    raw_values = np.zeros((len(teffs), len(loggs), len(metals), npoints))
    values = np.zeros((len(teffs), len(loggs), len(metals), npoints))
    values_norm = np.zeros((len(teffs), len(loggs), len(metals), npoints))
    
    """
    for i in tqdm(range(len(teffs))):
        for j in range(len(loggs)):
            for k in range(len(metals)):
                for a in range(len(carbons)):
                    for b in range(len(alphas)):
                        if metals[k] >= 0:
                            if carbons[a] >= 0:
                                if alphas[b] >= 0:
                                    path = basepath + 'metal_+{:.2f}/'.format(metals[k]) + 'carbon_+{:.2f}/'.format(carbons[a]) + 'alpha_+{:.2f}/'.format(alphas[b]) + 'amp{:02d}cp{:02d}op{:02d}t{}g{}v20modrt0b{}rs.fits'.format(int(10 * metals[k]), int(10 * carbons[a]), int(10 * alphas[b]), teffs[i], int(10*loggs[j]), R)
                                else:
                                    path = basepath + 'metal_+{:.2f}/'.format(metals[k]) + 'carbon_+{:.2f}/'.format(carbons[a]) + 'alpha_+{:.2f}/'.format(alphas[b]) + 'amp{:02d}cp{:02d}om{:02d}t{}g{}v20modrt0b{}rs.fits'.format(int(10 * metals[k]), int(10 * carbons[a]), int(-10 * alphas[b]), teffs[i], int(10*loggs[j]), R)
                            else:
                                if alphas[b] >= 0:
                                    path = basepath + 'metal_+{:.2f}/'.format(metals[k]) + 'carbon_+{:.2f}/'.format(carbons[a]) + 'alpha_+{:.2f}/'.format(alphas[b]) + 'amp{:02d}cm{:02d}op{:02d}t{}g{}v20modrt0b{}rs.fits'.format(int(10 * metals[k]), int(-10 * carbons[a]), int(10 * alphas[b]), teffs[i], int(10*loggs[j]), R)
                                else:
                                    path = basepath + 'metal_+{:.2f}/'.format(metals[k]) + 'carbon_+{:.2f}/'.format(carbons[a]) + 'alpha_+{:.2f}/'.format(alphas[b]) + 'amp{:02d}cm{:02d}om{:02d}t{}g{}v20modrt0b{}rs.fits'.format(int(10 * metals[k]), int(-10 * carbons[a]), int(-10 * alphas[b]), teffs[i], int(10*loggs[j]), R)
                        else:
                            if carbons[a] >= 0:
                                if alphas[b] >= 0:
                                    path = basepath + 'metal_{:.2f}/'.format(metals[k]) + 'carbon_+{:.2f}/'.format(carbons[a]) + 'alpha_+{:.2f}/'.format(alphas[b]) + 'amm{:02d}cp{:02d}op{:02d}t{}g{}v20modrt0b{}rs.fits'.format(int(-10 * metals[k]), int(10 * carbons[a]), int(10 * alphas[b]), teffs[i], int(10*loggs[j]), R)
                                else:
                                    path = basepath + 'metal_{:.2f}/'.format(metals[k]) + 'carbon_+{:.2f}/'.format(carbons[a]) + 'alpha_+{:.2f}/'.format(alphas[b]) + 'amp{:02d}cp{:02d}om{:02d}t{}g{}v20modrt0b{}rs.fits'.format(int(-10 * metals[k]), int(10 * carbons[a]), int(-10 * alphas[b]), teffs[i], int(10*loggs[j]), R)
                            else:
                                if alphas[b] >= 0:
                                    path = basepath + 'metal_{:.2f}/'.format(metals[k]) + 'carbon_+{:.2f}/'.format(carbons[a]) + 'alpha_+{:.2f}/'.format(alphas[b]) + 'amm{:02d}cm{:02d}op{:02d}t{}g{}v20modrt0b{}rs.fits'.format(int(-10 * metals[k]), int(-10 * carbons[a]), int(10 * alphas[b]), teffs[i], int(10*loggs[j]), R)
                                else:
                                    path = basepath + 'metal_{:.2f}/'.format(metals[k]) + 'carbon_+{:.2f}/'.format(carbons[a]) + 'alpha_+{:.2f}/'.format(alphas[b]) + 'amm{:02d}cm{:02d}om{:02d}t{}g{}v20modrt0b{}rs.fits'.format(int(-10 * metals[k]), int(-10 * carbons[a]), int(-10 * alphas[b]), teffs[i], int(10*loggs[j]), R)
                            
                            #path = basepath + 'metal_{:.2f}/'.format(metals[k]) + 'carbon_+0.00/alpha_+0.00/amm{:02d}cp00op00t{}g{}v20modrt0b{}rs.fits'.format(int(-10 * metals[k]), teffs[i], int(10*loggs[j]), R)
                                
                        try:
                            file = fits.open(path)
                        except:
                            print('[!!ERR] Could not find file: ' + path)
                            break
                            
                        fl = file[1].data['SpecificIntensity']
                        wl = file[1].data['Wavelength']
                               
                        smooth_fl = payne_utils.smoothing.smoothspec(wl,fl,resolution=R_target,smoothtype="R")[wavl_range]
                        wl = wl[wavl_range]
                        _, norm_fl = utils.continuum_normalize(wl_new,smooth_fl, avg_size = 500)
                        
                        
                        raw_values[i,j,k,a,b,r] = np.interp(wl_grid, wl_new, fl[nwavl_range])
                        values[i,j,k,a,b,r] = np.interp(wl_grid, wl_new, smooth_fl)
                        values_norm[i,j,k,a,b,r] = np.interp(wl_grid, wl_new, norm_fl)
                                                    
                        file.close()
    """
    
    for i in tqdm(range(len(teffs))):
        for j in range(len(loggs)):
            for k in range(len(metals)):
                if metals[k] >= 0:
                    path = basepath + 'metal_+{:.2f}/'.format(metals[k]) + 'carbon_+{:.2f}/'.format(0) + 'alpha_+{:.2f}/'.format(0) + 'amp{:02d}cp{:02d}op{:02d}t{}g{}v20modrt0b{}rs.fits'.format(int(10 * metals[k]), int(10 * 0), int(10 * 0), teffs[i], int(10*loggs[j]), R)     
                else:
                    path = basepath + 'metal_{:.2f}/'.format(metals[k]) + 'carbon_+{:.2f}/'.format(0) + 'alpha_+{:.2f}/'.format(0) + 'amm{:02d}cp{:02d}op{:02d}t{}g{}v20modrt0b{}rs.fits'.format(int(-10 * metals[k]), int(10 * 0), int(10 * 0), teffs[i], int(10*loggs[j]), R)
                            
                    #path = basepath + 'metal_{:.2f}/'.format(metals[k]) + 'carbon_+0.00/alpha_+0.00/amm{:02d}cp00op00t{}g{}v20modrt0b{}rs.fits'.format(int(-10 * metals[k]), teffs[i], int(10*loggs[j]), R)
                                
                try:
                    file = fits.open(path)
                except:
                    print('[!!ERR] Could not find file: ' + path)
                    break
                            
                fl = file[1].data['SpecificIntensity']
                wl = file[1].data['Wavelength']
                               
                smooth_fl = payne_utils.smoothing.smoothspec(wl,fl,resolution=R_target,smoothtype="R")[wavl_range]
                wl_new = wl[wavl_range]
                _, norm_fl = utils.continuum_normalize(wl_new,smooth_fl, avg_size = 500)
                        
                        
                raw_values[i,j,k,] = np.interp(wl_grid, wl_new, fl[wavl_range])
                values[i,j,k] = np.interp(wl_grid, wl_new, smooth_fl)
                values_norm[i,j,k] = np.interp(wl_grid, wl_new, norm_fl)
                                                    
                file.close()
                

    raw_bosz = RegularGridInterpolator((teffs,loggs,metals),raw_values)
    interp_bosz = RegularGridInterpolator((teffs,loggs,metals),values)
    interp_bosz_norm = RegularGridInterpolator((teffs,loggs,metals),values_norm)
    
    return wl_grid, raw_bosz, interp_bosz, interp_bosz_norm
    
def build_phoenix_grid(teff_grid = [3500, 7100, 250], logg_grid = [2.5, 5.5, 0.5], metal_grid = [0, 0, 1], alpha_grid = [0, 1, 1], rv_grid = [-1000, 1000, 25], 
               wl_range = [3600, 9000], R = 10000, R_target = 10000):

    teffs = np.arange(teff_grid[0], teff_grid[1], teff_grid[2])
    loggs = np.arange(logg_grid[0], logg_grid[1], logg_grid[2])
    metals = np.arange(metal_grid[0], metal_grid[1], metal_grid[2])
    alphas = np.arange(alpha_grid[0], alpha_grid[1], alpha_grid[2])
    rvs = np.arange(rv_grid[0], rv_grid[1], rv_grid[2])
        
    basepath = 'https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/FITS/'
    temppath = 'https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/FITS/lte012.0-2.5-0.0a+0.0.BT-Settl.spec.fits.gz'
    
    spec = fits.open(temppath)
    wvl = spec[1].data['Wavelength']
    wavl_range = (wl_range[0]<wvl)*(wvl<wl_range[1])
    
    wl_grid = wvl[wavl_range]
    npoints = len(wl_grid)
    
    raw_values = np.zeros((len(teffs), len(loggs), len(metals), len(carbons), len(alphas), len(rvs), npoints))
    values = np.zeros((len(teffs), len(loggs), len(metals), len(carbons), len(alphas), len(rvs), npoints))
    values_norm = np.zeros((len(teffs), len(loggs), len(metals), len(carbons), len(alphas), len(rvs), npoints))
        
    for i in tqdm(range(len(teffs))):
        for j in range(len(loggs)):
            for k in range(len(metals)):
                for a in range(len(alphas)):
                    if alphas[a] >= 0:
                        file = basepath + 'lte{:05.1f}-{:.2f}-{:.2f}a+{:.2f}.BT-Settl.spec.fits.gz'.format(teffs[i] / 100, loggs[j], metals[k], np.abs(alphas[a]))
                    else:
                        file = basepath + 'lte{:05.1f}-{:.2f}-{:.2f}a-{:.2f}.BT-Settl.spec.fits.gz'.format(teffs[i] / 100, loggs[j], metals[k], np.abs(alphas[a]))
                            
                    try:
                        file = fits.open(path)
                    except:
                        print('[!!ERR] Could not find file: ' + path)
                        break
                        
                    fl = file[1].data['SpecificIntensity']
                    wl = file[1].data['Wavelength']
                    
                    for r in range(len(rvs)):
                        wl_new = wl * np.sqrt((1 - rvs[r]/c_kms)/(1 + rvs[r]/c_kms))
                        nwavl_range = (wl_range[0]<wl_new)*(wl_new<wl_range[1])
                           
                        smooth_fl = payne_utils.smoothing.smoothspec(wl_new,fl,resolution=R_target,smoothtype="R")[nwavl_range]
                        wl_new = wl_new[nwavl_range]
                        _, norm_fl = utils.continuum_normalize(wl_new,smooth_fl, avg_size = 500)
                        
                        
                        raw_values[i,j,k,a,b,r] = np.interp(wl_grid, wl_new, fl[nwavl_range])
                        values[i,j,k,a,b,r] = np.interp(wl_grid, wl_new, smooth_fl)
                        values_norm[i,j,k,a,b,r] = np.interp(wl_grid, wl_new, norm_fl)
                                                
                    file.close()
                

    raw_bosz = RegularGridInterpolator((teffs,loggs,metals,alphas,rvs),raw_values)
    interp_bosz = RegularGridInterpolator((teffs,loggs,metals,alphas,rvs),values)
    interp_bosz_norm = RegularGridInterpolator((teffs,loggs,metals,alphas,rvs),values_norm)
    
    return wl_grid, raw_bosz, interp_bosz, interp_bosz_norm
    
    
    