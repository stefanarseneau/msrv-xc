import numpy as np
from Payne import utils as payne_utils
from astropy.io import fits

from . import utils 

# R500 --> 5769
# R1000 --> 
# R2000 --> 23074 | norm --> 4300
# R10000 --> 115367 | norm --> 21498

c_kms = 2.99792458e5 # speed of light in km/s

def build_grid():
    T_max = 7100
    T_min = 3500
    T_step = 250
    teffs = np.arange(T_min, T_max, T_step)
    
    logg_max = 5.5
    logg_min = 2.5
    logg_step = 0.5
    loggs = np.arange(logg_min, logg_max, logg_step)
    
    Z_max = 1.5
    Z_min = -2.5
    Z_step = 1
    metals = np.arange(Z_min, Z_max, Z_step)
    
    rv_max = 250
    rv_min = -250
    rv_step = 25
    rvs = np.arange(rv_min, rv_max, rv_step)
    
    R = 10000
    R_target = 2000
        
    basepath = 'https://archive.stsci.edu/missions/hlsp/bosz/fits/insbroad_{:06d}/'.format(R)
    temppath = basepath + 'metal_+{:.2f}/'.format(metals[0]) + 'carbon_+0.00/alpha_+0.00/amp{:02d}cp00op00t{}g{}v20modrt0b{}rs.fits'.format(int(10 * metals[0]), teffs[0], int(10*loggs[0]), R)
    
    spec = fits.open(temppath)
    wvl = spec[1].data['Wavelength']
    wavl_range = (3600<wvl)*(wvl<9000)
    wvl = wvl[wavl_range]
        
    values = np.zeros((len(teffs), len(loggs), len(metals), len(rvs), 115367))
    values_norm = np.zeros((len(teffs), len(loggs), len(metals), len(rvs), 115367))
    
    for i in range(len(teffs)):
        print('{} / {}'.format(i, len(teffs)))
        for j in range(len(loggs)):
            for k in range(len(metals)):
                if metals[k] >= 0:
                    path = basepath + 'metal_+{:.2f}/'.format(metals[k]) + 'carbon_+0.00/alpha_+0.00/amp{:02d}cp00op00t{}g{}v20modrt0b{}rs.fits'.format(int(10 * metals[k]), teffs[i], int(10*loggs[j]), R)
                else:
                    path = basepath + 'metal_{:.2f}/'.format(metals[k]) + 'carbon_+0.00/alpha_+0.00/amm{:02d}cp00op00t{}g{}v20modrt0b{}rs.fits'.format(int(-10 * metals[k]), teffs[i], int(10*loggs[j]), R)
                        
                try:
                    file = fits.open(path)
                except:
                    print(path)
                    
                fl = temp[1].data['SpecificIntensity'][wavl_range]
                
                for r in range(len(rvs)):
                    fl *= np.sqrt((1 - rvs[r]/c_kms)/(1 + rvs[r]/c_kms))
                    
                    values[i,j,k,r] = utils.smoothing.smoothspec(wvl,fl,resolution=R_target,smoothtype="R")
                    _, values_norm[i,j,k,r] = continuum_normalize(wvl,values[i1,i2,i3])

    