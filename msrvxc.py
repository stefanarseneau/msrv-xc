import numpy as np

# R = 3600

def build_grid():
    T_max = 7100
    T_min = 3500
    T_step = 100
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
    
    basepath = 'https://archive.stsci.edu/missions/hlsp/bosz/fits/insbroad_002000/'
    
    values = np.zeros((len(teffs), len(loggs), len(metals) len(rvs), 6000))
    values_norm = np.zeros((len(teffs), len(loggs), len(metals) len(rvs), 6000))
    
    for i in range(len(teffs)):
        for j in range(len(loggs)):
            for k in range(len(metals)):
                for r in range(len(rvs)):
                    if metals[k] >= 0:
                        path = basepath + 'metal_+{:1.2}'.format(metals[k]) + 'carbon_+0.00/alpha_+0.00/amp{}cp00op00t{}g{}v20modrt0b2000rs.fits'.format(int(10 * metals[k]), teffs[i], int(10*loggs[j]))
                        
                        print(path)
                    else:
                        path = basepath + 'metal_{:1.2}'.format(metals[k]) + 'carbon_+0.00/alpha_+0.00/amm{}cp00op00t{}g{}v20modrt0b2000rs.fits'.format(int(-10 * metals[k]), teffs[i], int(10*loggs[j]))
                        
                        print(path)
    