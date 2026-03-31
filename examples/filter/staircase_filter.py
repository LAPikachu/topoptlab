# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
from scipy.special import hyp2f1
import matplotlib.pyplot as plt

from topoptlab.filter.haeviside_projection import find_eta,\
                                                  eta_projection,\
                                                  find_multieta,\
                                                  multieta_projection

if __name__ == "__main__":
    #
    n=10
    volfrac = 0.2
    beta = 20
    # 
    x = np.linspace(0,1,1001)[:,None]
    print(x.mean())
    #
    eta = find_eta(eta0=0.5,
                   xTilde=x, 
                   volfrac=volfrac, 
                   beta=beta)
    xPhys = eta_projection(eta=eta,
                           xTilde=x, 
                           beta=beta)
    print(xPhys.mean())
    #
    weights = np.array([0.1,0.3,0.6])
    etas =   find_multieta(etas0=np.array([0.25, 0.5, 0.75]),
                           xTilde=x,
                           volfrac=volfrac,
                           beta=beta,
                           weights=weights,
                           mode="mse")
    xPhys_multi = multieta_projection(etas=etas,
                                      xTilde=x,
                                      beta=beta,
                                      weights=weights)
    print(etas)
    print(xPhys_multi.mean())
    #
    fig, ax = plt.subplots()
    # original
    ax.plot(x,x,label="x")
    # plot volume conservation
    ax.plot(x,
            xPhys,
            label="eta")
    ax.plot(x,
            xPhys_multi,
            label="multi-eta")
    ax.legend()
    plt.show()
    
