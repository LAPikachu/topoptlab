# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Dict,Tuple,Union

import numpy as np
from scipy.optimize import root_scalar

def find_eta(eta0: float, 
             xTilde: np.ndarray, 
             beta: float, 
             volfrac: float,
             root_args: Dict = {"fprime": True,
                                "method": "newton",
                                "maxiter": 1000,
                                "bracket": [-1/2,1/2]},
             **kwargs: Any) -> float:
    """
    Find volume preserving eta for the element-wiser elaxed Haeviside 
    as has been done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    eta0 : float
        initial guess for threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    volfrac : float
        volume fraction.
    root_args : dict
        arguments for root finding algorithm to find the volume conserving eta.

    Returns
    -------
    eta : float
        volume conserving eta.

    """
    # unfortunately scipy.optimize needs f to change sign between the
    # respective ends of the brackets, therefor the eta found by this function
    # is offset by -1/2 to the value later used
    result = root_scalar(f=_find_eta_root_func, 
                         x0=eta0-1/2, 
                         args=(xTilde,beta,volfrac),
                         x1=0.,
                         fprime=True, 
                         method="newton", 
                         maxiter=1000, 
                         bracket=[-1/2,1/2])
    #
    if result.converged:
        return result.root+1/2
    else:
        raise ValueError("volume conserving eta could not be found: ",result)

def _find_eta_root_func(eta: float, 
                        xTilde: np.ndarray, 
                        beta: float, 
                        volfrac: float) -> Tuple[float,float]:
    """
    Function whose root is the volume preserving threshold.

    Parameters
    ----------
    eta : float
        current threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    volfrac : float
        volume fraction.

    Returns
    -------
    res : float
        value of current volume fraction - intended volume fraction.
    gradient : float
        gradient for Newton procedure

    """
    #
    eta = eta + 1/2
    #
    xPhys = eta_projection(eta=eta,xTilde=xTilde,beta=beta)
    # helper terms
    tanh_bn = np.tanh(beta * eta)
    tanh_b1n = np.tanh(beta * (1 - eta))
    tanh_bx_n = np.tanh(beta * (xTilde - eta))
    tanh_bn_x = np.tanh(beta * (eta - xTilde))
    #
    sech2_bn = 1 - tanh_bn**2
    sech2_bx_n = 1 - tanh_bx_n**2
    sech2_b1n = 1 - tanh_b1n**2
    #
    denom = tanh_bn + tanh_b1n
    term =  sech2_bn * (tanh_b1n + tanh_bn_x) + sech2_b1n * (tanh_bn + tanh_bx_n)
    return xPhys.mean()-volfrac, -beta*(sech2_bx_n/denom + term/(denom**2)).mean()

def eta_projection(eta: float, 
                   xTilde: np.ndarray, 
                   beta: float) -> np.ndarray:
    """
    Perform a differentiable "relaxed" Haeviside projection as done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    eta : float
        threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : float
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity

    Returns
    -------
    xProj : np.ndarray
        projected densities.

    """
    return (np.tanh(beta * eta) + np.tanh(beta * (xTilde - eta))) / \
           (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))

def eta_projection_dx(eta: float, 
                      xTilde: np.ndarray, 
                      beta: float) -> np.ndarray:
    """
    Perform first derivative of differentiable "relaxed" Haeviside projection as 
    done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    eta : float
        threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : float
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity

    Returns
    -------
    xProj_dx : np.ndarray
        first derivative of projected densities.

    """
    return beta * (1 - np.tanh(beta * (x_filtered - eta))**2) /\
                  (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))    

def find_multieta(etas0: np.ndarray, 
                  xTilde: np.ndarray, 
                  beta: float, 
                  volfrac: float,
                  weights: Union[None,np.ndarray]=None,
                  root_args: Dict = {"fprime": True,
                                     "method": "newton",
                                     "maxiter": 1000,
                                     "bracket": [-1/2,1/2]},
                  **kwargs: Any) -> float:
    """
    Find volume preserving eta for the element-wiser elaxed Haeviside 
    as has been done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    etas0 : np.ndarray
        initial guess for threshold values.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    volfrac : float
        volume fraction.
    weights : None or np.ndarray
        weights for combining the multiple threshold projections. If None,
        uniform weights are used.
    root_args : dict
        arguments for root finding algorithm to find the volume conserving eta.

    Returns
    -------
    eta : float
        volume conserving eta.

    """
    #
    etas0=np.asarray(etas0)
    # unfortunately scipy.optimize needs f to change sign between the
    # respective ends of the brackets, therefor the eta found by this function
    # is offset by -1/2 to the value later used
    result = root_scalar(f=_find_multieta_root_func, 
                         x0=etas0-1/2, 
                         args=(xTilde,
                               beta,
                               weights,
                               volfrac),
                         x1=0.,
                         fprime=True, 
                         method="newton", 
                         maxiter=1000, 
                         bracket=[-1/2,1/2])
    #
    if result.converged:
        return result.root+1/2
    else:
        raise ValueError("volume conserving eta could not be found: ",result)

def _find_multieta_root_func(etas: np.ndarray, 
                             xTilde: np.ndarray, 
                             beta: float,
                             weights: Union[None,np.ndarray],
                             volfrac: float
                             ) -> Tuple[float,np.ndarray]:
    """
    Function whose root is the volume preserving threshold.

    Parameters
    ----------
    etas : np.ndarray
        current threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity.
    weights : None or np.ndarray
        weights for combining the multiple threshold projections. If None,
        uniform weights are used.
    volfrac : float
        volume fraction.

    Returns
    -------
    res : float
        value of current volume fraction - intended volume fraction.
    gradient : np.ndarray
        gradient for Newton procedure

    """
    #
    etas = etas + 1/2
    #
    xPhys = multieta_projection(etas=etas, 
                                xTilde=xTilde,
                                beta=beta, 
                                weights=weights)
    # helper terms
    tanh_bn = np.tanh(beta * etas)
    tanh_b1n = np.tanh(beta * (1 - etas))
    tanh_bx_n = np.tanh(beta * (xTilde[...,None] - etas[None,...]))
    tanh_bn_x = np.tanh(beta * (etas[None,...] - xTilde[...,None]))
    #
    sech2_bn = 1 - tanh_bn**2
    sech2_bx_n = 1 - tanh_bx_n**2
    sech2_b1n = 1 - tanh_b1n**2
    #
    denom = tanh_bn + tanh_b1n
    term =  sech2_bn[None,...] * (tanh_b1n[None,...] + tanh_bn_x) +\
            sech2_b1n[None,...] * (tanh_bn[None,...] + tanh_bx_n)
    return xPhys.mean()-volfrac,\
           -beta*(sech2_bx_n/denom + term/(denom**2)).mean()

def multieta_projection(etas: np.ndarray, 
                        xTilde: np.ndarray, 
                        beta: float, 
                        weights: Union[None,np.ndarray] = None
                        ) -> np.ndarray:
    """
    Perform a differentiable "relaxed" Haeviside projection as done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505
    
    but with multiple thresholds.
    
    Parameters
    ----------
    etas : np.ndarray
        threshold values.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : float
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    weights : None or np.ndarray
        weights for combining the multiple threshold projections. If None,
        uniform weights are used.

    Returns
    -------
    xProj : np.ndarray
        projected densities.

    """
    xProj = (np.tanh(beta*etas[None,...]) +\
             np.tanh(beta*(xTilde[...,None]-etas[None,...])))/\
            (np.tanh(beta*etas[None,...]) +\
             np.tanh(beta * (1 - etas[None,...])))
    if weights is None:
        weights = np.ones(etas.shape)/etas.shape[0]
    return np.sum( xProj * weights[None,...] ,axis=-1)

def multieta_projection_dx(etas: np.ndarray, 
                           xTilde: np.ndarray, 
                           beta: float, 
                           weights: Union[None,np.ndarray] = None
                           ) -> np.ndarray:
    """
    Perform a differentiable "relaxed" Haeviside projection as done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505
    
    but with multiple thresholds.
    
    Parameters
    ----------
    etas : np.ndarray
        threshold values.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : float
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    weights : None or np.ndarray
        weights for combining the multiple threshold projections. If None,
        uniform weights are used.  

    Returns
    -------
    xProj_dx : np.ndarray
        first derivative of projected densities.

    """
    xProj_dx = beta * (1 - np.tanh(beta * (x_filtered - eta[None,...]))**2) /\
                      (np.tanh(beta*eta[None,...])+np.tanh(beta*(1-eta[None,...])))
    
    if weights is None:
        weights = np.ones(etas.shape)/etas.shape[0]
    return np.sum( xProj_dx * weights[None,...] ,axis=-1)