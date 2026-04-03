# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Dict,Tuple,Union

import numpy as np
from scipy.optimize import root_scalar, minimize, Bounds,\
                           LinearConstraint, NonlinearConstraint

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
    xProj = eta_projection(eta=eta,xTilde=xTilde,beta=beta)
    #
    return xProj.mean()-volfrac, eta_projection_deta(eta = eta, 
                                                     xTilde=xTilde, 
                                                     xProj=xProj,
                                                     beta=beta).mean()

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
    return beta * (1 - np.tanh(beta * (xTilde - eta))**2) /\
                  (np.tanh(beta*eta)+np.tanh(beta*(1-eta))) 

def eta_projection_deta(eta: float, 
                        xTilde: np.ndarray, 
                        xProj: np.ndarray,
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
    xProj_deta : np.ndarray
        first derivative of projected densities.

    """
    return beta * ((1-xProj)*np.cosh(beta*eta)**(-2) -\
                   np.cosh(beta*(xTilde-eta))**(-2) + \
                   xProj*np.cosh(beta*(1-eta))**(-2)) /\
                  (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))    

def find_multieta(etas0: Union[float,np.ndarray], 
                  xTilde: np.ndarray, 
                  beta: float, 
                  volfrac: float,
                  weights: np.ndarray,
                  mode: str = "fixed",
                  etas_fixed: Union[None,np.ndarray] = None,
                  root_args: Dict = {"fprime": True,
                                     "method": "newton",
                                     "maxiter": 1000,
                                     "bracket": [-1/2,1/2]},
                  **kwargs: Any) -> float:
    """
    Find volume preserving eta multiple eta projections

    Parameters
    ----------
    etas0 : np.ndarray
        initial guess for threshold values.
    xTilde : np.ndarray
        intermediate densities.
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    volfrac : float
        volume fraction.
    weights : np.ndarray
        weights for combining the multiple threshold projections.
    root_args : dict
        arguments for root finding algorithm to find the volume conserving eta.

    Returns
    -------
    eta : float
        volume conserving eta.

    """
    #
    if mode == "fixed":
        etas0=np.hstack((etas_fixed, 0.5))
        func = None#_find_multieta_fixed
    elif mode == "equal":
        etas0=np.linstpace(0, 1, weights.shape[0]+2)[1:-1]
        func = None#_find_multieta_equalspaced
    elif mode == "mse":
        etas0 = np.linspace(0, 1, weights.shape[0]+2)[1:-1]
        func = mean_squared_error
    else:
        raise ValueError(f"mode must be 'fixed','equal' or 'mse'. Current value {mode}")
    # scalar problems
    if mode in ["fixed","equal"]:
        raise NotImplementedError("Not yet implemented.")
        # root_scalar needs f to change sign between the respective ends of the
        # brackets, therefor the eta found by this function is offset by -1/2 to the value later used
        result = root_scalar(f=func,
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
        if result.converged:
            result = result.root + 1/2
        else:
            raise ValueError("volume conserving eta could not be found: ", result)
    elif mode in ["mse"]:
        #
        n = etas0.shape[0]
        eps_bnd = 1e-6
        # bounds: each eta in (eps, 1-eps)
        bounds = Bounds(eps_bnd * np.ones(n), (1 - eps_bnd) * np.ones(n))
        # ordering constraint: eta[i+1] - eta[i] >= eps_bnd
        A = np.zeros((n - 1, n))
        A[:,:-1] = -np.eye(n-1)
        A[:,1:] += np.eye(n-1)
        order_constraint = LinearConstraint(A, lb=eps_bnd, ub=np.inf)
        # volume constraint: mean(xPhys) == volfrac
        volume_constraint = NonlinearConstraint(
            fun=lambda etas: volume_constraint_fun(etas, xTilde, beta, weights, volfrac),
            lb=0., ub=0.)
        result = minimize(fun=func,
                          x0=etas0,
                          args=(xTilde, beta, weights, volfrac),
                          method="trust-constr",
                          jac=None,
                          hess=None,
                          bounds=bounds,
                          constraints=[order_constraint, volume_constraint])
        #
        if result.success:
            result = result.x 
        else:
            raise ValueError("volume conserving eta could not be found: ", result)
    #
    return result

def mean_squared_error(etas: np.ndarray, 
                       xTilde: np.ndarray, 
                       beta: float,
                       weights: Union[None,np.ndarray],
                       volfrac: float
                       ) -> Tuple[float,np.ndarray]:
    xProj = multieta_projection(etas=etas,
                                xTilde=xTilde,
                                beta=beta,
                                weights=weights)
    return ((xProj - xTilde)**2).mean()

def volume_constraint_fun(etas: np.ndarray, 
                          xTilde: np.ndarray, 
                          beta: float,
                          weights: Union[None,np.ndarray],
                          volfrac: float
                          ) -> Tuple[float,np.ndarray]:
    xProj = multieta_projection(etas=etas,
                                xTilde=xTilde,
                                beta=beta,
                                weights=weights)
    return xProj.mean() - volfrac

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
    xProj_dx = beta * (1 - np.tanh(beta * (xTilde - eta[None,...]))**2) /\
                      (np.tanh(beta*eta[None,...])+np.tanh(beta*(1-eta[None,...])))
    
    if weights is None:
        weights = np.ones(etas.shape)/etas.shape[0]
    return np.sum( xProj_dx * weights[None,...] ,axis=-1)


def multieta_projection_deta(etas: np.ndarray,
                             xTilde: np.ndarray,
                             beta: float,
                             weights: Union[None,np.ndarray]
                             ) -> np.ndarray:
    """
    First derivative of the weighted multi-threshold Haeviside projection
    with respect to each threshold eta_n, as done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    etas : np.ndarray
        threshold values, shape (N_etas,).
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
    xPhys_deta : np.ndarray
        d(xPhys)/d(eta_n) = w_n * d(xProj_n)/d(eta_n),
        shape (..., N_etas).

    """
    # individual projections: shape (..., N_etas)
    xProj_n = (np.tanh(beta * etas[None,...]) +\
               np.tanh(beta * (xTilde[...,None] - etas[None,...]))) /\
              (np.tanh(beta * etas[None,...]) +\
               np.tanh(beta * (1 - etas[None,...])))
    # 
    return beta * weights[None,...] * ( 
            (1 - xProj_n) * np.cosh(beta * etas[None,...])**(-2) -
             np.cosh(beta * (xTilde[...,None] - etas[None,...]))**(-2) +
             xProj_n * np.cosh(beta * (1 - etas[None,...]))**(-2)) \
             / (np.tanh(beta * etas) + np.tanh(beta * (1 - etas)))[None,...]

if __name__ == "__main__":
    #
    eps = 1e-5
    #
    eta = 0.5
    beta = 10
    #
    xTilde = np.linspace(0,1.,11)[:-1]
    # finite difference
    xProj = eta_projection(eta=eta+eps,
                           xTilde=xTilde, 
                           beta=beta)
    xProj_deta = (eta_projection(eta=eta+eps, 
                                 xTilde=xTilde, 
                                 beta=beta) - \
                  eta_projection(eta=eta-eps, 
                                 xTilde=xTilde, 
                                 beta=beta))/(2*eps)
    print(xProj_deta)
    #
    xProj_deta = eta_projection_deta(xTilde=xTilde,
                                     eta=eta,
                                     xProj=xProj,
                                     beta=beta)
    print(xProj_deta)
    #
    # --- multi-eta test ---
    #
    etas = np.array([0.25, 0.5, 0.75])
    weights = np.array([1/3, 1/3, 1/3])
    #
    xPhys = multieta_projection(etas=etas, xTilde=xTilde, beta=beta, weights=weights)
    # finite difference: perturb each eta_n independently
    xPhys_deta_fd = np.zeros((xTilde.shape[0], etas.shape[0]))
    for n in range(etas.shape[0]):
        etas_p = etas.copy(); etas_p[n] += eps
        etas_m = etas.copy(); etas_m[n] -= eps
        xPhys_deta_fd[:, n] = (multieta_projection(etas=etas_p, xTilde=xTilde,
                                                    beta=beta, weights=weights) -
                                multieta_projection(etas=etas_m, xTilde=xTilde,
                                                    beta=beta, weights=weights)) / (2*eps)
    # analytic
    xPhys_deta = multieta_projection_deta(etas=etas, xTilde=xTilde,
                                          beta=beta, weights=weights)
    print("finite diff d(xPhys)/d(etas):")
    print(xPhys_deta_fd)
    print("analytic d(xPhys)/d(etas):")
    print(xPhys_deta.shape)
    print("max abs err:", np.max(np.abs(xPhys_deta - xPhys_deta_fd)))