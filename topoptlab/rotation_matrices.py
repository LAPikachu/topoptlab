# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

def R_2d(theta):
    """
    2D rotation matrix

    Parameters
    ----------
    theta : np.ndarray, shape (n)
        angle in radian.

    Returns
    -------
    R : np.ndarray shape (n,2,2)
        rotation matrices.

    """
    return np.column_stack((np.cos(theta),-np.sin(theta),
                            np.sin(theta),np.cos(theta)))\
          .reshape((theta.shape[0],2,2))

def dR2_dtheta(theta):
    """
    Derivative of R_2d w.r.t. theta

    Parameters
    ----------
    theta : np.ndarray, shape (n)
        angle in radian.

    Returns
    -------
    dR : np.ndarray, shape (n,2,2)
        derivatives of rotation matrices.
    """
    return np.column_stack((
        -np.sin(theta), -np.cos(theta),
         np.cos(theta), -np.sin(theta)
    )).reshape((theta.shape[0], 2, 2))
       
def Rv_2d(theta):
    """
    2D rotation matrix for tensors of 2nd order ("Voigt vectors") and 4th order 
    ("Voigt matrices"). 

    Parameters
    ----------
    theta : np.ndarray, shape (n)
        angle in radian.

    Returns
    -------
    R : np.ndarray shape (n,3,3)
        rotation matrices.

    """
    
    return np.column_stack((np.cos(theta)**2, np.sin(theta)**2, -np.sin(2*theta)/2, 
                            np.sin(theta)**2, np.cos(theta)**2, np.sin(2*theta)/2, 
                            np.sin(2*theta), -np.sin(2*theta), np.cos(2*theta)))\
          .reshape((theta.shape[0],3,3))

def dRvdtheta_2d(theta):
    """
    First order derivtive of 2D rotation matrix for tensors of 2nd order 
    ("Voigt vectors") and 4th order ("Voigt matrices"). 

    Parameters
    ----------
    theta : np.ndarray, shape (n)
        angle in radian.

    Returns
    -------
    dRvdtheta : np.ndarray shape (n,3,3)
        rotation matrices.

    """
    return np.column_stack((-np.sin(2*theta), np.sin(2*theta), -np.cos(2*theta),
                            np.sin(2*theta), -np.sin(2*theta), np.cos(2*theta), 
                            2*np.cos(2*theta), -2*np.cos(2*theta), -2*np.sin(2*theta)))\
          .reshape((theta.shape[0],3,3))

def R_3d(theta, phi):
    """
    3D rotation matrix.

    Parameters
    ----------
    theta : np.ndarray, shape (n,)
        angle in radian for rotation around z axis.
    phi : np.ndarray, shape (n,)
        angle in radian for rotation around y axis.

    Returns
    -------
    R : np.ndarray, shape (n, 3, 3)
        Rotation matrices for each (theta, phi) pair.
    """
    return np.column_stack((np.cos(theta)*np.cos(phi),-np.sin(theta),np.cos(theta)*np.sin(phi),
                            np.sin(theta)*np.cos(phi),np.cos(theta),np.sin(theta)*np.sin(phi),
                            -np.sin(phi),np.zeros(theta.shape[0]),np.cos(phi)))\
          .reshape((theta.shape[0],3,3))

def dR3_dtheta(theta, phi):
    """
    Derivative of R_3d w.r.t. theta
    """
    return np.column_stack((
        -np.sin(theta)*np.cos(phi), -np.cos(theta), -np.sin(theta)*np.sin(phi),
         np.cos(theta)*np.cos(phi), -np.sin(theta),  np.cos(theta)*np.sin(phi),
         np.zeros(theta.shape[0]),  np.zeros(theta.shape[0]),  np.zeros(theta.shape[0])
    )).reshape((theta.shape[0], 3, 3))

def dR3_dphi(theta, phi):
    """
    Derivative of R_3d w.r.t. phi
    """
    return np.column_stack((
        -np.cos(theta)*np.sin(phi),  np.zeros(theta.shape[0]),  np.cos(theta)*np.cos(phi),
        -np.sin(theta)*np.sin(phi),  np.zeros(theta.shape[0]),  np.sin(theta)*np.cos(phi),
        -np.cos(phi),                np.zeros(theta.shape[0]), -np.sin(phi)
    )).reshape((theta.shape[0], 3, 3))

def Rv_3d(theta, phi):
    return np.column_stack((
        np.cos(phi)**2*np.cos(theta)**2,
        np.sin(theta)**2,
        np.sin(phi)**2*np.cos(theta)**2,
        -np.sin(phi)*np.sin(theta)*np.cos(theta),
        np.sin(phi)*np.cos(phi)*np.cos(theta)**2,
        -np.sin(theta)*np.cos(phi)*np.cos(theta),
        np.sin(theta)**2*np.cos(phi)**2,
        np.cos(theta)**2,
        np.sin(phi)**2*np.sin(theta)**2,
        np.sin(phi)*np.sin(theta)*np.cos(theta),
        np.sin(phi)*np.sin(theta)**2*np.cos(phi),
        np.sin(theta)*np.cos(phi)*np.cos(theta),
        np.sin(phi)**2,
        np.zeros(theta.shape[0]),
        np.cos(phi)**2,
        np.zeros(theta.shape[0]),
        -np.sin(2*phi)/2,
        np.zeros(theta.shape[0]),
        -np.cos(2*phi - theta)/2 + np.cos(2*phi + theta)/2,
        np.zeros(theta.shape[0]),
        np.cos(2*phi - theta)/2 - np.cos(2*phi + theta)/2,
        np.cos(phi)*np.cos(theta),
        np.sin(theta)*np.cos(2*phi),
        -np.sin(phi)*np.cos(theta),
        -np.sin(2*phi - theta)/2 - np.sin(2*phi + theta)/2,
        np.zeros(theta.shape[0]),
        np.sin(2*phi - theta)/2 + np.sin(2*phi + theta)/2,
        -np.sin(theta)*np.cos(phi),
        np.cos(2*phi)*np.cos(theta),
        np.sin(phi)*np.sin(theta),
        2*np.sin(theta)*np.cos(phi)**2*np.cos(theta),
        -np.sin(2*theta),
        2*np.sin(phi)**2*np.sin(theta)*np.cos(theta),
        np.sin(phi)*np.cos(2*theta),
        np.cos(2*phi - 2*theta)/4 - np.cos(2*phi + 2*theta)/4,
        np.cos(phi)*np.cos(2*theta)
        )).reshape((theta.shape[0],6,6))


def dRvdtheta_3d(theta, phi):
    """
    Derivative of Rv_3d w.r.t. theta.

    Parameters
    ----------
    theta : np.ndarray, shape (n,)
        angle in radian for rotation around z axis.
    phi : np.ndarray, shape (n,)
        angle in radian for rotation around y axis.

    Returns
    -------
    dRv_dtheta : np.ndarray, shape (n,6,6)
        Derivative of engineering-Voigt rotation matrices w.r.t. theta.
    """
    return np.column_stack((
        -2*np.sin(theta)*np.cos(phi)**2*np.cos(theta),
        np.sin(2*theta),
        -2*np.sin(phi)**2*np.sin(theta)*np.cos(theta),
        -np.sin(phi)*np.cos(2*theta),
        -np.cos(2*phi - 2*theta)/4 + np.cos(2*phi + 2*theta)/4,
        -np.cos(phi)*np.cos(2*theta),
        2*np.sin(theta)*np.cos(phi)**2*np.cos(theta),
        -np.sin(2*theta),
        2*np.sin(phi)**2*np.sin(theta)*np.cos(theta),
        np.sin(phi)*np.cos(2*theta),
        np.cos(2*phi - 2*theta)/4 - np.cos(2*phi + 2*theta)/4,
        np.cos(phi)*np.cos(2*theta),
        np.zeros(theta.shape[0]),
        np.zeros(theta.shape[0]),
        np.zeros(theta.shape[0]),
        np.zeros(theta.shape[0]),
        np.zeros(theta.shape[0]),
        np.zeros(theta.shape[0]),
        -np.sin(2*phi - theta)/2 - np.sin(2*phi + theta)/2,
        np.zeros(theta.shape[0]),
        np.sin(2*phi - theta)/2 + np.sin(2*phi + theta)/2,
        -np.sin(theta)*np.cos(phi),
        np.cos(theta)*np.cos(2*phi),
        np.sin(phi)*np.sin(theta),
        np.cos(2*phi - theta)/2 - np.cos(2*phi + theta)/2,
        np.zeros(theta.shape[0]),
        -np.cos(2*phi - theta)/2 + np.cos(2*phi + theta)/2,
        -np.cos(phi)*np.cos(theta),
        -np.sin(theta)*np.cos(2*phi),
        np.sin(phi)*np.cos(theta),
        2*np.cos(phi)**2*np.cos(2*theta),
        -2*np.cos(2*theta),
        2*np.sin(phi)**2*np.cos(2*theta),
        -2*np.sin(phi)*np.sin(2*theta),
        np.sin(2*phi - 2*theta)/2 + np.sin(2*phi + 2*theta)/2,
        -2*np.sin(2*theta)*np.cos(phi)
    )).reshape((theta.shape[0], 6, 6))


def dRvdphi_3d(theta, phi):
    """
    Derivative of Rv_3d w.r.t. phi.

    Parameters
    ----------
    theta : np.ndarray, shape (n,)
        angle in radian for rotation around z axis.
    phi : np.ndarray, shape (n,)
        angle in radian for rotation around y axis.

    Returns
    -------
    dRv_dphi : np.ndarray, shape (n,6,6)
        Derivative of engineering-Voigt rotation matrices w.r.t. phi.
    """
    return np.column_stack((
        -2*np.sin(phi)*np.cos(phi)*np.cos(theta)**2,
        np.zeros(theta.shape[0]),
        2*np.sin(phi)*np.cos(phi)*np.cos(theta)**2,
        np.sin(phi - 2*theta)/4 - np.sin(phi + 2*theta)/4,
        np.cos(2*phi)*np.cos(theta)**2,
        np.sin(phi)*np.sin(theta)*np.cos(theta),
        -2*np.sin(phi)*np.sin(theta)**2*np.cos(phi),
        np.zeros(theta.shape[0]),
        2*np.sin(phi)*np.sin(theta)**2*np.cos(phi),
        np.sin(theta)*np.cos(phi)*np.cos(theta),
        np.sin(theta)**2*np.cos(2*phi),
        -np.cos(phi - 2*theta)/4 + np.cos(phi + 2*theta)/4,
        np.sin(2*phi),
        np.zeros(theta.shape[0]),
        -np.sin(2*phi),
        np.zeros(theta.shape[0]),
        -np.cos(2*phi),
        np.zeros(theta.shape[0]),
        2*(2*np.sin(phi)**2 - 1)*np.sin(theta),
        np.zeros(theta.shape[0]),
        -np.sin(2*phi - theta) + np.sin(2*phi + theta),
        -np.sin(phi)*np.cos(theta),
        -2*np.sin(2*phi)*np.sin(theta),
        -np.cos(phi)*np.cos(theta),
        2*(2*np.sin(phi)**2 - 1)*np.cos(theta),
        np.zeros(theta.shape[0]),
        np.cos(2*phi - theta) + np.cos(2*phi + theta),
        np.sin(phi)*np.sin(theta),
        -2*np.sin(2*phi)*np.cos(theta),
        np.sin(theta)*np.cos(phi),
        -np.cos(2*phi - 2*theta)/2 + np.cos(2*phi + 2*theta)/2,
        np.zeros(theta.shape[0]),
        np.cos(2*phi - 2*theta)/2 - np.cos(2*phi + 2*theta)/2,
        np.cos(phi)*np.cos(2*theta),
        -np.sin(2*phi - 2*theta)/2 + np.sin(2*phi + 2*theta)/2,
        -np.sin(phi)*np.cos(2*theta)
    )).reshape((theta.shape[0], 6, 6))
