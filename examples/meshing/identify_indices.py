from typing import Union

import numpy as np

from topoptlab.geometries import elids_in_mask, nodeids_in_mask, cube_mask

def demo_nodeids(nelx : int,
                 nely : int, 
                 nelz : Union[None,int]) -> None:
    """
    Short demo case to print all nodes located inside the mesh and not at any
    border. Compare with the mesh displayed in the tutorial for meshing. 
    
    Node coordinates by convention start from -0.5*l_i (sidelength of element 
    in the respective dimension). 
    
    Node IDs can be connected to degrees of freedom (dof) by multiplying the 
    node ID with the number of dofs per node (scalar field 1, vector field 
    typically ndim).
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.
    
    Returns
    -------
    None : 

    """
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    #
    ndids = np.arange(np.prod( np.array([nelx,nely,nelz][:ndim])+1))
    #
    ids = nodeids_in_mask(node_id = ndids, 
                          spatial_mask_fnc = cube_mask, 
                          mask_kw = {"low": np.zeros(ndim), 
                                     "upp": np.ones(ndim)*\
                                            np.array([nelx,nely,nelz][:ndim])-1}, 
                          nelx=nelx, 
                          nely=nely, 
                          nelz=nelz)
    print(ids)
    return

def demo_elids(nelx : int,
               nely : int, 
               nelz : Union[None,int]) -> None:
    """
    Short demo case to print all elements located inside the mesh and not at any
    border. Compare with the mesh displayed in the tutorial for meshing. 
    
    Element coordinates by convention start from 0. (sidelength of element 
    in the respective dimension). 
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.
    
    Returns
    -------
    None : 

    """
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    elids = np.arange(np.prod( np.array([nelx,nely,nelz][:ndim])))
    #
    ids = elids_in_mask(el = elids, 
                        spatial_mask_fnc = cube_mask, 
                        mask_kw = {"low": np.ones(ndim), 
                                   "upp": np.ones(ndim)*\
                                          np.array([nelx,nely,nelz][:ndim])-1-1e-9}, 
                        nelx=nelx, 
                        nely=nely, 
                        nelz=nelz)
    print(ids)
    return

if __name__ == "__main__":
    #
    nelx,nely,nelz = 4,3,2
    #
    demo_nodeids(nelx=nelx, 
                 nely=nely,
                 nelz=None)
    #
    demo_nodeids(nelx=nelx, 
                 nely=nely,
                 nelz=nelz)
    #
    nelx,nely,nelz = 4,3,3
    #
    demo_elids(nelx=nelx, 
               nely=nely,
               nelz=None)
    #
    demo_elids(nelx=nelx, 
               nely=nely,
               nelz=nelz)