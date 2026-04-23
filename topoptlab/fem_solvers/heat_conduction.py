# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.sparse import csc_array

from topoptlab.fem import (FEM_Phys, assemble_matrix, apply_bc,
                           create_matrixinds)
from topoptlab.elements.bilinear_quadrilateral import (
    create_edofMat as create_edofMat2d)
from topoptlab.elements.trilinear_hexahedron import (
    create_edofMat as create_edofMat3d)
from topoptlab.elements.poisson_2d import lk_poisson_2d
from topoptlab.elements.poisson_3d import lk_poisson_3d
from topoptlab.material_interpolation import simp, simp_dx
from topoptlab.solve_linsystem import solve_lin


class HeatConduction(FEM_Phys):
    """
    Stationary heat conduction solver.

    Solves the Poisson problem

        -div( k(x) grad T ) = f    in Omega
                          T = 0    on Gamma_D
                          

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction. 2D problem if None.
    bc : callable
        function that returns (T, f, fixed, free, springs) given
        (nelx, nely, nelz, ndof).
    l : float or np.ndarray
        element side length(s).
    lin_solver : str
        linear solver identifier passed to solve_lin.
    preconditioner : str or None
        preconditioner identifier passed to solve_lin.
    assembly_mode : str
        "full" or "lower" triangular assembly.
    """

    def __init__(self,
                 nelx: int,
                 nely: int,
                 bc: Callable,
                 nelz: Union[int, None] = None,
                 kmax: float = 1.0,
                 kmin: float = 1e-9,
                 penal: float = 3.0,
                 l: Union[float, np.ndarray] = 1.0,
                 lin_solver: str = "cvxopt-cholmod",
                 preconditioner: Union[str, None] = None,
                 assembly_mode: str = "lower",
                 **kwargs: Any) -> None:
        #
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.kmax = kmax
        self.kmin = kmin
        self.penal = penal
        self.lin_solver = lin_solver
        self.preconditioner = preconditioner
        self.assembly_mode = assembly_mode
        #
        self.ndim = 2 if nelz is None else 3
        #
        if isinstance(l, float):
            l = np.full(self.ndim, l)
        self.l = l
        #
        if self.ndim == 2:
            self._lk = lk_poisson_2d
            self._create_edofMat = create_edofMat2d
        else:
            self._lk = lk_poisson_3d
            self._create_edofMat = create_edofMat3d
        #
        n = np.array([nelx, nely] if nelz is None else [nelx, nely, nelz])
        self.nel = int(np.prod(n))
        # element stiffness matrix (unit conductivity)
        self._KE0 = self._lk(k=1.0, l=self.l)
        # nodal dofs per node
        nd_ndof = int(self._KE0.shape[0] / 2**self.ndim)
        self.ndof = int(np.prod(n + 1)) * nd_ndof
        # element DOF matrix
        self.edofMat, *_ = self._create_edofMat(nelx=nelx,
                                                 nely=nely,
                                                 nelz=nelz,
                                                 nnode_dof=nd_ndof)
        # sparse index arrays
        self.iK, self.jK = create_matrixinds(self.edofMat,
                                             mode=assembly_mode)
        if assembly_mode == "lower":
            assm = np.column_stack(np.tril_indices_from(self._KE0))
            self._assm_indcs = assm[np.lexsort((assm[:, 0], assm[:, 1]))]
        else:
            self._assm_indcs = None
        # boundary conditions
        self.T, self.f, self.fixed, self.free, self._springs = bc(
            nelx=nelx, nely=nely, nelz=nelz, ndof=self.ndof)
        # factorization / preconditioner cache (filled after first solve)
        self._fact = None
        self._precond = None
        return 
    
    def assemble_system(self,
                        Kes: np.ndarray,
                        **kwargs: Any) -> csc_array:
        """
        Assemble the global conductivity matrix K for densities xPhys.

        Parameters
        ----------
        Kes : np.ndarray
            element stiffness matrices shape (nel,m,m).

        Returns
        -------
        K : scipy.sparse.csc_array, shape (ndof, ndof)
            assembled global conductivity matrix.
        """
        if self.assembly_mode == "full":
            sK = Kes.reshape(np.prod(Kes.shape))
        else:
            sK = Kes[:, self._assm_indcs[:, 0],
                     self._assm_indcs[:, 1]].reshape(
                self.nel * int(self._KE0.shape[-1] / 2
                               * (self._KE0.shape[-1] + 1)))
        return assemble_matrix(sK=sK, iK=self.iK, jK=self.jK,
                               ndof=self.ndof, solver=self.lin_solver,
                               springs=self._springs)

    def coupling(self, **kwargs: Any) -> None:
        """
        Not used for single-physics heat conduction.
        """
        return None

    def to_interpolation(self,
                         xPhys: np.ndarray,
                         **kwargs: Any) -> np.ndarray:
        """
        SIMP conductivity scale factors for given densities.

        Parameters
        ----------
        xPhys : np.ndarray, shape (nel,) or (nel, 1)
            physical element densities.

        Returns
        -------
        scale : np.ndarray
            SIMP scale factors.
        """
        return simp(xPhys=xPhys, eps=self.kmin / self.kmax,
                    penal=self.penal)

    def _linsolve(self,
                  K: csc_array,
                  **kwargs: Any) -> np.ndarray:
        """
        Solve K[free,free] T[free] = f[free].

        Parameters
        ----------
        K : scipy.sparse.csc_array
            assembled, unreduced conductivity matrix.

        Returns
        -------
        T : np.ndarray, shape (ndof, 1)
            temperature field (full, including fixed DOFs at zero).
        """
        K_free = apply_bc(K=K, solver=self.lin_solver,
                          free=self.free, fixed=self.fixed)
        self.T[self.free], self._fact, self._precond = solve_lin(
            K=K_free,
            rhs=self.f[self.free],
            solver=self.lin_solver,
            preconditioner=self.preconditioner,
            factorization=self._fact,
            P=self._precond)
        return self.T

    def _nonlin_solve(self, **kwargs: Any) -> None:
        """
        Not applicable — heat conduction is linear.
        """
        raise NotImplementedError("Heat conduction is a linear problem.")

    def solve(self,
              xPhys: np.ndarray,
              **kwargs: Any) -> np.ndarray:
        """
        Assemble and solve for the temperature field.

        Parameters
        ----------
        xPhys : np.ndarray, shape (nel,) or (nel, 1)
            physical element densities.

        Returns
        -------
        T : np.ndarray, shape (ndof, 1)
            temperature field.
        """
        K = self.assemble_system(xPhys=xPhys)
        return self._linsolve(K=K)

    def time_evolve(self, **kwargs: Any) -> None:
        """
        Not applicable — stationary problem.
        """
        raise NotImplementedError("Not yet added.")
