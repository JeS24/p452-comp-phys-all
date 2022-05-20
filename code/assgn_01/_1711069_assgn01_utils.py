import warnings
import numpy as np
import plotly.graph_objects as go

# Gassian Elimination / Gauss-Jordan
def gauss_elim(A, b):
    """
    Solves Ax = b linear systems using Gassian Elimination

    Parameters
    ----------
    A: numpy.ndarray
        Contains coefficients of the variables (Matrix, A)
    b: numpy.ndarray
        Contains the constants on RHS (Vector, b)
    
    Returns
    -------
    bool
        Whether the process converged within ``iter_lim``
    numpy.ndarray
        Obtained solution
    float
        Error in the obtained solution
    
    """
    # Prepping the Augmented Matrix
    aug_mat = np.concatenate((A, np.reshape(b, (-1, 1))), axis=1)
    # Convergence Flag
    CONV_FLAG = True
    # Position of leading nonzero, nondiagonal-element in a row / pivot
    lead = 0
    
    # aug_mat.shape[0] == No. of rows
    # aug_mat[0].shape or aug_mat.shape[1] == Number of Columns
    rowCount = aug_mat.shape[0]
    columnCount = aug_mat.shape[1]

    for r in range(rowCount):
        if lead >= columnCount:
            CONV_FLAG = False
            break
        i = r

        # Finding the pivot in a column
        while aug_mat[i][lead] == 0:
            i += 1
            if i == rowCount:
                i = r
                lead += 1
                if columnCount == lead:
                    CONV_FLAG = False
                    break

        aug_mat[i], aug_mat[r] = aug_mat[r], aug_mat[i] # Swapping rows
        lv = aug_mat[r][lead]
        aug_mat[r] = [mrx / float(lv) for mrx in aug_mat[r]]
        for i in range(rowCount):
            if i != r:
                lv = aug_mat[i][lead]         
                aug_mat[i] = [iv - lv*rv for rv,iv in zip(aug_mat[r], aug_mat[i])]
        lead += 1
    
    if not CONV_FLAG:
        raise Exception(f"Solution did not converge.")
    
    # Returning convergence flag, solution and associated error
    return CONV_FLAG, aug_mat[:, -1], A @ aug_mat[:, -1] - b


# Jacobi Iteration
def jacobi_iter(A, b, init_val, iter_lim=100, tol=1e-8, info=False):
    """
    Solves Ax = b linear systems using Jacobi Iteration

    Parameters
    ----------
    A: numpy.ndarray
        Contains coefficients of the variables (Matrix, A)
    b: numpy.ndarray
        Contains the constants on RHS (Vector, b)
    init_val: numpy.ndarray
        Contains an initial guess for x
    iter_lim: int
        Maximum number of iterations
        Defaults to ``100``
    tol: float
        Tolerance value
        Defaults to ``1e-8``
    info: bool
        Whether to store residue & iteration steps
        Defaults to ``False``
    
    Returns
    -------
    bool
        Whether the process converged within ``iter_lim``
    numpy.ndarray
        Obtained solution
    float
        Error in the obtained solution
        
    Optionally Returns
    ------------------
    meta: list
        List containing iteration steps & residue per step
    
    """
    CONV_FLAG = False # Convergence Flag
    var = init_val # Vector, X
    
    # To store residue & iteration steps
    meta = []
    
    for i in range(iter_lim):
        var_new = np.zeros_like(var) # stores updated values of all variables (Vector, X)

        for j in range(A.shape[0]):
            # Matrix Multiplying all elements, before A's diagonal (in a row) with all corresponding vars (in Vector, X)
            d = np.dot(A[j, :j], var[:j])
            # Matrix Multiplying all elements, after A's diagonal (in a row) with all corresponding vars (in Vector, X)
            r = np.dot(A[j, j + 1:], var[j + 1:])
            # Updating values of vars
            var_new[j] = (b[j] - d - r) / A[j, j]

        meta.append([i, np.linalg.norm(var - var_new)]) # Storing iteration step and residue

        # Checks, how close the updated values are, to the previous iteration's values and breaks the loop, if close enough (defined by "tol")
        if np.allclose(var, var_new, atol=tol, rtol=0.):
            CONV_FLAG = True
            break

        var = var_new # Storing the new solution

    # If solution is not obtained (no convergence), after iter_lim iterations | Note that, this "else" block belongs to the previous "for" statement and not any "if" statement
    else:
        CONV_FLAG = False
        
    if not CONV_FLAG:
        raise Exception(f"Solution did not converge, after the specified limit of {iter_lim} iterations.")
    
    # Returning convergence flag, metadata, solution and associated error
    if info:
        return CONV_FLAG, meta, var, A @ var - b
    
    # Returning convergence flag, solution and associated error
    return CONV_FLAG, var, A @ var - b


# Gauss-Seidel Iteration
def gauss_seidel(A, b, init_val, iter_lim=100, tol=1e-8, info=False):
    """
    Solves Ax = b linear systems using Gauss-Seidel Iteration

    Parameters
    ----------
    A: numpy.ndarray
        Contains coefficients of the variables (Matrix, A)
    b: numpy.ndarray
        Contains the constants on RHS (Vector, b)
    init_val: numpy.ndarray
        Contains an initial guess for x
    iter_lim: int
        Maximum number of iterations
        Defaults to ``100``
    tol: float
        Tolerance value
        Defaults to ``1e-8``
    info: bool
        Whether to store residue & iteration steps
        Defaults to ``False``
    
    Returns
    -------
    bool
        Whether the process converged within ``iter_lim``
    numpy.ndarray
        Obtained solution
    float
        Error in the obtained solution
    
    Optionally Returns
    ------------------
    meta: list
        List containing iteration steps & residue per step
    
    """
    CONV_FLAG = False # Convergence Flag
    var = init_val # Vector, X
    
    # To store residue & iteration steps
    meta = []
    
    for i in range(iter_lim):
        var_new = np.zeros_like(var) # stores updated values of all variables (Vector, X)

        for j in range(A.shape[0]):
            # Matrix Multiplying all elements, before A's diagonal (in a row) with all corresponding vars (in Vector, X), that now have updated values
            l = np.dot(A[j, :j], var_new[:j]) # Note, the only change from jacobi_iter() is changing "var" to "var_new"
            # Matrix Multiplying all elements, after A's diagonal (in a row) with all corresponding vars (in Vector, X), that do not have updated values yet
            u = np.dot(A[j, j + 1:], var[j + 1:])
            # Updating values of vars
            var_new[j] = (b[j] - l - u) / A[j, j]

        meta.append([i, np.linalg.norm(var - var_new)]) # Storing iteration step and residue

        # Checks, how close the updated values are, to the previous iteration's values and breaks the loop, if close enough (defined by "tol")
        if np.allclose(var, var_new, atol=tol, rtol=0.):
            CONV_FLAG = True
            break

        var = var_new # Storing the new solution

    # If solution is not obtained (no convergence), after iter_lim iterations | Note that, this "else" block belongs to the previous "for" statement and not any "if" statement
    else:
        CONV_FLAG = False
    
    if not CONV_FLAG:
        raise Exception(f"Solution did not converge, after the specified limit of {iter_lim} iterations.")
    
    # Returning convergence flag, metadata, solution and associated error
    if info:
        return CONV_FLAG, meta, var, A @ var - b
    
    # Returning convergence flag, solution and associated error
    return CONV_FLAG, var, A @ var - b


# Forward & Backward Substitution
def forward_sub(A, b, init_val, iter_lim=100, tol=1e-8):
    """
    Solves Ax = b linear systems using Forward Substitutions

    Parameters
    ----------
    A: numpy.ndarray
        Contains coefficients of the variables (Matrix, A)
    b: numpy.ndarray
        Contains the constants on RHS (Vector, b)
    init_val: numpy.ndarray
        Contains an initial guess for x
    iter_lim: int
        Maximum number of iterations
        Defaults to ``100``
    tol: float
        Tolerance value
        Defaults to ``1e-8``
    
    Returns
    -------
    bool
        Whether the process converged within ``iter_lim``
    numpy.ndarray
        Obtained solution
    float
        Error in the obtained solution
    
    """
    CONV_FLAG = False # Convergence Flag
    var = init_val # Vector, X
    
    for i in range(iter_lim):
        var_new = np.zeros_like(var) # stores updated values of all variables (Vector, X)

        for j in range(A.shape[0]):
            # Matrix Multiplying all elements, after A's diagonal (in a row) with all corresponding vars (in Vector, X)
            l = np.dot(A[j, :j], var[:j])
            # Updating values of vars
            var_new[j] = (b[j] - l) / A[j, j]

        # Checks, how close the updated values are, to the previous iteration's values and breaks the loop, if close enough (defined by "tol")
        if np.allclose(var, var_new, atol=tol, rtol=0.):
            CONV_FLAG = True
            break

        var = var_new # Storing the new solution
    # If solution is not obtained (no convergence), after iter_lim iterations | Note that, this "else" block belongs to the previous "for" statement and not any "if" statement
    else:
        CONV_FLAG = False
        
    if not CONV_FLAG:
        raise Exception(f"Solution did not converge, after the specified limit of {iter_lim} iterations.")

    # Returning convergence flag, solution and associated error
    return CONV_FLAG, var, A @ var - b


def back_sub(A, b, init_val, iter_lim=100, tol=1e-8):
    """
    Solves Ax = b linear systems using Backward Substitutions

    Parameters
    ----------
    A: numpy.ndarray
        Contains coefficients of the variables (Matrix, A)
    b: numpy.ndarray
        Contains the constants on RHS (Vector, b)
    init_val: numpy.ndarray
        Contains an initial guess for x
    iter_lim: int
        Maximum number of iterations
        Defaults to ``100``
    tol: float
        Tolerance value
        Defaults to ``1e-8``
    
    Returns
    -------
    bool
        Whether the process converged within ``iter_lim``
    numpy.ndarray
        Obtained solution
    float
        Error in the obtained solution
    
    """
    CONV_FLAG = False # Convergence Flag
    var = init_val # Vector, X
    
    for i in range(iter_lim):
        var_new = np.zeros_like(var) # stores updated values of all variables (Vector, X)

        for j in range(A.shape[0]):
            # Matrix Multiplying all elements, after A's diagonal (in a row) with all corresponding vars (in Vector, X)
            u = np.dot(A[j, j + 1:], var[j + 1:])
            # Updating values of vars
            var_new[j] = (b[j] - u) / A[j, j]
        
        # Checks, how close the updated values are, to the previous iteration's values and breaks the loop, if close enough (defined by "tol")
        if np.allclose(var, var_new, atol=tol, rtol=0.):
            CONV_FLAG = True
            break

        var = var_new # Storing the new solution
    # If solution is not obtained (no convergence), after iter_lim iterations | Note that, this "else" block belongs to the previous "for" statement and not any "if" statement
    else:
        CONV_FLAG = False
        
    if not CONV_FLAG:
        raise Exception(f"Solution did not converge, after the specified limit of {iter_lim} iterations.")

    # Returning convergence flag, solution and associated error
    return CONV_FLAG, var, A @ var - b


# LU Decomposition
def LU(A):
    """
    Returns LU Decomposition of A

    Parameters
    ----------
    A: numpy.ndarray
        Matrix, to be decomposed

    Returns
    -------
    L: numpy.ndarray
        Lower Triangular Factor of A
    U: numpy.ndarray
        Upper Triangular Factor of A

    """
    U = A.copy()
    L = np.identity(A.shape[0], dtype=float)

    for i in range(A.shape[0]):
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i] # :, newaxis helps to turn factor into a 2D array of shape (N, 1) / column vector

    return L, U


def LUSolver(A, b, init_val, iter_lim=100, tol=1e-8):
    """
    Solves Ax = b linear systems using LU Decomposition
    This in turn uses Forward and Backward Substitutions

    Parameters
    ----------
    A: numpy.ndarray
        Contains coefficients of the variables (Matrix, A)
    b: numpy.ndarray
        Contains the constants on RHS (Vector, b)
    init_val: numpy.ndarray
        Contains an initial guess for x
    iter_lim: int
        Maximum number of iterations
        Defaults to ``100``
    tol: float
        Tolerance value
        Defaults to ``1e-8``
    
    Returns
    -------
    bool
        Whether the process converged within ``iter_lim``
    numpy.ndarray
        Obtained solution
    float
        Error in the obtained solution
    
    """
    L, U = LU(A)

    CONV_FLAG_FS, y, ERR_FS = forward_sub(L, b, init_val, iter_lim, tol)
    if CONV_FLAG_FS:
        CONV_FLAG_BS, x, ERR_BS = back_sub(U, y, init_val, iter_lim, tol)
    else:
        raise Exception(f"Solution did not converge, after the specified limit of {iter_lim} iterations.")

    if not CONV_FLAG_BS:
        raise Exception(f"Solution did not converge, after the specified limit of {iter_lim} iterations.")

    return CONV_FLAG_BS, x, ERR_BS


# Conjugate Gradient
def conj_grad(A, b, init_val, iter_lim=100, tol=1e-8, info=False):
    """
    Solves Ax = b linear systems using Conjugate Gradients

    Parameters
    ----------
    A: numpy.ndarray
        Contains coefficients of the variables (Matrix, A)
    b: numpy.ndarray
        Contains the constants on RHS (Vector, b)
    init_val: numpy.ndarray
        Contains an initial guess for x
    iter_lim: int
        Maximum number of iterations
        Defaults to ``100``
    tol: float
        Tolerance value
        Defaults to ``1e-8``
    info: bool
        Whether to store residue & iteration steps
        Defaults to ``False``
    
    Returns
    -------
    bool
        Whether the process converged within ``iter_lim``
    numpy.ndarray
        Obtained solution
    float
        Error in the obtained solution
        
    Optionally Returns
    ------------------
    meta: list
        List containing iteration steps & residue per step
    
    """
    CONV_FLAG = False # Convergence Flag

    if isinstance(A, OTFArray):
        A = A.dense
    
    # Conjugate Gradient only works with symmetric matrices
    if not np.allclose(A, A.T):
        raise ValueError("Input matrix is not symmetric.")

    xk = init_val
    rk = A @ xk - b
    pk = -rk
    # Stores all x vectors
    curve_x = [xk]

    # To store residue & iteration steps
    meta = []

    for i in range(iter_lim):
        apk = A @ pk
        rkrk = rk @ rk

        alpha = rkrk / (pk @ apk)
        xk += alpha * pk
        rk += alpha * apk

        curve_x.append(xk)
        rk_norm = np.linalg.norm(rk)

        meta.append([i, rk_norm]) # Storing iteration step and residue
        
        beta = (rk @ rk) / rkrk
        pk = -rk + beta * pk

        # Checks, how close the updated values are, to the previous iteration's values and breaks the loop, if close enough (defined by "tol")
        if rk_norm <= tol:
            CONV_FLAG = True
            break

    # If solution is not obtained (no convergence), after iter_lim iterations | Note that, this "else" block belongs to the previous "for" statement and not any "if" statement
    else:
        CONV_FLAG = False

    if not CONV_FLAG:
        raise Exception(f"Solution did not converge, after the specified limit of {iter_lim} iterations.")

    # Returning convergence flag, metadata, solution and associated error
    if info:
        return CONV_FLAG, meta, np.array(curve_x)[-1], A @ np.array(curve_x)[-1] - b

    # Returning convergence flag, solution and associated error
    return CONV_FLAG, np.array(curve_x)[-1], A @ np.array(curve_x)[-1] - b


# Matrix Inversion
def inverse(A, method="jacobi", iter_lim=100, tol=1e-8):
    """
    Finds the inverse of the input matrix

    Parameters
    ----------
    A: numpy.ndarray
        Input matrix
    method: str
        Method to use to calculate inverse
    iter_lim: int
        Maximum number of iterations
        Defaults to ``100``
    tol: float
        Tolerance value
        Defaults to ``1e-8``
    
    Returns
    -------
    numpy.ndarray
        Inverse of the input matrix

    """
    METHODS = {
        "jacobi": jacobi_iter,
        "gauss_seidel": gauss_seidel,
        "conj_grad": conj_grad
    }

    if method not in METHODS:
        raise NotImplementedError(f"{method} not found or not implemented!")

    B = np.identity(A.shape[0])
    init_val = np.zeros(A.shape[0])
    cols = []

    meta_overall = []
    for b in B:
        _, meta, var, _ = METHODS[method](A, b, init_val, iter_lim, tol, info=True)
        meta_overall.append(meta)
        cols.append(var)

    return meta_overall, np.c_[cols].T


# To visualize convergence
def viz_conv(meta_overall, name):
    """
    Visualizes convergence of matrix inversion methods

    Parameters
    ----------
    meta_overall: list
        List of lists that contains iterations steps and residue for each column
    name: str
        Name of the method
    
    Returns
    -------
    fig: plotly.graph_objs._figure.Figure
        Inverse of the input matrix

    """
    fig = go.Figure()

    for idx, meta_col in enumerate(meta_overall):
        meta_col = np.array(meta_col)

        fig.add_trace(go.Scatter(
            x=meta_col[:, 0], y=meta_col[:, 1],
            name=f'Column {idx}'
        ))

    fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)
    fig.add_hline(y=1e-4, line_width=1, line_dash="dash", line_color="red") # Tolerance
    fig.update_layout(title=f"Convergence plots for {name}", xaxis_title="Iteration", yaxis_title="Residue")

    return fig


# Matrix class that allows on-the-fly data access
# WIP
class OTFArray:
    def __init__(self, shape, logic):
        self.shape = shape
        self.logic = logic
        try:
            self.dense = np.fromfunction(np.vectorize(logic), shape)
        except MemoryError as e:
            print("Warning: Not generating dense array as available memory is insufficient.")

    def __getitem__(self, idx, **kwargs):
        if idx[0] >= self.shape[0] or idx[1] >= self.shape[1]:
            warnings.warn("One or both of the input indices is out of bounds. Ignore this warning if the matrix is periodic.")

        return self.logic(*idx, **kwargs)

    def __str__(self):
        st = "OTFArray([\n"
        I, J = self.shape
        
        
        for i in range(4):
            for j in range(4):
                st += f"\t{self.__getitem__((i, j))}"
            st += "\t......" * 4
            st += "\n"

        st += "\t.\t.\t.\t.\n" * 3

        for i in range(I - 4, I):
            st += "\t......" * 4
            for j in range(J - 4, J):
                st += f"\t{self.__getitem__((i, j))}"
            st += "\n"

        st += "\n])"

        return f"{st}"

    def __repr__(self):
        return f"{type(self)}, {self.shape}, {self.logic}\n" + self.__str__()

    def __matmul__(self, other):
        """
        Matrix Multiplication
        ``other`` needs to be a numpy.ndarray

        """
        if self.shape[1] != other.shape[0]:
            raise ValueError("These matrices are not compatible.")
        
        I, J = self.shape
        
        # With matrices - just to enable full usage of matmul, not space efficient
        if len(other.shape) == 2:
            cols = []
            for col in other.T:
                for i in range(I):
                    for j in range(J):
                        dotval[i] += self.__getitem__((i, j)) * other[j]
                        
                cols.append(dotval)
            
            return np.c_[cols]

        # With vectors
        dotval = np.zeros_like(other)
        for i in range(I):
            for j in range(J):
                dotval[i] += self.__getitem__((i, j)) * other[j]

        return dotval

    @property
    def T(self):
        """
        Returns the transpose of the matrix.

        """
        return self.dense.T
