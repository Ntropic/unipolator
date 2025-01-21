"""Compute the action of the matrix exponential onto a vector of a few columns of vectors."""
# Cythonized the _expm_multiply_simple_core function from scipy.sparse.linalg._expm_multiply
import numpy as np
from libc.math cimport exp
from scipy.linalg.cython_blas cimport zgemm, zaxpy
from unipolator.blas_functions cimport c_mat_scale
from unipolator.exp_and_log cimport copy_pointer 
from scipy.sparse.linalg._expm_multiply import LazyOperatorNormInfo, _fragment_3_1

# a function that prepares a expm_multiply_simple_core call by determining the number of iterations and matrix multiplications
cdef expm_multiply_prepare(A, m, t=1.0):
    """
    Prepare the computation of the action of the matrix exponential of A on B.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    m : number of wavevectors to use (B will be n x m)
    t : float (default=1.0)
    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)
           "Computing the Action of the Matrix Exponential,
           with an Application to Exponential Integrators."
           SIAM Journal on Scientific Computing,
           33 (2). pp. 488-511. ISSN 1064-8275
           http://eprints.ma.man.ac.uk/1591/

    .. [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)
           "Computing Matrix Functions."
           Acta Numerica,
           19. 159-208. ISSN 0962-4929
           http://eprints.ma.man.ac.uk/1451/
    """
    # Check the inputs.
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    ident = np.eye(A.shape[0], A.shape[1], dtype=A.dtype)
    n = A.shape[0]
    traceA =  A.trace()
    mu = traceA / float(n)
    A = A - mu * ident
    A_1_norm = np.linalg.norm(A, 1)  
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        norm_info = LazyOperatorNormInfo(t*A, A_1_norm=t*A_1_norm)
        m_star, s = _fragment_3_1(norm_info, m)
    return m_star, s, mu

cdef void c_mat_add_pointer(double complex *a0, double complex *b0, int nn) noexcept nogil : # A = A+B*c
    cdef int incz = 1
    cdef double complex c = 1.0 + 0.0j
    zaxpy(&nn, &c, b0, &incz, a0, &incz)

cpdef double norm_inf_complex(double complex[:, ::1] B) noexcept nogil :
    cdef Py_ssize_t i, j, n_rows, n_cols
    cdef double max_sum, row_sum
    
    n_rows = B.shape[0]
    n_cols = B.shape[1]
    
    max_sum = 0.0
    for i in range(n_rows):
        row_sum = 0.0
        for j in range(n_cols):
            row_sum += abs(B[i, j])
        if row_sum > max_sum:
            max_sum = row_sum
    return max_sum

cdef void MM_cdot_pointer_v_scaled(double complex *a0, double complex *v0, double complex *c0, double complex alpha, int n, int m) noexcept nogil :
    # matrix multiply 2 matrices A (n x n) and B (n x m)
    cdef char *ori = 'n'
    #cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    zgemm(ori, ori, &m, &n, &n, &alpha, v0, &m, a0, &n, &beta, c0, &m)
    
cpdef _expm_multiply_simple_core(double complex[:, ::1] A, double complex[:, ::1] B, double complex[:, ::1] F, double complex[:, ::1] G, double t, double mu, int m_star, int s):
    # A (nxn), B (nxm), F (nxm), G (nxm)
    # s is the number of iterations to perform, 
    # Set tolerance to machine precision
    cdef double complex *a0 = &A[0, 0]
    cdef double complex *b0 = &B[0, 0]
    cdef double complex *f0 = &F[0, 0]
    cdef double complex *g0 = &G[0, 0]
    cdef double tol = 2 ** -53 
    cdef int n = B.shape[0]
    cdef int m = B.shape[1]
    cdef int nm = n * m
    cdef double factor
    cdef double complex coeff = 1.0
    cdef double complex eta = 0.0
    copy_pointer(b0, f0, nm)  # Initialize F as B
    eta = exp(t*mu / float(s))  # Compute scaling factor eta
    
    # Perform s iterations
    for i in range(s):
        c1 = norm_inf_complex(B)            # Compute infinity norm of B
        for j in range(m_star):
            factor = float(s*(j+1))
            coeff.real = t / factor
            MM_cdot_pointer_v_scaled(a0, b0, g0, coeff, n, m)  # Update B ( B = coeff * A.dot(B) ) -> switch every iteration between B and G
            # switch pointers of B and G    #Old Approach: C = B; B = G; G = C
            B, G = G, B
            c2 = norm_inf_complex(B)        # Compute norm of updated B
            c_mat_add_pointer(f0, b0, nm)   # Update F ( F= F + B )
            if c1 + c2 <= tol * norm_inf_complex(F):  # Check if tolerance is reached
                break
            c1 = c2
        c_mat_scale(F, eta)  # Scale F
        copy_pointer(f0, b0, nm)  # Copy F to B
    return F  # Return final value of F
