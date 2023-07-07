#cython: language_level=3
from cython.operator import dereference
from libc.math cimport cos, sin, log
import numpy as np
#cimport numpy as np
from scipy.linalg.cython_blas cimport zgemm, zscal, zcopy
from scipy.sparse.linalg._expm_multiply import LazyOperatorNormInfo, _fragment_3_1

cdef void MM_cdot_pointer_v_scaled(double complex *a0, double complex *v0, double complex *c0, double complex alpha, int n, int m) nogil:
    # matrix multiply 2 matrices A (n x n) and B (n x m)
    cdef char *ori = 'n'
    #cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    zgemm(ori, ori, &m, &n, &n, &alpha, v0, &m, a0, &n, &beta, c0, &m)

cdef void MM_cdot_pointer_v(double complex *a0, double complex *v0, double complex *c0, int n, int m) nogil:
    # matrix multiply 2 matrices A (n x n) and B (n x m)
    cdef char *ori = 'n'
    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    zgemm(ori, ori, &m, &n, &n, &alpha, v0, &m, a0, &n, &beta, c0, &m)

cpdef void MM_cdot_v(double complex[:,::1] A, double complex[:,::1] v, double complex[:,::1] C): # C = A @ B
    # matrix multiply 2 matrices A (n x n) and B (n x m) -> C = A @ B
    cdef double complex *a0=&A[0,0]
    cdef double complex *v0=&v[0,0]
    cdef double complex *c0=&C[0,0]
    cdef int n = A.shape[0] # = v.shape[0]
    cdef int m = v.shape[1]
    #Mv_cdot_pointer(a0, v0, c0, n, m)
    cdef char *ori = 'n'
    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    zgemm(ori, ori, &m, &n, &n, &alpha, v0, &m, a0, &n, &beta, c0, &m)

cdef void DagM_M_cdot_pointer_v(double complex *a0, double complex *v0, double complex *c0, int n, int m) nogil:
    # matrix multiply 2 square matrices A (n x n) and B (n x n)
    cdef char *orientA = 'n'
    cdef char *orientB = 'c'
    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    zgemm(orientA, orientB, &m, &n, &n, &alpha, v0, &m, a0, &n, &beta, c0, &m)

cpdef void DagM_M_cdot_v(double complex[:,::1] A, double complex[:,::1] v, double complex[:,::1] C) nogil:
    # matrix multiply 2 square matrices A (n x n) and B (n x n)
    cdef double complex *a0=&A[0,0]
    cdef double complex *v0=&v[0,0]
    cdef double complex *c0=&C[0,0]
    cdef int n = A.shape[0]
    cdef int m = v.shape[1]
    DagM_M_cdot_pointer_v(a0, v0, c0, n, m)


cdef void v_exp_pointer_v(double *e0, double complex *v0, double t, int n, int m) nogil: # -i*E*amp*exp(-i*E*t)@v
    cdef double complex c
    cdef double ei
    cdef int incz = 1
    cdef int i
    ei = dereference(e0) * t
    c.real = cos(ei)
    c.imag = -sin(ei)
    zscal(&m, &c, v0, &incz)
    for i in range(1, n):
        v0 += m
        e0 += 1
        ei = dereference(e0) * t
        c.real = cos(ei)
        c.imag = -sin(ei)
        zscal(&m, &c, v0, &incz)

cdef void v_exp_v_pointer_v(double amp, double *e0, double complex *v0, double t, int n, int m) nogil: 
    cdef double complex c
    cdef double ei, f
    cdef int incz = 1
    cdef int i

    ei = dereference(e0) * t
    f = -dereference(e0) * amp
    c.real = f * sin(ei)
    c.imag = f * cos(ei)
    zscal(&m, &c, v0, &incz)
    for i in range(1, n):
        v0 += m
        e0 += 1
        ei = dereference(e0) * t
        f = -dereference(e0) * amp
        c.real = f * sin(ei)
        c.imag = f * cos(ei)
        zscal(&m, &c, v0, &incz)



##### Batch computations ############################################
cdef void MM_cdot_pointer_batch_v(double complex *a0, double complex *vi, double complex *vo, double complex *di, double complex *do, int n, int m, int nm, int batch_ind) nogil:
    # calculate all matrix products in a batch jumping by nm for batch_ind 
    cdef char *ori = 'n'
    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    # copy pointers
    cdef double complex *di2 = di 
    cdef double complex *do2 = do
    zgemm(ori, ori, &m, &n, &n, &alpha, vi, &m, a0, &n, &beta, vo, &m)
    for i in range(batch_ind):
        zgemm(ori, ori, &m, &n, &n, &alpha, di2, &m, a0, &n, &beta, do2, &m)
        di2 += nm
        do2 += nm

cdef void MM_cdot_batch_v(double complex[:,::1] A, double complex[:,::1] V_in, double complex[:,::1] V_out, double complex[:,:,::1] dV_in, double complex[:,:,::1] dV_out, int n, int m, int nm, int batch_ind) nogil:
    # calculate all matrix products in a batch jumping by nm for batch_ind 
    cdef double complex *a0=&A[0,0]
    cdef double complex *vi=&V_in[0,0]
    cdef double complex *vo=&V_out[0,0]
    cdef double complex *di=&dV_in[0,0,0]
    cdef double complex *do=&dV_out[0,0,0]
    MM_cdot_pointer_batch_v(a0, vi, vo, di, do, n, m, nm, batch_ind)
    
cdef void v_exp_v_and_v_exp_pointer_v_batch(double *e0, double complex *e0s, double complex *v0, double complex *dv0, double t, int n, int m, int batch_ind) nogil: 
    # use matrix exponential for every page in dv0
    # if batch_ind_increment=1 add a last element, copy the first element to the last element
    cdef double complex c, g
    cdef double ei
    cdef double complex f
    cdef int incz = 1
    cdef Py_ssize_t i, j
    # first calculate the array e0s once
    for i in range(n):
        #multiply the n'th element of e0 with t
        ei = t * dereference(e0)
        c.real = cos(ei)
        c.imag = -sin(ei)
        e0s[i] = c
        e0 += 1
    # now calculate the row wise exponential first of v0 and then of the pages of dv0
    for i in range(n):
        zscal(&m, e0s, v0, &incz)
        v0 += m
        e0s += 1
    # now calculate the row wise exponential of dv0
    for i in range(batch_ind): # page wise
        e0s -= n
        for j in range(n):
            zscal(&m, e0s, dv0, &incz)
            dv0 += m
            e0s += 1

cpdef void v_exp_v_and_v_exp_v_batch(double[::1] E, double complex[::1] Es, double complex[:,::1] V, double complex[:,:,::1] dV, double t, int batch_ind):
    cdef double *e0 = &E[0]
    cdef double complex *e0s = &Es[0]
    cdef double complex *v0 = &V[0,0]
    cdef double complex *dv0 = &dV[0,0,0]
    cdef int m = V.shape[1]
    cdef int n = V.shape[0]
    v_exp_v_and_v_exp_pointer_v_batch(e0, e0s, v0, dv0, t, n, m, batch_ind)

cdef void v_exp_v_and_v_exp_pointer_v_batch_increment(double amp, double *e0, double complex *e0s, double complex *v0, double complex *dv0, double t, int n, int m, int mn, int batch_ind) nogil: 
    # use matrix exponential for every page in dv0
    # if batch_ind_increment=1 add a last element, copy the first element to the last element
    cdef double complex c, g
    cdef double ei
    cdef double complex f
    cdef int incz = 1
    cdef Py_ssize_t i, j
    cdef int k = mn * batch_ind

    # first calculate the array e0s once
    for i in range(n):
        #multiply the n'th element of e0 with t
        ei = t * dereference(e0)
        c.real = cos(ei)
        c.imag = -sin(ei)
        e0s[i] = c
        e0 += 1
    # now calculate the row wise exponential first of v0 and then of the pages of dv0
    for i in range(n):
        zscal(&m, e0s, v0, &incz)
        v0 += m
        e0s += 1
    # precopy -> then rescale later
    v0 -= mn
    dv0 += k
    zcopy(&mn, v0, &incz, dv0, &incz)
    dv0 -= k
    # now calculate the row wise exponential of dv0
    for i in range(batch_ind): # page wise
        e0s -= n
        for j in range(n):
            zscal(&m, e0s, dv0, &incz)
            dv0 += m
            e0s += 1
    # scale the precopied elements
    e0 -= n
    f.real = 0.0
    for i in range(n): # copy and scale by f
        f.imag = - amp * dereference(e0)
        # complex multiplication
        zscal(&m, &f, dv0, &incz)
        dv0 += m
        e0 += 1

cpdef void v_exp_v_and_v_exp_v_batch_increment(double amp, double[::1] E, double complex[::1] Es, double complex[:,::1] V, double complex[:,:,::1] dV, double t, int batch_ind):
    cdef double *e0 = &E[0]
    cdef double complex *e0s = &Es[0]
    cdef double complex *v0 = &V[0,0]
    cdef double complex *dv0 = &dV[0,0,0]
    cdef int m = V.shape[1]
    cdef int n = V.shape[0]
    cdef int mn = m * n
    v_exp_v_and_v_exp_pointer_v_batch_increment(amp, e0, e0s, v0, dv0, t, n, m, mn, batch_ind)

#### Not really blas functions, but added here nonetheless
cdef double norm_inf_complex( double complex[:, ::1] B, int d, int m) nogil:
    cdef Py_ssize_t i, j, n_rows, n_cols
    cdef double row_sum
    cdef double max_sum = 0.0
    for i in range(d):
        row_sum = 0.0
        for j in range(m):
            row_sum += abs(B[i, j])
        if row_sum >  max_sum:
                max_sum = row_sum
    return max_sum

cpdef expm_multiply_prepare(double complex[:,::1] A0, int m, double tol):
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
    A = np.array(A0)
    #print(A.shape)
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    ident = np.eye(A.shape[0], A.shape[1], dtype=np.complex128)
    n = A.shape[0]
    traceA =  np.array(A).trace()
    mu = traceA / float(n)
    A = A - mu * ident
    A_1_norm = np.linalg.norm(A, 1)  
    if A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(A, A_1_norm=A_1_norm)
        m_star, s = _fragment_3_1(norm_info, m, tol)
    return m_star, s, np.real(mu)