#cython: language_level=3
from cython.operator import dereference
from libc.math cimport cos, sin, log, atan2, sqrt, fabs
import numpy as np
#cimport numpy as np
from scipy.linalg import schur
from scipy.linalg.cython_blas cimport zgemm, zscal, zdscal, zaxpy, zcopy, dcopy
from scipy.linalg.cython_lapack cimport zheevd
from .blas_functions cimport MM_cdot, DagM_M_cdot, M_DagM_cdot, MM_cdot_pointer, DagM_M_cdot_pointer, c_eigh_lapack

##### Basic complex number functions to avoid using complex.h #########################
cpdef double creal(double complex dc) nogil:
    cdef double complex* dcptr = &dc
    return (<double *>dcptr)[0]

cpdef double cimag(double complex dc) nogil:
    cdef double complex* dcptr = &dc
    return (<double *>dcptr)[1]

cpdef double complex cconj(double complex dc) nogil:
    dc.imag = -dc.imag 
    #a =  (<double *>dcptr)[0] -1j*(<double *>dcptr)[1]
    return dc

cpdef void cdiag(double complex[:,::1] A, double complex[::1] diagA) nogil:
    cdef int n = A.shape[0]
    cdef int i
    for i in range(n):
        diagA[i] = A[i,i]

cpdef void Conj_mat_copy(double complex[:,::1] A, double complex[:,::1] ConjA) nogil:
    cdef int n = A.shape[0]
    cdef int i,j
    for i in range(n):
        for j in range(n):
            ConjA[i,j].real =  A[i,j].real
            ConjA[i,j].imag = -A[i,j].imag

cpdef void Conj_mat(double complex[:,::1] A) nogil:
    cdef int n = A.shape[0]
    cdef int i,j
    for i in range(n):
        for j in range(n):
            A[i,j].imag = -A[i,j].imag

cpdef void Conj_vec_copy(double complex[::1] A, double complex[::1] ConjA) nogil:
    cdef int n = A.shape[0]
    cdef int i
    for i in range(n):
        ConjA[i].real =  A[i].real
        ConjA[i].imag = -A[i].imag

cpdef void Conj_vec(double complex[::1] A) nogil:
    cdef int n = A.shape[0]
    cdef int i
    for i in range(n):
        A[i].imag = -A[i].imag


cpdef Dag(double complex[:,::1] A):
    cdef int n = A.shape[0]
    cdef int i, j
    cdef double complex[:,::1] DagA = np.empty([n,n], dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            DagA[i,j].real =  A[j,i].real
            DagA[i,j].imag = -A[j,i].imag
    return DagA

cpdef void Dag_fast(double complex[:,::1] A, double complex[:,::1] DagA) nogil:
    cdef int n = A.shape[0]
    cdef int i, j
    for i in range(n):
        for j in range(n):
            DagA[i,j].real =  A[j,i].real
            DagA[i,j].imag = -A[j,i].imag

cdef double complex conj(double complex A) nogil:
    cdef double complex conjA
    conjA.real = A.real
    conjA.imag = -A.imag
    return conjA

cpdef double abs_2(double complex A) nogil: #c_dagc
    cdef double absA_2
    absA_2 = A.real * A.real + A.imag * A.imag
    return absA_2

##### Add Commutator function !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#### Vector exponential/logarithm times matrix

# Old skripts for this
cpdef void copy_v_exp_blas(double[::1] E, double complex[:,::1] V, double complex[:,::1] A, double t, int n) nogil:
    #cdef int n = V.shape[0]
    cdef double complex c
    cdef double ei
    cdef double complex *vi0
    cdef int incz = 1
    cdef int i
    A[:,:] = V
    for i in range(n):
        ei = E[i]*t
        c = cos(ei)-1j*sin(ei) #cexp(ei)
        vi0=&A[i,0]
        zscal(&n, &c, vi0, &incz)
cpdef void v_exp_blas(double[::1] E, double complex[:,::1] V, double t) nogil:
    cdef int n = V.shape[0]
    cdef double complex c
    cdef double ei
    cdef double complex *vi0
    cdef int incz = 1
    cdef int i
    for i in range(n):
        ei = E[i]*t
        c = cos(ei)-1j*sin(ei) #cexp(ei)
        vi0=&V[i,0]
        zscal(&n, &c, vi0, &incz)

cdef void copy_pointer(double complex *v0, double complex *a0, int n2) nogil:
    #cdef int n2 = n*n
    cdef int incz = 1
    zcopy(&n2, v0, &incz, a0, &incz)


cdef void copy_pointer_d(double *v0, double *a0, int n2) nogil:
    #cdef int n2 = n*n
    cdef int incz = 1
    dcopy(&n2, v0, &incz, a0, &incz)

# New skripts for this
cdef void expE_V_expE_pointer(double *e0, double complex *v0, double t, double complex *expe0, int n) nogil:
    # Construct the transformation:  A <- diag(exp(v)) @ A @ diag(exp(v))
    ## This should be auto simd
    cdef double complex c, d
    cdef double ei
    cdef int incz = 1
    cdef int i, j
    cdef double complex *expe0_0 = expe0 # .copy()
    for i in range(n):
        ei = dereference(e0) * t
        c = cos(ei) -1j*sin(ei) #cexp(ei)
        expe0[0] = c
        e0 += 1
        expe0 += 1
    for i in range(n):
        c = expe0_0[0]
        expe0_0 += 1
        expe0 -= n
        for j in range(n):
            d = c * expe0[0]
            v0[0] = d * v0[0]
            expe0 += 1
            v0 += 1
cdef void expE_V_expE(double[::1] E, double complex[:,::1] V, double t, double complex[::1] expE) nogil:
    cdef int n = V.shape[0]
    cdef double *e0 = &E[0]
    cdef double complex *expe0 = &expE[0]
    cdef double complex *v0 = &V[0,0]
    expE_V_expE_pointer(e0, v0, t, expe0, n)

cdef void d_dt_expE_V_expE_pointer(double amp, double *e0, double complex *v0, double complex *dv0, double t, double complex *expe0, int n) nogil:
    # Construct the derivative of the transformation:  A <- diag(exp(v)) @ A @ diag(exp(v))
    ## This should be auto simd
    cdef double complex c, d, ec, s
    cdef double ei
    cdef int incz = 1
    cdef int i, j
    cdef double complex *expe0_0 = expe0 # .copy()
    cdef double *e0_0 = e0 # .copy()
    cdef double complex iamp = -1j * amp
    for i in range(n):
        ei = dereference(e0) * t
        c.real = cos(ei)
        c.imag = -sin(ei) #cexp(ei)
        expe0[0] = c
        e0 += 1
        expe0 += 1
    for i in range(n):
        c = expe0_0[0]
        expe0_0 += 1
        ec = e0_0[0]
        e0_0 += 1
        expe0 -= n
        e0 -= n
        for j in range(n):
            d = c * expe0[0]
            s = iamp * (ec + e0[0])
            v0[0] = d * v0[0]
            dv0[0] = s * v0[0]
            expe0 += 1
            e0 += 1
            v0 += 1
            dv0 += 1
cdef void d_dt_expE_V_expE(double amp, double[::1] E, double complex[:,::1] V, double complex[:,::1] dV, double t, double complex[::1] expE) nogil:
    cdef int n = V.shape[0]
    cdef double *e0 = &E[0]
    cdef double complex *expe0 = &expE[0]
    cdef double complex *v0 = &V[0,0]
    cdef double complex *dv0 = &dV[0,0]
    d_dt_expE_V_expE_pointer(amp, e0, v0, dv0, t, expe0, n)

cdef void v_exp_pointer(double *e0, double complex *v0, double t, int n) nogil:
    cdef double complex c
    cdef double ei
    cdef int incz = 1
    cdef int i

    ei = dereference(e0) * t
    c.real = cos(ei)
    c.imag = -sin(ei)
    zscal(&n, &c, v0, &incz)
    for i in range(1, n):
        v0 += n
        e0 += 1
        ei = dereference(e0) * t
        c.real = cos(ei)
        c.imag = -sin(ei)
        zscal(&n, &c, v0, &incz)
cdef void v_exp_v_pointer(double amp, double *e0, double complex *v0, double t, int n) nogil: 
    cdef double complex c
    cdef double ei, f
    cdef int incz = 1
    cdef int i

    ei = dereference(e0) * t
    f = -dereference(e0) * amp
    c.real = f * sin(ei)
    c.imag = f * cos(ei)
    zscal(&n, &c, v0, &incz)
    for i in range(1, n):
        v0 += n
        e0 += 1
        ei = dereference(e0) * t
        f = -dereference(e0) * amp
        c.real = f * sin(ei)
        c.imag = f * cos(ei)
        zscal(&n, &c, v0, &incz)
cdef void d_dt_exp_v_pointer(double amp, double *e0, double complex *v0, double complex *dv0, double t, int n, int n_2) nogil: # -i*E*amp*exp(-i*E*t)
    cdef double complex c, dc
    cdef double ei, f
    cdef int incz = 1
    cdef int i
    copy_pointer(v0, dv0, n_2)
    ei = dereference(e0) * t
    f = -dereference(e0) * amp
    c.real = cos(ei)
    c.imag = -sin(ei)
    dc.real = f * c.real
    dc.imag = - f * c.imag
    zscal(&n, &c, v0, &incz)
    zscal(&n, &dc, dv0, &incz)
    for i in range(1, n):
        v0 += n
        dv0 += n
        e0 += 1
        ei = dereference(e0) * t
        f = -dereference(e0) * amp
        c.real = cos(ei)
        c.imag = -sin(ei)
        dc.real = f * c.real
        dc.imag = - f * c.imag
        zscal(&n, &c, v0, &incz)
        zscal(&n, &dc, dv0, &incz)

cdef void phase_shift_matrix_pointer(double *e0, double complex *v0, double complex *a0, double dt, int n) nogil: # Only calculates upper triangle of symmetric matrix
    cdef int i, j
    cdef double complex idt = - 1j*dt
    cdef double ei, de
    cdef double complex *a1 = a0
    ei = dereference(e0) * dt
    v0[0].real = cos(ei)
    v0[0].imag = -sin(ei)
    for i in range(1, n):
        e0 += 1
        v0 += 1
        ei = dereference(e0) * dt
        v0[0].real = cos(ei)
        v0[0].imag = -sin(ei)
    e0 -= n - 1
    v0 -= n - 1
    for i in range(n):
        # Diagonal case
        #print(idt * v0[i])
        a0[0] = idt * v0[i]
        for j in range(i+1, n):
            a0 += 1
            a1 += n
            de = e0[i] - e0[j]
            if fabs(de)< 10**-14:
                a0[0] = idt * v0[i]
            else:
                a0[0] = 1/de * (v0[i] - v0[j])
            a1[0] = a0[0]
        a0 += i+2
        a1 = a0
cpdef void phase_shift_matrix(double[::1] E, double complex[::1] v, double complex[:,::1] A, double dt) nogil: # Only calculates upper triangle of symmetric matrix
    cdef double *e0 = &E[0]
    cdef double complex *v0 = &v[0]
    cdef double complex *a0 = &A[0,0]
    cdef int n = A.shape[0]
    phase_shift_matrix_pointer(e0, v0, a0, dt, n)



##### Add differential!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

###### Matrix exponentials from eigenvalue decomposition of Hamiltonians and Unitary logarithms / decompositions

cdef void V_expE_W_pointer(double complex *v0, double complex *w0, double *e0, double t, double complex *a0, double complex *b0, int n, int n_2) nogil:
    # Calculates B = V @ diag(exp(-1j*E*t)) @ W
    copy_pointer(w0 , a0, n_2)
    v_exp_pointer(e0, a0, t, n)
    MM_cdot_pointer(v0, a0, b0, n)
cpdef void V_expE_W(double complex[:,::1] V, double complex[:,::1] W, double[::1] E, double t, double complex[:,::1] A, double complex[:,::1] B, int n, int n_2):
    # Calculates B = V @ diag(exp(-1j*E*t)) @ W
    cdef double *e0 = &E[0]
    cdef double complex *v0 = &V[0,0]
    cdef double complex *w0 = &W[0,0]
    cdef double complex *a0 = &A[0,0]
    cdef double complex *b0 = &B[0,0]
    V_expE_W_pointer(v0, w0, e0, t, a0, b0, n, n_2)

cdef void V_EexpE_W_pointer(double amp, double complex *v0, double complex *w0, double *e0, double t, double complex *a0, double complex *b0, int n, int n_2) nogil:
    # Calculates amp*dB/dt = -1j * amp @ V @ diag(E) * diag(exp(-1j*E*t)) @ W   -> with helper variable *a0
    copy_pointer(w0 , a0, n_2)
    v_exp_v_pointer(amp, e0, a0, t, n)
    MM_cdot_pointer(v0, a0, b0, n)
cdef void d_dt_V_expE_W_pointer(double amp, double complex *v0, double complex *w0, double *e0, double t, double complex *a0, double complex *b0, double complex *db0, int n, int n_2) nogil:
    # Calculates B = V @ diag(exp(-1j*E*t)) @ W    -> with helper variable *a0
    # and its derivative amp*dB/dt = -1j * amp @ V @ diag(E) * diag(exp(-1j*E*t)) @ W
    copy_pointer(w0, a0, n_2)
    d_dt_exp_v_pointer(amp, e0, a0, b0, t, n, n_2)
    MM_cdot_pointer(v0, b0, db0, n)
    MM_cdot_pointer(v0, a0,  b0, n)
cdef void DagV_expE_V_pointer(double complex *v0, double *e0, double t, double complex *a0, double complex *b0, int n, int n_2) nogil:
    # Calculates B = Dag(V) @ diag(exp(-1j*E*t)) @ V
    copy_pointer(v0 , a0, n_2)
    v_exp_pointer(e0, a0, t, n)
    DagM_M_cdot_pointer(v0, a0, b0, n)

cpdef void c_expmH(double complex[:,::1] H, double dt, double complex[:,::1] U, int lwork, int lrwork, int liwork):
    cdef int n = H.shape[0]
    cdef double complex *h0=&H[0,0]
    cdef double complex[:,::1] V = np.empty([n, n], dtype=np.complex128)
    cdef double complex[:,::1] R = np.empty([n, n], dtype=np.complex128)
    cdef double complex *v0=&V[0,0]
    cdef double complex *r0=&V[0,0]
    cdef double[::1] E = np.empty([n], dtype=np.double)
    cdef double *e0=&E[0]
    c_eigh_lapack(H, V, E , lwork, lrwork, liwork)
    copy_v_exp_blas(E, V, R, dt, n)
    DagM_M_cdot(V, R, U)

cpdef void eigU(double complex[:,::1] U, double complex[::1] Ec, double complex[:,::1] V): # alternative to np.eig returning unitary matrices V
    cdef int n = U.shape[0]
    cdef int i, j
    Emat, Vmat = schur(U, output='complex')
    for i in range(n):
        Ec[i] = Emat[i, i]
        for j in range(n):
            V[i,j].real =  Vmat[j,i].real
            V[i,j].imag = -Vmat[j,i].imag

cpdef double complex clog(double complex c) nogil:   # logarithm of complex number
    cdef double x = c.real
    cdef double y = c.imag
    cdef double complex d
    cdef double squares = x*x + y*y
    cdef double abs_val = sqrt(squares)
    d.real = log(abs_val)
    d.imag = atan2(y,x)
    return d

cpdef void copy_v_log_blas(double complex[::1] E, double complex[:,::1] V, double complex[:,::1] A):
    cdef int n = E.shape[0]
    cdef double complex c
    cdef int i
    cdef double complex *vi0
    cdef int incz = 1
    A[:,:] = V
    for i in range(n):
        c = 1j*clog(E[i])
        vi0=&A[i,0]
        zscal(&n, &c, vi0, &incz)

cpdef void logmU(double complex[:,::1] U, double complex[:,::1] logU, double complex[:,::1] Emat, double complex[::1] E, double complex[:,::1] V):# Helper variables Emat, E, V
    cdef double dt = 1.0
    eigU(U, E, V)
    copy_v_log_blas(E, V, Emat)
    DagM_M_cdot(V, Emat, logU)

cpdef void ilogmU_eig(double complex[:,::1] U, double complex[::1] Ec, double[::1] logE_tmp, double complex[:,::1] V_tmp, double[::1] logE, double complex[:,::1] V):# Helper variables E, output variables: logE, V
    cdef int s = Ec.shape[0]
    cdef int i, j
    eigU(U, Ec, V_tmp)
    for i in range( s ):
        logE_tmp[i] = -clog(Ec[i]).imag
    # Sort the values
    cdef long long[::1] order
    order = np.argsort(logE_tmp)
    for i in range( s ):
        j = order[i]
        logE[i] = logE_tmp[j]
        V[i,:] = V_tmp[j,:]


cpdef void Make_U_Partial(double complex[:,::1] U_1, double complex[:,::1] U_0, double complex[:,::1] Emat, double complex[::1] Ec, double[::1] logE_tmp, double complex[:,::1] V_tmp, double[::1] logE, double complex[:,::1] V):
    # Input matrices:                   U_1 @ Dag(U_0) -> using full steps U_0, U_1   ### double complex[:,::1] logU,
    # Intermediate matrices (vectors):  lugU, Emat, (Ec)
    # Result matrices (vectors):        V, (E)
    cdef int s = U_1.shape[0]
    M_DagM_cdot(U_1, U_0, Emat)
    ilogmU_eig(Emat, Ec, logE_tmp, V_tmp, logE, V)

cpdef void Partialize_Center(double complex[:,::1] U_1, double complex[:,::1] U_0, double complex[:,::1] Emat2, double complex[:,::1] Emat, double complex[::1] Ec, double[::1] logE_tmp, double complex[:,::1] V_tmp, double[::1] logE, double complex[:,::1] V):
    # Input matrices:                   Dag(U_0) @ U_1^2 @ Dag(U_0) -> using half steps U_0, U_1
    # Intermediate matrices (vectors):  lugU, Emat, (Ec)
    # Result matrices (vectors):        V, (E)
    MM_cdot(U_1, U_1, V)  # lugU = U_1 ** 2
    DagM_M_cdot(U_0, V, Emat)
    M_DagM_cdot(Emat, U_0, Emat2)
    ilogmU_eig(Emat2, Ec, logE_tmp, V_tmp, logE, V)

cpdef void Partialize_Sides(double complex[:,::1] U_1, double complex[:,::1] U_0, double complex[:,::1] Emat, double complex[::1] Ec, double[::1] logE_tmp, double complex[:,::1] V_tmp, double[::1] logE, double complex[:,::1] V_L, double complex[:,::1] V_R):
    # Input matrices:                   Dag(U_0) @ U_1     &     U_1 @ Dag(U_0)    ->    using half steps U_0, U_1
    # Intermediate matrices (vectors):  lugU, Emat, V_tmp, (Ec)
    # Result matrices (vectors):        V_L, V_R, (E_L, E_R)
    DagM_M_cdot(U_0, U_1, Emat)
    ilogmU_eig(Emat, Ec, logE, V_tmp, logE, V_L)
    M_DagM_cdot(U_1, U_0, Emat)
    ilogmU_eig(Emat, Ec, logE_tmp, V_tmp, logE, V_R)