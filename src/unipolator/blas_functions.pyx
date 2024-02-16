#cython: language_level=3
from libc.math cimport cos, sin, log
import numpy as np
#cimport numpy as np
from scipy.linalg.cython_blas cimport zgemm, zscal, zdscal, zaxpy, zcopy, zdotu
from scipy.linalg.cython_lapack cimport zheevd

cdef void AxB_elementwise_pointer(double complex *a0, double complex *b0, double complex *c0, int n2) nogil:
    cdef int i
    for i in range(n2):
        c0[i] = a0[i]*b0[i]

cdef void MM_cdot_pointer(double complex *a0, double complex *b0, double complex *c0, int n) nogil:
    # matrix multiply 2 square matrices A (n x n) and B (n x n)
    cdef char *ori = 'n'
    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    zgemm(ori, ori, &n, &n, &n, &alpha, b0, &n, a0, &n, &beta, c0, &n)
cdef void MM_cdot_scale_pointer(double alpha, double complex *a0, double complex *b0, double complex *c0, int n) nogil:
    # matrix multiply 2 square matrices A (n x n) and B (n x n)
    cdef char *ori = 'n'
    cdef double complex alpha_c
    cdef double complex beta = 0.0
    alpha_c.real = alpha
    alpha_c.imag = 0
    zgemm(ori, ori, &n, &n, &n, &alpha_c, b0, &n, a0, &n, &beta, c0, &n)
cpdef void MM_cdot(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C) nogil: # C = A @ B
    # matrix multiply 2 square matrices A (n x n) and B (n x n) -> C = A @ B
    cdef char *ori = 'n'
    cdef double complex *a0=&A[0,0]
    cdef double complex *b0=&B[0,0]
    cdef double complex *c0=&C[0,0]
    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    cdef int n
    n = A.shape[0]
    zgemm(ori, ori, &n, &n, &n, &alpha, b0, &n, a0, &n, &beta, c0, &n)

cdef void MMM_cdot_pointer(double complex *a0, double complex *b0, double complex *c0, double complex *d0, double complex *e0, int n) nogil: # A @ B @ C = D
    cdef char *ori = 'n'
    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    zgemm(ori, ori, &n, &n, &n, &alpha, b0, &n, a0, &n, &beta, e0, &n)   
    zgemm(ori, ori, &n, &n, &n, &alpha, c0, &n, e0, &n, &beta, d0, &n)
cdef void MMM_cdot(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C, double complex[:,::1] D, double complex[:,::1] E) nogil:  # A @ B @ C = D (helper variable E)
    cdef double complex *a0 = &A[0, 0]
    cdef double complex *b0 = &B[0, 0]
    cdef double complex *c0 = &C[0, 0]
    cdef double complex *d0 = &D[0, 0]
    cdef double complex *e0 = &E[0, 0]
    cdef int n = A.shape[0]
    MMM_cdot_pointer(a0, b0, c0, d0, e0, n)

cdef void DagM_M_cdot_pointer(double complex *a0, double complex *b0, double complex *c0, int n) nogil:
    # matrix multiply 2 square matrices A (n x n) and B (n x n)
    cdef char *orientA = 'n'
    cdef char *orientB = 'c'
    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    zgemm(orientA, orientB, &n, &n, &n, &alpha, b0, &n, a0, &n, &beta, c0, &n)
cdef void DagM_M_cdot_scale_pointer(double alpha, double complex *a0, double complex *b0, double complex *c0, int n) nogil:
    # matrix multiply 2 square matrices A (n x n) and B (n x n)
    cdef char *orientA = 'n'
    cdef char *orientB = 'c'
    cdef double complex alpha_c
    cdef double complex beta = 0.0
    alpha_c.real = alpha
    alpha_c.imag = 0
    zgemm(orientA, orientB, &n, &n, &n, &alpha_c, b0, &n, a0, &n, &beta, c0, &n)
cpdef void DagM_M_cdot(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C) nogil:
    # matrix multiply 2 square matrices A (n x n) and B (n x n)
    cdef char *orientA = 'n'
    cdef char *orientB = 'c'
    cdef double complex *a0=&A[0,0]
    cdef double complex *b0=&B[0,0]
    cdef double complex *c0=&C[0,0]
    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    cdef int n = A.shape[0]
    zgemm(orientA, orientB, &n, &n, &n, &alpha, b0, &n, a0, &n, &beta, c0, &n)

cdef void M_DagM_cdot_pointer(double complex *a0, double complex *b0, double complex *c0, int n) nogil:
    # matrix multiply 2 square matrices A (n x n) and B (n x n)
    cdef char *orientA = 'c'
    cdef char *orientB = 'n'
    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    zgemm(orientA, orientB, &n, &n, &n, &alpha, b0, &n, a0, &n, &beta, c0, &n)
cdef void M_DagM_cdot_scale_pointer(double alpha, double complex *a0, double complex *b0, double complex *c0, int n) nogil:
    # matrix multiply 2 square matrices A (n x n) and B (n x n)
    cdef char *orientA = 'c'
    cdef char *orientB = 'n'
    cdef double complex alpha_c
    cdef double complex beta = 0.0
    alpha_c.real = alpha
    alpha_c.imag = 0
    zgemm(orientA, orientB, &n, &n, &n, &alpha_c, b0, &n, a0, &n, &beta, c0, &n)
cpdef void M_DagM_cdot(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C) nogil:
    # matrix multiply 2 square matrices A (n x n) and B (n x n)
    cdef char *orientA = 'c'
    cdef char *orientB = 'n'
    cdef double complex *a0=&A[0,0]
    cdef double complex *b0=&B[0,0]
    cdef double complex *c0=&C[0,0]
    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    cdef int n = A.shape[0]
    zgemm(orientA, orientB, &n, &n, &n, &alpha, b0, &n, a0, &n, &beta, c0, &n)

# Triple products
cpdef void DagA_B_A_cdot(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C, double complex[:,::1] D) nogil:
    # calculates the product: D = Dag(A) @ B @ A
    # uses C to store intermediate results
    DagM_M_cdot(A, B, C)
    MM_cdot(C,A, D)
cdef void DagA_B_A_cdot_pointer(double complex *a0, double complex *b0, double complex *c0, double complex *d0, int n) nogil:
    # calculates the product: D = Dag(A) @ B @ A
    # uses C to store intermediate results
    DagM_M_cdot_pointer(a0, b0, c0, n)
    MM_cdot_pointer(c0, a0, d0, n)

cpdef void A_B_DagA_cdot(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C, double complex[:,::1] D) nogil:
    # calculates the product: D = A @ B @ Dag(A)
    # uses C to store intermediate results
    MM_cdot(A, B, C)
    M_DagM_cdot(C, A, D)
cdef void A_B_DagA_cdot_pointer(double complex *a0, double complex *b0, double complex *c0, double complex *d0, int n) nogil:
    # calculates the product: D = A @ B @ Dag(A)
    # uses C to store intermediate results
    MM_cdot_pointer(a0, b0, c0, n)
    M_DagM_cdot_pointer(c0, a0, d0, n)

cdef double complex tr_dot_pointer(double complex *v0, double complex *w0, int n) nogil: # -i*E*amp*exp(-i*E*t)
    cdef double complex c
    cdef double abstr
    cdef int inc = 1
    cdef int i
    cdef double complex trAB = zdotu(&n, v0, &inc, w0, &n)
    for i in range(1,n):
        v0 += n
        w0 += 1
        trAB += zdotu(&n, v0, &inc, w0, &n)
    return trAB
cpdef double complex tr_dot(double complex[:,::1] V, double complex[:,::1] W, int n ):
    cdef double complex *v0 = &V[0,0]
    cdef double complex *w0 = &W[0,0]
    return tr_dot_pointer(v0, w0, n)

cdef double complex tr_dot_pointer_target_indexes(double complex *v0, double complex *w0, int n, int[::1] target_indexes) nogil: # -i*E*amp*exp(-i*E*t)
    cdef double complex c
    cdef int i, i_s
    cdef int inc = 1
    i = target_indexes[0]
    cdef double complex trAB = zdotu(&n, v0 + i * n, &inc, w0 + i, &n)
    for i_s in range(1,target_indexes.shape[0]):
        i = target_indexes[i_s]
        trAB += zdotu(&n, v0+i*n, &inc, w0+i, &n)
    return trAB
cpdef double complex tr_dot_target_indexes(double complex[:,::1] V, double complex[:,::1] W, int n, int[::1] target_indexes ):
    cdef double complex *v0 = &V[0,0]
    cdef double complex *w0 = &W[0,0]
    return tr_dot_pointer_target_indexes(v0, w0, n, target_indexes)

cdef double complex target_indexes_trace_pointer(double complex *v0, int n, int[::1] target_indexes) nogil:
    cdef int i
    cdef double complex c = 0
    for i in range(target_indexes.shape[0]):
        c += v0[i*n+i]
    return c

cpdef (int, int, int) c_eigh_lapack_workspace_sizes(double complex[:,::1] H):  # H will be returned as the
    # matrix multiply 2 square matrices A (n x n) and B (n x n)
    cdef char *jobz = 'v' #eigenvectors and values -> v
    cdef char *uplo = 'l' # upper triangle
    cdef int n = H.shape[0]
    cdef double complex *h0=&H[0,0]
    cdef int info = 0
    cdef double[:] w = np.array([n], dtype=np.double)
    cdef double complex[:] work1 = np.empty([1], dtype=np.complex128)
    cdef double [:] rwork1 = np.empty([1], dtype=np.double)
    cdef int[:] iwork1 = np.empty([1], dtype=np.int32)
    cdef double complex *work0 = &work1[0]
    cdef double *rwork0 = &rwork1[0]
    cdef int *iwork0 = &iwork1[0]
    cdef int lrwork = -1
    cdef int lwork = -1
    cdef int liwork = -1
    zheevd(jobz, uplo, &n, h0, &n, &w[0], work0, &lwork, rwork0, &lrwork, iwork0, &liwork, &info)
    cdef int work = <int>(work1[0].real + 1e-9)
    cdef int rwork = <int>(rwork1[0] + 1e-9)
    cdef int iwork = <int>(iwork1[0] +1e-9)
    return work, rwork, iwork

cpdef void c_eigh_lapack(double complex[:,::1] H, double complex[:,::1] V, double[::1] E, int lwork, int lrwork, int liwork):  # H will be returned as the #int nwork, int nrwork, int niwork,
    # matrix multiply 2 square matrices A (n x n) and B (n x n)
    cdef int n = H.shape[0]
    V[:,:] = H
    cdef double complex *v0 = &V[0,0]
    cdef double *e0=&E[0]

    cdef char *jobz = 'v' #eigenvectors and values -> v
    cdef char *uplo = 'l' # upper triangle

    #cdef int lwork = 2*n + n*n+1
    cdef double complex[:] work = np.empty([lwork], dtype=np.complex128)
    cdef double complex *work0 = &work[0]
    cdef double[:] rwork = np.empty([lrwork], dtype=np.double)
    cdef double *rwork0 = &rwork[0]

    #cdef int liwork = 3 + 5*n +1
    cdef int[:] iwork = np.empty([liwork], dtype=np.int32)
    cdef int *iwork0 = &iwork[0]

    cdef int info = 0
    zheevd(jobz, uplo, &n, v0, &n, e0, work0, &lwork, rwork0, &lrwork, iwork0, &liwork, &info)
    #return info


###### Consider using https://epubs.siam.org/doi/10.1137/100788860 ---- for expmH * vec   --> for vector optimization

###### These could be useful for a future additional unitary interpolation approach ####################################
###### Additions #######################################################################################################
cpdef void d_third_order_tensor_scale(double complex[:,:,::1] A, double d) nogil:
    cdef int s = A.shape[0]
    cdef int n = A.shape[1]
    cdef int snn = s*n*n
    cdef int incz = 1
    cdef double complex *a0=&A[0,0,0]
    zdscal(&snn, &d, a0, &incz)

cpdef void d_mat_scale(double complex[:,::1] A, double d) nogil:
    cdef int n = A.shape[0]
    cdef int nn = n*n
    cdef int incz = 1
    cdef double complex *a0=&A[0,0]
    zdscal(&nn, &d, a0, &incz)

cpdef void c_mat_scale(double complex[:,::1] A, double complex c) nogil:
    cdef int n = A.shape[0]
    cdef int nn = n*n
    cdef int incz = 1
    cdef double complex *a0=&A[0,0]
    zscal(&nn, &c, a0, &incz)

cpdef void d_mat_add_first(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C, double d): # nogil:
    cdef int n = A.shape[0]
    cdef int nn = n*n
    cdef int incz = 1
    cdef double complex *a0=&A[0,0]
    C[:,:] = B
    d_mat_scale(C, d)
    cdef double complex *c0=&C[0,0]
    zcopy(&nn, c0, &incz, a0, &incz)

cpdef void c_mat_add_first(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C, double complex c): # nogil:
    # 
    cdef int n = A.shape[0]
    cdef int nn = n*n
    cdef int incz = 1
    cdef double complex *a0=&A[0,0]
    C[:,:] = B
    c_mat_scale(C, c)
    cdef double complex *c0=&C[0,0]
    zcopy(&nn, c0, &incz, a0, &incz)

cpdef void d_mat_add(double complex[:,::1] A, double complex[:,::1] B, double d) nogil: # A = A+B*d
    cdef int n = A.shape[0]
    cdef int nn = n*n
    cdef int incz = 1
    cdef double complex *a0=&A[0,0]
    cdef double complex *b0=&B[0,0]
    cdef double complex c = <double complex> d
    zaxpy(&nn, &c, b0, &incz, a0, &incz)

cdef void d_mat_add_pointer(double complex *a0, double complex *b0, double d, int n2) nogil: # A = A+B*d
    cdef int incz = 1
    cdef double complex c = <double complex> d
    zaxpy(&n2, &c, b0, &incz, a0, &incz)

cpdef void c_mat_add(double complex[:,::1] A, double complex[:,::1] B, double complex c) nogil: # A = A+B*c
    cdef int n = A.shape[0]
    cdef int nn = n*n
    cdef int incz = 1
    cdef double complex *a0=&A[0,0]
    cdef double complex *b0=&B[0,0]
    zaxpy(&nn, &c, b0, &incz, a0, &incz)

cdef void c_mat_add_pointer(double complex *a0, double complex *b0, int nn) nogil: # A = A+B*c
    cdef int incz = 1
    cdef double complex c = 1.0 + 0.0j
    zaxpy(&nn, &c, b0, &incz, a0, &incz)

