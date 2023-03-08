cpdef double creal(double complex dc) nogil

cpdef double cimag(double complex dc) nogil

cpdef double complex cconj(double complex dc) nogil

cpdef void cdiag(double complex[:,::1] A, double complex[::1] diagA) nogil

cpdef void Conj_mat_copy(double complex[:,::1] A, double complex[:,::1] ConjA) nogil

cpdef void Conj_mat(double complex[:,::1] A) nogil

cpdef void Conj_vec_copy(double complex[::1] A, double complex[::1] ConjA) nogil

cpdef void Conj_vec(double complex[::1] A) nogil

cpdef Dag(double complex[:,::1] A)
cpdef void Dag_fast(double complex[:,::1] A, double complex[:,::1] DagA) nogil
cdef double complex conj(double complex A) nogil
cpdef double abs_2(double complex A) nogil

cpdef void copy_v_exp_blas(double[::1] E, double complex[:,::1] V, double complex[:,::1] A, double t, int n) nogil
cpdef void v_exp_blas(double[::1] E, double complex[:,::1] V, double t) nogil

cdef void copy_pointer(double complex *v0, double complex *a0, int n2) nogil
cdef void copy_pointer_d(double *v0, double *a0, int n2) nogil
cdef void expE_V_expE_pointer(double *e0, double complex *v0, double t, double complex *expe0, int n) nogil
cdef void expE_V_expE(double[::1] E, double complex[:,::1] V, double t, double complex[::1] expE) nogil
cdef void d_dt_expE_V_expE_pointer(double amp, double *e0, double complex *v0, double complex *dv0, double t, double complex *expe0, int n) nogil
cdef void d_dt_expE_V_expE(double amp, double[::1] E, double complex[:,::1] V, double complex[:,::1] dV, double t, double complex[::1] expE) nogil
cdef void v_exp_pointer(double *e0, double complex *v0, double t, int n) nogil
cdef void v_exp_v_pointer(double amp, double *e0, double complex *v0, double t, int n) nogil
cdef void d_dt_exp_v_pointer(double amp, double *e0, double complex *v0, double complex *dv0, double t, int n, int n_2) nogil

cdef void phase_shift_matrix_pointer(double *e0, double complex *v0, double complex *a0, double dt, int n) nogil
cpdef void phase_shift_matrix(double[::1] E, double complex[::1] v, double complex[:,::1] A, double dt) nogil

cdef void V_expE_W_pointer(double complex *v0, double complex *w0, double *e0, double t, double complex *a0, double complex *b0, int n, int n2) nogil
cpdef void V_expE_W(double complex[:,::1] V, double complex[:,::1] W, double[::1] E, double t, double complex[:,::1] A, double complex[:,::1] B, int n, int n_2)
cdef void V_EexpE_W_pointer(double amp, double complex *v0, double complex *w0, double *e0, double t, double complex *a0, double complex *b0, int n, int n2) nogil
cdef void d_dt_V_expE_W_pointer(double amp, double complex *v0, double complex *w0, double *e0, double t, double complex *a0, double complex *b0, double complex *db0, int n, int n_2) nogil
cdef void DagV_expE_V_pointer(double complex *v0, double *e0, double t, double complex *a0, double complex *b0, int n, int n_2) nogil
cpdef void c_expmH(double complex[:,::1] H, double dt, double complex[:,::1] U, int lwork, int lrwork, int liwork)

cpdef void eigU(double complex[:,::1] U, double complex[::1] Ec, double complex[:,::1] V)

cpdef double complex clog(double complex c) nogil

cpdef void copy_v_log_blas(double complex[::1] E, double complex[:,::1] V, double complex[:,::1] A)

cpdef void logmU(double complex[:,::1] U, double complex[:,::1] logU, double complex[:,::1] Emat, double complex[::1] E, double complex[:,::1] V)
cpdef void ilogmU_eig(double complex[:,::1] U, double complex[::1] E, double[::1] logE_tmp, double complex[:,::1] V_tmp, double[::1] logE, double complex[:,::1] V)

cpdef void Make_U_Partial(double complex[:,::1] U_1, double complex[:,::1] U_0, double complex[:,::1] Emat, double complex[::1] Ec, double[::1] logE_tmp, double complex[:,::1] V_tmp, double[::1] logE, double complex[:,::1] V)
cpdef void Partialize_Center(double complex[:,::1] U_1, double complex[:,::1] U_0, double complex[:,::1] Emat2, double complex[:,::1] Emat, double complex[::1] Ec, double[::1] logE_tmp, double complex[:,::1] V_tmp, double[::1] logE, double complex[:,::1] V)
cpdef void Partialize_Sides(double complex[:,::1] U_1, double complex[:,::1] U_0, double complex[:,::1] Emat, double complex[::1] Ec, double[::1] logE_tmp, double complex[:,::1] V_tmp, double[::1] logE, double complex[:,::1] V_L, double complex[:,::1] V_R)
