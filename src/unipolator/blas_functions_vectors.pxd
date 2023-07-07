cdef void MM_cdot_pointer_v_scaled(double complex *a0, double complex *v0, double complex *c0, double complex alpha, int n, int m) nogil
cdef void MM_cdot_pointer_v(double complex *a0, double complex *v0, double complex *c0, int n, int m) nogil
cpdef void MM_cdot_v(double complex[:,::1] A, double complex[:,::1] v, double complex[:,::1] C)
cdef void DagM_M_cdot_pointer_v(double complex *a0, double complex *v0, double complex *c0, int n, int m) nogil
cpdef void DagM_M_cdot_v(double complex[:,::1] A, double complex[:,::1] v, double complex[:,::1] C) nogil

cdef void v_exp_pointer_v(double *e0, double complex *v0, double t, int n, int m) nogil
cdef void v_exp_v_pointer_v(double amp, double *e0, double complex *v0, double t, int n, int m) nogil

cdef void MM_cdot_pointer_batch_v(double complex *a0, double complex *vi, double complex *vo, double complex *di, double complex *do, int n, int m, int nm, int batch_ind) nogil
cdef void MM_cdot_batch_v(double complex[:,::1] A, double complex[:,::1] V_in, double complex[:,::1] V_out, double complex[:,:,::1] dV_in, double complex[:,:,::1] dV_out, int n, int m, int nm, int batch_ind) nogil

cdef void v_exp_v_and_v_exp_pointer_v_batch(double *e0, double complex *e0s, double complex *v0, double complex *dv0, double t, int n, int m, int batch_ind) nogil
cpdef void v_exp_v_and_v_exp_v_batch(double[::1] E, double complex[::1] Es, double complex[:,::1] V, double complex[:,:,::1] dV, double t, int batch_ind)
cdef void v_exp_v_and_v_exp_pointer_v_batch_increment(double amp, double *e0, double complex *e0s, double complex *v0, double complex *dv0, double t, int n, int m, int mn, int batch_ind) nogil
cpdef void v_exp_v_and_v_exp_v_batch_increment(double amp, double[::1] E, double complex[::1] Es, double complex[:,::1] V, double complex[:,:,::1] dV, double t, int batch_ind)

cdef double norm_inf_complex( double complex[:, ::1] B, int d, int m) nogil
cpdef expm_multiply_prepare(double complex[:,::1] A0, int m, double tol)