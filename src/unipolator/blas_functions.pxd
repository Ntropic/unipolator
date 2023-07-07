cdef void AxB_elementwise_pointer(double complex *a0, double complex *b0, double complex *c0, int n2) nogil

cdef void MM_cdot_pointer(double complex *a0, double complex *b0, double complex *c0, int n) nogil
cdef void MM_cdot_scale_pointer(double alpha, double complex *a0, double complex *b0, double complex *c0, int n) nogil
cpdef void MM_cdot(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C) nogil

cdef void MMM_cdot_pointer(double complex *a0, double complex *b0, double complex *c0, double complex *d0, double complex *e0, int n) nogil
cdef void MMM_cdot(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C, double complex[:,::1] D, double complex[:,::1] E) nogil

cdef void DagM_M_cdot_pointer(double complex *a0, double complex *b0, double complex *c0, int n) nogil
cdef void DagM_M_cdot_scale_pointer(double alpha, double complex *a0, double complex *b0, double complex *c0, int n) nogil
cpdef void DagM_M_cdot(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C) nogil

cpdef void M_DagM_cdot(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C) nogil
cdef void M_DagM_cdot_pointer(double complex *a0, double complex *b0, double complex *c0, int n) nogil
cdef void M_DagM_cdot_scale_pointer(double alpha, double complex *a0, double complex *b0, double complex *c0, int n) nogil

# Triple products
cpdef void DagA_B_A_cdot(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C, double complex[:,::1] D) nogil
cdef void DagA_B_A_cdot_pointer(double complex *a0, double complex *b0, double complex *c0, double complex *d0, int n) nogil
cpdef void A_B_DagA_cdot(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C, double complex[:,::1] D) nogil
cdef void A_B_DagA_cdot_pointer(double complex *a0, double complex *b0, double complex *c0, double complex *d0, int n) nogil

cdef double complex tr_dot_pointer(double complex *v0, double complex *w0, int n) nogil
cpdef double complex tr_dot(double complex[:,::1] V, double complex[:,::1] W, int n )
cdef double complex tr_dot_pointer_target_indexes(double complex *v0, double complex *w0, int n, int[::1] target_indexes) nogil
cpdef double complex tr_dot_target_indexes(double complex[:,::1] V, double complex[:,::1] W, int n, int[::1] target_indexes )
cdef double complex target_indexes_trace_pointer(double complex *v0, int n, int[::1] target_indexes) nogil

cpdef (int, int, int) c_eigh_lapack_workspace_sizes(double complex[:,::1] H)

cpdef void c_eigh_lapack(double complex[:,::1] H, double complex[:,::1] V, double[::1] E, int lwork, int lrwork, int liwork)

cpdef void d_third_order_tensor_scale(double complex[:,:,::1] A, double d) nogil

cpdef void d_mat_scale(double complex[:,::1] A, double d) nogil

cpdef void c_mat_scale(double complex[:,::1] A, double complex c) nogil

cpdef void d_mat_add_first(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C, double d) # nogil:

cpdef void c_mat_add_first(double complex[:,::1] A, double complex[:,::1] B, double complex[:,::1] C, double complex c) # nogil:

cpdef void d_mat_add(double complex[:,::1] A, double complex[:,::1] B, double d) nogil
cdef void d_mat_add_pointer(double complex *a0, double complex *b0, double d, int n2) nogil

cpdef void c_mat_add(double complex[:,::1] A, double complex[:,::1] B, double complex c) nogil
cdef void c_mat_add_pointer(double complex *a0, double complex *b0, int nn) nogil
