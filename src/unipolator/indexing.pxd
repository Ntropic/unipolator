cpdef Bin_Parameters(double[::1] c_mins, double[::1] c_maxs, long[::1] c_bins)

cpdef void next_in_mrange(long[:] A, long[:] n_s) nogil

cpdef reverse_cum_prod(long[:] s, int n)

cpdef int findex_0(long[:] i, long[:] cum_prod, int n)

cpdef int findex(long[:] i, long[:] cum_prod, int n, int a) nogil

cpdef flat_index_sizes_E(long[::1] c_bins)

cpdef flat_index_sizes_V(long[::1] c_bins)

cpdef int is_odd(int num) nogil

cpdef void is_odd_array(long[::1] num, int n) nogil

cpdef int is_even(int num) nogil

cpdef void is_even_array(long[::1] num, int n) nogil

cpdef void asym_sign(double[::1] nums, long[::1] signs, int n) nogil

cpdef void asym_sign_2D(double[:,::1] nums, long[:,::1] signs, int n0, int n1) nogil

cpdef void rounder(double a, int b) nogil

cpdef void rounder_array(double[::1] a, long[::1] b, int n) nogil

cpdef void rounder_array_2d(double[:,::1] a, long[:,::1] b, int n0, int n1) nogil

cpdef int is_in_interval(double[::1] c, long[::1] maxs, int n) nogil

cpdef int int_sum(long[::1] i_s, int n) nogil

cpdef int int_max(long[::1] i_s, int n) nogil

cpdef int int_prod(long[::1] i_s, int n) nogil

cpdef int_prod_array(long[:,::1] i_s, int n)
cpdef int_prod_asym_array(long[:,::1] i_s)

cpdef void c_abs_sum(double[::1] c, double[::1] abs_c, int n) nogil

cpdef void c_int_sum(long[::1] c, int sum_c, int n) nogil

cpdef void dvec_minus_ivec(double[::1] c, double[::1] dvec, long[:] ivec, int n) nogil

cpdef void elementwise_grid_parameters(double[::1] c, double[::1] c_min, double[::1] dc, long[::1] c_bins, double[::1] alpha, int sum_location, long[::1] location, double[::1] alpha_rest, double[::1] abs_alpha_rest, long[::1] d_location) nogil

cpdef int c_argmax(double[::1] c, int max_ind, double c_max, int n) nogil

#cpdef Parameters2OddGrid(double[::1] alpha, double[::1] c, double[::1] c_min, double[::1] dc, int[::1] c_bins)

