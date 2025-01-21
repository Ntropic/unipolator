# pxd file for compatibility with updated pyx file

cimport numpy as npc

cpdef Bin_Parameters(double[::1] c_mins, double[::1] c_maxs, npc.intp_t[::1] c_bins)

cpdef void next_in_mrange(npc.intp_t[:] A, npc.intp_t[:] n_s) noexcept nogil 

cpdef reverse_cum_prod(npc.intp_t[:] s, int n)

cpdef int findex_0(npc.intp_t[:] i, npc.intp_t[:] cum_prod, int n)

cpdef int findex(npc.intp_t[:] i, npc.intp_t[:] cum_prod, int n, int a) noexcept nogil 

cpdef flat_index_sizes_E(npc.intp_t[::1] c_bins)

cpdef flat_index_sizes_V(npc.intp_t[::1] c_bins)

cpdef int is_odd(int num) noexcept nogil 

cpdef void is_odd_array(npc.intp_t[::1] num, int n) noexcept nogil 

cpdef int is_even(int num) noexcept nogil 

cpdef void is_even_array(npc.intp_t[::1] num, int n) noexcept nogil 

cpdef void asym_sign(double[::1] nums, npc.intp_t[::1] signs, int n) noexcept nogil 

cpdef void asym_sign_2D(double[:,::1] nums, npc.intp_t[:,::1] signs, int n0, int n1) noexcept nogil 

cpdef void rounder(double a, int b) noexcept nogil 

cpdef void rounder_array(double[::1] a, npc.intp_t[::1] b, int n) noexcept nogil 

cpdef void rounder_array_2d(double[:,::1] a, npc.intp_t[:,::1] b, int n0, int n1) noexcept nogil 

cpdef int is_in_interval(double[::1] c, npc.intp_t[::1] maxs, int n) noexcept nogil 

cpdef int int_sum(npc.intp_t[::1] i_s, int n) noexcept nogil 

cpdef int int_max(npc.intp_t[::1] i_s, int n) noexcept nogil 

cpdef int int_prod(npc.intp_t[::1] i_s, int n) noexcept nogil 

cpdef int_prod_array(npc.intp_t[:,::1] i_s, int n)

cpdef int_prod_asym_array(npc.intp_t[:,::1] i_s)

cpdef void c_abs_sum(double[::1] c, double[::1] abs_c, int n) noexcept nogil 

cpdef void c_int_sum(npc.intp_t[::1] c, int sum_c, int n) noexcept nogil 

cpdef void dvec_minus_ivec(double[::1] c, double[::1] dvec, npc.intp_t[:] ivec, int n) noexcept nogil 

cpdef void elementwise_grid_parameters(
    double[::1] c, 
    double[::1] c_min, 
    double[::1] dc, 
    npc.intp_t[::1] c_bins, 
    double[::1] alpha, 
    int sum_location, 
    npc.intp_t[::1] location, 
    double[::1] alpha_rest, 
    double[::1] abs_alpha_rest, 
    npc.intp_t[::1] d_location
) noexcept nogil 

cpdef int c_argmax(double[::1] c, int max_ind, double c_max, int n) noexcept nogil 

# cpdef Parameters2OddGrid(double[::1] alpha, double[::1] c, double[::1] c_min, double[::1] dc, int[::1] c_bins)
