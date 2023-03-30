#cython: language_level=3
from libc.math cimport fabs
import numpy as np
cimport numpy as npc

#### Construct Parameters of bounds #############################################################

cpdef Bin_Parameters(double[::1] c_mins, double[::1] c_maxs, long[::1] c_bins):
    # Correct input parameters
    cdef int n = c_mins.shape[0]
    cdef int i
    cdef double c_min, c_max, dc, ac
    cdef int c_bin
    cdef npc.ndarray[npc.double_t, ndim=1] dcs = np.empty(n, dtype=np.double)
    cdef npc.ndarray[npc.double_t, ndim=1] das = np.empty(n, dtype=np.double)
    cdef npc.ndarray[npc.double_t, ndim=1] c_minis = np.empty(n, dtype=np.double)
    cdef npc.ndarray[npc.double_t, ndim=1] c_maxis = np.empty(n, dtype=np.double)
    for i in range(n):
        if c_mins[i] <= c_maxs[i]:
            c_min = c_mins[i]
            c_max = c_maxs[i]
        else:
            c_min = c_maxs[i]
            c_max = c_mins[i]
        c_bin = c_bins[i]
        if abs(c_max -c_min) < 10**-15:
            print('This parameter might not be necessary! (c_max - c_min < 10**-15)')
            c_max = c_max + 10**-15
        dc = (c_max-c_min) / c_bin
        ac = c_bin / (c_max-c_min)
        c_minis[i] = c_min
        c_maxis[i] = c_max
        dcs[i] = dc
        das[i] = ac # For scaling the derivatives
    return c_minis, c_maxis, dcs, das

##### Iterate over multiple dimensions
cpdef void next_in_mrange(long[:] A, long[:] n_s) nogil:
    cdef int i = n_s.shape[0] - 1
    while i >= 0:
        A[i] = (A[i]+1) % n_s[i]
        if A[i]:
            break
        i += -1

##### Index Transformation to (partially) flattened arrays
cpdef reverse_cum_prod(long[:] s, int n):
    if n < 0:
        n = s.shape[0]
    cdef long[:] s_red = s[:n]
    cdef npc.ndarray[long, ndim=1] cum_prod = np.empty(n, dtype=long)
    cdef int curr_index = n-1
    cum_prod[curr_index] = 1 #s_red[-1]
    for i in range(n-1):
        curr_index += -1
        cum_prod[curr_index] = cum_prod[curr_index+1]*s_red[curr_index+1]
    cdef int tot_prod = cum_prod[0]*s[0]
    return cum_prod, tot_prod

cpdef int findex_0(long[:] i, long[:] cum_prod, int n): # n is length of i, cum_prod #### Inline this function and similar functions
    cdef int i_s
    cdef int a = i[0] * cum_prod[0]
    for i_s in range(1,n):
        a += i[i_s]*cum_prod[i_s]
    return a

cpdef int findex(long[:] i, long[:] cum_prod, int n, int a) nogil: # n is length of i, cum_prod #### Inline this function and similar functions
    # in this variant a is initialized via an a_min
    cdef int i_s
    for i_s in range(n):
        a += i[i_s]*cum_prod[i_s]
    return a

# Convert indices to flattened arrays (works equivalently to ndarrays)
cpdef flat_index_sizes_E(long[::1] c_bins):   # For eigene-enery vectors
    # Construct indexes for a flattened cache
    cdef int l = c_bins.shape[0]
    cdef int i, j
    cdef int tot_prod0 = 1
    cdef int tot_ind
    cdef long[::1] tot_prods = np.empty(l, dtype = long)
    cdef long[::1] d_bins = np.empty(l, dtype = long)
    cdef long[::1] first_elements = np.empty(l, dtype = long)
    cdef long[:,::1] array_sizes = np.empty([l,l], dtype = long)
    cdef long[:,::1] array_strides = np.empty([l,l], dtype = long)
    for i in range(l):
        d_bins[i] = c_bins[i] + 1 # Should copy data
        tot_prod0 = tot_prod0*d_bins[i]
    # Construct the total sizes of the caches for every dimensions
    for i in range(l): # One arrow cache for every direction in grid
        d_bins[i] = c_bins[i] # Should copy data
        if i > 0:
            d_bins[i-1] = c_bins[i-1]+1
        tot_prods[i] =  (tot_prod0*c_bins[i]) // (c_bins[i]+1)
        curr_cum_prod, _ = reverse_cum_prod(d_bins, l)
        for j in range(l):
            array_strides[i,j] = curr_cum_prod[j]
        for j in range(l):
            array_sizes[i,j] = d_bins[j]
    first_elements[0] = 0
    for i in range(1,l):
        first_elements[i] = first_elements[i-1]+tot_prods[i-1]
    tot_ind = first_elements[l-1]+tot_prods[l-1]
    return tot_ind, first_elements, array_strides, array_sizes

cpdef flat_index_sizes_V(long[::1] c_bins): # For orthonormal matrices and their products (left to center (low-high) to right)
    # Construct indexes for a flattened cache
    cdef int l = c_bins.shape[0]
    cdef int i, j
    cdef int tot_prod0 = 1
    cdef long[::1] d_bins = np.empty(l, dtype = long)
    cdef int tot_ind_L, tot_int_R, tot_ind_C

    cdef long[::1] tot_prods_C = np.empty(l-1, dtype = long)
    cdef long[::1] first_elements_C = np.empty(l-1, dtype = long)
    cdef long[:,::1] array_sizes_C = np.empty([l-1,l], dtype = long)
    cdef long[:,::1] array_strides_C = np.empty([l-1,l], dtype = long)

    cdef long[::1] array_sizes_L = np.empty(l, dtype = long)
    cdef long[::1] array_strides_L = np.empty(l, dtype = long)
    cdef long[::1] array_sizes_R = np.empty(l, dtype = long)
    cdef long[::1] array_strides_R = np.empty(l, dtype = long)
    for i in range(l):
        d_bins[i] = c_bins[i] + 1 # Should copy data
        tot_prod0 = tot_prod0*d_bins[i]
    # Construct the total sizes of the caches for every dimensions
    for i in range(l): # One arrow cache for every direction in grid
        tot_prod0 = (tot_prod0 * c_bins[i]) // (c_bins[i]+1)
        d_bins[i] = c_bins[i] # Should copy data
        if i > 1:
            d_bins[i-2] = c_bins[i-2]+1
            tot_prod0 = (tot_prod0 * (c_bins[i-2]+1)) // (c_bins[i-2])
        if i == 0: # R
            tot_ind_R = tot_prod0
            curr_cum_prod, _ = reverse_cum_prod(d_bins, l)
            for j in range(l):
                array_strides_R[j] = curr_cum_prod[j]
                array_sizes_R[j] = d_bins[j]
        elif i > 0: # C
            tot_prods_C[i-1] = tot_prod0
            curr_cum_prod, _ = reverse_cum_prod(d_bins, l)
            for j in range(l):
                array_strides_C[i-1, j] = curr_cum_prod[j]
                array_sizes_C[i-1, j] = d_bins[j]
        if i == l - 1: # L
            if i > 0:
                tot_prod0 = ( tot_prod0 * (c_bins[i-1]+1) ) // (c_bins[i-1])
                d_bins[i-1] = c_bins[i-1]+1
            tot_ind_L = tot_prod0
            curr_cum_prod, _ = reverse_cum_prod(d_bins, l)
            for j in range(l):
                array_strides_L[j] = curr_cum_prod[j]
                array_sizes_L[j] = d_bins[j]
    if l > 1:
        first_elements_C[0] = 0
        for i in range(1,l-1):
            first_elements_C[i] = first_elements_C[i-1] + tot_prods_C[i-1]
        tot_ind_C = first_elements_C[l-2]+tot_prods_C[l-2]
    else:
        tot_ind_C = 0
    return tot_ind_L, array_strides_L, array_sizes_L, tot_ind_C, first_elements_C, array_strides_C, array_sizes_C, tot_ind_R, array_strides_R, array_sizes_R

##### Interpolation Parameters ########################################################################
cpdef int is_odd(int num) nogil:
    return (num & 0x1) #bitwise and with hexadecimal 1

cpdef void is_odd_array(long[::1] num, int n) nogil:
    #cdef int n = num.shape[0]
    cdef int i
    for i in range(n):
        num[i] = (num[i] & 0x1)

cpdef int is_even(int num) nogil:
    return 1-(num & 0x1)

cpdef void is_even_array(long[::1] num, int n) nogil:
    #cdef int n = num.shape[0]
    cdef int i
    for i in range(n):
        num[i] = 1-(num[i] & 0x1)

cpdef void asym_sign(double[::1] nums, long[::1] signs, int n) nogil:
    #cdef int n = nums.shape[0]
    cdef int i
    for i in range(n):
        signs[i] = 1 if nums[i] >= 0 else -1

cpdef void asym_sign_2D(double[:,::1] nums, long[:,::1] signs, int n0, int n1) nogil:
    #cdef int n0 = nums.shape[0]
    #cdef int n1 = nums.shape[1]
    cdef int i, j
    for i in range(n0):
        for j in range(n1):
            signs[i,j] = 1 if nums[i,j] >= 0 else -1

cpdef void rounder(double a, int b) nogil:
    b = <int> a
    if a - b > 0.5:
        b += 1

cpdef void rounder_array(double[::1] a, long[::1] b, int n) nogil:
    #cdef int n = a.shape[0]
    cdef int i
    for i in range(n):
        b[i] = <int> a[i]
        if a[i] - b[i] > 0.5:
            b[i] += 1

cpdef void rounder_array_2d(double[:,::1] a, long[:,::1] b, int n0, int n1) nogil:
    #cdef int n0 = a.shape[0]
    #cdef int n1 = a.shape[1]
    cdef int i, j
    for i in range(n0):
        for j in range(n1):
            b[i,j] = <int> a[i,j]
            if a[i,j] - b[i,j] > 0.5:
                b[i,j] += 1

cpdef int is_in_interval(double[::1] c, long[::1] maxs, int n) nogil:
    cdef int inside = 1
    #cdef int n = c.shape[0]
    for i in range(n):
        if c[i] < 0 or c[i] > maxs[i]:
            inside = 0
            break
    return inside

cpdef int int_sum(long[::1] i_s, int n) nogil:
    #cdef int n = i_s.shape[0]
    cdef int s_i = 0
    cdef int i
    for i in range(n):
        s_i += i_s[i]
    return s_i

cpdef int int_max(long[::1] i_s, int n) nogil:
    #cdef int n = i_s.shape[0]
    cdef int m_i
    cdef int i
    if n > 0:
        m_i= i_s[0]
        for i in range(1,n):
            if i_s[i] > m_i:
                m_i = i_s[i]
    else:
        m_i = 0
    return m_i

cpdef int int_prod(long[::1] i_s, int n) nogil:
    #cdef int n = i_s.shape[0]
    cdef int p_i = i_s[0]
    cdef int i
    for i in range(1,n):
        p_i = p_i*i_s[i]
    return p_i

cpdef int_prod_array(long[:,::1] i_s, int n): # square
    cdef int i, j
    cdef long[::1] p_i = np.empty(i_s.shape[0], dtype=long)
    for i in range(n):
        p_i[i] = i_s[i, 0]
        for j in range(1,n):
            p_i[i] *= i_s[i, j]
    return p_i

cpdef int_prod_asym_array(long[:,::1] i_s): # square
    cdef int i, j
    cdef int[::1] p_i = np.empty(i_s.shape[0], dtype=int)
    for i in range(i_s.shape[0]):
        p_i[i] = i_s[i, 0]
        for j in range(1,i_s.shape[1]):
            p_i[i] *= i_s[i, j]
    return p_i

cpdef void c_abs_sum(double[::1] c, double[::1] abs_c, int n) nogil:
    cdef int i
    for i in range(n):
        abs_c[i] = fabs(c[i])

cpdef void c_int_sum(long[::1] c, int sum_c, int n) nogil:
    cdef int i
    sum_c = c[0]
    for i in range(1, n):
        sum_c += c[i]

cpdef void dvec_minus_ivec(double[::1] c, double[::1] dvec, long[:] ivec, int n) nogil:
    cdef int i
    for i in range(n):
        c[i] = dvec[i]-ivec[i]

cpdef void elementwise_grid_parameters(double[::1] c, double[::1] c_min, double[::1] dc, long[::1] c_bins, double[::1] alpha, int sum_location, long[::1] location, double[::1] alpha_rest, double[::1] abs_alpha_rest, long[::1] d_location) nogil:
    cdef int n = alpha.shape[0]
    cdef int i
    #sum_location = 0
    for i in range(n):
        # Transform
        alpha[i] = (c[i] - c_min[i]) / dc[i]
        # Round
        location[i] = <int> alpha[i]
        if alpha[i] - location[i] > 0.5:
            location[i] += 1
        sum_location += location[i]
        alpha_rest[i] = alpha[i] - location[i]
        abs_alpha_rest[i] = fabs(alpha_rest[i])
        d_location[i] = 1 if alpha_rest[i] >= 0 else -1
        if alpha[i] < 0 or alpha[i] > c_bins[i]:
            inside = 0
            break

cpdef int c_argmax(double[::1] c, int max_ind, double c_max, int n) nogil:
    cdef int i
    max_ind = 0
    c_max = c[0]
    for i in range(1,n):
        if c_max < c[i]:
            c_max = c[i]
            max_ind = i
    return max_ind
