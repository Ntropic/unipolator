#cython: language_level=3
from __future__ import print_function
cimport cython
import numpy as np
from .blas_functions cimport c_eigh_lapack_workspace_sizes, d_mat_add, M_DagM_cdot, MM_cdot, DagM_M_cdot
from .exp_and_log cimport c_expmH, Make_U_Partial, Dag, Partialize_Center, Partialize_Sides, copy_pointer  # Find correct files for these functions
from .indexing cimport next_in_mrange, findex_0, reverse_cum_prod, flat_index_sizes_E, flat_index_sizes_V, int_max, int_prod_array, int_sum, is_even         # Find correct files for these functions


##### Create Interpolation Cache #######################################################################################
cpdef Unitary_Grid(double complex[:,:,::1] H, double[::1] c_mins, double[::1] dcs, long[::1] c_bins): # s is the number of interpolation points
    cdef int n = H.shape[1]
    cdef int l = c_bins.shape[0]
    if not H.shape[0]  == l + 1:
        print('Error: The number of Hamiltonians needs to be one higher than the number of control parameters')
        raise ValueError
    cdef long[:] d_bins = np.empty(l, dtype=long)
    cdef int i, j # int
    for i in range(l):
        d_bins[i] = c_bins[i] + 1
    cdef long[:] cum_prod = np.empty(l, dtype=long)
    cdef int tot_prod
    cum_prod, tot_prod = reverse_cum_prod(d_bins, l)
    cdef double complex[:,:,::1] U_grid = np.empty([tot_prod, n, n], dtype=np.complex128)
    cdef double complex[:,::1] H_i_s = np.empty([n, n], dtype=np.complex128)
    cdef long[:] i_s = np.zeros(l, dtype=long)
    cdef double d
    cdef double dt = 1.0
    work, rwork, iwork = c_eigh_lapack_workspace_sizes(H_i_s)
    for i in range(tot_prod):
        H_i_s[:,:] = H[0, :, :]
        for j in range(l):
            d = (c_mins[j] + dcs[j] * i_s[j])
            d_mat_add(H_i_s, H[<int> j+1,:,:], d)
        #print(str(findex_0(i_s, cum_prod, l)))
        c_expmH(H_i_s, dt, U_grid[findex_0(i_s, cum_prod, l), :, :], work, rwork, iwork)
        next_in_mrange(i_s, d_bins) # returns new i_s -> always apply at the end of the loop
    return U_grid, cum_prod


#### Needs to be updates
cpdef Create_Interpolation_Cache(double complex[:,:,::1] U_grid, long[::1] grid_cum_prod, long[::1] c_bins):
    # Create interpolation arrows
    cdef int n = U_grid.shape[1]
    cdef int l = c_bins.shape[0]
    cdef int ind, i_1
    cdef long[::1] i_s = np.empty(l, dtype = long)
    cdef long[::1] i_s2 = np.empty(l, dtype = long)
    cdef long[::1] j_s = np.empty(l, dtype = long)
    cdef long[::1] k_s = np.empty(l, dtype = long)
    cdef long[::1] curr_arrow_index = np.empty(l, dtype = long)
    cdef int i, j  # int
    cdef int tot_ind_E, tot_ind_L, tot_ind_C, tot_ind_R
    cdef long[::1] first_elements_E = np.empty(l, dtype = long)
    cdef long[::1] first_elements_C = np.empty(l-1, dtype = long)

    cdef long[:,::1] array_strides_E = np.empty([l,l], dtype = long)
    cdef long[::1] array_strides_L = np.empty(l, dtype = long)
    cdef long[::1] array_strides_R = np.empty(l, dtype = long)
    cdef long[:,::1] array_strides_C = np.empty([l-1,l], dtype = long)

    cdef long[:,::1] array_sizes_E = np.empty([l,l], dtype = long)
    cdef long[::1] array_sizes_L = np.empty(l, dtype = long)
    cdef long[::1] array_sizes_R = np.empty(l, dtype = long)
    cdef long[:,::1] array_sizes_C = np.empty([l-1,l], dtype = long)

    tot_ind_E, first_elements_E, array_strides_E, array_sizes_E = flat_index_sizes_E(c_bins)
    tot_ind_L, array_strides_L, array_sizes_L, tot_ind_C, first_elements_C, array_strides_C, array_sizes_C, tot_ind_R, array_strides_R, array_sizes_R = flat_index_sizes_V(c_bins)
    # Initialize cache matrices
    cdef double[:, ::1] E = np.empty([tot_ind_E, n], dtype = np.double)
    cdef double complex[:,:,::1] Vr = np.empty([tot_ind_R, n, n], dtype = np.complex128)
    cdef double complex[:,:,::1] Vl = np.empty([tot_ind_L, n, n], dtype = np.complex128)
    cdef double complex[:,:,::1] CL = np.empty([tot_ind_C, n, n], dtype = np.complex128)
    cdef double complex[:,:,::1] CH = np.empty([tot_ind_C, n, n], dtype = np.complex128)
    # Initialize Intermediate variables
    cdef double complex[:,::1] U_0     = np.empty([n,n], dtype = np.complex128)
    cdef double complex[:,::1] U_1     = np.empty([n,n], dtype = np.complex128)
    cdef double complex[:,::1] V       = np.empty([n,n], dtype = np.complex128)
    cdef double complex[:,::1] logU    = np.empty([n,n], dtype = np.complex128)
    cdef double complex[:,::1] V_tmp   = np.empty([n,n], dtype = np.complex128)
    cdef double [::1]         logE_tmp = np.empty(n, dtype = np.double)
    cdef double[::1]                Es = np.empty(n, dtype = np.double)
    cdef double complex[::1]        Ec = np.empty(n, dtype = np.complex128)
    cdef double complex[:,::1]    Emat = np.empty([n,n], dtype = np.complex128)
    cdef long[::1] strides       = np.empty(l, dtype = long)
    cdef long[::1] prev_strides  = np.empty(l, dtype = long)
    cdef long[::1] strides_V     = np.empty(l, dtype = long)
    cdef double complex[:,::1] L = np.empty([n,n], dtype = np.complex128)
    cdef long[::1] sub_mat_length = int_prod_array(array_sizes_E, l)
    cdef int max_tot_ind = int_max(sub_mat_length, l)
    cdef int prod_array_sizes
    cdef double complex[:,:,::1] curr_V = np.empty([max_tot_ind,n,n], dtype = np.complex128)
    cdef double complex[:,:,::1] prev_V = np.empty([max_tot_ind,n,n], dtype = np.complex128)
    # Get eig parameters
    for i in range(l): # One arrow cache for every direction in grid
        sizes = array_sizes_E[i,:]
        strides = array_strides_E[i,:]
        i_1 = i - 1
        if i > 0:
            prev_strides[:] = array_strides_E[i_1, :]
            strides_V[:] = array_strides_C[i_1, :]
        i_s = np.zeros(l, dtype = long) # Initialize as zero -> update at end of loop via next_in_mrange
        prod_array_sizes = sub_mat_length[i]
        for j in range(prod_array_sizes):
            for k in range(l):
                if not k == i:
                    j_s[k] = i_s[k]
                else:
                    j_s[k] = i_s[k]+1
            if is_even(int_sum(i_s, l)):
                curr_arrow_index = j_s
                U_1 = U_grid[findex_0(i_s, grid_cum_prod, l), :, :]
                U_0 = U_grid[findex_0(j_s, grid_cum_prod, l), :, :]
            else:
                curr_arrow_index = i_s
                U_0 = U_grid[findex_0(i_s, grid_cum_prod, l), :, :]
                U_1 = U_grid[findex_0(j_s, grid_cum_prod, l), :, :]
            Make_U_Partial(U_1, U_0, Emat, Ec, logE_tmp, V_tmp, Es, V) # returns E, V
            curr_V[findex_0(i_s, strides, l), :, :] = V
            ind = findex_0(i_s, strides, l) + first_elements_E[i]
            E[ind, :] = Es

            if i == 0: # Add right side only for i == 0
                MM_cdot(V, U_0, Emat)
                Vr[findex_0(i_s, strides, l),:,:] = Emat
            else:
                if i_s[i_1] > 0: #Lowering is possible
                    for k in range(l):
                        if not k == i-1:
                            k_s[k] = curr_arrow_index[k]
                            i_s2[k] = i_s[k]
                        else:
                            k_s[k] = curr_arrow_index[k] - 1
                            i_s2[k] = i_s[k] - 1
                    i_s2 = i_s.copy()
                    i_s2[i_1] += -1
                    M_DagM_cdot(V, prev_V[findex_0(k_s, prev_strides, l), :, :], Emat)
                    ind = findex_0(i_s2, strides_V, l) + first_elements_C[i_1]
                    CL[ind,:,:] =  Emat
                if i_s[i_1] < c_bins[i_1]: # Increasing is possible
                    M_DagM_cdot(V, prev_V[findex_0(curr_arrow_index, prev_strides,l), :, :], Emat)
                    ind = findex_0(i_s, strides_V, l) + first_elements_C[i_1]
                    CH[ind,:,:] = Emat      #dot(Dag(V), prev_VH)
            if i == l-1:
                L = Dag(V)
                Vl[findex_0(i_s, strides, l),:,:] = L #Dag(V)
            next_in_mrange(i_s, sizes)   # Increase i_s counter
        prev_V[:,:,::1] = curr_V
    return E, Vl, Vr, CL, CH, array_strides_E, array_strides_L, array_strides_R, array_strides_C, first_elements_E, first_elements_C


#### Symmetric Unitary Interpolation
cpdef Create_Sym_Interpolation_Cache(double complex[:,:,::1] U_grid2, long[::1] grid_cum_prod, long[::1] c_bins):
    # U_grid2 = sqrt(U_grid)   -> for half steps
    # Create interpolation arrows
    U_grid = 1
    cdef int n = U_grid2.shape[1]
    cdef int l = c_bins.shape[0]
    cdef int n2 = n*n
    cdef int ind, i_1
    cdef long[::1] i_s = np.empty(l, dtype = long)
    cdef long[::1] i_s2 = np.empty(l, dtype = long)
    cdef long[::1] j_s = np.empty(l, dtype = long)
    cdef long[::1] k_s = np.empty(l, dtype = long)
    cdef long[::1] curr_arrow_index = np.empty(l, dtype = long)
    cdef int i, j  # int
    cdef int tot_ind_E, tot_ind_C, tot_ind_L
    cdef long[::1] first_elements_E = np.empty(l, dtype = long)
    cdef long[::1] first_elements_C = np.empty(l-1, dtype = long)

    cdef long[:,::1] array_strides_E = np.empty([l,l], dtype = long)
    cdef long[::1] array_strides_L   = np.empty(l, dtype = long)  # Also valid for L
    cdef long[:,::1] array_strides_C = np.empty([l-1,l], dtype = long)

    cdef long[:,::1] array_sizes_E = np.empty([l,l], dtype = long)
    cdef long[::1] array_sizes_L   = np.empty(l, dtype = long)    # Also valid for L
    cdef long[:,::1] array_sizes_C = np.empty([l-1,l], dtype = long)

    tot_ind_E, first_elements_E, array_strides_E, array_sizes_E = flat_index_sizes_E(c_bins)
    tot_ind_L, array_strides_L, array_sizes_L, tot_ind_C, first_elements_C, array_strides_C, array_sizes_C, _, _, _ = flat_index_sizes_V(c_bins)

    # Initialize cache matrices
    cdef double[:, ::1] E   = np.empty([tot_ind_E, n], dtype = np.double)
    cdef double complex[:,:,::1] Vr      = np.empty([tot_ind_L, n, n], dtype = np.complex128)
    cdef double complex[:,:,::1] Vl      = np.empty([tot_ind_L, n, n], dtype = np.complex128)
    cdef double complex[:,:,::1] CL_R    = np.empty([tot_ind_C, n, n], dtype = np.complex128)
    cdef double complex[:,:,::1] CH_R    = np.empty([tot_ind_C, n, n], dtype = np.complex128)
    cdef double complex[:,:,::1] CL_L    = np.empty([tot_ind_C, n, n], dtype = np.complex128)
    cdef double complex[:,:,::1] CH_L    = np.empty([tot_ind_C, n, n], dtype = np.complex128)
    # Initialize Intermediate variables
    cdef double complex[:,::1] U_0       = np.empty([n,n], dtype = np.complex128)
    cdef double complex[:,::1] U_1       = np.empty([n,n], dtype = np.complex128)
    cdef double complex[:,::1] V_L       = np.empty([n,n], dtype = np.complex128)
    cdef double complex            *v_l0 = &V_L[0, 0]
    cdef double complex[:,::1] V_R       = np.empty([n,n], dtype = np.complex128)
    cdef double complex            *v_r0 = &V_R[0, 0]
    cdef double complex[:,::1] logU      = np.empty([n,n], dtype = np.complex128)
    cdef double complex[:,::1]   V_tmp   = np.empty([n,n], dtype = np.complex128)
    cdef double [::1]           logE_tmp = np.empty(n, dtype = np.double)
    cdef double[::1]                Es   = np.empty([n], dtype = np.double)
    cdef double complex[::1]        Ec   = np.empty([n], dtype = np.complex128)    # Intermediate variable
    cdef double complex[:,::1]    Emat   = np.empty([n,n], dtype = np.complex128)  # Intermediate variable
    #cdef double complex[:,::1]    U1     = np.empty([n,n], dtype = np.complex128)  # Intermediate variable
    cdef long[::1] strides        = np.empty(l, dtype = long)
    cdef long[::1] prev_strides   = np.empty(l, dtype = long)
    cdef long[::1] strides_V      = np.empty(l, dtype = long)
    cdef long[::1] sub_mat_length = int_prod_array(array_sizes_E, l)
    cdef int max_tot_ind         = int_max(sub_mat_length, l)

    cdef int prod_array_sizes
    cdef double complex[:,:,::1] curr_V_L  = np.empty([max_tot_ind,n,n], dtype = np.complex128)
    cdef double complex[:,:,::1] prev_V_L  = np.empty([max_tot_ind,n,n], dtype = np.complex128)
    cdef double complex[:,:,::1] curr_V_R  = np.empty([max_tot_ind,n,n], dtype = np.complex128)
    cdef double complex[:,:,::1] prev_V_R  = np.empty([max_tot_ind,n,n], dtype = np.complex128)

    for i in range(l): # One arrow cache for every direction in grid
        sizes = array_sizes_E[i, :]
        strides = array_strides_E[i, :]
        i_1 = i - 1
        if i > 0:
            prev_strides[:] = array_strides_E[i_1, :]
            strides_V[:] = array_strides_C[i_1, :]
        i_s = np.zeros(l, dtype = long) # Initialize as zero -> update at end of loop via next_in_mrange
        prod_array_sizes = sub_mat_length[i]

        for j in range(prod_array_sizes):
            for k in range(l):
                if not k == i:
                    j_s[k] = i_s[k]
                else:
                    j_s[k] = i_s[k]+1
            # Select matrices from grid (half step matrices)
            if is_even(int_sum(i_s, l)):
                curr_arrow_index = j_s
                U_1 = U_grid2[findex_0(i_s, grid_cum_prod, l), :, :]
                U_0 = U_grid2[findex_0(j_s, grid_cum_prod, l), :, :]
            else:
                curr_arrow_index = i_s
                U_0 = U_grid2[findex_0(i_s, grid_cum_prod, l), :, :]
                U_1 = U_grid2[findex_0(j_s, grid_cum_prod, l), :, :]

            if i == 0: # Central element (sqrt(U0) @ U1 @ sqrt(U0)) -> sqrt disappears in formulas as we are passing half steps to this function in the first place
                Partialize_Center(U_1, U_0, logU, Emat, Ec, logE_tmp, V_tmp, Es, V_L)  # returns E, V from decomposition of product: Dag(U_0) @ U_1^2 @ Dag(U_0)
                copy_pointer(v_l0, v_r0, n2)
            else:  # Calculate left and right hand side of this
                Partialize_Sides(U_1, U_0, Emat, Ec, logE_tmp, V_tmp, Es, V_L, V_R)
            ind = findex_0(i_s, strides, l) + first_elements_E[i]
            E[ind, :] = Es
            curr_V_L[findex_0(i_s, strides, l), :, :] = V_L
            curr_V_R[findex_0(i_s, strides, l), :, :] = V_R

            if i > 0:
                if i_s[i_1] > 0: # Lowering is possible
                    for k in range(l):
                        if not k == i-1:
                            k_s[k] = curr_arrow_index[k]
                            i_s2[k] = i_s[k]
                        else:
                            k_s[k] = curr_arrow_index[k] - 1
                            i_s2[k] = i_s[k] - 1
                    i_s2 = i_s.copy()
                    i_s2[i_1] += -1
                    ind = findex_0(i_s2, strides_V, l) + first_elements_C[i_1]
                    M_DagM_cdot(V_L, prev_V_L[findex_0(k_s, prev_strides, l), :, :], Emat)
                    CL_L[ind, :, :] = Emat
                    M_DagM_cdot(prev_V_R[findex_0(k_s, prev_strides, l), :, :], V_R, Emat)
                    CL_R[ind,:,:] =  Emat
                if i_s[i_1] < c_bins[i_1]: # Increasing is possible
                    ind = findex_0(i_s, strides_V, l) + first_elements_C[i_1]
                    M_DagM_cdot(V_L, prev_V_L[findex_0(curr_arrow_index, prev_strides, l), :, :], Emat)
                    CH_L[ind, :, :] = Emat
                    M_DagM_cdot(prev_V_R[findex_0(curr_arrow_index, prev_strides, l), :, :], V_R, Emat)
                    CH_R[ind, :, :] = Emat
            
            if i == l-1:
                M_DagM_cdot(U_0, V_L, Emat)
                Vl[findex_0(i_s, strides, l), :, :] = Emat
                MM_cdot(V_R, U_0, Emat)
                Vr[findex_0(i_s, strides, l), :, :] = Emat
            next_in_mrange(i_s, sizes)    # Increase i_s counter
        prev_V_L[:,:,::1] = curr_V_L
        prev_V_R[:,:,::1] = curr_V_R
    return E, Vl, Vr, CL_L, CL_R, CH_L, CH_R, array_strides_E, array_strides_L, array_strides_C, first_elements_E, first_elements_C
