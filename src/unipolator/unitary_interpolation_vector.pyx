#cython: language_level=3
import numpy as np
cimport numpy as npc
from libc.math cimport fabs
from .exp_and_log cimport *
from .indexing cimport *
from .caching cimport *
from .blas_functions cimport *
from .blas_functions_vectors cimport *
from .autobinning import optimal_binning


# Unitary Interpolation
cdef class UI_vector:
    # Initialize variables, to quickly calculate interpolations while minimizing memmory allocation overheads
    cdef double[::1] c_mins, c_maxs, dcs, das
    cdef long[::1] c_bins
    cdef int n_dims, d, d2, n_dims_1, n_d_di_1, n_d_di, m, dm
    cdef double[:, ::1] E
    cdef double complex[::1] Es
    cdef double complex[:,:,::1] Vr, Vl, CL, CH, dU1, dU2
    cdef double complex[:,::1] Ur, Ur1
    cdef double* ei
    cdef double complex *vl
    cdef double complex *vr
    cdef double complex *c
    cdef double complex *ur0
    cdef double complex *ur1
    cdef double complex *du1
    cdef double complex *du2
    cdef double complex *es0
    cdef long[::1] strides_L, strides_R,
    cdef long[:,::1] strides_E, strides_C
    cdef long[::1] location, d_location
    cdef double[::1] abs_alpha_rest, alpha
    cdef long[::1] first_elements_E, first_elements_C, L
    cdef long[::1] d_di
    def __cinit__(self, double complex[:,:,::1] H_s, double[::1] c_min_s, double[::1] c_max_s, long[::1] c_bins, long[::1] which_diffs = np.array([], dtype=long), int m = 1):
        # Construct parameters
        self.n_dims = c_min_s.shape[0]
        self.n_dims_1 = self.n_dims - 1
        self.d = H_s.shape[1]
        self.d2 = self.d * self.d
        if not H_s.shape[0] == self.n_dims + 1:
            print('Requires n+1 Hamiltonians for n dimensional interpolation. Check lenths of Hs, c_mins, c_maxs, c_bins')
            raise ValueError
        self.c_bins = np.empty(self.n_dims, dtype=long)
        for i in range(self.n_dims):
            self.c_bins[i] = c_bins[i]
        self.c_mins, self.c_maxs, self.dcs, self.das = Bin_Parameters(c_min_s, c_max_s, self.c_bins)
        # Single Step Indexing Parameters  --> Add multiple indexes later
        self.location = np.empty(self.n_dims, dtype=long)
        self.abs_alpha_rest = np.empty(self.n_dims, dtype=np.double)
        self.d_location = np.empty(self.n_dims, dtype=long)
        self.alpha = np.empty(self.n_dims, dtype=np.double)
        if which_diffs.shape[0] == 0:
            self.d_di = np.arange(self.n_dims)
        else:
            self.d_di = which_diffs
        self.n_d_di = self.d_di.shape[0]

        # Construct interpolation grid points
        U_grid, cum_prod = Unitary_Grid(H_s, self.c_mins, self.dcs, self.c_bins)
        ## Construct interpolation cache
        self.E, self.Vl, self.Vr, self.CL, self.CH, self.strides_E, self.strides_L, self.strides_R, self.strides_C, self.first_elements_E, self.first_elements_C = Create_Interpolation_Cache( U_grid, cum_prod, self.c_bins)
        self.ei = &self.E[0, 0]
        self.vl = &self.Vl[0, 0, 0]
        self.vr = &self.Vr[0, 0, 0]
        self.c = &self.CH[0, 0, 0]
        self.L = np.empty(self.n_dims, dtype=long)
        self.change_m(m)

    def change_m(self, int m):
        self.m = m
        self.dm = self.d * m
        # also change self.vi's
        self.Ur = np.empty([self.d, self.m], dtype=np.complex128)
        self.ur0 = &self.Ur[0, 0]
        self.Ur1 = np.empty([self.d, self.m], dtype=np.complex128)
        self.ur1 = &self.Ur1[0, 0]
        self.dU1 = np.empty([self.n_d_di, self.d, self.m], dtype=np.complex128)
        self.du1 = &self.dU1[0, 0, 0]
        self.dU2 = np.empty([self.n_d_di, self.d, self.m], dtype=np.complex128)
        self.du2 = &self.dU2[0, 0, 0]
        self.Es = np.empty(self.d, dtype=np.complex128)
        self.es0 = &self.Es[0]

    cdef single_parameters2oddgrid(self, double[::1] c):
        cdef int sum_location = 0
        cdef double alpha_max = 0.0
        cdef double alpha_rest
        cdef Py_ssize_t i
        cdef int max_alpha_ind = 0
        cdef int max_vals
        for i in range(self.n_dims):
            # Transform
            self.alpha[i] = (c[i] - self.c_mins[i]) / self.dcs[i]
            # Round
            self.location[i] = <int> self.alpha[i]     # floor -> 
            alpha_rest = self.alpha[i] - self.location[i]
            if alpha_rest > 0.5:
                self.location[i] += 1
                alpha_rest -= 1
            sum_location += self.location[i]

            self.abs_alpha_rest[i] = fabs(alpha_rest)
            self.d_location[i] = 1 if alpha_rest >= 0 else -1
            if self.alpha[i] < 0 or self.alpha[i] > self.c_bins[i]:
                print('Warning: These parameters lie outside of interpolation grid!')
                break
        if is_even(sum_location):
            max_alpha_ind = c_argmax(self.abs_alpha_rest, max_alpha_ind, alpha_max, self.n_dims)
            if self.location[max_alpha_ind] + self.d_location[max_alpha_ind] > self.c_bins[max_alpha_ind]:
                self.d_location[max_alpha_ind] = -1
            self.location[max_alpha_ind] += self.d_location[max_alpha_ind]
            self.d_location[max_alpha_ind] = - self.d_location[max_alpha_ind]
            self.abs_alpha_rest[max_alpha_ind] = 1-self.abs_alpha_rest[max_alpha_ind]
        for i in range(self.n_dims):
            self.d_location[i] -= 1
            self.d_location[i] >>= 1 # via bitshift operation
            #d_location[i] /= 2 #(d_location[i]-1)/2 ### in two steps with optimized in-place operations
        for i in range(self.n_dims):
            max_vals = self.location[i]+self.d_location[i]
            if max_vals > self.c_bins[i]-1:
                self.d_location[i] = -1

    def set_which_diffs(self, long[::1] which_diffs):
        self.d_di = which_diffs
        self.n_d_di_1 = self.d_di.shape[0] - 1
        self.n_d_di = self.d_di.shape[0]

    def get_single_param(self, double[::1] c): # To verify interpolation grid
        self.single_parameters2oddgrid(c)
        return np.asarray(self.location), np.asarray(self.d_location), np.asarray(self.abs_alpha_rest)

    def get_cache(self): # To verify interpolation grid
        return np.asarray(self.E), np.asarray(self.Vl), np.asarray(self.Vr), np.asarray(self.CL), np.asarray(self.CH), np.asarray(self.strides_E), np.asarray(self.strides_L), np.asarray(self.strides_R), np.asarray(self.strides_C)

    cdef interpolate_single_u(self, double complex *u0, double complex *v0): #u0 => input vectors, v0 => output vectors
        cdef Py_ssize_t i, j
        cdef Py_ssize_t ind
        if self.n_dims == 1:
            ind = self.location[0] + self.d_location[0]
            self.vr = &self.Vr[ind,0,0] # In 1D strides are 1
            self.vl = &self.Vl[ind,0,0]
            self.ei = &self.E[ind,0]
            MM_cdot_pointer_v(self.vr, u0, self.ur0, self.d, self.m)
            v_exp_pointer_v(self.ei, self.ur0, self.abs_alpha_rest[0], self.d, self.m)
            MM_cdot_pointer_v(self.vl, self.ur0, v0, self.d, self.m)
        else:
            # Right side first
            self.L[0] = self.location[0] + self.d_location[0]
            for i in range(1, self.n_dims):
                self.L[i] = self.location[i]
            ind = findex_0(self.L, self.strides_E[0,:], self.n_dims)
            self.vr = &self.Vr[ind, 0, 0]  # In 1D strides are 1
            MM_cdot_pointer_v(self.vr, u0, self.ur0, self.d, self.m)
            self.ei = &self.E[ind, 0]
            v_exp_pointer_v(self.ei, self.ur0, self.abs_alpha_rest[0], self.d, self.m)
            self.L[0] = self.location[0]
            # Center multiplications
            for i in range(self.n_dims_1):
                j = i + 1
                self.L[j] += self.d_location[j]
                ind = findex_0(self.L, self.strides_E[j, :], self.n_dims) + self.first_elements_E[j]
                self.ei = &self.E[ind, 0]
                if self.d_location[i]:  ### Higher
                    self.L[i] += -1
                    ind = findex_0(self.L, self.strides_C[i, :], self.n_dims) + self.first_elements_C[i]
                    self.c = &self.CL[ind, 0, 0]
                    self.L[i] += 1  # Restore value
                else:                        ### Lower
                    ind = findex_0(self.L, self.strides_C[i, :], self.n_dims)+self.first_elements_C[i]
                    self.c = &self.CH[ind,0,0]
                self.L[j] = self.location[j]
                MM_cdot_pointer_v(self.c, self.ur0, self.ur1, self.d, self.m)
                v_exp_pointer_v(self.ei, self.ur1, self.abs_alpha_rest[j], self.d, self.m)
                self.ur1, self.ur0 = self.ur0, self.ur1
            # Left side multiplication
            self.L[self.n_dims_1] += self.d_location[self.n_dims_1]
            ind = findex_0(self.L, self.strides_L, self.n_dims)
            self.vl = &self.Vl[ind,0,0]
            MM_cdot_pointer_v(self.vl, self.ur0, v0, self.d, self.m)

    cdef interpolate_single_u_du(self, double complex *u0, double complex *v0, double complex *du0): #u0, du0 => input the matrices for output   ##int[::1] d_di, -> define earlier
        # d_di contains the indexes of the derivatives that we want to calculate
        cdef Py_ssize_t i, j, ind, curr_d_di
        cdef int curr_d_ind = 0
        cdef double complex *du1 = self.du1 #.copy()
        cdef double complex *du2 = self.du2 #.copy()
        cdef double complex *ur0 = self.ur0
        cdef double complex *ur1 = self.ur1
        cdef double complex *es0 = self.es0
        if self.n_dims == 1:
            ind = self.location[0] + self.d_location[0]
            self.vr = &self.Vr[ind, 0, 0] # In 1D strides are 1
            self.vl = &self.Vl[ind, 0, 0]
            self.ei = &self.E[ind, 0]
            MM_cdot_pointer_v(self.vr, u0, ur0,  self.d, self.m)
            if self.d_di[curr_d_ind] == 0:
                copy_pointer(ur0, du1, self.dm)
                v_exp_v_pointer_v( (self.d_location[0]*2+1)*self.das[0], self.ei, du1, self.abs_alpha_rest[0], self.d, self.m) # -i*E*amp*exp(-i*E*t)
                MM_cdot_pointer_v(self.vl, self.du1, du0, self.d, self.m)
            v_exp_pointer_v(self.ei, ur0, self.abs_alpha_rest[0], self.d, self.m)
            MM_cdot_pointer_v(self.vl, ur0, v0, self.d, self.m)
        else:
            # Track the number of performed derivatives via curr_d_ind, the indexes to perform with self.d_di and the next derivative to perform with self.d_di[curr_d_ind] == curr_d_di
            # Right side first
            self.L[0] = self.location[0] + self.d_location[0]
            for i in range(1, self.n_dims):
                self.L[i] = self.location[i]
            ind = findex_0(self.L, self.strides_E[0,:], self.n_dims)
            self.vr = &self.Vr[ind, 0, 0]  # In 1D strides are 1
            self.ei = &self.E[ind, 0]

            MM_cdot_pointer_v(self.vr, u0, ur0, self.d, self.m) # now ur0 and du1
            curr_d_di = self.d_di[curr_d_ind]
            if curr_d_di == 0:
                copy_pointer(ur0, du1, self.dm)
                v_exp_v_pointer_v( (self.d_location[0]*2+1)*self.das[0], self.ei, du1, self.abs_alpha_rest[0], self.d, self.m) # -i*E*amp*exp(-i*E*t)
                curr_d_ind += 1
                curr_d_di = self.d_di[curr_d_ind]
            v_exp_pointer_v(self.ei, ur0, self.abs_alpha_rest[0], self.d, self.m) # still ur0 and du1

            self.L[0] = self.location[0]
            # Center multiplications
            for i in range(self.n_dims_1):
                j = i + 1
                self.L[j] += self.d_location[j]
                ind = findex_0(self.L, self.strides_E[j, :], self.n_dims) + self.first_elements_E[j]
                self.ei = &self.E[ind, 0]
                if self.d_location[i]:  ### Higher
                    self.L[i] += -1
                    ind = findex_0(self.L, self.strides_C[i, :], self.n_dims) + self.first_elements_C[i]
                    self.c = &self.CL[ind, 0, 0]
                    self.L[i] += 1  # Restore value
                else:                        ### Lower
                    ind = findex_0(self.L, self.strides_C[i, :], self.n_dims)+self.first_elements_C[i]
                    self.c = &self.CH[ind,0,0]
                self.L[j] = self.location[j]
                # Batch matrix multiply
                MM_cdot_pointer_batch_v(self.c, ur0, ur1, du1, du2, self.d, self.m, self.dm, curr_d_ind) # now ur1 and du2
                # Perform the row wise complex scaling, with potential derivative
                if curr_d_di == j: # Add derivative here
                    v_exp_v_and_v_exp_pointer_v_batch_increment((self.d_location[curr_d_di]*2+1)*self.das[curr_d_di], self.ei, es0, ur1, du2, self.abs_alpha_rest[j], self.d, self.m, self.dm, curr_d_ind)
                    curr_d_ind += 1
                    curr_d_di = self.d_di[curr_d_ind]
                else:
                    v_exp_v_and_v_exp_pointer_v_batch(self.ei, self.es0, ur1, du2, self.abs_alpha_rest[j], self.d, self.m, curr_d_ind)
                ur1, ur0 = ur0, ur1 # now flip them
                du2, du1 = du1, du2
            # Left side multiplication
            self.L[self.n_dims_1] = self.location[self.n_dims_1] + self.d_location[self.n_dims_1]
            ind = findex_0(self.L, self.strides_L, self.n_dims)
            self.vl = &self.Vl[ind,0,0]
            MM_cdot_pointer_batch_v(self.vl, ur0, v0, du1, du0, self.d, self.m, self.dm, curr_d_ind) # now ur1 and du2
            

    cdef expmH_pointer(self, double[::1] c, double complex *u0, double complex *v0):
        self.single_parameters2oddgrid(c)
        self.interpolate_single_u(u0, v0)
    def expmH(self, double[::1] c, double complex[:,::1] V_in, double complex[:,::1] V_out):
        if not c.shape[0] == self.n_dims:
            raise ValueError('c.shape[0] needs to be equal to H_s[0].shape[0]-1.')
        # check V_in size
        if not V_in.shape[0] == self.d:
            raise ValueError('V_in.shape[0] needs to be equal to H_s[0].shape[0].')
        if not V_in.shape[1] == self.m:
            # change m
            self.change_m(V_in.shape[1])
        # check V_out size
        if not V_out.shape[0] == self.d:
            raise ValueError('V_out.shape[0] needs to be equal to H_s[0].shape[0].')
        if not V_out.shape[1] == self.m:
            raise ValueError('V_out.shape[1] needs to be equal to V_in.shape[1].')
        cdef double complex *u0 = &V_in[0, 0]
        cdef double complex *v0 = &V_out[0, 0]
        self.expmH_pointer(c, u0, v0)

    cdef dexpmH_pointer(self, double[::1] c, double complex *u0, double complex *v0, double complex *du0):  #int[::1] d_di,
        # d_di contains the indexes of the derivatives that we want to calculate (needs to be in ascending order with a negative value at the end)
        self.single_parameters2oddgrid(c)
        self.interpolate_single_u_du(u0, v0, du0)

    def dexpmH(self, double[::1] c, double complex[:,::1] V_in, double complex[:,::1] V_out, double complex[:,:,::1] dU):  #int[::1] d_di,
        # d_di contains the indexes of the derivatives that we want to calculate (needs to be in ascending order with a negative value at the end)
        cdef double complex *u0 = &V_in[0,0]
        cdef double complex *v0 = &V_out[0,0]
        cdef double complex *du0 = &dU[0,0,0]
        if not c.shape[0] == self.n_dims:
            raise ValueError('The coefficient c must be of size [interpolation_dimensions].')
        if not self.n_d_di == dU.shape[0]:
            raise ValueError('Inputs must fulfill: which_diffs.shape[0] = dU.shape[0].')
        if not V_in.shape[0] == self.d:
            raise ValueError('The input vector must be of size [d].')
        if not V_out.shape[0] == self.d:
            raise ValueError('The output vector must be of size [d].')
        if not V_in.shape[1] == V_out.shape[1] == self.m:
            raise ValueError('The input and output vectors must have the same number of columns, and equal to m.')
        if not dU.shape[1] == self.d:
            raise ValueError('The input derivative vector must be of size [d].')
        if not dU.shape[2] == self.m:
            raise ValueError('The input derivative vector must have the same number of columns as the input and output vectors, and equal to m.')
        if not self.n_d_di == dU.shape[0]:
            raise ValueError('The input derivative vector must have the same number of rows as the number of derivatives to calculate.')
        self.dexpmH_pointer(c, u0, v0, du0)


    cdef expmH_pulse_pointer(self, double[:,::1] cs, double complex *u0, double complex *v0):
        cdef Py_ssize_t i = 0
        cdef int steps = cs.shape[0]
        cdef int steps_2 = steps >> 1  # floor (steps/2)
        cdef int is_odd_steps = steps % 2
        if not cs.shape[1] == self.n_dims:
            raise ValueError('The coefficient matrix must be of size [n_time_steps, interpolation_dimensions].')
        for i in range(0, steps_2, 2):
            self.single_parameters2oddgrid(cs[i,:])
            self.interpolate_single_u(u0, v0)
            self.single_parameters2oddgrid(cs[i+1,:])
            self.interpolate_single_u(v0, u0)
        if is_odd_steps:
            self.single_parameters2oddgrid(cs[steps_2,:])
            self.interpolate_single_u(u0, v0)
        else:  # copy to v0 
            copy_pointer(u0, v0, self.dm)

    def expmH_pulse(self, double[:,::1] cs, double complex[:,::1] V_in, double complex[:,::1] V_out):
        cdef double complex *u0 = &V_in[0, 0]
        cdef double complex *v0 = &V_out[0, 0]
        self.expmH_pulse_pointer(cs, u0, v0)


    def expmH_pulse_no_multiply(self, double[:,::1] cs, double complex[:,::1] V_in, double complex[:,:,::1] V_out):
        cdef double complex *u0 = &V_in[0,0]
        cdef double complex *v0 = &V_out[0,0,0]
        cdef n = cs.shape[0]
        for i in range(n):
            self.expmH_pointer(cs[i,:], u0, v0)
            v0 += self.dm
    """
    def grape(self, double[:,::1] cs, double complex[:,::1] U_target, int[::1] target_indexes, double complex[:,::1] U, double complex[:,:,::1] dU, double[:,::1] dI_dj):
        # Calculate fidelity for a pulse and the differentials of the fidelity at every timestep using the grape trick
        cdef Py_ssize_t i, j
        cdef int steps = cs.shape[0]

        cdef double complex *new_p0 = self.ur0
        cdef double complex *new_q0 = self.ur1
        cdef double complex *p0 = self.ur4
        cdef double complex *q0 = self.ur5
        cdef double complex *u0 = &U[0, 0]
        cdef double complex *u_tar = &U_target[0, 0]
        cdef double complex *curr_u = self.ur3
        cdef double complex *du = &dU[0,0,0]

        cdef double complex trM, trdM, two_nni
        cdef double n = <double> target_indexes.shape[0]
        cdef double nni = 1.0 / (n * (n + 1))
        cdef double one_minus = 1.0 - n * nni

        if not dI_dj.shape[0] == steps:
            raise ValueError('Inputs must fulfill: cs.shape[0] = dI_dj.shape[0].')
        if not dI_dj.shape[1] == self.n_d_di:
            raise ValueError('Inputs must fulfill: which_diffs.shape[0] = dI_dj.shape[1].')
        if not cs.shape[1] == self.n_d_di:
            raise ValueError('Inputs must fulfill: which_diffs.shape[0] = cs.shape[1].')
        # Fidelity constants

        self.expmH_pulse_pointer(cs, u0)
        #Dag_fast(U_target, self.Ur3) # -> Ur3 is identical to Q  -> replaced by new approach
        DagM_M_cdot_pointer(u_tar, u0, q0, self.d)
        trM = target_indexes_trace_pointer(q0, self.d, target_indexes)
        two_nni = - 2 / (n * (n + 1)) * conj(trM)
        cdef double I0 = one_minus - nni * abs_2(trM) # Sub Fidelity

        # Do GRAPE trick
        self.dexpmH_pointer(cs[0,:], curr_u, du) # Calculate Unitary and its derivatives
        M_DagM_cdot_pointer(q0, curr_u, new_q0, self.d)  # Reduce q0
        copy_pointer(new_q0, q0, self.d2)
        for j in range(self.n_d_di):
            #MM_cdot_pointer(curr_du, p0, new_p0, self.d)  # p0 is not defined yet
            trdM = tr_dot_pointer_target_indexes(q0, du, self.d, target_indexes)
            dI_dj[0, j] = ( two_nni * trdM ).real
            du += self.d2
        copy_pointer(curr_u, p0, self.d2)  # Define p0 from curr_u
        for i in range(1, steps):
            du -= self.d2 * self.n_d_di
            self.dexpmH_pointer(cs[i, :], curr_u, du)  # Calculate Unitary and its derivatives
            M_DagM_cdot_pointer(q0, curr_u, new_q0, self.d)  # Remove current unitary (curr_u) from Q
            for j in range(self.n_d_di):
                MM_cdot_pointer(du, p0, new_p0, self.d)
                trdM = tr_dot_pointer_target_indexes(new_q0, new_p0, self.d, target_indexes)
                dI_dj[i, j] = ( two_nni * trdM ).real
                du += self.d2
            MM_cdot_pointer(curr_u, p0, new_p0, self.d)  # Add current unitary (curr_u) to P
            #copy_pointer(new_p0, p0, self.d2)
            #copy_pointer(new_q0, q0, self.d2)
            # flip pointers
            self.ur0, self.ur4 = self.ur4, self.ur0
            self.ur1, self.ur5 = self.ur5, self.ur1
            new_p0, p0 = p0, new_p0
            new_q0, q0 = q0, new_q0
        return I0
    """


def UI_vector_auto(H_s, c_min_s, c_max_s, I_tar=1e-10, which_diffs = np.array([], dtype=np.compat.long), m = 1):
    opt_bins  = optimal_binning(H_s, c_mins=c_min_s, c_maxs=c_max_s, I_tar=I_tar)
    return UI_vector(H_s, c_min_s, c_max_s, opt_bins, which_diffs, m = 1)
