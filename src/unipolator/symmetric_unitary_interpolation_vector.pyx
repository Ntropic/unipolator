#cython: language_level=3
import numpy as np
cimport numpy as npc
from libc.math cimport fabs
from .exp_and_log cimport *
from .indexing cimport *
from .caching cimport *
from .blas_functions cimport *
from .blas_functions_vectors cimport *

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# Unitary Interpolation
cdef class Sym_UI_vector:
    # Initialize variables, to quickly calculate interpolations while minimizing memory allocation overheads
    cdef double[::1] c_mins, c_maxs, dcs, das
    cdef npc.intp_t[::1] c_bins               # CHANGED: was long[::1]
    cdef int n_dims, d, d2, n_dims_1, n_dims_2, n_d_di_1, n_d_di, m, dm
    cdef double[:, ::1] E
    cdef double complex[:,:,::1] Vr, Vl, CL_L, CH_L, CL_R, CH_R, dUl, dUr
    cdef double complex[:,::1] Ur, Ur1, dUc
    cdef double complex[::1] Es   # Consider removing one of the two
    cdef double *ei
    cdef double complex *expe
    cdef double complex *vl
    cdef double complex *vr
    cdef double complex *cl
    cdef double complex *cr
    cdef double complex *ur0
    cdef double complex *ur1
    cdef double complex *dul
    cdef double complex *dur
    cdef double complex *duc
    cdef npc.intp_t[::1] strides_L        # CHANGED: was long[::1]
    cdef npc.intp_t[:,::1] strides_E, strides_C   # CHANGED: was long[:,::1]
    cdef npc.intp_t[::1] location, d_location       # CHANGED: was long[::1]
    cdef double[::1] abs_alpha_rest, alpha
    cdef npc.intp_t[::1] first_elements_E, first_elements_C, L    # CHANGED: was long[::1]
    cdef npc.intp_t[::1] d_di                    # CHANGED: was long[::1]

    def __cinit__(self, double complex[:,:,::1] H_s, double[::1] c_min_s, double[::1] c_max_s, npc.intp_t[::1] c_bins, npc.intp_t[::1] which_diffs = np.array([], dtype=np.intp), int m = 1):
        # Construct parameters
        self.n_dims = c_min_s.shape[0]
        self.n_dims_1 = self.n_dims - 1
        self.n_dims_2 = self.n_dims - 2
        self.d = H_s.shape[1]
        self.d2 = self.d * self.d

        if not H_s.shape[0] == self.n_dims + 1:
            print('Requires n+1 Hamiltonians for n dimensional interpolation. Check lengths of Hs, c_mins, c_maxs, c_bins')
            raise ValueError

        self.c_bins = np.empty(self.n_dims, dtype=np.intp)    # CHANGED: dtype=np.intp
        for i in range(self.n_dims):
            if c_bins[i] > 0:
                self.c_bins[i] = c_bins[i]
            else:
                print('Need at least 1 bin per dimension, corrected bins[' + str(i) + '] to 1.')
                self.c_bins[i] = 1  # Ensure at least 1 bin

        self.c_mins, self.c_maxs, self.dcs, self.das = Bin_Parameters(c_min_s, c_max_s, self.c_bins)

        # Single Step Indexing Parameters  --> Add multiple indexes later
        self.location = np.empty(self.n_dims, dtype=np.intp)           # CHANGED: dtype=np.intp
        self.abs_alpha_rest = np.empty(self.n_dims, dtype=np.double)
        self.d_location = np.empty(self.n_dims, dtype=np.intp)         # CHANGED: dtype=np.intp
        self.alpha = np.empty(self.n_dims, dtype=np.double)

        if which_diffs.shape[0] == 0:
            self.d_di = np.arange(self.n_dims, dtype=np.intp)          # CHANGED: dtype=np.intp
        else:
            self.d_di = which_diffs

        self.n_d_di_1 = self.d_di.shape[0] - 1
        self.n_d_di = self.d_di.shape[0]

        # Construct interpolation grid points
        H_s2 = H_s.copy()
        d_third_order_tensor_scale(H_s2, 0.5 )
        U_grid2, cum_prod = Unitary_Grid(H_s2, self.c_mins, self.dcs, self.c_bins)  # Half grid -> H_s2 = H_s / 2

        ## Construct interpolation cache
        (self.E, self.Vl, self.Vr, self.CL_L, self.CL_R, self.CH_L, self.CH_R,
         self.strides_E, self.strides_L, self.strides_C, 
         self.first_elements_E, self.first_elements_C) = Create_Sym_Interpolation_Cache(
            U_grid2, cum_prod, self.c_bins
         )

        self.ei = &self.E[0, 0]
        self.expe = &self.Es[0]              # Assuming Es is initialized properly elsewhere
        self.vl = &self.Vl[0, 0, 0]
        self.vr = &self.Vr[0, 0, 0]
        self.cl = &self.CL_L[0, 0, 0]
        self.cr = &self.CL_R[0, 0, 0]
        self.L = np.empty(self.n_dims, dtype=np.intp)   # CHANGED: dtype=np.intp

    def change_m(self, int m):
        self.m = m
        self.dm = self.d * m
        # also change self.vi's
        self.Ur = np.empty([self.d, self.m], dtype=np.complex128)
        self.ur0 = &self.Ur[0, 0]
        self.Ur1 = np.empty([self.d, self.m], dtype=np.complex128)
        self.ur1 = &self.Ur1[0, 0]
        #self.dU1 = np.empty([self.n_d_di, self.d, self.m], dtype=np.complex128)
        #self.du1 = &self.dU1[0, 0, 0]
        #self.dU2 = np.empty([self.n_d_di, self.d, self.m], dtype=np.complex128)
        #self.du2 = &self.dU2[0, 0, 0]
        #self.Es = np.empty(self.d, dtype=np.complex128)
        #self.es0 = &self.Es[0]
        # Split into left and right side
        self.dUl = np.empty([self.n_dims, self.d, self.m], dtype=np.complex128)
        self.dul = &self.dUl[0, 0, 0]
        self.dUr = np.empty([self.n_dims, self.d, self.m], dtype=np.complex128)
        self.dur = &self.dUr[0, 0, 0]
        self.dUc = np.empty([self.d, self.m], dtype=np.complex128)
        self.duc = &self.dUc[0,0]

    cdef single_parameters2oddgrid(self, double[::1] c):
        cdef npc.intp_t sum_location = 0          # CHANGED: long -> npc.intp_t
        cdef double alpha_max = 0.0
        cdef double alpha_rest
        cdef npc.intp_t i                        # CHANGED: long -> npc.intp_t
        cdef npc.intp_t max_alpha_ind = 0        # CHANGED: long -> npc.intp_t
        cdef npc.intp_t max_vals                 # CHANGED: long -> npc.intp_t

        for i in range(self.n_dims):
            # Transform
            self.alpha[i] = (c[i] - self.c_mins[i]) / self.dcs[i]
            # Round
            self.location[i] = <npc.intp_t> self.alpha[i]   # CHANGED: <long> -> <npc.intp_t>
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
            self.abs_alpha_rest[max_alpha_ind] = 1 - self.abs_alpha_rest[max_alpha_ind]

        for i in range(self.n_dims):
            self.d_location[i] -= 1
            self.d_location[i] >>= 1  # via bitshift operation
            #d_location[i] /= 2 #(d_location[i]-1)/2 ### in two steps with optimized in-place operations

        for i in range(self.n_dims):
            max_vals = self.location[i] + self.d_location[i]
            if max_vals > self.c_bins[i] - 1:
                self.d_location[i] = -1

    def set_which_diffs(self, npc.intp_t[::1] which_diffs):   # CHANGED: was long[::1]
        self.d_di = which_diffs
        self.n_d_di_1 = self.d_di.shape[0] - 1
        self.n_d_di = self.d_di.shape[0]

    def get_single_param(self, double[::1] c):  # To verify interpolation grid
        self.single_parameters2oddgrid(c)
        return np.asarray(self.location), np.asarray(self.d_location), np.asarray(self.abs_alpha_rest)

    def get_cache(self): # To verify interpolation grid
        return np.asarray(self.E), np.asarray(self.Vl), np.asarray(self.Vr), np.asarray(self.CL_L), np.asarray(self.CL_R), np.asarray(self.CH_L), np.asarray(self.CH_R), np.asarray(self.strides_E), np.asarray(self.strides_L), np.asarray(self.strides_C), np.asarray(self.first_elements_E), np.asarray(self.first_elements_C)

    cdef interpolate_single_u(self, double complex *u0, double complex *v0):   #u0 => input vectors, v0 => output vectors
        cdef Py_ssize_t i, j
        cdef Py_ssize_t ind, indE
        if self.n_dims == 1:  # Shouldn't make any difference if you use symmetric 1D or non symmetric 1D interpolation
            ind = self.location[0] + self.d_location[0]
            self.vr = &self.Vr[ind, 0, 0]  # In 1D strides are 1
            self.vl = &self.Vl[ind, 0, 0]
            self.ei = &self.E[ind, 0]
            #V_expE_W_pointer(self.vl, self.vr, self.ei, self.abs_alpha_rest[0], self.ur0, u0, self.d, self.d2)
            MM_cdot_pointer_v(self.vr, u0, self.ur0, self.d, self.m)  # u0 --> ur0
            v_exp_pointer_v(self.ei, self.ur0, self.abs_alpha_rest[0], self.d, self.m) 
            MM_cdot_pointer_v(self.vl, self.ur0, v0, self.d, self.m)  # ur0 --> v0
        else:
            # From right to center to left
            # right to center
            for i in range(self.n_dims):
                self.L[i] = self.location[i]
            self.L[self.n_dims_1] += self.d_location[self.n_dims_1]
            indE = findex_0(self.L, self.strides_E[self.n_dims_1,:], self.n_dims)   + self.first_elements_E[self.n_dims_1]
            ind = findex_0(self.L, self.strides_L, self.n_dims)
            self.L[self.n_dims_1] = self.location[self.n_dims_1] 
            self.vr = &self.Vr[ind, 0, 0]
            MM_cdot_pointer_v(self.vr, u0, self.ur0, self.d, self.m)  # u0 --> ur0
            self.ei = &self.E[indE, 0]
            v_exp_pointer_v(self.ei, self.ur0, self.abs_alpha_rest[self.n_dims_1], self.d, self.m)
            """
            for i in range(self.n_dims_2,0,-1):  # i is between [n_dims_2, 1]
                j = i + 1  # between [n_dims_1, 2]
                self.ei = &self.E[indE, 0]
                v_exp_pointer_v(self.ei, self.ur0, self.abs_alpha_rest[i], self.d, self.m)
                if self.d_location[i]:  ### Higher
                    self.L[i] -= 1
                    ind = findex_0(self.L, self.strides_C[i, :], self.n_dims) + self.first_elements_C[i]
                    self.cr = &self.CL_R[ind, 0, 0]
                    self.L[i] += 1 # Restore value
                else:  ### Lower
                    ind = findex_0(self.L, self.strides_C[i, :], self.n_dims) + self.first_elements_C[i]
                    self.cr = &self.CH_R[ind, 0, 0]
                self.L[j] = self.location[j]
                self.L[i] += self.d_location[i]
                #Continue here
                MM_cdot_pointer_v(self.cr, self.ur0, self.ur1, self.d, self.m)  # ur0 --> ur1
                indE = findex_0(self.L, self.strides_E[i, :], self.n_dims) + self.first_elements_E[i]  # For E   
                # switch ur1 and ur0
                self.ur1, self.ur0 = self.ur0, self.ur1  # ur1 --> ur0
            """
            # Central element
            self.L[0] = self.location[0] + self.d_location[0]
            indE = findex_0(self.L, self.strides_E[0,:], self.n_dims) 
            self.L[0] = self.location[0]
            self.ei = &self.E[indE, 0]
            self.L[1] += self.d_location[1]
            if self.d_location[0]:  ### Higher
                self.L[0] += -1
                ind = findex_0(self.L, self.strides_C[0, :], self.n_dims)
                self.cl = &self.CL_L[ind, 0, 0]
                self.cr = &self.CL_R[ind, 0, 0]
                self.L[0] += 1
            else:  ### Lower
                ind = findex_0(self.L, self.strides_C[0, :], self.n_dims)
                self.cl = &self.CH_L[ind, 0, 0]
                self.cr = &self.CH_R[ind, 0, 0]
            self.L[1] = self.location[1]
            self.L[0] = self.location[0]
            MM_cdot_pointer_v(self.cr, self.ur0, self.ur1, self.d, self.m)  # ur0 --> ur1
            v_exp_pointer_v(self.ei, self.ur1, self.abs_alpha_rest[0], self.d, self.m)
            MM_cdot_pointer_v(self.cl, self.ur1, self.ur0, self.d, self.m)  # ur1 --> v0
            # from center to left (similar to non vector approach)
            """
            for i in range(1,self.n_dims_1):  # All except for the last element
                j = i + 1  # between [2, n_dims_1]
                self.ei = &self.E[indE, 0]
                self.L[j] += self.d_location[j]
                indE = findex_0(self.L, self.strides_E[j, :], self.n_dims) + self.first_elements_E[j]  # For E
                if self.d_location[i]:  ### Higher
                    self.L[i] += -1
                    ind = findex_0(self.L, self.strides_C[i, :], self.n_dims) + self.first_elements_C[i]
                    self.cl = &self.CL_L[ind, 0, 0]
                    self.L[i] += 1 # Restore value
                else:  ### Lower
                    ind = findex_0(self.L, self.strides_C[i, :], self.n_dims) + self.first_elements_C[i]
                    self.cl = &self.CH_L[ind, 0, 0]
                self.L[j] = self.location[j]
                #Continue here
                v_exp_pointer_v(self.ei, self.ur0, self.abs_alpha_rest[i], self.d, self.m)
                MM_cdot_pointer_v(self.cl, self.ur0, self.ur1, self.d, self.m)  # ur0 --> ur1
                # switch ur1 and ur0
                self.ur1, self.ur0 = self.ur0, self.ur1  # ur1 --> ur0
            """
            # Last element
            
            self.L[self.n_dims_1] += self.d_location[self.n_dims_1]
            indE = findex_0(self.L, self.strides_E[self.n_dims_1,:], self.n_dims)  + self.first_elements_E[self.n_dims_1]
            ind = findex_0(self.L, self.strides_L, self.n_dims)
            self.ei = &self.E[indE, 0]
            self.vl = &self.Vl[ind, 0, 0]
            v_exp_pointer_v(self.ei, self.ur0, self.abs_alpha_rest[self.n_dims_1], self.d, self.m)
            MM_cdot_pointer_v(self.vl, self.ur0, v0, self.d, self.m)  # ur0 --> v0

    cdef expmH_pointer(self, double[::1] c, double complex *u0, double complex *v0):
        if not c.shape[0] == self.n_dims:
            raise ValueError('The coefficient c must be of size [interpolation_dimensions].')
        self.single_parameters2oddgrid(c)
        self.interpolate_single_u(u0, v0)
    def expmH(self, double[::1] c, double complex[:,::1] V_in, double complex[:,::1] V_out):
        cdef double complex *u0 = &V_in[0, 0]
        cdef double complex *v0 = &V_out[0, 0]
        if self.n_dims > 2:
            raise ValueError('This function is only implemented for 1 and 2 dimensions.')
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
        self.expmH_pointer(c, u0, v0)

            
    def expmH_pulse_no_multiply(self, double[:,::1] cs, double complex[:,::1] V_in, double complex[:,:,::1] V_out):
        cdef double complex *u0 = &V_in[0,0]
        cdef double complex *v0 = &V_out[0,0,0]
        cdef n = cs.shape[0]
        for i in range(n):
            self.expmH_pointer(cs[i,:], u0, v0)
            v0 += self.dm