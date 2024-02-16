#cython: language_level=3
import numpy as np
cimport numpy as npc
from .exp_and_log cimport *
from .indexing cimport *
from .caching cimport *
from .blas_functions cimport *
#from scipy.linalg.cython_lapack cimport zheevd

# Trotter System
cdef class Trotter_System:
    # Initialize variables, to quickly calculate interpolations while minimizing memmory allocation overheads
    cdef int n_dims, n_dims_1, n_dims_2, n_dims_3, d, d2, n_d_di_1, n_d_di, i
    cdef long[::1] d_di
    cdef double[::1] E
    cdef double[:,::1] Es
    cdef double complex[::1] C1
    cdef double complex[:,:,::1] H, L
    cdef double complex[:,::1] V, U1, U2, U3, U4, U5
    cdef double complex *h0
    cdef double complex *l0
    cdef double complex *l1
    cdef double complex *v0
    cdef double complex *curr_h0
    cdef double complex *u1
    cdef double complex *u2
    cdef double complex *u3
    cdef double complex *u4
    cdef double complex *u5
    cdef double *e0
    cdef double *e0s
    cdef double *e1s
    cdef double complex *c1
    cdef int lwork, lrwork, liwork
    cdef double complex[::1] work
    cdef double complex *work0
    cdef double[::1] rwork
    cdef double *rwork0
    cdef int[::1] iwork
    cdef int *iwork0
    cdef char *jobz
    cdef char *uplo
    cdef int info, n_times, m_times
    def __cinit__(self, double complex[:,:,::1] H_s, long[::1] which_diffs = np.array([], dtype=long), int m_times=0): # n_times = 2**m_times ==> m_times = 0 --> n_times = 1
        # Construct parameters
        self.n_dims = H_s.shape[0]
        self.n_dims_1 = self.n_dims - 1
        self.n_dims_2 = self.n_dims - 2
        self.n_dims_3 = self.n_dims - 3
        self.m_times = m_times
        self.n_times = int(2**m_times)
        self.d = H_s.shape[1]
        self.d2 = self.d * self.d
        if which_diffs.shape[0] == 0:
            self.d_di = np.arange(self.n_dims_1)
        else:
            self.d_di = which_diffs
        self.n_d_di = self.d_di.shape[0]

        self.H = np.empty([self.n_dims, self.d, self.d], dtype=np.complex128)
        self.h0 = &self.H[0,0,0]
        cdef int i
        cdef double complex *h_s0 = &H_s[0,0,0]
        copy_pointer(h_s0, self.h0, self.d2*self.n_dims)

        self.E = np.empty([self.d], dtype=np.double)
        self.e0 = &self.E[0]
        self.C1 = np.empty([self.d], dtype=np.complex128)
        self.c1 = &self.C1[0]
        self.U1 = np.empty([self.d, self.d], dtype=np.complex128)
        self.u1 = &self.U1[0, 0]
        self.U2 = np.empty([self.d, self.d], dtype=np.complex128)
        self.u2 = &self.U2[0, 0]
        self.U3 = np.empty([self.d, self.d], dtype=np.complex128)
        self.u3 = &self.U3[0, 0]
        self.U4 = np.empty([self.d, self.d], dtype=np.complex128)
        self.u4 = &self.U4[0, 0]
        self.U5 = np.empty([self.d, self.d], dtype=np.complex128)
        self.u5 = &self.U5[0, 0]
        self.V = np.empty([self.d, self.d], dtype=np.complex128)
        self.v0 = &self.V[0, 0]
        self.curr_h0 = &self.H[0, 0, 0]
        # Cache variables
        self.L = np.empty([self.n_dims, self.d, self.d], dtype=np.complex128)
        self.l0 = &self.L[0, 0, 0] 
        self.Es = np.empty([self.n_dims_1, self.d], dtype=np.double)
        self.e0s = &self.Es[0,0] 

        self.lwork, self.lrwork, self.liwork = c_eigh_lapack_workspace_sizes(self.V)
        self.work = np.empty([self.lwork], dtype=np.complex128)
        self.work0 = &self.work[0]
        self.rwork = np.empty([self.lrwork], dtype=np.double)
        self.rwork0 = &self.rwork[0]
        self.iwork = np.empty([self.liwork], dtype=np.int32)
        self.iwork0 = &self.iwork[0]
        self.jobz = 'v'  #eigenvectors and values -> v
        self.uplo = 'l'  # upper triangle
        self.info = 0

        # Cache matrices
        # First the H0 element
        c_expmH(self.H[0], 1.0/self.n_times, self.U1, self.lwork, self.lrwork, self.liwork) # store in U1
        self.l1 = self.l0 # temporary pointer - moving along self.L, but backwards
        self.e1s = self.e0s  # temporary pointer - moving along self.Es, but backwards
        for i in range(1, self.n_dims):
            c_eigh_lapack(self.H[i], self.V, self.Es[i-1], self.lwork, self.lrwork, self.liwork) # store in 
            M_DagM_cdot_pointer(self.u1, self.v0, self.l1, self.d) # store in U1
            copy_pointer(self.v0 , self.u1, self.d2)
            self.l1 += self.d2
        copy_pointer(self.v0, self.l1, self.d2)
        self.l1 = self.l0  # Restore original value of pointer

        for i in range(self.n_dims_1):  # half steps 
            for j in range(self.d):
                self.Es[i,j] = self.Es[i,j]/self.n_times
    
    def get_cache(self):
        return np.array(self.L), np.array(self.Es)

    cdef expmH_pointer(self, double[::1] c, double complex *u0):
        cdef int i 
        if self.n_dims_1 % 2 == 0:
            copy_pointer(self.l1, u0, self.d2)
            for i in range(0, self.n_dims_1, 2):
                self.l1 += self.d2
                V_expE_W_pointer(u0, self.l1, self.e1s, c[i], self.v0, self.u1, self.d, self.d2)
                self.e1s += self.d
                self.l1 += self.d2
                V_expE_W_pointer(self.u1, self.l1, self.e1s, c[i+1], self.v0, u0, self.d, self.d2)
                self.e1s += self.d
        else:
            copy_pointer(self.l1, self.u1, self.d2)
            for i in range(0, self.n_dims_3, 2):
                self.l1 += self.d2
                V_expE_W_pointer(self.u1, self.l1, self.e1s, c[i], self.v0, u0, self.d, self.d2)
                self.e1s += self.d
                self.l1 += self.d2
                V_expE_W_pointer(u0, self.l1, self.e1s, c[i+1], self.v0, self.u1, self.d, self.d2)
                self.e1s += self.d
            self.l1 += self.d2
            V_expE_W_pointer(self.u1, self.l1, self.e1s, c[self.n_dims_2], self.v0, u0, self.d, self.d2)
        self.l1 = self.l0  # Restore original value of pointer
        self.e1s = self.e0s
        for i in range(1, self.m_times, 2):
            MM_cdot_pointer(u0, u0, self.u1, self.d)
            MM_cdot_pointer(self.u1, self.u1, u0, self.d)
        if self.m_times % 2:
            MM_cdot_pointer(u0, u0, self.u1, self.d)
            copy_pointer(self.u1, u0, self.d2)

    def expmH(self, double[::1] c, double complex[:,::1] U):
        # Construct Hamiltonian
        if not c.shape[0] == self.n_dims_1:
            raise ValueError('c.shape[0] needs to be equal to H_s[0].shape[0]-1.')
        cdef double complex *u0 = &U[0,0]
        self.expmH_pointer(c, u0)
            
    def expmH_pulse_no_multiply(self, double[:,::1] cs, double complex[:,:,::1] U):
        cdef double complex *u0 = &U[0, 0, 0]
        cdef how_many = cs.shape[0]
        cdef int i
        for i in range(how_many):
            self.expmH_pointer(cs[i,:], u0)
            u0 += self.d2

    """
    
    def set_which_diffs(self, long[::1] which_diffs):
        self.d_di = which_diffs
        self.n_d_di = self.d_di.shape[0]

    def get_which_diffs(self):
        return np.asarray(self.d_di), self.n_d_di

    cdef dexpmH_pointer(self, double[::1] c, double complex *u0, double complex *du0, dt=1.0):  #int[::1] d_di,
        # Construct Hamiltonian
        cdef int i
        cdef double complex *h0
        cdef int curr_d_di_1
        self.weighted_hamiltonian(c)
        zheevd(self.jobz, self.uplo, &self.d, self.v0, &self.d, self.e0, self.work0, &self.lwork, self.rwork0, &self.lrwork, self.iwork0, &self.liwork, &self.info)
        copy_pointer(self.v0, self.u1, self.d2)
        v_exp_pointer(self.e0, self.u1, dt, self.d)
        DagM_M_cdot_pointer(self.v0, self.u1, u0, self.d)
        # Calculate the differentials
        for i in range(self.n_d_di): # According to https://arxiv.org/pdf/2006.00935.pdf (Daalgard, Motzoi paper --> Original source?)
            curr_d_di_1 = self.d_di[i] + 1
            h0 = &self.H[curr_d_di_1, 0, 0]
            MM_cdot_scale_pointer(dt, self.v0, h0, self.u1, self.d) #self.u1, self.d)
            M_DagM_cdot_pointer(self.u1, self.v0, du0, self.d)

            phase_shift_matrix_pointer(self.e0, self.c1, self.u1, dt, self.d)
            AxB_elementwise_pointer(du0, self.u1, self.u2, self.d2)

            MM_cdot_pointer( self.u2, self.v0, self.u1, self.d)
            DagM_M_cdot_pointer(self.v0, self.u1, du0, self.d)
            du0 += self.d2
    def dexpmH(self, double[::1] c, double complex[:,::1] U, double complex[:,:,::1] dU, double dt=1.0):  #int[::1] d_di,
        # d_di contains the indexes of the derivatives that we want to calculate (needs to be in ascending order with a negative value at the end)
        cdef double complex *u0 = &U[0, 0]
        cdef double complex *du0 = &dU[0,0,0]
        if not c.shape[0] == self.n_dims_1:
            raise ValueError('The coefficient c must be of size [interpolation_dimensions].')
        if not self.d_di.shape[0] == dU.shape[0]:
            raise ValueError('Inputs must fulfill: which_diffs.shape[0] = dU.shape[0].')
        self.dexpmH_pointer(c, u0, du0, dt)

    cdef dexpmH_pointer(self, double[::1] c, double complex *u0, double complex *du0, dt=1.0):  #int[::1] d_di,
        # Construct Hamiltonian
        cdef int i
        cdef double complex *h0
        cdef int curr_d_di_1
        self.weighted_hamiltonian(c)
        zheevd(self.jobz, self.uplo, &self.d, self.v0, &self.d, self.e0, self.work0, &self.lwork, self.rwork0, &self.lrwork, self.iwork0, &self.liwork, &self.info)
        copy_pointer(self.v0, self.u1, self.d2)
        v_exp_pointer(self.e0, self.u1, dt, self.d)
        DagM_M_cdot_pointer(self.v0, self.u1, u0, self.d)
        # Calculate the differentials
        for i in range(self.n_d_di): # According to https://arxiv.org/pdf/2006.00935.pdf (Daalgard, Motzoi paper --> Original source?)
            curr_d_di_1 = self.d_di[i] + 1
            h0 = &self.H[curr_d_di_1, 0, 0]
            MM_cdot_scale_pointer(dt, self.v0, h0, self.u1, self.d) #self.u1, self.d)
            M_DagM_cdot_pointer(self.u1, self.v0, du0, self.d)

            phase_shift_matrix_pointer(self.e0, self.c1, self.u1, dt, self.d)
            AxB_elementwise_pointer(du0, self.u1, self.u2, self.d2)

            MM_cdot_pointer( self.u2, self.v0, self.u1, self.d)
            DagM_M_cdot_pointer(self.v0, self.u1, du0, self.d)
            du0 += self.d2
    def dexpmH(self, double[::1] c, double complex[:,::1] U, double complex[:,:,::1] dU, double dt=1.0):  #int[::1] d_di,
        # d_di contains the indexes of the derivatives that we want to calculate (needs to be in ascending order with a negative value at the end)
        cdef double complex *u0 = &U[0, 0]
        cdef double complex *du0 = &dU[0,0,0]
        if not c.shape[0] == self.n_dims_1:
            raise ValueError('The coefficient c must be of size [interpolation_dimensions].')
        if not self.d_di.shape[0] == dU.shape[0]:
            raise ValueError('Inputs must fulfill: which_diffs.shape[0] = dU.shape[0].')
        self.dexpmH_pointer(c, u0, du0, dt)

    cdef expmH_pulse_pointer(self, double[:,::1] cs, double complex *u0, double dt=1.0):
        cdef int i
        cdef int steps = cs.shape[0]

        self.weighted_hamiltonian(cs[0,:])
        zheevd(self.jobz, self.uplo, &self.d, self.v0, &self.d, self.e0, self.work0, &self.lwork, self.rwork0, &self.lrwork, self.iwork0, &self.liwork, &self.info)
        copy_pointer(self.v0, self.u1, self.d2)
        v_exp_pointer(self.e0, self.v0, dt, self.d)
        DagM_M_cdot_pointer(self.u1, self.v0, u0, self.d)

        for i in range(1,steps):
            self.weighted_hamiltonian(cs[i, :])
            zheevd(self.jobz, self.uplo, &self.d, self.v0, &self.d, self.e0, self.work0, &self.lwork, self.rwork0, &self.lrwork, self.iwork0, &self.liwork, &self.info)
            MM_cdot_pointer(self.v0, u0, self.u1, self.d)
            v_exp_pointer(self.e0, self.u1, dt, self.d)
            DagM_M_cdot_pointer(self.v0, self.u1, u0, self.d)
    def expmH_pulse(self, double[:,::1] cs, double complex[:,::1] U, double dt=1.0):
        # Construct Hamiltonian
        cdef double complex *u0 = &U[0, 0]
        self.expmH_pulse_pointer(cs, u0, dt)

    def grape(self, double[:,::1] cs, double complex[:,::1] U_target, int[::1] target_indexes, double complex[:,::1] U, double complex[:,:,::1] dU, double[:,::1] dI_dj):
        # Calculate fidelity for a pulse and the differentials of the fidelity at every timestep using the grape trick
        cdef int i, j
        cdef int steps = cs.shape[0]

        cdef double complex *new_p0 = self.u1
        cdef double complex *new_q0 = self.u2
        cdef double complex *p0 = self.u4
        cdef double complex *q0 = self.u5
        cdef double complex *u0 = &U[0, 0]
        cdef double complex *u_tar = &U_target[0, 0]
        cdef double complex *curr_u = self.u3
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
            # flip pointers
            self.u1, self.u4 = self.u4, self.u1
            self.u2, self.u5 = self.u5, self.u2
            new_p0, p0 = p0, new_p0
            new_q0, q0 = q0, new_q0
        return I0
    """
