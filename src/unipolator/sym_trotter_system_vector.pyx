#cython: language_level=3
import numpy as np
cimport numpy as npc
from .exp_and_log cimport *
from .indexing cimport *
from .caching cimport *
from .blas_functions cimport *
from .blas_functions_vectors cimport *

# Unitary Interpolation
cdef class Sym_Trotter_System_vector:
    # Initialize variables, to quickly calculate interpolations while minimizing memory allocation overheads
    cdef int n_dims, n_dims_1, n_dims_2, n_dims_3, d, d2, i, m, dm
    cdef double[:,::1] Es
    cdef double complex[:,:,::1] H, C
    cdef double complex[:,::1] V, U1, U2, Unn, L, R
    cdef double complex *h0
    cdef double complex *c0
    cdef double complex *c1
    cdef double complex *l0
    cdef double complex *r0
    cdef double complex *v0
    cdef double complex *curr_h0
    cdef double complex *u1
    cdef double complex *u2
    cdef double complex *unn
    cdef double *e0s
    cdef double *e1s
    cdef int lwork, lrwork, liwork
    cdef double complex[::1] work
    cdef double complex *work0
    cdef double[::1] rwork
    cdef double *rwork0
    cdef npc.intp_t[::1] iwork
    cdef npc.intp_t *iwork0
    cdef char *jobz
    cdef char *uplo
    cdef int info, n_times, m_times

    def __cinit__(self, double complex[:,:,::1] H_s, npc.intp_t[::1] which_diffs=np.array([], dtype=np.intp), int m_times=0, int n_times=-1, int m=1):
        # Construct a Trotter object for state evolution of
        # H_s - the Hamiltonian of the system H_s[0] = H0, H_s[1] = H1, H_s[2] = H2, etc.
        # m_times - do 2**m_times trotter steps to approximate a unitary
        # if n_times is not <0, then m_times is ignored and n_times is used instead
        # m - number of wavevectors to calculate (default is 1)

        # Construct parameters
        self.n_dims = H_s.shape[0]
        self.n_dims_1 = self.n_dims - 1
        self.n_dims_2 = self.n_dims - 2
        self.n_dims_3 = self.n_dims - 3
        if n_times < 0:
            self.n_times = int(2**m_times)
        else:
            self.n_times = n_times
        self.d = H_s.shape[1]
        self.d2 = self.d * self.d
        self.m = m
        self.dm = self.d * self.m

        self.H = np.empty([self.n_dims, self.d, self.d], dtype=np.complex128)
        self.h0 = &self.H[0, 0, 0]
        cdef double complex *h_s0 = &H_s[0, 0, 0]
        copy_pointer(h_s0, self.h0, self.d2 * self.n_dims)

        self.U1 = np.empty([self.d, self.d], dtype=np.complex128)
        self.u1 = &self.U1[0, 0]
        self.U2 = np.empty([self.d, self.d], dtype=np.complex128)
        self.u2 = &self.U2[0, 0]

        self.V = np.empty([self.d, self.d], dtype=np.complex128)
        self.v0 = &self.V[0, 0]
        self.curr_h0 = &self.H[0, 0, 0]

        # Cache variables
        self.Es = np.empty([self.n_dims_1, self.d], dtype=np.double)
        self.e0s = &self.Es[0, 0]
        self.C = np.empty([self.n_dims_2, self.d, self.d], dtype=np.complex128)
        self.c0 = &self.C[0, 0, 0]
        self.L = np.empty([self.d, self.d], dtype=np.complex128)
        self.l0 = &self.L[0, 0]
        self.R = np.empty([self.d, self.d], dtype=np.complex128)
        self.r0 = &self.R[0, 0]
        self.Unn = np.empty([self.d, self.d], dtype=np.complex128)
        self.unn = &self.Unn[0, 0]
        self.change_m(m)

        self.lwork, self.lrwork, self.liwork = c_eigh_lapack_workspace_sizes(self.V)
        self.work = np.empty([self.lwork], dtype=np.complex128)
        self.work0 = &self.work[0]
        self.rwork = np.empty([self.lrwork], dtype=np.double)
        self.rwork0 = &self.rwork[0]
        self.iwork = np.empty([self.liwork], dtype=np.intp)
        self.iwork0 = &self.iwork[0]
        self.jobz = 'v'
        self.uplo = 'l'
        self.info = 0

        # Cache matrices
        self.c1 = self.c0
        self.e1s = self.e0s

        c_expmH(self.H[0], 0.5 / self.n_times, self.Unn, self.lwork, self.lrwork, self.liwork)
        for i in range(1, self.n_dims):
            c_eigh_lapack(self.H[i], self.V, self.Es[i - 1], self.lwork, self.lrwork, self.liwork)
            if i == 1:
                M_DagM_cdot(self.Unn, self.V, self.L)
                MM_cdot(self.V, self.Unn, self.R)
            else:
                M_DagM_cdot(self.V, self.Unn, self.C[i - 2])
            copy_pointer(self.v0, self.unn, self.d2)
        MM_cdot(self.R, self.L, self.V)

    def change_m(self, int m):
        self.m = m
        self.dm = self.d * m
        self.U1 = np.empty([self.d, self.m], dtype=np.complex128)
        self.u1 = &self.U1[0, 0]
        self.U2 = np.empty([self.d, self.m], dtype=np.complex128)
        self.u2 = &self.U2[0, 0]

    
    """
    def set_which_diffs(self, int[::1] which_diffs):
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
