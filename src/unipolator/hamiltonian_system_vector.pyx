#cython: language_level=3
import numpy as np
cimport numpy as npc
from libc.math cimport exp
from .blas_functions cimport *
from .blas_functions_vectors cimport *
from .exp_and_log cimport copy_pointer
from scipy.linalg.cython_blas cimport zgemm, zaxpy, zscal


# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
cdef class Hamiltonian_System_vector:
    # Initialize variables, to quickly calculate interpolations while minimizing memory allocation overheads
    cdef int n_dims, n_dims_1, d, d2, m, dm
    cdef double complex[:,:,::1] H
    cdef double complex[:,::1] V, V_inter, V_in
    cdef double complex *h0
    cdef double complex *v0
    cdef double complex *v_inter
    cdef double complex *v_in
    cdef double complex *curr_h0
    cdef double tol
    cdef double[::1] max_amp
    cdef int m_star, s
    cdef double mu

    def __cinit__(self, double complex[:,:,::1] H_s, int m=1, double tol=2**-53, double[::1] max_amp=np.array([-1.0], dtype=np.double)):
        # Construct exponential using scaling and squaring
        # H_s is a 3D array of shape (n_dims, d, d) where n_dims is the number of dimensions of the Hamiltonian and d is the dimension of the Hilbert space
        # m is the number of wavevectors to compute at once (can be changed using change_m)
        # tol is the tolerance of the approximation (can be changed using change_tolerance)
        # max_amp are the maximum coefficient of the Hamiltonian parts (1.0 by default for each dimension)

        # Construct parameters
        self.n_dims = H_s.shape[0]
        self.n_dims_1 = self.n_dims - 1
        self.d = H_s.shape[1]
        self.d2 = self.d * self.d
        self.m = m
        if max_amp[0] == -1:
            self.max_amp = np.ones(self.n_dims, dtype=np.double)
        else:
            self.max_amp = max_amp

        self.H = np.empty([self.n_dims, self.d, self.d], dtype=np.complex128)
        self.h0 = &self.H[0, 0, 0]
        # Change elements in H to -i * H
        for i in range(self.n_dims):
            for j in range(self.d):
                for k in range(self.d):
                    self.H[i, j, k] = -1j * H_s[i, j, k]

        self.V = np.empty([self.d, self.d], dtype=np.complex128)
        self.v0 = &self.V[0, 0]

        self.curr_h0 = &self.H[0, 0, 0]
        self.change_m(m)
        self.change_tolerance(tol)

    def change_m(self, int m):
        self.m = m
        self.dm = self.d * m
        self.V_in = np.empty([self.d, self.dm], dtype=np.complex128)
        self.v_in = &self.V_in[0, 0]
        self.V_inter = np.empty([self.d, self.dm], dtype=np.complex128)
        self.v_inter = &self.V_inter[0, 0]

    def change_tolerance(self, double tol):
        self.tol = tol
        self.weighted_hamiltonian(self.max_amp)
        self.m_star, self.s, self.mu = expm_multiply_prepare(self.V, self.m, self.tol)

    def get_parameters(self):
        return self.m_star, self.s, self.mu

    def set_m_star(self, int m_star):
        self.m_star = m_star

    def set_s(self, int s):
        self.s = s

    def set_mu(self, double mu):
        self.mu = mu

    cdef weighted_hamiltonian(self, double[::1] c):
        self.curr_h0 = self.h0
        copy_pointer(self.curr_h0, self.v0, self.d2)
        for i in range(self.n_dims_1):
            self.curr_h0 += self.d2
            d_mat_add_pointer(self.v0, self.curr_h0, c[i], self.d2)

    cdef void expm_multiply_cython(self, double complex[:, ::1] V_out) noexcept nogil:
        # A (nxn), B (nxm), F (nxm), G (nxm)
        # s is the number of iterations to perform,
        # Set tolerance to machine precision
        cdef double complex *vout = &V_out[0, 0]  # V_out
        cdef int temp_int, i, j
        cdef int incz = 1
        cdef double factor, c1, c2, max_amp
        cdef double complex coeff = 1.0
        cdef double complex eta = exp(self.mu / float(self.s))  # Compute scaling factor eta

        copy_pointer(self.v_in, vout, self.dm)  # Initialize F as B
        # Perform s iterations
        for i in range(self.s):
            c1 = norm_inf_complex(self.V_in, self.d, self.m)  # Compute infinity norm of V_in
            for j in range(self.m_star):
                temp_int = self.s * (j + 1)
                factor = <double> float(temp_int)
                coeff = 1 / factor
                MM_cdot_pointer_v_scaled(self.v0, self.v_in, self.v_inter, coeff, self.d, self.m)  # Update B
                copy_pointer(self.v_inter, self.v_in, self.dm)  # Switch pointers
                c2 = norm_inf_complex(self.V_in, self.d, self.m)  # Compute norm of updated B
                c_mat_add_pointer(vout, self.v_in, self.dm)  # Update F
                max_amp = norm_inf_complex(V_out, self.d, self.m)
                if c1 + c2 <= self.tol * max_amp:  # Check tolerance
                    break
                c1 = c2
            zscal(&self.dm, &eta, vout, &incz)  # Scale v_out by eta
            copy_pointer(vout, self.v_in, self.dm)  # Copy v_out to v_in

    def expmH(self, double[::1] c, double complex[:,::1] V_in, double complex[:,::1] V_out):
        # Construct Hamiltonian
        if not c.shape[0] == self.n_dims_1:
            raise ValueError('c.shape[0] needs to be equal to H_s[0].shape[0]-1.')
        if not V_in.shape[0] == V_out.shape[0] == self.d:
            raise ValueError('V_in.shape[0] and V_out.shape[0] need to be equal to H_s[0].shape[1].')
        if not V_in.shape[1] == V_out.shape[1] == self.m:
            raise ValueError('V_in.shape[1] and V_out.shape[1] need to be equal to m.')
        cdef double complex *vin = &V_in[0, 0]  # V_in
        copy_pointer(vin, self.v_in, self.dm)
        self.weighted_hamiltonian(c)
        self.expm_multiply_cython(V_out)

    def expmH_pulse_no_multiply(self, double[:,::1] cs, double complex[:,::1] V_in, double complex[:,:,::1] V_out):
        # Repeatedly calculate expmH for different c and store them in different V_out
        cdef int n = cs.shape[0]
        cdef double complex *vin = &V_in[0, 0]  # V_in
        copy_pointer(vin, self.v_in, self.dm)
        for i in range(n):
            self.weighted_hamiltonian(cs[i, :])
            self.expm_multiply_cython(V_out[i, :, :])
