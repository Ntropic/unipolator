from numba import njit
from numpy import dot, transpose, conj, abs, diag, sum, angle, complex128, arange, exp, log, empty, random, trace, zeros, take, int_
from numpy.linalg import norm, eigh, qr
from scipy.linalg import logm
from scipy.linalg import schur

# ༼ つ ◕_◕༽つ
#### Basic Operations ####################################################
@njit(fastmath=True, nogil=True, cache=True) 
def Dagger(U):
    return transpose(conj(U))
@njit(fastmath=True, nogil=True, cache=True) 

def Dag(U):
    return transpose(conj(U))

@njit(fastmath=True, nogil=True, cache=True) 
def Commutator(A,B):
    return dot(A,B)-dot(B,A)

#### Fidelitity #########################################################
@njit(fastmath=True, nogil=True, cache=True)  # cache=False only for performance comparison
def numba_ix(arr, rows, cols):
    lr = len(rows)
    lc = len(cols)
    one_d_index = zeros(lr*lc, dtype=int_)
    for i, r in enumerate(rows):
        start = i * lc
        one_d_index[start: start + lc] = cols + arr.shape[1] * r
    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = take(arr_1d, one_d_index)
    return slice_1d.reshape((lr, lc))

@njit(fastmath=True, nogil=True, cache=True)
def Av_Fidelity(U_ex, U_ap, indexes=empty(0)):
    # Substate Average Fidelity with diagonal elements not covered by indexes
    s = len(indexes)
    sU = U_ex.shape[0]
    if s > 0:
        B = dot(Dagger(numba_ix(U_ex, indexes, indexes)), numba_ix(U_ap, indexes, indexes))
    else:
        B = dot(Dagger(U_ex), U_ap)
        s = sU
    F_av = 1 / (s * (s + 1)) * (trace(dot(B, Dagger(B))) + abs(trace(B)) ** 2)
    F_av = abs(F_av)
    return F_av
@njit(fastmath=True, nogil=True, cache=True)
def Av_Infidelity(U_ex, U_ap, indexes=empty(0)):
    return abs(1 - Av_Fidelity(U_ex, U_ap, indexes))

#### Exponentials #######################################################
@njit()#(fastmath=True, nogil=True)
def vecxvec( E, wf, alpha): # Many times faster than multiply
    s = E.size
    for i in range(s):
        wf[i] = exp(-1j * E[i] * alpha) * wf[i]
    return wf
# Construct matrix exponentials and products faster than in numpy or scipy
@njit(fastmath=True, nogil=True)
def vm_exp_mul(E, V, dt = 1.0): # expm(diag(vector))*matrix  multiplication via exp(vector)*matrix
    s = E.size
    A = empty((s, s), complex128)
    for i in range(s):
        A[i,:] = exp(-1j * E[i] * dt) * Dag(V[:,i])
    return A
@njit(fastmath=True, nogil=True)
def expmH_from_Eig(E, V, dt = 1.0):
    U = dot(V, vm_exp_mul(E, V, dt))
    return U
@njit(fastmath=True, nogil=True)
def expmH(H, dt = 1.0):
    E, V = eigh(H)
    return expmH_from_Eig( E, V, dt)

##### Logarithms #########################################################
def unitary_eig(A): # alternative to np.eig returning unitary matrices V
    Emat, V = schur(A, output='complex')
    return diag(Emat), V
@njit(fastmath=True, nogil=True)
def vm_log_mul(E, V):
    s = E.size
    A = empty((s, s), complex128)
    for i in range(s):
        A[i,:] = log(E[i])*Dag(V[:,i])
    return A
def logmU(U):
    E, V = unitary_eig(U)
    return dot(V, vm_log_mul(E, V))

#### Random Unitaries ###################################################
def randU_Haar( s, rng=random.default_rng()):
    X = rng.normal(1, 1, [s,s]) +1j*rng.normal(1, 1, [s,s])
    Q, R = qr(X) # QR decomospition -> Gram-Schmidt orthogonalisation and normalisation
    return Q

def randH_Haar( s, amp=-1, rng=random.default_rng()):
    U = randU_Haar( s, rng)
    H = -1j*logmU(U)
    if amp > 0:
        H = amp/norm(H)*H
    return H

def Random_parametric_Hamiltonian_Haar(n, s, amps, rng=random.default_rng()):
    # Generates random parametric Hamiltonians with n parameters and an s dimensional Hilbert space using the vector amps to specify the norm of the Hamiltonians
    # Outputs a 3d tensor of the Hamiltonians, where the first index is the Hamiltonian number, and the second and third index are the Hamiltonians themselves
    l = len(amps)
    H_s = empty([n+1, s, s], dtype=complex)
    H_s[0] = randH_Haar(s, amps[0], rng=rng)
    for i in range(1, n+1):
        if i >= l-1:
            curr_amp = amps[-1]
        else:
            curr_amp = amps[i]
        H_s[i] = randH_Haar(s, curr_amp, rng=rng)
    return H_s