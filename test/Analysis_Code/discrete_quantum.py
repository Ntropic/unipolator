from numpy import ix_, dot, transpose, conj, abs, diag, sum, angle, complex128, arange, exp, log, empty, random, trace, zeros, take, int_, multiply
from numpy.linalg import norm, eigh, qr
from scipy.linalg import logm
from scipy.linalg import schur

# ༼ つ ◕_◕༽つ
#### Basic Operations ####################################################
def Dagger(U):
    return transpose(conj(U))

def Dag(U):
    return transpose(conj(U))

def Commutator(A,B):
    return dot(A,B)-dot(B,A)

#### Fidelitity #########################################################
def Av_Fidelity(U_ex, U_ap, indexes=empty(0)):
    # Substate Average Fidelity with diagonal elements not covered by indexes
    s = len(indexes)
    sU = U_ex.shape[0]
    if s > 0:
        B = dot(Dagger(ix_(U_ex, indexes, indexes)), ix_(U_ap, indexes, indexes))
    else:
        B = dot(Dagger(U_ex), U_ap)
        s = sU
    F_av = 1 / (s * (s + 1)) * (trace(dot(B, Dagger(B))) + abs(trace(B)) ** 2)
    F_av = abs(F_av)
    return F_av
def Av_Infidelity(U_ex, U_ap, indexes=empty(0)):
    return abs(1 - Av_Fidelity(U_ex, U_ap, indexes))

def State_Fidelity(v_ex, v_ap):   # |<v_ex|v_ap>|^2 = F(v_ex, v_ap)   ==>  calculate column wise dot product of v_ex and v_ap and take the absolute value squared of the result (if matrix)
    # currently only supports all indexes
    return abs(dot(v_ex.conj().transpose(), v_ap))**2
def State_Infidelity(v_ex, v_ap):
    return 1 - abs(dot(v_ex.conj().transpose(), v_ap))**2
### For unnormalized states
def State_Fidelity_noramalize_ap(v_ex, v_ap):   # |<v_ex|v_ap>|^2 = F(v_ex, v_ap)   ==>  calculate column wise dot product of v_ex and v_ap and take the absolute value squared of the result (if matrix)
    # currently only supports all indexes
    v_ap2 = v_ap / norm(v_ap)
    return abs(dot(v_ex.conj().transpose(), v_ap2))**2
def State_Infidelity_noramalize_ap(v_ex, v_ap):
    v_ap2 = v_ap / norm(v_ap)
    return 1 - abs(dot(v_ex.conj().transpose(), v_ap2))**2
### For multiple states (matrix of columns vectors v_ex and v_ap of states)
def State_Fidelities(v_ex, v_ap):
    return abs(sum(multiply(v_ex.conj(), v_ap), axis=0))**2
def State_Infidelities(v_ex, v_ap):
    return 1 - abs(sum(multiply(v_ex.conj(), v_ap), axis=0))**2
### For multiple states (matrix of columns vectors v_ex and v_ap of states)
def State_Fidelities_noramalize_ap(v_ex, v_ap):
    # noramlize v_ap column wise
    v_ap2 = v_ap / norm(v_ap, axis=0)
    return abs(sum(multiply(v_ex.conj(), v_ap2), axis=0))**2
def State_Infidelities_noramalize_ap(v_ex, v_ap):
    v_ap2 = v_ap / norm(v_ap, axis=0)
    return 1 - abs(sum(multiply(v_ex.conj(), v_ap2), axis=0))**2


#### Exponentials #######################################################
def vecxvec( E, wf, alpha): # Many times faster than multiply
    s = E.size
    for i in range(s):
        wf[i] = exp(-1j * E[i] * alpha) * wf[i]
    return wf
# Construct matrix exponentials and products faster than in numpy or scipy
def vm_exp_mul(E, V, dt = 1.0): # expm(diag(vector))*matrix  multiplication via exp(vector)*matrix
    s = E.size
    A = empty((s, s), complex128)
    for i in range(s):
        A[i,:] = exp(-1j * E[i] * dt) * Dag(V[:,i])
    return A
def expmH_from_Eig(E, V, dt = 1.0):
    U = dot(V, vm_exp_mul(E, V, dt))
    return U
def expmH(H, dt = 1.0):
    E, V = eigh(H)
    return expmH_from_Eig( E, V, dt)

##### Logarithms #########################################################
def unitary_eig(A): # alternative to np.eig returning unitary matrices V
    Emat, V = schur(A, output='complex')
    return diag(Emat), V
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
def rand_state( s, how_many=1, rng=random.default_rng()):
    # every column is a random state with how_many columns and s rows
    X = rng.normal(1, 1, [s,how_many]) +1j*rng.normal(1, 1, [s,how_many])
    return X / norm(X, axis=0)

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

def randH_gauss(s, E0=0, sigma=1, rng=random.default_rng()):
    # According to an idea by Matthias Müller
    Es = rng.normal(E0, sigma, s)
    U = randU_Haar(s, rng)
    H = U @ diag(Es) @ Dag(U)
    return H

def Random_parametric_Hamiltonian_gauss(n, s, E0=0, sigmas=[1,0.1], rng=random.default_rng()):
    # Generates random parametric Hamiltonians with n parameters and an s dimensional Hilbert space using the vector amps to specify the norm of the Hamiltonians
    # Outputs a 3d tensor of the Hamiltonians, where the first index is the Hamiltonian number, and the second and third index are the Hamiltonians themselves
    l = len(sigmas)
    H_s = empty([n+1, s, s], dtype=complex)
    H_s[0] = randH_gauss(s, E0=E0, sigma=sigmas[0], rng=rng)
    for i in range(1, n+1):
        if i >= l-1:
            curr_amp = sigmas[-1]
        else:
            curr_amp = sigmas[i]
        H_s[i] = randH_gauss(s, E0=0, sigma=curr_amp, rng=rng)
    return H_s