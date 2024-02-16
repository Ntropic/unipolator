from unipolator import *
import Analysis_Code.discrete_quantum as dq
from numpy import random, ones, zeros, array

# Test generating a random hamiltonian and then interpolating it
def test_random_hamiltonian():
    # Generate a random hamiltonian
    n = 1 # how many parameters
    s = 16 # dimension of the hamiltonian
    rng = random.default_rng(123)
    amps = np.pi/2*array([1.0, 0.025])
    #H_s = dq.Random_parametric_Hamiltonian_Haar(n, s, amps, rng)
    H_s = dq.Random_parametric_Hamiltonian_gauss(n, s, sigmas=amps, rng=rng)
    c_mins = zeros(n)
    c_maxs = ones(n)
    c_bins = ones(n, dtype=int)
    # Construct the integrators
    trotter = Trotter_System(H_s)
    sym_trotter = Sym_Trotter_System(H_s)
    ui = UI(H_s, c_mins, c_maxs, c_bins)
    #sym = Sym_UI(H_s, c_mins, c_maxs, c_bins)
    system = Hamiltonian_System(H_s)
    c = ones(n)*0.3
    U_ex = np.zeros((s,s), dtype=complex)
    U_ap = np.zeros((s,s), dtype=complex)
    system.expmH(c, U_ex)
    ui.expmH(c, U_ap)
    I_ui = dq.Av_Infidelity(U_ap, U_ex)

    trotter.expmH(c, U_ap)
    
    #U_ap = dq.expmH(H0)@ dq.expmH(H1*c[0])
    I_trotter = dq.Av_Infidelity(U_ap, U_ex)

    sym_trotter.expmH(c, U_ap)
    #U_ap = dq.expmH(H0/2)@ dq.expmH(H1*c[0])@dq.expmH(H0/2)
    I_sym_trotter = dq.Av_Infidelity(U_ap, U_ex)

    ### assert that this function runs without error
    assert I_ui < 1e-2
    assert I_trotter < 1e-2
    assert I_sym_trotter < 1e-2
    #assert False
    return
