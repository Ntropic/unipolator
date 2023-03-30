import discrete_quantum as dq
from unipolator import *
import numpy as np
from hashlib import sha1

def infidelities_2d(k, bins, n, s, I, I_sym):
    rng = np.random.default_rng(100)
    amps = np.array([1.0, 0.1])
    H_s = dq.Random_parametric_Hamiltonian_Haar(n, s, amps, rng=rng)
    c_mins = np.ones(n) * 0
    c_maxs = np.ones(n) * 1.0
    c_bins = bins*np.ones(n, dtype=int)
    ui = UI(H_s, c_mins, c_maxs, c_bins)
    sym_ui = Sym_UI(H_s, c_mins, c_maxs, c_bins)
    system = Hamiltonian_System(H_s)
    U = np.empty([s,s], dtype=complex)
    U_sym = np.empty([s,s], dtype=complex)
    U_classic = np.empty([s,s], dtype=complex)
    for i, (d,e) in k.kronprod(change=False, progress=True):
        c = np.array([d, e])
        ui.expmH(c, U)
        sym_ui.expmH(c, U_sym)
        system.expmH(c, U_classic)
        I[i] = dq.Av_Infidelity(U, U_classic)
        I_sym[i] = dq.Av_Infidelity(U_sym, U_classic)
    return I, I_sym
