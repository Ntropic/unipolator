from Analysis_Code.discrete_quantum import *
from unipolator import *
import numpy as np
import Analysis_Code.discrete_quantum as dq
import numpy as np
from hashlib import sha1

def Infidelities_2d_UI_and_Trotter(k, s, n, amps, bins, H_s, I_ui, I_ui_sym, I_trotter, I_sym_trotter):
    c_mins = np.zeros(n)
    c_maxs = np.ones(n)
    c_bins = bins*np.ones(n, dtype=int)
    U_ap = np.empty([s,s], dtype=complex)
    U_ex = np.empty([s,s], dtype=complex)
    trotter = Trotter_System(H_s)
    sym_trotter = Sym_Trotter_System(H_s)
    ui = UI(H_s, c_mins, c_maxs, c_bins)
    sym_ui = Sym_UI(H_s, c_mins, c_maxs, c_bins)
    system = Hamiltonian_System(H_s)
    for i, c in k.kronprod(change=False, progress=True):
        c = np.array(c)
        system.expmH(c, U_ex)
        ui.expmH(c, U_ap)
        I_ui[i] = Av_Infidelity(U_ap, U_ex)
        sym_ui.expmH(c, U_ap)
        I_ui_sym[i] = Av_Infidelity(U_ap, U_ex)
        trotter.expmH(c, U_ap)
        I_trotter[i] = Av_Infidelity(U_ap, U_ex)
        sym_trotter.expmH(c, U_ap)
        I_sym_trotter[i] = Av_Infidelity(U_ap, U_ex)
