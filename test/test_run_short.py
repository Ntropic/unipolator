import numpy as np
import discrete_quantum as dq
from unipolator import *

c_mins = np.zeros(3)
c_maxs = np.ones(3)
c_bins = np.array([1,2,3], dtype=int)
s = 4
amps = [1.0, 0.1]
H_s = dq.Random_parametric_Hamiltonian_Haar(3, s, amps)                   
U = np.empty((s, s), dtype=complex)

sym_ui = Sym_UI(H_s, c_mins, c_maxs, c_bins) 
sym_ui.expmH(np.array([0.5, 0.5, 0.5]), U)

ui = UI(H_s, c_mins, c_maxs, c_bins) 
ui.expmH(np.array([0.5, 0.5, 0.5]), U)

trotter = Trotter_System(H_s) 
trotter.expmH(np.array([0.5, 0.5, 0.5]), U)

sym_trotter = Sym_Trotter_System(H_s) 
sym_trotter.expmH(np.array([0.5, 0.5, 0.5]), U)

exact = Hamiltonian_System(H_s) 
exact.expmH(np.array([0.5, 0.5, 0.5]), U)

Hamiltonian_System

print(U)