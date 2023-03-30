from Average_Infidelity_Integrator import *
from kronbinations import *
from unipolator import *
import discrete_quantum as dq
import numpy as np
from numba import jit
import numpy as np
from hashlib import sha1

def Infidelities(k, methods, *args):
    reps = 50
    for i, c, v in k.kronprod(do_index=True, do_change=True):
        if k.changed('num_controls'):
            num_controls = k.value('num_controls')
            c_mins = np.zeros(num_controls)
            c_maxs = np.ones(num_controls)
            c_bins = np.ones(num_controls, dtype=int)
            pr = 2.5
            integrators = []
            for name, method in methods.items():
                min_order = 1
                max_order = 4
                if 'UI' in name:
                    curr_args = [k.value('num_controls'), 'UI', name]
                    seed = int(sha1(str(curr_args).encode('utf-8')).hexdigest(), 16)
                    rng = np.random.default_rng(seed)
                    integrator = I_mean_UI(c_mins, c_maxs, c_bins, min_order=min_order, max_order=max_order, rng=rng, repeats=5, add_points=4, point_ratio=pr)
                else:
                    curr_args = [k.value('num_controls')]
                    seed = int(sha1(str(curr_args).encode('utf-8')).hexdigest(), 16)
                    rng = np.random.default_rng(seed)
                    integrator = I_mean_trotter(c_mins, c_maxs, min_order=min_order, max_order=max_order, rng=rng, repeats=5, add_points=4, point_ratio=pr)
                integrators.append(integrator)
        curr_args = [i, c, v]
        seed = int(sha1(str(curr_args).encode('utf-8')).hexdigest(), 16)
        rng = np.random.default_rng(seed)
        if k.changed('dim_hilbert'):
            dim_hilbert = k.value('dim_hilbert')
            U_ex1, U_ex1_2, U_ex2, U_ap = [np.empty((dim_hilbert, dim_hilbert), dtype=complex) for i in range(4)]
        amp_ratio = k.value('amp_ratio')
        bins = k.value('bins')
        amps = [1.0] + [amp_ratio/bins for i in range(num_controls)]
        max_infidelity_values = np.empty((len(methods), reps))
        mean_infidelity_values = np.empty((len(methods), reps))
        for r in range(reps):
            H_s = dq.Random_parametric_Hamiltonian_Haar(num_controls, dim_hilbert, amps, rng)
            c1 = np.ones(num_controls)
            c1_2 = np.ones(num_controls) / 2
            exact = Hamiltonian_System(H_s)
            exact.expmH(c1, U_ex1)
            exact.expmH(c1_2, U_ex1_2)
            for i_method, (name, method) in enumerate(methods.items()):
                approx = method(H_s, c_mins, c_maxs, c_bins) if 'UI' in name else method(H_s)
                approx.expmH(c1_2, U_ap)
                max_infidelity_values[i_method, r] = dq.Av_Infidelity(U_ex1_2, U_ap)
                mean_infidelity_values[i_method, r], x = integrators[i_method].Mean_Average_Infidelity(exact.expmH, approx.expmH, U_ex2, U_ap)
        len_args = len(args)
        ind = 0
        repeats = 0
        while ind < len_args:
            max_I_mean, max_I_std, mean_I_mean, mean_I_std = args[ind:ind+4]
            max_I_mean[i] = np.mean(max_infidelity_values[repeats,:])
            max_I_std[i] = np.std(max_infidelity_values[repeats,:])
            mean_I_mean[i] = np.mean(mean_infidelity_values[repeats,:])
            mean_I_std[i] = np.std(mean_infidelity_values[repeats,:])
            repeats += 1
            ind += 4
    return args
