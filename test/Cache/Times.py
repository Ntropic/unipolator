from Average_Infidelity_Integrator import *
from kronbinations import *
from unipolator import *
import discrete_quantum as dq
import numpy as np
from numba import jit
import timeit
import numpy as np
from hashlib import sha1

def timeit_autorange(func, min_time=2.0):
    # Warm up
    time = timeit.timeit(func, number=1)
    if time < min_time:
        n = int(np.ceil(min_time/time))
        time = timeit.timeit(func, number=n)/n
    return time


def Times(k, methods, *args):
    for i, c, v in k.kronprod(do_index=True, do_change=True):
        curr_args = [i, c, v]
        seed = int(sha1(str(curr_args).encode('utf-8')).hexdigest(), 16)
        rng = np.random.default_rng(seed)
        if k.changed('num_controls'):
            num_controls = k.value('num_controls')
            c_mins = np.zeros(num_controls)
            c_maxs = np.ones(num_controls)
            c_bins = np.ones(num_controls, dtype=int)
        if k.changed('dim_hilbert'):
            dim_hilbert = k.value('dim_hilbert')
            U_ex1, U_ex1_2, U_ex2, U_ap = [np.empty((dim_hilbert, dim_hilbert), dtype=complex) for i in range(4)]
        amp_ratio = k.value('amp_ratio')
        bins = k.value('bins')
        amps = [1.0] + [amp_ratio/bins for i in range(num_controls)]
        times = np.empty(len(methods)+1)
        H_s = dq.Random_parametric_Hamiltonian_Haar(num_controls, dim_hilbert, amps, rng)
        c1_2 = np.ones(num_controls) / 2
        exact = Hamiltonian_System(H_s)
        for i_method, (name, method) in enumerate(methods.items()):
            if 'UI' in name:
                approx = method(H_s, c_mins, c_maxs, c_bins)
            else:
                approx = method(H_s)
            times[i_method] = timeit_autorange(lambda: approx.expmH(c1_2, U_ap))
        times[-1] = timeit_autorange(lambda: exact.expmH(c1_2, U_ex2))
        len_args = len(args)
        for j in range(len_args):
            args[j][i] = times[j]
    return args
