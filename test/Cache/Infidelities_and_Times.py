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


def initialize_rep_arrays(to_calculate, reps, methods):
    max_infidelity_values, mean_infidelity_values, times = [], [], []
    for name, method_ in methods.items():
        if 'max' in to_calculate:
            max_infidelity_values.append(np.empty(reps))
        if 'mean' in to_calculate:
            mean_infidelity_values.append(np.empty(reps))
        if 'times' in to_calculate:
            times.append(np.empty(1))
    if 'times' in to_calculate:
        times.append(np.empty((1)))
    return max_infidelity_values, mean_infidelity_values, times


def intitialize_integrators(methods, c_mins, c_maxs, c_bins, rng):
    repeats = 5
    add_points = 4
    point_ratio = 2.0
    integrators = []
    ui_done, trotter_done = False, False
    for name, method in methods.items():
        if 'UI' in name: # only create the UI integrator once and copy it for every method using ui
            if not ui_done:
                ui_integrator = I_mean_UI(c_mins, c_maxs, c_bins, min_order=1, max_order=4, rng=rng, repeats=repeats, add_points=add_points, point_ratio=point_ratio, progress=False)
                ui_done = True
            integrators.append(ui_integrator)
        else: # Trotter
            if not trotter_done:
                trotter_integrator = I_mean_trotter(c_mins, c_maxs, min_order=1, max_order=4, rng=rng, repeats=repeats, add_points=add_points, point_ratio=point_ratio, progress=False)
                trotter_done = True
            integrators.append(trotter_integrator)
    return integrators


def mean_std(x): # Calculate the mean and standard deviation of an array summing over the second axis
    n = x.ndim - 1
    return np.mean(x, axis=n), np.std(x, axis=n)


def Infidelities_and_Times(k, methods, to_calculate, *args):
    reps = 25
    max_infidelity_values, mean_infidelity_values, times = initialize_rep_arrays(to_calculate, reps, methods)
    do_times, do_max, do_mean = 'times' in to_calculate, 'max' in to_calculate, 'mean' in to_calculate
    for i, v, c in k.kronprod(do_index=True, do_change=True):
        if k.changed('num_controls'):
            num_controls = k.value('num_controls')
            c_mins, c_maxs, c_bins = np.zeros(num_controls), np.ones(num_controls), np.ones(num_controls, dtype=int)
            if 'mean' in to_calculate:
                curr_args = [k.value('num_controls'), 'mean', to_calculate]
                seed = int(sha1(str(curr_args).encode('utf-8')).hexdigest(), 16)
                rng = np.random.default_rng(seed)
                integrators = intitialize_integrators(methods, c_mins, c_maxs, c_bins, rng)
        curr_args = [i, v, c]
        seed = int(sha1(str(curr_args).encode('utf-8')).hexdigest(), 16)
        rng = np.random.default_rng(seed)
        if k.changed('dim_hilbert'):
            dim_hilbert = k.value('dim_hilbert')
            curr_U_ex, U_ex2, U_ap = [np.empty((dim_hilbert, dim_hilbert), dtype=complex) for i in range(3)]
        amp0 = np.pi*4
        amps = [amp0] + [amp0*k.value('amp_ratio') for i in range(num_controls)]
        m_times = k.value('m_times')
        for r in range(reps):
            H_s = dq.Random_parametric_Hamiltonian_Haar(num_controls, dim_hilbert, amps, rng)
            for i_method, (name, method_) in enumerate(methods.items()):
                method = method_['method']
                if 'UI' in name:
                    H_s_ui = H_s.copy()
                    H_s_ui[1:] = H_s_ui[1:] /k.value('bins') / 2**m_times
                    exact = Hamiltonian_System(H_s_ui)
                    curr_c = np.ones(num_controls) / 2
                    approx = method(H_s_ui, c_mins, c_maxs, c_bins)
                else:
                    exact = Hamiltonian_System(H_s)
                    curr_c = np.ones(num_controls)
                    approx = method(H_s, m_times=m_times)
                if do_max:
                    exact.expmH(curr_c, curr_U_ex)
                    approx.expmH(curr_c, U_ap)
                    max_infidelity_values[i_method][r] = dq.Av_Infidelity(curr_U_ex, U_ap)
                if do_mean:
                    mean_infidelity_values[i_method][r], x = integrators[i_method].Mean_Average_Infidelity(exact.expmH, approx.expmH, U_ex2, U_ap)
                if r == 0 and 'times' in to_calculate:
                    times[i_method] = timeit_autorange(lambda: approx.expmH(curr_c, U_ap))
            if r == 0 and do_times:
                times[-1] = timeit_autorange(lambda: exact.expmH(curr_c, U_ex2))
        len_args = len(args)
        if do_times:
            len_args -= 1
        ind = 0
        repeats = 0
        while ind < len_args:
            if do_max:
                args[ind][i], args[ind+1][i] = mean_std(max_infidelity_values[repeats])
                ind += 2
            if do_mean:
                args[ind][i], args[ind+1][i] = mean_std(mean_infidelity_values[repeats])
                ind += 2
            if do_times:
                args[ind][i] = times[repeats]
                ind += 1
            repeats += 1
        if do_times:
            args[-1][i] = times[-1]
    return args
