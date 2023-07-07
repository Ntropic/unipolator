from Analysis_Code.Average_Infidelity_Integrator import *
from kronbinations import *
from unipolator import *
import Analysis_Code.discrete_quantum as dq
from numba import jit
import timeit
import numpy as np
from hashlib import sha1

def timeit_autorange_vector(func, dim_hilbert, curr_c, v_in, v_out, min_time=2.0, pass_U=False):
    # Warm up
    if pass_U:
        U = np.empty((dim_hilbert, dim_hilbert), dtype=complex)
        def func2(curr_c, v_in):
            func.expmH(curr_c, U)
            return U @ v_in
        time = timeit.timeit(lambda: func2(curr_c, v_in), number=1)
    else:    
        time = timeit.timeit(lambda: func.expmH(curr_c, v_in, v_out), number=1)
    if time < min_time:
        n = int(np.ceil(2.0/time))
        n = np.min([10**4, n])
        # create n repetitions of c array
        cs = np.tile(curr_c, (n, 1))
        if pass_U:
            def func2(curr_c, v_in):
                for c in curr_c:
                    func.expmH(c, U)
                    v_out = U @ v_in
                return v_out
            time = timeit.timeit(lambda: func2(cs, v_in), number=1)/n
        else:
            v_out_s = np.empty((n, dim_hilbert, 1), dtype=complex)
            time = timeit.timeit(lambda: func.expmH_pulse_no_multiply(cs, v_in, v_out_s), number=1)/n
    return time


def initialize_rep_arrays(to_calculate, reps, how_many_samples, methods):
    max_infidelity_values, mean_infidelity_values, std_infidelity_values_min, std_infidelity_values_max, times = [], [], [], [], []
    for name, method_ in methods.items():
        m_times = method_['m_times']
        len_m = len(m_times)
        if 'max' in to_calculate:
            max_infidelity_values.append(np.empty((len_m, reps)))
        if 'mean' in to_calculate:
            mean_infidelity_values.append(np.empty((len_m, reps, how_many_samples)))
        if 'times' in to_calculate:
            times.append(np.empty((len_m)))
    if 'times' in to_calculate:
        times.append(np.empty((1)))
    return max_infidelity_values, mean_infidelity_values, times


def intitialize_integrators(methods, c_mins, c_maxs, c_bins, rng):
    repeats = 5
    point_ratio = 2.0
    integrators = []
    ui_done, trotter_done, expm_done = False, False, False
    for name, method in methods.items():
        if 'UI' in name: # only create the UI integrator once and copy it for every method using ui
            if not ui_done:
                ui_integrator = I_mean_UI(c_mins, c_maxs, c_bins, min_order=1, max_order=2, rng=rng, repeats=repeats, point_ratio=point_ratio)
                ui_done = True
            integrators.append(ui_integrator)
        if 'Expm' in name: # only create the Expm integrator once and copy it for every method using Expm
            if not expm_done:
                expm_integrator = I_mean_expm(c_mins, c_maxs, rng=rng)
                expm_done = True
            integrators.append(expm_integrator)
        else: # Trotter
            if not trotter_done:
                trotter_integrator = I_mean_trotter(c_mins, c_maxs, min_order=1, max_order=2, rng=rng, repeats=repeats, point_ratio=point_ratio)
                trotter_done = True
            integrators.append(trotter_integrator)
    return integrators


def Binned_Infidelities_and_Times_vector(k, methods, to_calculate, *args):
    reps = 100
    how_many_samples = 100
    how_many_states = 1
    max_infidelity_values, mean_infidelity_values, times = initialize_rep_arrays(to_calculate, reps, how_many_samples, methods)
    do_times, do_max, do_mean = 'times' in to_calculate, 'max' in to_calculate, 'mean' in to_calculate
    for i, v, c in k.kronprod(do_index=True, do_change=True):
        if k.changed('num_controls'):
            num_controls = k.value('num_controls')
            c_mins, c_maxs, c_bins = np.zeros(num_controls), np.ones(num_controls), np.ones(num_controls, dtype=int)
            if do_mean:
                curr_args = [k.value('num_controls'), do_mean]
                seed = int(sha1(str(curr_args).encode('utf-8')).hexdigest(), 16)
                rng = np.random.default_rng(seed)
                integrators = intitialize_integrators(methods, c_mins, c_maxs, c_bins, rng)
        curr_args = [i, v, c]
        seed = int(sha1(str(curr_args).encode('utf-8')).hexdigest(), 16)
        rng = np.random.default_rng(seed)
        if k.changed('dim_hilbert'):
            dim_hilbert = k.value('dim_hilbert')
        if k.changed('bins'):
            bins = k.value('bins')
        amp0 = np.pi/2
        amps = [amp0] + [amp0*k.value('amp_ratio') for i in range(num_controls)]
        v_in = dq.rand_state(dim_hilbert, how_many_states, rng=rng)
        v_ex = np.empty((dim_hilbert, how_many_states), dtype=complex)
        v_ap = np.empty((dim_hilbert, how_many_states), dtype=complex)
        U_ex = np.empty((dim_hilbert, dim_hilbert), dtype=complex)
        for r in k.tqdm(range(reps)):
            H_s = dq.Random_parametric_Hamiltonian_gauss(num_controls, dim_hilbert, sigmas=amps, rng=rng)
            for i_method, (name, method_) in enumerate(methods.items()):
                method = method_['method']
                m_times = method_['m_times']
                for m, curr_m_times in enumerate(m_times):
                    exact = Hamiltonian_System(H_s)
                    if 'UI' in name:
                        factor = int(bins * curr_m_times)
                        curr_c = np.ones(num_controls) / 2 / factor
                        approx = method(H_s, c_mins, c_maxs, c_bins*factor)
                    elif 'Expm' in name:
                        factor = 1
                        approx = method(H_s)
                        curr_c = np.ones(num_controls)
                        approx.set_m_star(curr_m_times)
                    elif 'Trotter':
                        factor = 1
                        curr_c = np.ones(num_controls)
                        approx = method(H_s, n_times=curr_m_times)
                    if do_mean:
                        if not 'Expm' in name:
                            mean_infidelity_values[i_method][m, r, :] = integrators[i_method].InfidelityFit2Samples_vector(lambda c, U_ex: exact.expmH(c/factor, U_ex), lambda c, vin, vout: approx.expmH(c/factor, vin, vout), U_ex, v_in, v_ex, v_ap, how_many_samples=how_many_samples)
                        else:
                            mean_infidelity_values[i_method][m, r, :] = integrators[i_method].Sample_Infidelities_vector(lambda c, U_ex: exact.expmH(c/factor, U_ex), lambda c, vin, vout: approx.expmH(c/factor, vin, vout), U_ex, v_in, v_ex, v_ap, how_many_samples=how_many_samples)
                    if r == 0 and do_times:
                        times[i_method][m] = timeit_autorange_vector(approx, dim_hilbert, curr_c, v_in, v_ap )
            if r == 0 and do_times:
                times[-1] = timeit_autorange_vector(exact, dim_hilbert, curr_c, v_in, v_ap, pass_U=True)
        len_args = len(args)
        if do_times:
            len_args -= 1
        ind = 0
        i_method = 0
        while ind < len_args:
            if do_max:
                args[ind][i,:], args[ind+1][i,:], args[ind+2][i,:] = mean_std_asym(max_infidelity_values[i_method], axis=1)
                ind += 3
            if do_mean:
                args[ind][i,:], args[ind+1][i,:], args[ind+2][i,:]  = mean_std_asym(mean_infidelity_values[i_method], axis=(1,2))
                ind += 3
            if do_times:
                pass
                args[ind][i,:] = times[i_method]
                ind += 1
            i_method += 1
        if do_times:
            args[-1][i] = times[-1]
            pass
    return args
