from Analysis_Code.Average_Infidelity_Integrator import *
from kronbinations import *
from unipolator import *
import Analysis_Code.discrete_quantum as dq
import numpy as np
from numba import jit
import timeit
import numpy as np
from hashlib import sha1

def timeit_autorange(func, dim_hilbert, curr_c, min_time=2.0):
    # Warm up
    U = np.empty((dim_hilbert, dim_hilbert), dtype=complex)
    time = timeit.timeit(lambda: func.expmH(curr_c, U), number=1)
    if time < min_time:
        n = int(np.ceil(2.0/time))
        n = np.min([10**4, n])
        # create n repetitions of c array
        cs = np.tile(curr_c, (n, 1))
        Us = np.empty((n, dim_hilbert, dim_hilbert), dtype=complex)
        time = timeit.timeit(lambda: func.expmH_pulse_no_multiply(cs, Us), number=1)/n
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


def mean_std(x): # Calculate the mean and standard deviation of an array summing over the second axis
    n = x.ndim - 1
    return np.mean(x, axis=n), np.std(x, axis=n)


def mean_std_asym(arr, axis=-1):
    # Calculate the mean and two std's for each side of the mean (asymmetric std)
    # first check if axis is a tuple
    if isinstance(axis, (tuple, list, np.ndarray)):
        # create a shape_out list 
        shape = arr.shape
        shape_out = []
        for i in range(len(shape)):
            if not i in axis:
                shape_out.append(shape[i])
        mean = np.zeros(shape_out)
        std_min = np.zeros(shape_out)
        std_max = np.zeros(shape_out)
        for ind, ind_reduced in zip(nditer_slices(shape, slice_dimensions=axis), nditer_slices(shape_out)):
            mean[ind_reduced], std_min[ind_reduced], std_max[ind_reduced] = mean_std_asym(arr[ind], axis=-1) 
        return mean, std_min, std_max
    elif axis < 0:
        mean = np.mean(arr)
        smaller_than = arr[arr<mean]
        larger_than = arr[arr>mean]
        std_min = np.sqrt(np.sum((mean - smaller_than)**2)/(len(smaller_than) - 1))
        std_max = np.sqrt(np.sum((larger_than - mean)**2)/(len(larger_than) -1))
        return mean, std_min, std_max
    else:
        # iterate over all axis except the one specified in axis
        #define output arrays and their shape
        shape = arr.shape
        shape_out = shape[:axis] + shape[axis+1:]
        mean = np.zeros(shape_out)
        std_min = np.zeros(shape_out)
        std_max = np.zeros(shape_out)
        # iterate over all axis except the one specified in axis, get the index for every iteration in loop
        for ind in np.ndindex(shape_out):
            # modify the index to include slice along axis
            ind_post = ind[:axis] + (slice(None),) + ind[axis:]
            # calculate mean and asymmetric std for this slice
            mean[ind], std_min[ind], std_max[ind] = mean_std_asym(arr[ind_post], axis=-1)
        return mean, std_min, std_max


def Infidelities_and_Times(k, methods, to_calculate, *args):
    reps = 100
    how_many_samples = 100
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
            curr_U_ex, U_ex2, U_ap = [np.empty((dim_hilbert, dim_hilbert), dtype=complex) for i in range(3)]
        amp0 = np.pi/2
        amps = [amp0] + [amp0*k.value('amp_ratio') for i in range(num_controls)]
        for r in k.tqdm(range(reps)):
            H_s = dq.Random_parametric_Hamiltonian_gauss(num_controls, dim_hilbert, sigmas=amps, rng=rng)
            for i_method, (name, method_) in enumerate(methods.items()):
                method = method_['method']
                m_times = method_['m_times']
                for m, curr_m_times in enumerate(m_times):
                    if 'UI' in name:
                        H_s_ui = H_s.copy()
                        H_s_ui[1:] = H_s_ui[1:] /k.value('bins') / 2**curr_m_times
                        exact = Hamiltonian_System(H_s_ui)
                        curr_c = np.ones(num_controls) / 2
                        approx = method(H_s_ui, c_mins, c_maxs, c_bins)
                    else:
                        exact = Hamiltonian_System(H_s)
                        curr_c = np.ones(num_controls)
                        approx = method(H_s, m_times=curr_m_times)
                    if do_max:
                        exact.expmH(curr_c, curr_U_ex)
                        approx.expmH(curr_c, U_ap)
                        max_infidelity_values[i_method][m, r] = dq.Av_Infidelity(curr_U_ex, U_ap)
                    if do_mean:
                        mean_infidelity_values[i_method][m, r, :] = integrators[i_method].InfidelityFit2Samples(exact.expmH, approx.expmH, U_ex2, U_ap, how_many_samples=how_many_samples)
                    if r == 0 and do_times:
                        times[i_method][m] = timeit_autorange(approx, dim_hilbert, curr_c)
            if r == 0 and do_times:
                times[-1] = timeit_autorange(exact, dim_hilbert, curr_c)
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
                args[ind][i,:] = times[i_method]
                ind += 1
            i_method += 1
        if do_times:
            args[-1][i] = times[-1]
    return args
