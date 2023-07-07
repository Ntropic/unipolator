from numpy import zeros, ones, array, empty, random
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import axes3d
from kronbinations import *
from timeit import *
from tikzplotlib import get_tikz_code, save as tikz_save
import os

from unipolator import *
from Analysis_Code.Average_Infidelity_Integrator import *
import Analysis_Code.discrete_quantum as dq
from Analysis_Code.useful import *


# A script that checks whether a file needs to be updated, by comparing it's modification time to the modification time of the file it was created from
def time2update(filename, filename_of_original):
    if not os.path.isfile(filename): # not yet created
        return True
    if not os.path.isfile(filename_of_original): # cannot create from a file that does not exist
        raise ValueError('The file '+filename_of_original+' does not exist.')
    if os.path.getmtime(filename) < os.path.getmtime(filename_of_original): # Needs to be updated
        return True
    return False

# a function that transforms lists into numpy arrays for lists inside of dictionaries (recursively)
def list_in_dict2array(dictionary):
    for key in dictionary.keys():
        if isinstance(dictionary[key], dict):
            dictionary[key] = list_in_dict2array(dictionary[key])
        elif isinstance(dictionary[key], list):
            vals = [float(val) if val is not None else val for val in dictionary[key]]
            dictionary[key] = np.array(vals, dtype=float)
    return dictionary

def closest_match(array, value):  # find the index of the closest value in the array and the value at that index
    index = np.argmin(np.abs(array-value))
    return index, array[index]

def first_smaller(array, value):  # find the index of the first value in the array that is smaller than the value 
    truth_array = array<value
    if np.sum(truth_array) == 0:
        return None
    else:
        index = np.argmax(truth_array)
    return index

def map_index2value(index, values, none_val):
    return values[index] if index is not None else none_val

# a function that rounds a float to n significant digits
def round_sig(x, sig=2):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(np.floor(np.log10(abs(x))))-1)

def index_partial_minimum(array, std=None):
    n = len(array)
    if std is None:
        ind_partial_minimum = np.zeros(n,dtype=int)
        for i in range(n):
            ind_partial_minimum[i] = np.argmin(array[:i+1])
        ind_partial_minimum = np.unique(ind_partial_minimum)
    else: # find minimum considering spread
        ind_partial_minimum = [0]
        i = 0
        for a in array[1:]:
            i += 1
            if a < array[ind_partial_minimum[-1]] - std[ind_partial_minimum[-1]]:
                ind_partial_minimum.append(i)
            else:
                break
    return ind_partial_minimum

def steps_xy(x,y):
    fx = [item for item in x[:-1] for i in range(2)] + [x[-1]]
    fy = [y[0]] + [item for item in y[1:] for i in range(2)] 
    return fx, fy

def get_parameters(combination, all_parameters):
    # A function that from the name of the combination and the list of parameter dictionaries 
    # generates a parameter dictionary for the JIT_kroncombination
    # also outputs the axis labels and types
    parameters = {}
    axis_labels = []
    axis_types = []
    for parameter in all_parameters:
        if parameter['name'] in combination:
            # check if parameter has key 'weight'
            if 'weight' in parameter.keys():
                parameters[parameter['name']] = {'value': parameter['variation_vals'], 'weight': parameter['weight']}
            if 'weights' in parameter.keys():
                parameters[parameter['name']] = {'value': parameter['variation_vals'], 'weight': parameter['weights']}
            else:
                parameters[parameter['name']] = parameter['variation_vals']
            axis_labels.append(parameter['label'])
            axis_types.append(parameter['variation_type'])
        else:
            parameters[parameter['name']] = [parameter['def_val']]
    return parameters, axis_labels, axis_types
    
def create_variables(k, method_dict, to_do=['mean', 'max', 'times']):
    variable_dicts = {}
    type_names = []
    if 'max' in to_do:
        type_names += ['max_I_mean', 'max_I_std_min', 'max_I_std_max']    # using asymmetric std
    if 'mean' in to_do:
        type_names += ['mean_I_mean', 'mean_I_std_min', 'mean_I_std_max']   # using asymmetric std
    if 'times' in to_do:
        type_names += ['times']
    for method_name, method in method_dict.items():
        if 'm_times' in method:
            m_times = method['m_times']
            args = (len(m_times),)
        else:
            method_dict[method_name]['m_times'] = [0] # add this argument for simplicity in later data processing
            args = (1, )
        variable_dicts[method_name] = {type_name: k.empty(*args, name=method_name+'_'+type_name) for type_name in type_names}
    if 'times' in to_do:
        variable_dicts['times'] = k.empty(len(m_times), name='times')
    return variable_dicts, method_dict

# Calculate the average and maximum infidelities between exact and approximate unitaries as a function 
# of system size, of number of control parameters, and the amplitudes of the Hamiltonians
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

def mean_std(x): # Calculate the mean and standard deviation of an array summing over the second axis
    n = x.ndim - 1
    return np.mean(x, axis=n), np.std(x, axis=n)

def Infidelities_and_Times(k, methods, to_calculate, *args, rng):
    # Simplifies grid creation to a single grid interval to speed up caching, use script below (Binned version) to create a grid with multiple intervals, necessary for accurate timing
    reps = 100 
    how_many_samples = 100
    # Inititalize intermediate arrays, tpo store results of reps, to average over
    max_infidelity_values, mean_infidelity_values, times = initialize_rep_arrays(to_calculate, reps, how_many_samples, methods)
    do_times, do_max, do_mean = 'times' in to_calculate, 'max' in to_calculate, 'mean' in to_calculate
    for i, v, c in k.kronprod(do_index=True, do_change=True):
        if k.changed('num_controls'):
            num_controls = k.value('num_controls')
            c_mins, c_maxs, c_bins = np.zeros(num_controls), np.ones(num_controls), np.ones(num_controls, dtype=int)
            # Generate the mean value approximator
            if do_mean:
                integrators = intitialize_integrators(methods, c_mins, c_maxs, c_bins, rng)
        if k.changed('dim_hilbert'):
            dim_hilbert = k.value('dim_hilbert')
            curr_U_ex, U_ex2, U_ap = [np.empty((dim_hilbert, dim_hilbert), dtype=complex) for i in range(3)]
        amp0 = np.pi/2
        amps = [amp0] + [amp0*k.value('amp_ratio') for i in range(num_controls)] # /k.value('bins')

        for r in k.tqdm(range(reps)):     
            # Generate the random Hamiltonian
            #H_s = dq.Random_parametric_Hamiltonian_Haar(num_controls, dim_hilbert, amps, rng)
            H_s = dq.Random_parametric_Hamiltonian_gauss(num_controls, dim_hilbert, sigmas=amps, rng=rng)
            # Approximate calculation
            for i_method, (name, method_) in enumerate(methods.items()):   # Calculate max infidelities, average infidelities, and time
                method = method_['method']
                m_times = method_['m_times']
                for m, curr_m_times in enumerate(m_times):
                    if 'UI' in name:
                        H_s_ui = H_s.copy() 
                        H_s_ui[1:] = H_s_ui[1:] /k.value('bins') / 2**curr_m_times
                        exact = Hamiltonian_System(H_s_ui)                        
                        curr_c = np.ones(num_controls) / 2 
                        approx = method(H_s_ui, c_mins, c_maxs, c_bins)  
                    else: # Trotteresque
                        exact = Hamiltonian_System(H_s)
                        curr_c = np.ones(num_controls)
                        approx = method(H_s, m_times=curr_m_times)
                    ## Max Infidelities ####################################################################################################
                    if do_max:
                        exact.expmH(curr_c, curr_U_ex)
                        approx.expmH(curr_c, U_ap)
                        max_infidelity_values[i_method][m, r] = dq.Av_Infidelity(curr_U_ex, U_ap)  # Max infidelity of approximation
                    ## Mean Infidelities ###################################################################################################
                    if do_mean:
                        mean_infidelity_values[i_method][m, r, :] = integrators[i_method].InfidelityFit2Samples(exact.expmH, approx.expmH, U_ex2, U_ap, how_many_samples=how_many_samples)
                    ## Times ###############################################################################################################
                    # use timeit to time the function approx.expmH with the arguments c1_2
                    if r == 0 and do_times:
                        times[i_method][m] = timeit_autorange(approx, dim_hilbert, curr_c)
            if r == 0 and do_times:
                times[-1] = timeit_autorange(exact, dim_hilbert, curr_c)
        # Save values onto *args = [max_I_mean, max_I_str, mean_I_mean, mean_I_std, times]
        len_args = len(args)
        if do_times:
            len_args -= 1
        ind = 0
        i_method = 0
        while ind < len_args: # last one for exact times
            if do_max:
                args[ind][i,:], args[ind+1][i,:], args[ind+2][i,:] = mean_std_asym(max_infidelity_values[i_method], axis=1) #max_I_mean, max_I_std
                ind += 3
            if do_mean:
                args[ind][i,:], args[ind+1][i,:], args[ind+2][i,:]  = mean_std_asym(mean_infidelity_values[i_method], axis=(1,2)) #mean_I_mean, mean_I_std
                ind += 3
            if do_times:
                args[ind][i,:] = times[i_method] #time
                ind += 1
            i_method += 1
        if do_times:
            args[-1][i] = times[-1]
    return args


def Binned_Infidelities_and_Times(k, methods, to_calculate, *args, rng):
    ### With complete grid cache (for timing purposes) ### Use non binned version above for faster results, whenever timing is not necessary (cache creation is slower for full grids)
    reps = 100
    how_many_samples = 100
    # Inititalize intermediate arrays, too store results of reps, to average over
    max_infidelity_values, mean_infidelity_values, times = initialize_rep_arrays(to_calculate, reps, how_many_samples, methods)
    do_times, do_max, do_mean = 'times' in to_calculate, 'max' in to_calculate, 'mean' in to_calculate
    for i, v, c in k.kronprod(do_index=True, do_change=True):
        if k.changed('num_controls'):
            num_controls = k.value('num_controls')
            c_mins, c_maxs, c_bins = np.zeros(num_controls), np.ones(num_controls), np.ones(num_controls, dtype=int)
            # Generate the mean value approximator
            if do_mean:
                integrators = intitialize_integrators(methods, c_mins, c_maxs, c_bins, rng)
        if k.changed('dim_hilbert'):
            dim_hilbert = k.value('dim_hilbert')
        if k.changed('bins'):
            bins = k.value('bins')
        amp0 = np.pi/2
        amps = [amp0] + [amp0*k.value('amp_ratio') for i in range(num_controls)] # /k.value('bins')

        for r in k.tqdm(range(reps)):     
            # Generate the random Hamiltonian
            #H_s = dq.Random_parametric_Hamiltonian_Haar(num_controls, dim_hilbert, amps, rng)
            H_s = dq.Random_parametric_Hamiltonian_gauss(num_controls, dim_hilbert, sigmas=amps, rng=rng)
            curr_U_ex, U_ex2, U_ap = [np.empty((dim_hilbert, dim_hilbert), dtype=complex) for i in range(3)]
            
            # Approximate calculation
            for i_method, (name, method_) in enumerate(methods.items()):   # Calculate max infidelities, average infidelities, and time
                method = method_['method']
                m_times = method_['m_times']
                for m, curr_m_times in enumerate(m_times):
                    exact = Hamiltonian_System(H_s) 
                    if 'UI' in name:
                        factor = int(bins * 2**curr_m_times)      
                        curr_c = np.ones(num_controls) / 2 / factor 
                        approx = method(H_s, c_mins, c_maxs, c_bins*factor)  
                    else: # Trotteresque
                        factor = 1
                        curr_c = np.ones(num_controls)
                        approx = method(H_s, m_times=curr_m_times)
                    ## Max Infidelities ####################################################################################################
                    if do_max:
                        exact.expmH(curr_c, curr_U_ex)
                        approx.expmH(curr_c, U_ap)
                        max_infidelity_values[i_method][m, r] = dq.Av_Infidelity(curr_U_ex, U_ap)  # Max infidelity of approximation
                    ## Mean Infidelities ###################################################################################################
                    if do_mean:
                        mean_infidelity_values[i_method][m, r, :] = integrators[i_method].InfidelityFit2Samples(lambda c, U: exact.expmH(c/factor, U),lambda c, U: approx.expmH(c/factor, U), U_ex2, U_ap, how_many_samples=how_many_samples)
                    ## Times ###############################################################################################################
                    # use timeit to time the function approx.expmH with the arguments c1_2
                    if r == 0 and do_times:
                        times[i_method][m] = timeit_autorange(approx, dim_hilbert, curr_c)
            if r == 0 and do_times:
                times[-1] = timeit_autorange(exact, dim_hilbert, curr_c)
        # Save values onto *args = [max_I_mean, max_I_str, mean_I_mean, mean_I_std, times]
        len_args = len(args)
        if do_times:
            len_args -= 1
        ind = 0
        i_method = 0
        while ind < len_args: # last one for exact times
            if do_max:
                args[ind][i,:], args[ind+1][i,:], args[ind+2][i,:] = mean_std_asym(max_infidelity_values[i_method], axis=1) #max_I_mean, max_I_std
                ind += 3
            if do_mean:
                args[ind][i,:], args[ind+1][i,:], args[ind+2][i,:]  = mean_std_asym(mean_infidelity_values[i_method], axis=(1,2)) #mean_I_mean, mean_I_std
                ind += 3
            if do_times:
                pass
                args[ind][i,:] = times[i_method] #time
                ind += 1
            i_method += 1
        if do_times:
            args[-1][i] = times[-1]
            pass
    return args


def Binned_Infidelities_and_Times_vector(k, methods, to_calculate, *args, rng):
    ### With complete grid cache (for timing purposes) ### Use non binned version above for faster results, whenever timing is not necessary (cache creation is slower for full grids)
    # Does not use 2**m_times, but m_times directly for binning / stepping
    reps = 100
    how_many_samples = 100
    how_many_states = 1  # makes analysis easier. simply run more reps
    # Inititalize intermediate arrays, too store results of reps, to average over
    max_infidelity_values, mean_infidelity_values, times = initialize_rep_arrays(to_calculate, reps, how_many_samples, methods)
    do_times, do_max, do_mean = 'times' in to_calculate, 'max' in to_calculate, 'mean' in to_calculate
    for i, v, c in k.kronprod(do_index=True, do_change=True):
        if k.changed('num_controls'):
            num_controls = k.value('num_controls')
            c_mins, c_maxs, c_bins = np.zeros(num_controls), np.ones(num_controls), np.ones(num_controls, dtype=int)
            # Generate the mean value approximator
            if do_mean:
                integrators = intitialize_integrators(methods, c_mins, c_maxs, c_bins, rng)
        if k.changed('dim_hilbert'):
            dim_hilbert = k.value('dim_hilbert')
        if k.changed('bins'):
            bins = k.value('bins')
        amp0 = np.pi/2
        amps = [amp0] + [amp0*k.value('amp_ratio') for i in range(num_controls)] # /k.value('bins')

        # Generate random states
        v_in = dq.rand_state(dim_hilbert, how_many_states, rng=rng)
        v_ex = np.empty((dim_hilbert, how_many_states), dtype=complex)
        v_ap = np.empty((dim_hilbert, how_many_states), dtype=complex)
        U_ex = np.empty((dim_hilbert, dim_hilbert), dtype=complex)
        for r in k.tqdm(range(reps)):     
            # Generate the random Hamiltonian
            #H_s = dq.Random_parametric_Hamiltonian_Haar(num_controls, dim_hilbert, amps, rng)
            H_s = dq.Random_parametric_Hamiltonian_gauss(num_controls, dim_hilbert, sigmas=amps, rng=rng)
            # Approximate calculation
            for i_method, (name, method_) in enumerate(methods.items()):   # Calculate max infidelities, average infidelities, and time
                method = method_['method']
                m_times = method_['m_times']
                for m, curr_m_times in enumerate(m_times):
                    exact = Hamiltonian_System(H_s)      
                    if 'UI' in name:
                        factor = int(bins * curr_m_times)                  
                        curr_c = np.ones(num_controls) / 2 / factor 
                        approx = method(H_s, c_mins, c_maxs, c_bins*factor) 
                    elif 'Expm' in name: # Expm Multiply
                        factor = 1
                        approx = method(H_s) 
                        curr_c = np.ones(num_controls)
                        # Change the number of steps for the approximation
                        approx.set_m_star(curr_m_times)
                    elif 'Trotter': # Trotteresque
                        factor = 1
                        curr_c = np.ones(num_controls)
                        approx = method(H_s, n_times=curr_m_times)
                    ## Mean Infidelities ###################################################################################################
                    if do_mean:
                        if not 'Expm' in name:
                            mean_infidelity_values[i_method][m, r, :] = integrators[i_method].InfidelityFit2Samples_vector(lambda c, U_ex: exact.expmH(c/factor, U_ex), lambda c, vin, vout: approx.expmH(c/factor, vin, vout), U_ex, v_in, v_ex, v_ap, how_many_samples=how_many_samples)
                        else: # Sample Random points
                            mean_infidelity_values[i_method][m, r, :] = integrators[i_method].Sample_Infidelities_vector(lambda c, U_ex: exact.expmH(c/factor, U_ex), lambda c, vin, vout: approx.expmH(c/factor, vin, vout), U_ex, v_in, v_ex, v_ap, how_many_samples=how_many_samples)
                    ## Times ###############################################################################################################
                    # use timeit to time the function approx.expmH with the arguments c1_2
                    if r == 0 and do_times:
                        times[i_method][m] = timeit_autorange_vector(approx, dim_hilbert, curr_c, v_in, v_ap )
            if r == 0 and do_times:
                times[-1] = timeit_autorange_vector(exact, dim_hilbert, curr_c, v_in, v_ap, pass_U=True)
        # Save values onto *args = [max_I_mean, max_I_str, mean_I_mean, mean_I_std, times]
        len_args = len(args)
        if do_times:
            len_args -= 1
        ind = 0
        i_method = 0
        while ind < len_args: # last one for exact times
            if do_max:
                args[ind][i,:], args[ind+1][i,:], args[ind+2][i,:] = mean_std_asym(max_infidelity_values[i_method], axis=1) #max_I_mean, max_I_std
                ind += 3
            if do_mean:
                args[ind][i,:], args[ind+1][i,:], args[ind+2][i,:]  = mean_std_asym(mean_infidelity_values[i_method], axis=(1,2)) #mean_I_mean, mean_I_std
                ind += 3
            if do_times:
                pass
                args[ind][i,:] = times[i_method] #time
                ind += 1
            i_method += 1
        if do_times:
            args[-1][i] = times[-1]
            pass
    return args

