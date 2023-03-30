from numpy import zeros, ones, array, empty, random
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from tqdm import tqdm

from Average_Infidelity_Integrator import *
import discrete_quantum as dq
from kronbinations import *
from unipolator import *
from timeit import *
import os

def line(title='', left=5, right=-1, line_style='—'):
    if right < 0:
        try:
            width, _ = os.get_terminal_size()
        except:
            terminal_size = shutil.get_terminal_size()
            width = terminal_size.columns
        right = width - left
    if len(title):
        title = ' '+title+' '
        len_title = len(title)
        right = right-len_title
    if right < 0:
        right = 0
    print(line_style*left+title+line_style*right,end='\n')
def dline(title='', left=5, right=-1):
    line(title, left, right, line_style='═')
def tline(title='', left=5, right=-1):
    line(title, left, right, line_style='▬')

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

def first_smaller(array, value):  # find the index of the first value in the array that is smaller than the value and the value at that index
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

# get partial minima of an array
def index_partial_minimum(array):
    n = len(array)
    ind_partial_minimum = np.zeros(n,dtype=int)
    for i in range(n):
        ind_partial_minimum[i] = np.argmin(array[:i+1])
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
            parameters[parameter['name']] = parameter['variation_vals']
            axis_labels.append(parameter['label'])
            axis_types.append(parameter['variation_type'])
        else:
            parameters[parameter['name']] = [parameter['def_val']]
    return parameters, axis_labels, axis_types
    
def create_variables(k, method_dict, to_do=['mean', 'max', 'times']):
    variable_dicts = {}
    type_names = []
    if 'mean' in to_do:
        type_names += ['mean_I_mean', 'mean_I_std']
    if 'max' in to_do:
        type_names += ['max_I_mean', 'max_I_std']
    if 'times' in to_do:
        type_names += ['times']
    for method_name, method in method_dict.items():
        variable_dicts[method_name] = {type_name: k.empty(name=method_name+'_'+type_name) for type_name in type_names}
    if 'times' in to_do:
        variable_dicts['times'] = k.empty(name='times')
    return variable_dicts, method_dict

# Calculate the average and maximum infidelities between exact and approximate unitaries as a function 
# of system size, of number of control parameters, and the amplitudes of the Hamiltonians
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

def Infidelities_and_Times(k, methods, to_calculate, *args, rng):#
    reps = 25
    # Inititalize intermediate arrays, tpo store results of reps, to average over
    max_infidelity_values, mean_infidelity_values, times = initialize_rep_arrays(to_calculate, reps, methods)
    do_times, do_max, do_mean = 'times' in to_calculate, 'max' in to_calculate, 'mean' in to_calculate
    for i, v, c in k.kronprod(do_index=True, do_change=True):
        if k.changed('num_controls'):
            num_controls = k.value('num_controls')
            c_mins, c_maxs, c_bins = np.zeros(num_controls), np.ones(num_controls), np.ones(num_controls, dtype=int)
            # Generate the mean value approximator
            if 'mean' in to_calculate:
                integrators = intitialize_integrators(methods, c_mins, c_maxs, c_bins, rng)
        if k.changed('dim_hilbert'):
            dim_hilbert = k.value('dim_hilbert')
            curr_U_ex, U_ex2, U_ap = [np.empty((dim_hilbert, dim_hilbert), dtype=complex) for i in range(3)]
        amp0 = np.pi*4
        amps = [amp0] + [amp0*k.value('amp_ratio') for i in range(num_controls)] # /k.value('bins')
        m_times = k.value('m_times')

        for r in range(reps):     
            # Generate the random Hamiltonian
            H_s = dq.Random_parametric_Hamiltonian_Haar(num_controls, dim_hilbert, amps, rng)
            # Approximate calculation
            for i_method, (name, method_) in enumerate(methods.items()):   # Calculate max infidelities, average infidelities, and time
                method = method_['method']
                if 'UI' in name:
                    H_s_ui = H_s.copy() 
                    H_s_ui[1:] = H_s_ui[1:] /k.value('bins') / 2**m_times
                    exact = Hamiltonian_System(H_s_ui)                        
                    curr_c = np.ones(num_controls) / 2 
                    approx = method(H_s_ui, c_mins, c_maxs, c_bins)  
                else: # Trotteresque
                    exact = Hamiltonian_System(H_s)
                    curr_c = np.ones(num_controls)
                    approx = method(H_s, m_times=m_times)
                ## Max Infidelities ####################################################################################################
                if do_max:
                    exact.expmH(curr_c, curr_U_ex)
                    approx.expmH(curr_c, U_ap)
                    max_infidelity_values[i_method][r] = dq.Av_Infidelity(curr_U_ex, U_ap)  # Max infidelity of approximation
                ## Mean Infidelities ###################################################################################################
                if do_mean:
                    mean_infidelity_values[i_method][r], x = integrators[i_method].Mean_Average_Infidelity(exact.expmH, approx.expmH, U_ex2, U_ap)
                ## Times ###############################################################################################################
                # use timeit to time the function approx.expmH with the arguments c1_2
                if r == 0 and 'times' in to_calculate:
                    times[i_method] = timeit_autorange(lambda: approx.expmH(curr_c, U_ap))
            if r == 0 and do_times:
                times[-1] = timeit_autorange(lambda: exact.expmH(curr_c, U_ex2))
        # Save values onto *args = [max_I_mean, max_I_str, mean_I_mean, mean_I_std, times]
        len_args = len(args)
        if do_times:
            len_args -= 1
        ind = 0
        repeats = 0
        while ind < len_args: # last one for exact times
            if do_max:
                args[ind][i], args[ind+1][i] = mean_std(max_infidelity_values[repeats]) #max_I_mean, max_I_std
                ind += 2
            if do_mean:
                args[ind][i], args[ind+1][i] = mean_std(mean_infidelity_values[repeats]) #mean_I_mean, mean_I_std
                ind += 2
            if do_times:
                args[ind][i] = times[repeats] #time
                ind += 1
            repeats += 1
        if do_times:
            args[-1][i] = times[-1]
    return args

redo = False
figsize=(8,8)
width = 8.5
# Variation by different variables, 
# specify the variables by a dictionary 
num_controls = {'label': 'Number of controls', 'name': 'num_controls', 'def_val': 2, 'variation_vals': np.arange(1, 6), 'variation_type': 'int'}
hilbert_dim = {'label':'Hilbert space dimension', 'name': 'dim_hilbert', 'def_val': 32, 'variation_vals': 2**np.arange(1, 10), 'variation_type': 'log_int'}
control_amplitudes = {'label': 'Control amplitude ratio ($|H_i|>0 / |H_0|$)', 'name': 'amp_ratio', 'def_val': 0.01, 'variation_vals': np.logspace(-3, 0, 61), 'variation_type': 'log'} # ([H_i>0| / |H_0|)
bins = {'label': 'Number of bins', 'name': 'bins', 'def_val': 1, 'variation_vals': np.arange(1,17), 'variation_type': 'log_int'}
m_times = {'label': 'Number of Trotter steps', 'name': 'm_times', 'def_val': 0, 'variation_vals': np.arange(0, 11), 'variation_type': 'log'}
all_parameter_dicts = [num_controls, hilbert_dim, control_amplitudes, bins, m_times] 
# Methods to use 
method_dict = {}
method_dict['UI'] = {'method': UI}
method_dict['Sym UI'] = {'method': Sym_UI}
method_dict['Trotter'] = {'method': Trotter_System}
method_dict['Sym Trotter'] = {'method': Sym_Trotter_System}

import_statements = ['Average_Infidelity_Integrator', 'kronbinations', 'unipolator', ['discrete_quantum', 'dq'], ['numpy', 'np'], 'from numba import jit', 'import timeit']
other_func = [timeit_autorange, initialize_rep_arrays, intitialize_integrators, mean_std]
combinations = ['amp_ratio', 'num_controls', 'dim_hilbert', 'bins']  
to_do = ['max', 'mean', 'times'] 
combinations = ['amp_ratio', 'dim_hilbert']
for i, combination in enumerate(combinations):
    parameters, axis_labels, axis_types = get_parameters(combination, all_parameter_dicts)  
    k = JIT_kronbinations(parameters, func=Infidelities_and_Times, other_func=other_func, import_statements=import_statements, other_arguments=[method_dict, to_do], redo=redo, progress=True)
    method_dict_results, method_dict = create_variables(k, method_dict, to_do=to_do)
    k.calculate_all()
    print('Done with '+combination)