from kronbinations import *
from unipolator import *

from Analysis_Code.parameter_sweep import *
from Analysis_Code.useful import *
from Analysis_Code.iplots import *
import Analysis_Code.discrete_quantum as dq
from Analysis_Code.Average_Infidelity_Integrator import *

checksum_pre = 'pgi_'
## Calculate the time it takes to get to a target fidelity via kronbinations parameter sweep
redo = False
# Variation by different variables, 
# specify the variables by a dictionary 
num_controls = {'label': 'Number of controls', 'name': 'num_controls', 'def_val': 2, 'variation_vals': np.arange(1, 8), 'variation_type': 'int'}
hilbert_dim = {'label':'Hilbert space dimension', 'name': 'dim_hilbert', 'def_val': 16, 'variation_vals': 2**np.arange(1, 10), 'variation_type': 'log_int'}
control_amplitudes = {'label': 'Control amplitude ratio ($|H_i| / |H_0|$)', 'name': 'amp_ratio', 'def_val': 0.025, 'variation_vals': np.logspace(-3, -1, 61), 'variation_type': 'log'} # ([H_i>0| / |H_0|)
bins = {'label': 'Number of bins', 'name': 'bins', 'def_val': 1, 'variation_vals': np.arange(1,64), 'variation_type': 'log_int'}
all_parameter_dicts = [num_controls, hilbert_dim, control_amplitudes, bins] 
# Methods to use 
m_times_list_ui = [0,1,2,3,4,5,6,7] 
m_times_list    = [0,1,2,3,4,5,6,7,8,9,10]
method_dict = {}
method_dict['UI'] = {'method': UI, 'm_times': m_times_list_ui}
method_dict['Sym UI'] = {'method': Sym_UI, 'm_times': m_times_list_ui}
method_dict['Trotter'] = {'method': Trotter_System, 'm_times': m_times_list}
method_dict['Sym Trotter'] = {'method': Sym_Trotter_System, 'm_times': m_times_list}

import_statements = ['Analysis_Code.Average_Infidelity_Integrator', 'kronbinations', 'unipolator', ['Analysis_Code.discrete_quantum', 'dq'], ['numpy', 'np'], 'from numba import jit', 'import timeit']
other_func = [timeit_autorange, initialize_rep_arrays, intitialize_integrators, mean_std]
combinations = ['amp_ratio']  
to_do = ['max', 'mean', 'times'] 
for i, combination in enumerate(combinations):
    parameters, axis_labels, axis_types = get_parameters(combination, all_parameter_dicts)  
    k = JIT_kronbinations(parameters, func=Binned_Infidelities_and_Times, other_func=other_func, import_statements=import_statements, other_arguments=[method_dict, to_do], redo=redo, checksum_pre=checksum_pre)
    method_dict_results, method_dict = create_variables(k, method_dict, to_do=to_do)
    k.calculate_all()
    line('Done with '+combination)

# Print completion
print('Done with all combinations')