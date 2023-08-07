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
num_controls = {'label': 'Number of controls', 'name': 'num_controls', 'def_val': 2, 'variation_vals': np.arange(1, 3), 'variation_type': 'int'}
hilbert_dim = {'label':'Hilbert space dimension', 'name': 'dim_hilbert', 'def_val': 16, 'variation_vals': 2**np.arange(1, 10), 'variation_type': 'log_int'}
control_amplitudes = {'label': 'Eigenvalue ratio', 'name': 'amp_ratio', 'def_val': 0.025, 'variation_vals': np.logspace(-3, 0, 61), 'variation_type': 'log'} # ([H_i>0| / |H_0|)
bins = {'label': 'Number of bins', 'name': 'bins', 'def_val': 1, 'variation_vals': np.arange(1,64), 'variation_type': 'log_int'}
all_parameter_dicts = [num_controls, hilbert_dim, control_amplitudes, bins] 
# Methods to use 
m_times_list_ui = [1,2,4,8,16,32,64,128] 
m_times_list_multiply    = [i for i in range(25)]
m_times_list_trotter = list(np.round(np.logspace(0,np.log10(1024),16)).astype(int))
m_times_list_sym_trotter = list(np.round(np.logspace(0,np.log10(128),11)).astype(int))
method_dict = {}
method_dict['Expm Multiply'] = {'method': Hamiltonian_System_vector, 'm_times': m_times_list_multiply}
method_dict['UI'] = {'method': UI_vector, 'm_times': m_times_list_ui}
method_dict['Sym UI'] = {'method': Sym_UI_vector, 'm_times': m_times_list_ui}
method_dict['Trotter'] = {'method': Trotter_System_vector, 'm_times': m_times_list_trotter}
method_dict['Sym Trotter'] = {'method': Sym_Trotter_System_vector, 'm_times': m_times_list_sym_trotter}

import_statements = ['Analysis_Code.Average_Infidelity_Integrator', 'kronbinations', 'unipolator', ['Analysis_Code.discrete_quantum', 'dq'], 'from numba import jit', 'import timeit']
other_func = [timeit_autorange_vector, initialize_rep_arrays, intitialize_integrators]
combinations = ['num_controls']  
to_do = ['mean', 'times'] 
for i, combination in enumerate(combinations):
    parameters, axis_labels, axis_types = get_parameters(combination, all_parameter_dicts)  
    k = JIT_kronbinations(parameters, func=Binned_Infidelities_and_Times_vector, other_func=other_func, import_statements=import_statements, other_arguments=[method_dict, to_do], redo=redo, checksum_pre=checksum_pre)
    method_dict_results, method_dict = create_variables(k, method_dict, to_do=to_do)
    print(k.checksum)
    k.calculate_all()
    line('Done with '+combination)

# Print completion
print('Done with all combinations')