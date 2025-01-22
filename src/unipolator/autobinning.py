import numpy as np
from numpy.linalg import norm, eigh, qr
from scipy.linalg import schur



##### Matrix operations #########################################################
def Dag(U):
    return np.transpose(np.conj(U))

def vecxvec( E, wf, alpha): # Many times faster than multiply
    s = E.size
    for i in range(s):
        wf[i] = np.exp(-1j * E[i] * alpha) * wf[i]
    return wf
# Construct matrix exponentials and products faster than in numpy or scipy
def vm_exp_mul(E, V, dt = 1.0): # expm(diag(vector))*matrix  multiplication via exp(vector)*matrix
    s = E.size
    A = np.empty((s, s), np.complex128)
    for i in range(s):
        A[i,:] = np.exp(-1j * E[i] * dt) * Dag(V[:,i])
    return A
def expmH_from_Eig(E, V, dt = 1.0):
    U = np.dot(V, vm_exp_mul(E, V, dt))
    return U
def expmH(H, dt = 1.0):
    E, V = eigh(H)
    return expmH_from_Eig( E, V, dt)

##### Logarithms 
def unitary_eig(A): # alternative to np.eig returning unitary matrices V
    Emat, V = schur(A, output='complex')
    return np.diag(Emat), V
def vm_log_mul(E, V):
    s = E.size
    A = np.empty((s, s), np.complex128)
    for i in range(s):
        A[i,:] = np.log(E[i])*Dag(V[:,i])
    return A
def logmU(U):
    E, V = unitary_eig(U)
    return np.dot(V, vm_log_mul(E, V))

def logmU_parts(U):
    E, V = unitary_eig(U)
    return -np.log(E).imag, V

##### Scripts to create test interpolation #######################################
def H_from_c(H_s, c):
    H = H_s[0].copy()
    for i in range(len(c)):
        H = H + c[i]*H_s[i+1]
    return H

# Get unitary from weights c and H_s
def expm_H_s(H_s, c):
    H = H_from_c(H_s, c)
    E, V = eigh(H)
    return expmH_from_Eig( E, V )

# Construct the terms for corners of an interpolation (hyper-) cube
def make_border_unitaries(H_s, c_mins, c_maxs):
    if isinstance(c_mins, (list)):
        c_mins = np.array(c_mins, dtype=np.float64)
    if isinstance(c_maxs, (list)):
        c_maxs = np.array(c_maxs, dtype=np.float64)
    n = len(c_mins)
    s = H_s.shape[-1]
    U0 = expm_H_s(H_s, c_mins)
    Ui = np.empty((n, s, s), dtype=np.complex128)
    for i in range(n):
        curr_c = c_mins.copy()
        curr_c[i] = c_maxs[i]
        Ui[i] = expm_H_s(H_s, curr_c)
    n_ij = n * (n-1) // 2
    Uij_2 = np.empty((n_ij, s, s), dtype=np.complex128)
    Ui_2 = np.empty((n, s, s), dtype=np.complex128)
    ind = 0
    dc = (c_maxs - c_mins) 
    for i in range(n):
        curr_c = c_mins.copy()
        curr_c[i] += dc[i]/2
        Ui_2[i] = expm_H_s(H_s, curr_c)
        for j in range(i+1, n):
            curr_c = c_mins.copy()
            curr_c[i] += dc[i]/2
            curr_c[j] += dc[j]/2
            Uij_2[ind] = expm_H_s(H_s, curr_c)
            ind += 1
    curr_c = c_mins + dc/2
    U_2 = expm_H_s(H_s, curr_c)
    return U0, Ui, Ui_2, Uij_2, U_2

# Get the unitary interpolation along 
# Construct interpolation terms from U0 to Ui
def make_interpolation_terms(U0, Ui):
    n = Ui.shape[0]
    s = Ui.shape[-1]
    Es = np.empty((n,s), dtype=np.float64)
    Vs = np.empty((n,s,s), dtype=np.complex128)
    for i in range(n):
        U = np.dot(Ui[i], Dag(U0))
        E, V = logmU_parts(U)
        Es[i] = E
        Vs[i] = V
    return Es, Vs

def interpolation_core_term(Vs, Es, index, alpha):
    E = Es[index]
    V = Vs[index]
    return V @ np.diag(np.exp(-1j*E*alpha)) @ Dag(V)

def nd_interpolation_core_term(Vs, Es, U0, alphas, indices):
    U = interpolation_core_term(Vs, Es, indices[0], alphas[0])
    for i in range(1, len(indices)):
        U = np.dot(interpolation_core_term(Vs, Es, indices[i], alphas[i]), U)
    return U @ U0

###### Scripts for caching values for polynomial approximation of Infidelities from Test binning ######################
# test directional accuracy of interpolation
def UI_directional_overlap_traces(U0, Ui_2, Uij_2, U_2, Es, Vs):
    # Test the quality of the interpolation for every combination (i,j) of directions at point x_i = 0.5 = x_j
    # exception 1d: test only center
    # Calculate and return the trace of the overlap operator M = U_exact^dagger @ U_interpolated (-d) , where d is the hilbert space dimension, which is subtracted except for all tr_M except the central element tr_M_2
    n = Es.shape[0]
    d = U0.shape[0] 
    if n == 1:
        alphas = np.array([0.5])
        indices = np.array([0])
        U = nd_interpolation_core_term(Vs, Es, U0, alphas, indices)
        tr_M = np.array([])
        tr_M_single = np.array([np.trace(Dag(Ui_2[0]) @ U) - d])
        tr_M_2 = tr_M_single[0]
        indexes = np.empty((0,0), dtype=np.intp)
    else:
        n_ij = n * (n-1) // 2
        tr_M = np.empty(n_ij, dtype=np.complex128)
        tr_M_single = np.empty(n, dtype=np.complex128)
        indexes = np.empty((n_ij, 2), dtype=np.intp)
        ind = 0
        alphas = np.array([0.5])
        for i in range(n):
            U = nd_interpolation_core_term(Vs, Es, U0, alphas, np.array([i]))
            tr_M_single[i] = np.trace(Dag(Ui_2[i]) @ U) - d
        alphas = np.array([0.5, 0.5])
        for i in range(n):
            for j in range(i+1, n):
                indices = np.array([i,j])
                U = nd_interpolation_core_term(Vs, Es, U0, alphas, indices)
                tr_M[ind] = np.trace(Dag(Uij_2[ind]) @ U) - d - tr_M_single[i] - tr_M_single[j]
                indexes[ind] = np.array([i,j], dtype=np.intp)
                ind += 1
    tr_M = tr_M + tr_M.conj()
    tr_M = tr_M.real.astype(np.float64)
    tr_M_single = tr_M_single + tr_M_single.conj()
    tr_M_single = tr_M_single.real.astype(np.float64)
    # compare central element
    if n > 2:
        alphas = np.array([0.5]*n)
        indices = np.arange(n)
        U = nd_interpolation_core_term(Vs, Es, U0, alphas, indices)
        tr_M_2 = np.trace(Dag(U_2) @ U) - d
        tr_M_2 = tr_M_2 + tr_M_2.conj()
        tr_M_2 = tr_M_2.real.astype(np.float64)
    elif n == 2:
        tr_M_2 = tr_M[0]
    else:
        tr_M_2 = tr_M_single[0]
    pref = -1/(d+1)
    tr_M = pref * tr_M
    tr_M_single = pref * tr_M_single
    tr_M_2 = pref * tr_M_2
    return tr_M_single, tr_M, tr_M_2, indexes

def get_bidirectional_overlap_operators(H_s, c_mins=None, c_maxs=None, bins=None): # Calculate the infidelity parameters for a given interpolation
    n = H_s.shape[0]-1
    if c_mins is None:
        c_mins = np.zeros(n)
    if c_maxs is None:
        c_maxs = np.ones(n)
    if bins is None:
        bins = np.ones(n, dtype=np.intp)
    else:
        bins = bins.astype(np.intp)
    dc = (c_maxs - c_mins)
    c_maxs = c_mins + dc / bins
    U0, Ui, Ui_2, Uij_2, U_2_exact = make_border_unitaries(H_s, c_mins, c_maxs)
    Es, Vs = make_interpolation_terms(U0, Ui)
    tr_M_single, tr_M, tr_M_2, indexes = UI_directional_overlap_traces(U0, Ui_2, Uij_2, U_2_exact, Es, Vs)  # , tr_M_2
    alphas = np.array([0.5]*n)
    indices = np.arange(n)
    U_2_approx = nd_interpolation_core_term(Vs, Es, U0, alphas, indices)
    return tr_M_single, tr_M, tr_M_2, indexes, U_2_exact, U_2_approx

def I_from_tr_M_bins(tr_M_single, tr_M, indexes, n, param=None, bins=None):
    if bins is None:
        bins = np.ones(n, dtype=np.intp)
    if param is None:
        param = np.ones(n)
    param = param / bins
    res = 0.0
    if len(indexes) > 0:
        multipliers_2 = param[indexes[:,0]]**2*param[indexes[:,1]]**2
        res += np.sum(multipliers_2 * tr_M)
    multipliers_4 = param**4
    res +=  np.sum(multipliers_4 * tr_M_single)
    return res

def I_from_tr_M(tr_M_single, tr_M, indexes, n, param=None):
    return I_from_tr_M_bins(tr_M_single, tr_M, indexes, n, param=param, bins=None)

##### Scripts for Optimization ########################################################
def cache_size_by_bins(bins):
    # calculate the number of cache elements for a given binning
    n = bins.shape[0] #len(bins)
    if n > 1:
        n_C = 2**(n-1) * np.prod(bins)
        n_L = bins[0] * np.prod(bins[1:] + 1)
        n_R = bins[-1] * np.prod(bins[:-1] + 1) 
        N_tot = n_C + n_L + n_R
    else:
        N_tot = bins[0]
    return N_tot

def cache_size_change_by_dir(bins, negative=False):
    # check by how much the cache grows for every direction it could change in
    n = bins.shape[0] #len(bins)
    curr_size = cache_size_by_bins(bins)
    differences = np.zeros(len(bins))
    for i in range(n):
        curr_bins = bins.copy()
        if not negative:
            curr_bins[i] += 1
        else:
            curr_bins[i] -= 1
        new_size = cache_size_by_bins(curr_bins)
        differences[i] = new_size - curr_size
    return differences

def infidelity_change_by_dir(tr_M_single, tr_M, indexes, n, bins, negative=False):
    # check by how much the infidelity changes for every direction it could change in
    curr_I = I_from_tr_M_bins(tr_M_single, tr_M, indexes, n, bins=bins)
    differences = np.zeros(len(bins))
    for i in range(n):
        curr_bins = bins.copy()
        if not negative:
            curr_bins[i] += 1
        else:
            curr_bins[i] -= 1
        new_I = I_from_tr_M_bins(tr_M_single, tr_M, indexes, n, bins=curr_bins)
        differences[i] = curr_I - new_I 
    return differences

def _find_solution(tr_M_single, tr_M, indexes, n, bins, I_tar=10**-14):
    # increase to find solution
    curr_I = I_from_tr_M_bins(tr_M_single, tr_M, indexes, n, bins=bins)
    while curr_I > I_tar:
        inf_diff = infidelity_change_by_dir(tr_M_single, tr_M, indexes, n, bins)
        cache_differences = cache_size_change_by_dir(bins)
        curr_prices_of_optimization = inf_diff / cache_differences # infidelity decrease per cache increase
        # any of the new points smaller than the target?
        new_Is = curr_I - inf_diff
        if np.any(new_Is < I_tar):
            # find the one with the smallest increase in cache size
            inds = np.where(new_Is < I_tar)
            min_index = np.argmin(cache_differences[inds])
            min_dir = inds[0][min_index]
            bins[min_dir] += 1
            curr_I = new_Is[min_dir]
        else:
            # find the one with the smallest cache price increase for the largest infidelity decrease
            max_ind = np.argmax(curr_prices_of_optimization)
            bins[max_ind] += 1
            curr_I = new_Is[max_ind]
    return bins, curr_I
def _optimize_solution(tr_M_single, tr_M, indexes, n, bins, I_tar=10**-14):
    # decrease for better solution
    optimizing = True # while the current binnning is changing keep optimizing= True
    curr_I = I_from_tr_M_bins(tr_M_single, tr_M, indexes, n, bins=bins)
    curr_cache = cache_size_by_bins(bins)
    if curr_I > I_tar:
        raise Exception('Initial solution not < I_tar')
    new_Is = np.zeros(n)
    new_caches = np.zeros(n, dtype=np.intp)
    while optimizing:
        # check neighbors I
        for i in range(n):
            curr_bins = bins.copy()
            curr_bins[i] -= 1
            if curr_bins[i] < 1:
                new_Is[i] = 1.0
            else:
                new_Is[i] = I_from_tr_M_bins(tr_M_single, tr_M, indexes, n, bins=curr_bins)
            # cache smaller?
            new_caches[i] = cache_size_by_bins(curr_bins)
        # find I < I_tar among now_Is
        inds = np.where(new_Is < I_tar)[0]
        if len(inds) > 0:
            min_index = np.argmin(new_caches[inds])
            potential_ind = inds[min_index]
            if new_caches[potential_ind] < curr_cache:
                bins[potential_ind] -= 1
                curr_I = new_Is[potential_ind]
                curr_cache = new_caches[potential_ind]
            else:
                optimizing = False
        else:
            optimizing = False
    return bins, curr_I

def separate_indexes(i, indexes, tr_M):
    # get indexes that contain i
    inds_i = np.where(indexes[:,0] == i)[0]
    inds_i = np.concatenate((inds_i, np.where(indexes[:,1] == i)[0]))
    # get indexes that dont contain i, the inds not in inds_i
    inds_not_i = np.setdiff1d(np.arange(len(indexes)), inds_i)
    indexes_not_i = indexes[inds_not_i]
    ind_i = indexes[inds_i]
    # reduce indexes_i to not contain i, make it a one d array
    indexes_i = np.empty(len(ind_i), dtype=np.intp)
    for j in range(len(ind_i)):
        indexes_i[j] = ind_i[j][0] if ind_i[j][0] != i else ind_i[j][1]
    tr_M_not_i = tr_M[inds_not_i]
    tr_M_i = tr_M[inds_i]
    return indexes_i, indexes_not_i, tr_M_i, tr_M_not_i

def separate_all_indexes(indexes, tr_M, n):	
    indexes_i = []
    indexes_not_i = []
    tr_M_i = []
    tr_M_not_i = []
    for i in range(n):
        inds_i, inds_not_i, tr_M_i_, tr_M_not_i_ = separate_indexes(i, indexes, tr_M)
        indexes_i.append(inds_i)
        indexes_not_i.append(inds_not_i)
        tr_M_i.append(tr_M_i_)
        tr_M_not_i.append(tr_M_not_i_)
    return indexes_i, indexes_not_i, tr_M_i, tr_M_not_i

def I_rest_from_tr_M_bins(i, tr_M_single, tr_M_not_i, indexes_not_i, n, bins):
    if bins is None:
        bins = np.ones(n, dtype=np.intp)
    param = 1 / bins
    res = 0.0
    if len(indexes_not_i) > 0:
        multipliers_2 = param[indexes_not_i[:,0]]**2*param[indexes_not_i[:,1]]**2
        res += np.sum(multipliers_2 * tr_M_not_i)
    multipliers_4 = param**4
    res +=  np.sum(multipliers_4[:i] * tr_M_single[:i]) + np.sum(multipliers_4[i+1:] * tr_M_single[i+1:])
    return res

def I_i_from_tr_M_bins(i, tr_M_single, tr_M_i, indexes_i, n, bins):
    if bins is None:
        bins = np.ones(n, dtype=np.intp)
    param = 1 / bins
    res = 0.0
    if len(indexes_i) > 0:
        multipliers_2 = param[indexes_i]**2
        res += np.sum(multipliers_2 * tr_M_i) * param[i]**2
    res +=  param[i]**4 * tr_M_single[i]
    return res

# optimize solution by reducing one bin direction by one and then finding the optimum along the other directions 
def optimum_along_single_direction(i, tr_M_single, tr_M_i, tr_M_not_i, indexes_i, indexes_not_i, n, bins, I_tar=10**-14):
    # rewrite as qudratic expression, to get optimum along one direction
    new_bins = bins.copy()
    success = False
    s = 1/bins**2
    Pii =tr_M_single[i]
    I = I_rest_from_tr_M_bins(i, tr_M_single, tr_M_not_i, indexes_not_i, n, bins) - I_tar
    if I < 0:
        Ci = np.sum(tr_M_i * s[indexes_i])
        # find optimum along s_i -> quadratic expression in s_i
        Ci_Pii = Ci/Pii/2
        sqrt_internal = Ci_Pii**2 - I/Pii
        if sqrt_internal > 0:
            sqrt_term = np.sqrt(sqrt_internal)
            s_i = - Ci_Pii - sqrt_term
            if s_i < 0:
                s_i = - Ci_Pii + sqrt_term
            # find the 
            success = True if s_i > 0 else False
            if success:
                # from s_i to_bins[i]
                bins_i = np.sqrt(1/s_i)
                new_bins = bins.copy()
                new_bins[i] = np.ceil(bins_i).astype(np.intp)
    return new_bins, success

##### The script we need to run to optimize binning
def optimize_binning(tr_M_single, tr_M, indexes, n, bins=None, I_tar=10**-14):
    # optimize binning to get minimum cache for target infidelity
    # while I > I_tar: add bins in a direction that increases infidelity the most per cache size, if more than one direction would achieve the target fidelity 
    if bins is None:
        bins = np.ones(n, dtype=np.intp)
    bins, curr_I = _find_solution(tr_M_single, tr_M, indexes, n, bins, I_tar)
    # check surrounding points for smaller cache size
    bins, curr_I = _optimize_solution(tr_M_single, tr_M, indexes, n, bins, I_tar)

    # separate indexes and tr_M
    if len(indexes) > 0:
        indexes_i, indexes_not_i, tr_M_i, tr_M_not_i = separate_all_indexes(indexes, tr_M, n)
        # for every index i, reduce by one, check if solution is still in target range,
        # if not, find which directions first inner point has lower cache size. if non has lower cache size, stop and go back to previous solution
        last_bins = bins.copy()
        found_lower = True
        curr_min_cache = cache_size_by_bins(bins)
        while found_lower:
            found_lower = False
            for i in range(n):
                bin_red = last_bins.copy()
                if bin_red[i] > 1:
                    bin_red[i] -= 1
                    #I_bin_red = I_from_tr_M_bins(tr_M_single, tr_M, indexes, n, bin_red)
                    #cache_bin_red = cache_size_from_bins(bin_red)
                    # check for every other direction, which point leads to I < I_tar, and save the cache needed
                    for j in range(n):
                        if j == i:
                            continue
                        else:
                            bin_j, success = optimum_along_single_direction(j, tr_M_single, tr_M_i[j], tr_M_not_i[j], indexes_i[j], indexes_not_i[j], n, bin_red, I_tar)
                            if success:
                                I_bin_j = I_from_tr_M_bins(tr_M_single, tr_M, indexes, n, bins=bin_j)
                                if not I_bin_j < I_tar:
                                    raise ValueError('I_bin_j='+str(I_bin_j)+' should be smaller than I_tar, algorithm is broken. (' + str(bin_red) + '->' + str(bin_j) + ')')
                                cache_bin_j = cache_size_by_bins(bin_j)
                                if cache_bin_j < curr_min_cache:
                                    curr_min_cache = cache_bin_j
                                    bins = bin_j
                                    found_lower = True
            if found_lower:
                last_bins = bins.copy()
        bins = last_bins
    return bins, curr_I


def optimal_binning(H_s, c_mins, c_maxs, I_tar):
    n = H_s.shape[0] 
    n -= 1
    bins = np.ones(n, dtype=np.intp)
    tr_M_single, tr_M, tr_M_2, indexes, U_exact, U_approx = get_bidirectional_overlap_operators(H_s, c_mins, c_maxs, bins=bins)
    opt_bins, curr_I = optimize_binning(tr_M_single, tr_M, indexes, n, bins=bins, I_tar=I_tar)
    return opt_bins