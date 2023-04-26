import numpy as np
from scipy import integrate
from Analysis_Code.analytical_integration_vector_functions import *


# A class that generates points from which to fit coefficients of basis functions 
# with which to represent the sqrt of the infidelity function for a control parameter space and calculate the average infidelity on a volume of those parameters
class I_mean_Gen():
    def __init__(self, x0, dx0, limit_fun, edge_points, basis_fun, min_order=2, max_order=3, one_zero=False, which_orders=None, repeats=10, point_ratio=1.0, rng=None, add_points=None, integrator=None, do_std=True):
        # Initiate the object, generate evaluation points on the parameter space defined by x0, dx0 and limit_fun
        # x0 is the origin, dx0 the size in every direction and limit_fun specifies whether a vector in the cube is also within the parameter space
        # rng is the random number generator for the random evaluation points
        # Create basis functions of orders between min_order and max_order, using a basis function generator basis_fun
        # Save input variables
        self.x0 = x0
        self.dim = len(x0)
        self.dx0 = dx0
        # Check for consistency of input
        if len(dx0) != self.dim:
            raise Exception('dx0 must have the same length as x0')
        if len(limit_fun) != self.dim:
            raise Exception('limit_fun must return a vector of the same length as x0')
        self.limit_fun = limit_fun
        self.basis_fun = basis_fun
        if max_order < min_order:
            print('Warning: max_order is smaller than min_order, switching them')
            min_order, max_order = max_order, min_order
        if not isinstance(min_order, int):
            print('Warning: min_order is not an integer, rounding it')
            min_order = int(min_order)
        if not isinstance(max_order, int):
            print('Warning: max_order is not an integer, rounding it')
            max_order = int(max_order)
        self.min_order = min_order
        self.max_order = max_order
        if rng is None:
            print('Warning: rng is None, using np.random.default_rng(). This will remove reproducibility of your results.')
            rng = np.random.default_rng()
        self.rng = rng
        self.edge_points = edge_points
        self.rel_edge_points = self.relative_location(edge_points)
        self.repeats = repeats
        self.integrator = integrator
        self.do_std = do_std
        
        # Construct basis functions
        # First construct combinations of orders of the basis functions
        self.combinations = self.construct_combinations()
        # there are multiplicities in the combinations! create the relevant unique combinations of combinations
        self.combinations_a, self.combinations_b = self.find_unique_basis_functions(self.combinations, one_zero=one_zero, which_orders=which_orders)
        # integrate the infidelity for product of 2 basis functions over the volume to construct the average infidelity matrix
        self.integral_combinations = self.integral_coefficient_combinations()
        
        # Then construct the evaluation points
        self.n_points = self.combinations_a.shape[0]
        if not add_points is None:
            self.n_points += add_points
        elif point_ratio > 1.0:
            self.n_points = int(self.n_points * point_ratio)
        
        self.points = self.random_points_in_volume()
        self.relative_points = self.relative_location()

        # Construct basis function values at the evaluation points self.points (relative_points)
        self.coeff2amplitude_a = self.basis_fun(self.combinations_a, self.relative_points)
        self.coeff2amplitude_b = self.basis_fun(self.combinations_b, self.relative_points)
        self.coeff2amplitude = self.coeff2amplitude_a * self.coeff2amplitude_b
        self.amplitude2coeff = self.fit_amplitudes2coefficients()   # Construct fit matrix to convert amplitudes to coefficients
        # Remove the last components of the self.amplitude2coeff fit matrix, as those points are already known to be zero
        if len(self.edge_points) > 1:
            self.amplitude2coeff = self.amplitude2coeff[:, :-(self.edge_points.shape[0]-1)]
            self.points = self.points[:-(self.edge_points.shape[0]-1),:]
            self.n_points = self.n_points - self.edge_points.shape[0] + 1
        # Make array for the results of the fits
        self.fit_coeff = np.empty(self.n_points)    # Fit coefficients of the basis functions to the sqrt of the infidelity
        self.n_eval_points = 0
        self.prepared_std = False
        
    # Functions to return variables by name of variable
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError(name + ' is not an attribute of this class')

    #### Useful side calculations
    def Dag(self, U):
        return U.conj().T
    def Av_Fidelity(self, U_ex, U_ap):
        # Substate Average Fidelity with diagonal elements not covered by indexes
        s = U_ex.shape[0]
        B = self.Dag(U_ex) @ U_ap
        F_av = 1 / (s * (s + 1)) * (np.trace(B @ self.Dag(B)) + np.abs(np.trace(B)) ** 2)
        F_av = np.abs(F_av)
        return F_av
    def Av_Infidelity(self, U_ex, U_ap):
        return np.abs(1 - self.Av_Fidelity(U_ex, U_ap))

    #### Main functions
    def Sample_at_Points(self, points, U_fun, approx_U_fun, U_ex, U_ap, passU2fun = True):
        n_points = points.shape[0]
        I = np.empty(n_points)
        for i in range(n_points):
            point = points[i,:]
            if passU2fun:
                approx_U_fun(point, U_ap)
                U_fun(point, U_ex)
                I[i] = self.Av_Infidelity(U_ex, U_ap)
            else:
                I[i] = self.Av_Infidelity(U_fun(point), approx_U_fun(point))
        return I

    # Functions to calculate the average infidelity
    def Mean_Average_Infidelity(self, U_fun, approx_U_fun, U_ex, U_ap, passU2fun=True):
        # U_fun is a function that takes a vector of coefficients and returns the exact unitary matrix
        # approx_U_fun is a function that takes a vector of coefficients and returns the approximated unitary matrix
        # we then calculate and return the mean of the average infidelity integrated over the integration volume (and divided by it's volume)
        # passU2fun specifies whether U_fun and approx_U_fun return a unitary matrix (False) or the unitary is passed as a second argument
        # Evaluate integrator_fun, at the evaluation points
        I = self.Sample_at_Points(self.points, U_fun, approx_U_fun, U_ex, U_ap, passU2fun)
        # Using linear regression --> Fit the coefficients of the basis functions of the sqrt(Infidelities)
        self.fit_coeff = self.amplitude2coeff @ I
        # Calculate the average infidelity
        I_mean = np.sum( self.integral_combinations @ self.fit_coeff)
        do_std = self.do_std
        # Calculate STD?
        if do_std:
            if not self.prepared_std: # Prepare STD
                self.integral_combinations_std = self.integral_coefficient_combinations_std()
            I_std = np.sqrt( self.fit_coeff.T @ self.integral_combinations_std @ self.fit_coeff )
            return I_mean, I_std, I 
        else:
            return I_mean, I
    
    def eval_basis_term(self, combination, x_s):
        # Evaluate the basis function term combination at the point x_s
        x_s_rel = self.relative_location(x_s)
        return self.basis_fun(combination, x_s_rel)

    def eval_fit_at_points(self, x_s):
        # Evaluate the fit at the points x_s
        # x_s is a 2D array of shape (n_points, dim)
        # First evaluate the basis functions at the points x_s
        x_s_rel = self.relative_location(x_s)
        basis_a = self.basis_fun(self.combinations_a, x_s_rel)
        basis_b = self.basis_fun(self.combinations_b, x_s_rel)
        basis = basis_a * basis_b
        # Then calculate the fit
        fit = basis @ self.fit_coeff
        return fit
    
    def loc2rel_loc(self, x_s):
        # Convert the location x_s to the relative location x_s_rel
        return self.relative_location(x_s)
    
    def is_in_list(self, array, list_of_arrays):
        # array is an array
        # list_of_arrays is a list of arrays
        # returns True if array is in list_of_arrays, False otherwise
        for i in range(len(list_of_arrays)):
            if np.array_equal(array, list_of_arrays[i]):
                return True
        return False

    def find_unique_basis_functions(self, combinations, one_zero=False, which_orders=None):
        # Check which combinations of 2 basis functionx c_1 * c_2 are unique!   
        # if one_zero, then also the number of zeros in the combination has to be the same
        # if which_orders is not None, then only the combinations of the specified orders are considered
        if isinstance(which_orders, int):
            which_orders = [i for i in range(which_orders+1)]
        unique_combinations_indexes = []
        unique_combinations_vectors = [] # stores vectors associated with a unique combination
        n = combinations.shape[0]
        dim = combinations.shape[1]
        if one_zero:
            # append the zeros zo combinations
            combinations_new = np.zeros((combinations.shape[0], combinations.shape[1]*2), dtype=int)
            for i in range(n):
                c = combinations[i]
                c_zero = (c == 0).astype(int)
                combinations_new[i] = np.concatenate((c, c_zero), axis = 0)
        else:
            combinations_new = combinations.copy()
        do_combination = True
        for i in range(n):
            c = combinations_new[i]
            c_old = combinations[i]
            for j in range(i, n):
                d = combinations_new[j]
                d_old = combinations[j]
                cpd = c + d 
                if one_zero:
                    cpd[dim:] = cpd[dim:]
                # check if cpd is already in the unique_combinations_vectors list
                if which_orders is not None:
                    sum_cpd = np.sum(cpd[:dim])
                    if self.is_in_list(sum_cpd, which_orders):
                        do_combination = True
                    else:
                        do_combination = False
                if do_combination and not self.is_in_list(cpd, unique_combinations_vectors):
                    unique_combinations_indexes.append([i,j])
                    unique_combinations_vectors.append(cpd)
        # create an array of combinations
        unique_combinations_a = np.empty((len(unique_combinations_indexes), dim), dtype=int)
        unique_combinations_b = np.empty((len(unique_combinations_indexes), dim), dtype=int)
        for ind, [i,j] in enumerate(unique_combinations_indexes):
            unique_combinations_a[ind] = combinations[i]
            unique_combinations_b[ind] = combinations[j]
        #unique_combinations_indexes = np.array(unique_combinations_indexes)
        return unique_combinations_a, unique_combinations_b #, unique_combinations_indexes

    # Initialisation functions
    def relative_location(self, x=None):
        # Calculates the relative position in an interpolation volume
        # x is the position (vector)
        # x0 is the start of the interval (vector)
        # dx0 is the length of the interval (vector)
        # returns a vector with the relative position in the interpolation volume
        if x is None:
            x = self.points
        x0 = self.x0
        dx0 = self.dx0
        return (x - x0)/dx0

    def construct_combinations(self):
        # a function that constructs an array of combinations, so that the sum of the elements of each row is between lowest_degree and highest_degree
        # the number of rows is equal to the number of combinations
        # the number of columns is equal to n
        highest_degree = self.max_order
        lowest_degree = self.min_order
        n = self.dim
        def construct_combinations_of_order(sum_of_rows=2, n=1):
            # sum_of_rows is the sum of the elements of each row
            # n is the number of columns
            # returns an array of combinations
            if n < 1:
                print('Exception: n must be larger than 0')
                n = 1
            if sum_of_rows < 0:
                print('Exception: sum_of_rows must be larger than or equal to 0')
                sum_of_rows = 0

            if n == 1:
                combinations = np.array([[sum_of_rows]], dtype=int)
            else:
                combinations = np.empty((0, n), dtype=int)
                for i in range(0, sum_of_rows+1):
                    new_combinations = construct_combinations_of_order(sum_of_rows-i, n-1)
                    new_columns = np.concatenate((np.ones((new_combinations.shape[0], 1), dtype=int)*i, new_combinations), axis=1)
                    combinations = np.concatenate((combinations, new_columns), axis=0)
            return combinations
        if lowest_degree > highest_degree:
            lowest_degree, highest_degree = highest_degree, lowest_degree
        for i in range(lowest_degree, highest_degree+1):
            if i == lowest_degree:  
                new_combinations = construct_combinations_of_order(i, n)
                combinations = new_combinations
            else:
                new_combinations = construct_combinations_of_order(i, n)
                combinations = np.concatenate((combinations, new_combinations), axis=0)
        return combinations

    def random_points_in_volume(self, repeats=None, n_points=None, ratio_out=False):
        # returns n_points random points in the volume defined by x0 and dx0
        x0 = self.x0
        dx0 = self.dx0
        limits = self.limit_fun
        rng= self.rng
        if repeats is None:
            repeats = self.repeats
        if n_points is None:
            n_points = self.n_points
        n_points = n_points - self.edge_points.shape[0] + 1   # the points are already 0
        all_points_done = 0
        dim = len(x0)
        #ratio = 1/2**dim # ratio of points to be generated in each dimension - volume ratio
        points = np.empty((n_points, dim), dtype=float)
        # Function that checks if a point is in the volume
        def in_interval(x, x_min, x_max): #check which elements in x are between their corresponding elements in x_min and x_max
            return np.logical_and(x >= x_min, x <= x_max)
        def in_interval_fun(x, lim_fun): 
            if len(x.shape) == 1:
                x = x.reshape(1, x.shape[0])
            x_len = x.shape[1]
            x_min = np.empty(x_len, dtype=float)
            x_max = np.empty(x_len, dtype=float)
            for i in range(x_len):
                curr_interval = lim_fun[i](*list(x[0,:i]))
                x_min[i] = curr_interval[0]
                x_max[i] = curr_interval[1]
                if x_min[i] > x_max[i]:
                    x_min[i], x_max[i] = x_max[i], x_min[i]
            return in_interval(x, x_min, x_max)
        ind = 0
        attempts = 0
        if repeats == 1:  # Don't need to calculate distances -> unnecessary overhead
            while ind < n_points:
                new_point = rng.random((1, dim)) * dx0 + x0
                rel_new_point = self.relative_location(new_point)
                attempts += 1
                if in_interval_fun(rel_new_point, limits).all():
                    points[ind,:] = new_point
                    ind += 1
        else:
            for ind in range(n_points):
                curr_points = np.empty((repeats, dim)) 
                min_distances = np.empty(repeats)
                for i in range(repeats):
                    while True:
                        new_point = rng.random((1, dim)) * dx0 + x0
                        rel_new_point = self.relative_location(new_point)
                        attempts += 1
                        if in_interval_fun(rel_new_point, limits).all():
                            curr_points[i,:] = new_point
                            points_array = np.concatenate((points[:ind], self.rel_edge_points), axis=0)
                            min_distances[i] = np.min(np.linalg.norm(points_array - new_point, axis=1))
                            break
                # Choose the best repeat
                ind_best = np.argmax(min_distances)
                points[ind,:] = curr_points[ind_best,:]
        ratio = n_points/attempts
        # append the edge points (except for the first one) to the points
        points = np.concatenate((points, self.edge_points[1:]), axis=0)
        if ratio_out:
            return points, ratio
        else:
            return points
    
    def add_evaluation_points(self, how_many_points=100):
        # Adds points to evaluate the quality of the fit
        self.n_eval_points = how_many_points
        self.eval_points = self.random_points_in_volume(n_points=self.n_eval_points)
        self.eval_rel_points = self.relative_location(self.eval_points)

    def compare_fit_at_evaluation_points(self, U_fun, approx_U_fun, U_ex, U_ap, passU2fun = True, how_many_points=100):
        if self.n_eval_points == 0:
            self.add_evaluation_points(how_many_points)
        # Compares the fit at the evaluation points
        # returns the mean squared error, and its standard deviation
        I_fit = self.eval_fit_at_points(self.eval_points)
        I_method = self.Sample_at_Points(self.eval_points, U_fun, approx_U_fun, U_ex, U_ap) 
        # compare I_method and I_fit
        return np.mean(np.abs(I_fit - I_method)), np.std(np.abs(I_fit - I_method))

    def distances_in_volume(self, points):
        # calculates the distances
        # minimum distance between point and list of points
        edge_points = self.edge_points
        def min_distance(point, point_array):
            return np.min(np.linalg.norm(point_array - point, axis=1))
        # calculate for every point in points the minimum distance to the other points
        min_distances = np.empty(len(points))
        for i in range(len(points)):
            # construct other points from points and edge points
            other_points = np.concatenate((points[:i,:], points[i+1:,:], edge_points), axis=0)
            curr_min = min_distance(points[i], other_points)
            min_distances[i] = curr_min
        return min_distances

    def fit_amplitudes2coefficients(self):
        # Fit the amplitudes of the basis functions to the coefficients of the polynomial
        # returns a matrix, so that the amplitudes at positions (x_s used in the construction of coeff2amplitude) are trasformed 
        # into the coefficients of the polynomial
        # Using linear regression:
        coeff2amplitude = self.coeff2amplitude
        amplitude2coeff = np.linalg.inv(coeff2amplitude.T @ coeff2amplitude) @ coeff2amplitude.T
        return amplitude2coeff

    def integral_coefficient_combinations(self):
        # A function that computes the integrals of the products of basis functions over an interpolation volume
        # fun creates the value of a basis function using a combination from the combinations array(to define the basis functio) and x_s (to define the position)
        # x0 and dx0 define the interpolation volume over which the integration is performed using the nquad function of scipy.integrate
        # Define product of basis functions
        integrator = self.integrator
        combinations_a = self.combinations_a
        combinations_b = self.combinations_b
        # Compute integrals
        n = combinations_a.shape[0]
        integral_combinations = np.zeros(n)
        for i in range(n):
            c1 = combinations_a[i,:]
            c2 = combinations_b[i,:]    
            polynomials = combinate_polynomial_vectors(c1, c2)
            integral_combinations[i] = integrator(polynomials) 
        return integral_combinations

    def integral_coefficient_combinations_std(self):
        # Calcualate the combinations of the integral terms, to enable the calculation of standard deviations
        integrator = self.integrator
        combinations_a = self.combinations_a
        combinations_b = self.combinations_b
        n = combinations_a.shape[0]
        n_poly = combinations_a.shape[1]
        integral_combinations_std = np.zeros((n,n))
        for i in range(n):
            ic1 = combinations_a[i,:]
            ic2 = combinations_b[i,:]
            mean_i = self.integral_combinations[i]
            i_polynomials = combinate_polynomial_vectors(ic1, ic2)
            i_polynomials[0][0] = i_polynomials[0][0] - mean_i # remove mean_i from the polynomial
            for j in range(i,n):     
                jc1 = combinations_a[j,:]
                jc2 = combinations_b[j,:]    
                mean_j = self.integral_combinations[i]
                j_polynomials = combinate_polynomial_vectors(jc1, jc2)
                j_polynomials[0][0] = j_polynomials[0][0] - mean_j # remove mean_j from the polynomial
                # multiply the polynomials
                ij_polynomials = [multiply_polynomial_vectors(i_polynomials[k], j_polynomials[k]) for k in range(n_poly)]
                integral_combinations_std[i,j] = integrator(ij_polynomials)
                integral_combinations_std[j,i] = integral_combinations_std[i,j]
        return integral_combinations_std
                
###############################################################################################################################################################

class I_mean_UI(I_mean_Gen):
    # Specialized class for calculating the mean average infidelities of Unitary Interpolations
    def __init__(self, c_mins, c_maxs, c_bins, min_order=1, max_order=3, which_orders=None, repeats=4, point_ratio=1.0, rng=None, add_points=None, do_std=True):
        x0 = c_mins.copy()
        dx0 = (c_maxs - c_mins)/c_bins.astype(float)
        x0[0] += dx0[0]
        dx0[0] = -dx0[0]
        n_dim = len(x0)
        edge_points = np.empty((n_dim+1, n_dim), dtype=float)
        edge_points[0] = x0
        for i in range(n_dim):
            edge_points[i+1] = x0.copy()
            edge_points[i+1,i] += dx0[i]
        super().__init__(x0, dx0, self.limits_ui(n_dim), edge_points, self.evaluate_ui_polynomial_basis, min_order, max_order, False, which_orders, repeats, point_ratio, rng, add_points, integrator=integrate_ui_polynomial, do_std=do_std)
    def limits_ui(self, n_dim):    
        # Construct limits of an integration interval for the unitary interpolation - in relative coordinates
        limits = []
        def max_(x):
            if len(x) == 0:
                return 0.0
            else:
                return np.max(x)
        for j in range(n_dim):
            limits.append(lambda *vals: [0, 1-max_(vals)])
        return limits
    def evaluate_ui_polynomial_basis(self, combinations, x_s): # relative coordinates
        if not isinstance(x_s, np.ndarray):
            x_s = np.array([x_s])
        if len(x_s.shape) == 1:
            x_s = x_s.reshape(1, x_s.shape[0])
        if len(combinations.shape) == 1:
            combinations = combinations.reshape(1, combinations.shape[0])
        coeff2amplitude = np.empty((x_s.shape[0], combinations.shape[0]))
        #combinations_1 = (combinations > 0).astype(int)
        #x_s_1 = (1 - x_s)
        for i in range(combinations.shape[0]): # All the coefficients
            p = combinations[i,:]
            #p_1 = combinations_1[i,:]
            coeff2amplitude[:,i] = np.prod(x_s**p, axis=1) #x_s_1**p_1*
        return coeff2amplitude


class I_mean_trotter(I_mean_Gen):
    # Specialized class for calculating the mean average infidelities of Trotterizations
    def __init__(self, x0, dx0, min_order=1, max_order=3, which_orders=None, repeats=4, point_ratio=1.0, rng=None, add_points=None, do_std=True):
        n_dim = len(x0)
        edge_points = np.empty((1, n_dim), dtype=float)
        edge_points[0] = x0
        super().__init__(x0, dx0, self.limits_trotter(n_dim), edge_points, self.evaluate_trotter_polynomial_basis, min_order, max_order, False, which_orders, repeats, point_ratio, rng, add_points, integrator=integrate_trotter_polynomial, do_std=do_std)
    def limits_trotter(self, n_dim):    
        # Construct limits of an integration interval for the unitary interpolation - in relative coordinates
        limits = []
        for j in range(n_dim):
            limits.append(lambda *vals: [0, 1])
        return limits
    def evaluate_trotter_polynomial_basis(self, combinations, x_s): # relative coordinates
        # Evaluate the polynomial basis of the Trotter coefficients
        if not isinstance(x_s, np.ndarray):
            x_s = np.array([x_s])
        if len(x_s.shape) == 1:
            x_s = x_s.reshape(1, x_s.shape[0])
        if len(combinations.shape) == 1:
            combinations = combinations.reshape(1, combinations.shape[0])   
        coeff2amplitude = np.empty((x_s.shape[0], combinations.shape[0]))
        for i in range(combinations.shape[0]): # All the coefficients  
            p = combinations[i,:]
            coeff2amplitude[:,i] = np.prod(x_s**p, axis=1)
        return coeff2amplitude
    