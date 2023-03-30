import numpy as np
from scipy import integrate
from tqdm import tqdm
from analytical_integration_vector_functions import *


# A class that generates points from which to fit coefficients of basis functions 
# with which to represent the sqrt of the infidelity function for a control parameter space and calculate the average infidelity on a volume of those parameters
class I_mean_Gen():
    def __init__(self, x0, dx0, limit_fun, edge_points, basis_fun, min_order=2, max_order=3, opts={'epsrel': 1e-10, 'epsabs': 1e-10}, repeats=10, point_ratio=1.0, rng=None, add_points=None, integrator=None, progress=True):
        # Initiate the object, generate evaluation points on the parameter space defined by x0, dx0 and limit_fun
        # x0 is the origin, dx0 the size in every direction and limit_fun specifies whether a vector in the cube is also within the parameter space
        # rng is the random number generator for the random evaluation points
        # opts specifies the options for the integration of the infidelity function (scipy.integrate.nquad opts)
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
        self.opts = opts
        self.edge_points = edge_points
        self.rel_edge_points = self.relative_location(edge_points)
        self.repeats = repeats
        self.integrator = integrator
        self.progress = progress

        # Construct basis functions
        # First construct combinations of orders of the basis functions
        self.combinations = self.construct_combinations()
        # Then construct the evaluation points
        self.n_points = self.combinations.shape[0]
        if not add_points is None:
            self.n_points += add_points
        elif point_ratio > 1.0:
            self.n_points = int(self.n_points * point_ratio)
        
        self.points = self.random_points_in_volume()
        self.relative_points = self.relative_location()

        # Construct basis function values at the evaluation points self.points (relative_points)
        self.coeff2amplitude = self.basis_fun(self.combinations, self.relative_points)
        self.amplitude2coeff = self.fit_amplitudes2coefficients()   # Construct fit matrix to convert amplitudes to coefficients

        # integrate the infidelity for product of 2 basis functions over the volume to construct the average infidelity matrix
        self.integral_combinations = self.integral_coefficient_combinations()
        self.fit_coeff = np.empty(self.n_points)    # Fit coefficients of the basis functions to the sqrt of the infidelity

    # Functions to return variables by name of variable
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError(name + ' is not an attribute of this class')

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

    # Functions to calculate the average infidelity
    def Mean_Average_Infidelity(self, U_fun, approx_U_fun, U_ex, U_ap, passU2fun = True):
        # U_fun is a function that takes a vector of coefficients and returns the exact unitary matrix
        # approx_U_fun is a function that takes a vector of coefficients and returns the approximated unitary matrix
        # we then calculate and return the mean of the average infidelity integrated over the integration volume (and divided by it's volume)
        # passU2fun specifies whether U_fun and approx_U_fun return a unitary matrix (False) or the unitary is passed as a second argument
        # Evaluate integrator_fun, at the evaluation points
        I = np.empty(self.n_points)
        for i in range(self.n_points):
            point = self.points[i,:]
            if passU2fun:
                approx_U_fun(point, U_ap)
                U_fun(point, U_ex)
                I[i] = self.Av_Infidelity(U_ex, U_ap)
            else:
                I[i] = self.Av_Infidelity(U_fun(point), approx_U_fun(point))
        # Calculate sqrt of infidelities
        sqrt_I = np.sqrt(I)
        # Fit the coefficients of the basis functions of the sqrt(Infidelities)
        self.fit_coeff = self.amplitude2coeff @ sqrt_I
        # Calculate the average infidelity
        I_mean = np.sum( self.fit_coeff.T @ self.integral_combinations @ self.fit_coeff)
        return I_mean, I
    
    def eval_basis_term(self, combination, x_s):
        # Evaluate the basis function term combination at the point x_s
        if isinstance(combination, int):
            combination = self.combinations[combination, :]
        return self.basis_fun(combination, x_s)

    def eval_fit_at_points(self, x_s, squared=False):
        # Evaluate the fit at the points x_s
        # x_s is a 2D array of shape (n_points, dim)
        # First evaluate the basis functions at the points x_s
        basis = self.basis_fun(self.combinations, x_s)
        # Then calculate the fit
        fit = basis @ self.fit_coeff
        if squared:
            fit = fit**2
        return fit

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
        if ratio_out:
            return points, ratio
        else:
            return points

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
        fun = self.basis_fun
        lim_fun_array = self.limit_fun
        combinations = self.combinations
        x0 = self.x0
        dx0 = self.dx0
        opts = self.opts
        dim_shift = 2
        integrator = self.integrator

        if 'dim_shift' in opts:
            dim_shift = opts['dim_shift']
        reduced_opts = opts.copy()
        # remove elements from reduced_opts that are not needed for nquad
        for key in ['dim_shift', 'n_points', 'repeats', 'ratio_out']:
            if key in reduced_opts:
                del reduced_opts[key]

        def square_fun(combination1, combination2, x_s):
            return fun(combination1, x_s) * fun(combination2, x_s)
        # Define integration over volume specified by limits
        def integrate_over_volume(fun):
            return integrate.nquad(fun, lim_fun_array, opts=reduced_opts)[0]
        def sample_over_volume(fun, integral_points):
            # Find n points in the volume and average the values of the function
            integral_values = np.array([fun(*x_s) for x_s in integral_points])
            return np.mean(integral_values)

        
        # Determine size of integration volume, for averaging values of integrals
        if integrator == None:
            if self.dim <= dim_shift:
                vol = integrate_over_volume(lambda *x_s: 1)
            else: 
                n_points = 1000            
                repeats_opt = 1
                if 'n_points' in opts:
                    n_points = opts['n_points']
                if 'repeats' in opts:
                    repeats_opt = opts['repeats']
                integral_points, ratio = self.random_points_in_volume(repeats=repeats_opt, n_points=n_points, ratio_out=True)

        # Compute integrals
        n = combinations.shape[0]
        integral_combinations = np.zeros((n,n))
        if integrator == None:
            for i in range(n):
                c1 = combinations[i,:]
                for j in range(i,n):
                    c2 = combinations[j,:]
                    if self.dim <= dim_shift:
                        integral_combinations[i,j] = integrate_over_volume(lambda *x_s: square_fun(c1, c2, x_s))/vol
                    else:
                        integral_combinations[i,j] = sample_over_volume(lambda *x_s: square_fun(c1, c2, x_s), integral_points) * ratio
        else:
            if self.progress:
                for i in tqdm(range(n), desc='Calculating integrals'):
                    c1 = combinations[i,:]
                    for j in range(i,n):
                        c2 = combinations[j,:]        
                        integral_combinations[i,j] = integrator(c1,c2)
                        integral_combinations[j,i] = integral_combinations[i,j]
            else:
                for i in range(n):
                    c1 = combinations[i,:]
                    for j in range(i,n):
                        c2 = combinations[j,:]        
                        integral_combinations[i,j] = integrator(c1,c2)
                        integral_combinations[j,i] = integral_combinations[i,j]
        return integral_combinations

class I_mean_UI(I_mean_Gen):
    # Specialized class for calculating the mean average infidelities of Unitary Interpolations
    def __init__(self, c_mins, c_maxs, c_bins, min_order=2, max_order=3, opts={'epsrel': 1e-10, 'epsabs': 1e-10}, repeats=4, point_ratio=1.0, rng=None, add_points=None, progress=True):
        x0 = c_mins.copy()
        dx0 = (c_maxs - c_mins)/c_bins.astype(float)
        x0[0] += dx0[0]
        dx0[0] = -dx0[0]
        n_dim = len(x0)
        edge_points = np.empty((n_dim+1, n_dim), dtype=float)
        edge_points[0] = x0
        vol = 1.0/2.0**(n_dim - 1)
        for i in range(n_dim):
            edge_points[i+1] = x0.copy()
            edge_points[i+1,i] += dx0[i]
        super().__init__(x0, dx0, self.limits_ui(n_dim), edge_points, self.evaluate_ui_polynomial_basis, min_order, max_order, opts, repeats, point_ratio, rng, add_points, integrator=lambda c1,c2: integrate_ui_polynomial(c1,c2)/vol, progress=progress)

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
        combinations_1 = (combinations > 0).astype(int)
        x_s_1 = (1 - x_s)
        for i in range(combinations.shape[0]): # All the coefficients
            p = combinations[i,:]
            p_1 = combinations_1[i,:]
            coeff2amplitude[:,i] = np.prod(x_s_1**p_1*x_s**p, axis=1)
        return coeff2amplitude


class I_mean_trotter(I_mean_Gen):
    # Specialized class for calculating the mean average infidelities of Trotterizations
    def __init__(self, x0, dx0, min_order=1, max_order=3, opts={'epsrel': 1e-10, 'epsabs': 1e-10}, repeats=4, point_ratio=1.0, rng=None, add_points=None, progress=True):
        n_dim = len(x0)
        edge_points = np.empty((1, n_dim), dtype=float)
        edge_points[0] = x0
        super().__init__(x0, dx0, self.limits_trotter(n_dim), edge_points, self.evaluate_trotter_polynomial_basis, min_order, max_order, opts, repeats, point_ratio, rng, add_points, integrator=integrate_trotter_polynomial, progress=progress)

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
        combinations_1 = (combinations > 0).astype(int)
        x_s_1 = (1 - x_s)
        for i in range(combinations.shape[0]): # All the coefficients  
            p = combinations[i,:]
            coeff2amplitude[:,i] = np.prod(x_s**p, axis=1)
        return coeff2amplitude