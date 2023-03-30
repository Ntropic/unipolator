# -*- coding: utf-8 -*-
import numpy as np
from scipy import integrate

# A class that generates points from which to fit coefficients of basis functions 
# with which to represent the sqrt of the infidelity function for a control parameter space and calculate the average infidelity on a volume of those parameters
class I_mean_Gen():
    def __init__(self, x0, dx0, limit_fun, basis_fun, min_order=2, max_order=3, opts={'epsrel': 1e-10, 'epsabs': 1e-10}, rng=None):
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
        if len(limit_fun(x0)) != self.dim:
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

        # Construct basis functions
        # First construct combinations of orders of the basis functions
        self.combinations = self.construct_combinations()
        # Then construct the evaluation points
        self.n_points = self.combinations.shape[0]
        self.points = random_points_in_volume(self.n_points, x0, dx0, limit_fun, rng)
        self.relative_points = self.relative_location()

        # Construct basis function values at the evaluation points self.points (relative_points)
        self.coeff2amplitude = self.basis_fun(self.combinations, self.relative_points)
        self.amplitude2coeff = self.fit_amplitude2coeff()   # Construct fit matrix to convert amplitudes to coefficients

        # integrate the infidelity for product of 2 basis functions over the volume to construct the average infidelity matrix
        self.integral_combinations = self.integral_coefficient_combinations()

    # Functions to return variables
    def get_relative_points(self):
        return self.relative_points
    def get_points(self):
        return self.points
    def get_combinations(self):
        return self.combinations
    def get_coeff2amplitude(self):
        return self.coeff2amplitude
    def get_amplitude2coeff(self):
        return self.amplitude2coeff
    def get_integral_combinations(self):
        return self.integral_combinations

    # Functions to calculate the average infidelity
    def Mean_Average_Infidelity(self, U_fun, approx_U_fun):
        # U_fun is a function that takes a vector of coefficients and returns the exact unitary matrix
        # approx_U_fun is a function that takes a vector of coefficients and returns the approximated unitary matrix
        # we then calculate and return the mean of the average infidelity integrated over the integration volume (and divided by it's volume)
        
        # Evaluate integrator_fun, at the evaluation points
        I = np.empty(self.n_points)
        for i in range(self.n_points):
            point = self.points[i,:]
            I[i] = Av_Infidelity(U_fun(point), approx_U_fun(point))
        # Calculate sqrt of infidelities
        sqrt_I = np.sqrt(I)
        # Fit the coefficients of the basis functions of the sqrt(Infidelities)
        fit_coeff = self.amplitude2coeff @ sqrt_I
        # Calculate the average infidelity
        I_mean = np.sum(self.integral_combinations @ fit_coeff)
        return I_mean, I, fit_coeff

    def Av_Fidelity(U_ex, U_ap):
        # Substate Average Fidelity with diagonal elements not covered by indexes
        s = U_ex.shape[0]
        B = dot(Dagger(U_ex), U_ap)
        s = sU
        F_av = 1 / (s * (s + 1)) * (trace(dot(B, Dagger(B))) + abs(trace(B)) ** 2)
        F_av = abs(F_av)
        return F_av
    def Av_Infidelity(U_ex, U_ap):
        return abs(1 - Av_Fidelity(U_ex, U_ap, indexes))

    # Initialisation functions
    def relative_location(self):
        # Calculates the relative position in an interpolation volume
        # x is the position (vector)
        # x0 is the start of the interval (vector)
        # dx0 is the length of the interval (vector)
        # returns a vector with the relative position in the interpolation volume
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
                combinations = array([[sum_of_rows]], dtype=int)
            else:
                combinations = empty((0, n), dtype=int)
                for i in range(0, sum_of_rows+1):
                    new_combinations = construct_combinations_of_order(sum_of_rows-i, n-1)
                    new_columns = np.concatenate((ones((new_combinations.shape[0], 1), dtype=int)*i, new_combinations), axis=1)
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

    def random_points_in_volume(self):
        # returns n_points random points in the volume defined by x0 and dx0
        n_points = self.n_points
        x0 = self.x0
        dx0 = self.dx0
        limits = self.limit_fun
        rng= self.rng
        
        all_points_done = 0
        dim = len(x0)
        #ratio = 1/2**dim # ratio of points to be generated in each dimension - volume ratio
        points = []

        # Function that checks if a point is in the volume
        def in_interval(x, x_min, x_max): #check which elements in x are between their corresponding elements in x_min and x_max
            return np.logical_and(x >= x_min, a <= x_max)
        def in_interval_fun(x, lim_fun): 
            is_in_interval = np.empty(len(x), dtype=bool)
            x_min = np.empty(len(x), dtype=float)
            x_max = np.empty(len(x), dtype=float)
            for i in range(len(x)):
                curr_interval = lim_fun[i](x)
                x_min[i] = curr_interval[0]
                x_max[i] = curr_interval[1]
                if x_min[i] > x_max[i]:
                    x_min[i], x_max[i] = x_max[i], x_min[i]
            return in_interval(x, x_min, x_max)

        while not all_points_done:
            new_points = rng.rand(n_points - len(points), dim) * dx0 + x0 
            for i in range(new_points.shape[0]):
                if in_interval_fun(new_points[i,:], limits):
                    points.append(new_points[i,:])
            all_points_done = len(points) >= n_points
        return np.random.rand(n_points, len(x0)) * dx0 + x0

    def fit_amplitudes2coefficients(self):
        # Fit the amplitudes of the basis functions to the coefficients of the polynomial
        # returns a matrix, so that the amplitudes at positions (x_s used in the construction of coeff2amplitude) are trasformed 
        # into the coefficients of the polynomial
        # Using linear regression:
        coeff2amplitude = self.coeff2amplitude
        amplitude2coeff = np.linalg.inv(coeff2amplitude @ coeff2amplitude.T) @ coeff2amplitude
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

        def square_fun(combination1, combination2, x_s):
            return fun(combination1, x_s) * fun(combination2, x_s)
        # Define integration over volume specified by limits
        def integrate_over_volume(fun, x0, dx0):
            return integrate.nquad(fun, lim_fun_array, opts=opts)[0]
        
        # Determine size of integration volume, for averaging values of integrals
        vol = integrate_over_volume(lambda x_s: 1, x0, dx0)

        # Compute integrals
        n = combinations.shape[0]
        integral_combinations = np.zeros((n,n))
        for i in range(n):
            c1 = combinations[i,:]
            for j in range(i,n):
                c2 = combinations[j,:]
                integral_combinations[i,j] = integrate_over_volume(lambda x_s: square_fun(c1, c2, x_s), x0, dx0)/vol
                integral_combinations[j,i] = integral_combinations[i,j]
        return integral_combinations


class I_mean_UI(I_mean_Gen):
    # Specialized class for calculating the mean average infidelities of Unitary Interpolations
    def __init__(self, x0, dx0, limit_fun, min_order=2, max_order=3, opts={'epsrel': 1e-10, 'epsabs': 1e-10}, rng=None):
        super().__init__(x0, dx0, limits_ui, evaluate_ui_polynomial_basis, min_order, max_order, opts, rng)

    def limits_ui(n_dim):    
        # Construct limits of an integration interval for the unitary interpolation - in relative coordinates
        limits = []
        def max_other(x, j):
            y = np.delete(x, j)
            return np.max(y)
        for j in range(n_dim):
            limits.append(lambda *vals: [0, 1-max_other(vals, j)])
        return limits
    def evaluate_ui_polynomial_basis(combinations, x_s): # relative coordinates
        coeff2amplitude = empty((x_s.shape[0], combinations.shape[0]))
        combinations_1 = (combinations > 0).astype(int)
        x_s_1 = (1 - x_s)
        for i in range(combinations.shape[0]): # All the coefficients
            p = combinations[i,:]
            p_1 = combinations_1[i,:]
            coeff2amplitude[:,i] = np.prod(x_s_1**p_1*x_s**p, axis=1)
        return coeff2amplitude

class I_mean_trotter(I_mean_Gen):
    # Specialized class for calculating the mean average infidelities of Trotterizations
    def __init__(self, x0, dx0, limit_fun, min_order=1, max_order=3, opts={'epsrel': 1e-10, 'epsabs': 1e-10}, rng=None):
        super().__init__(x0, dx0, limits_ui, evaluate_trotter_polynomial_basis, min_order, max_order, opts, rng)

    def limits_trotter(n_dim):    
        # Construct limits of an integration interval for the unitary interpolation - in relative coordinates
        limits = []
        for j in range(n_dim):
            limits.append(lambda *vals: [0, 1])
        return limits
    def evaluate_trotter_polynomial_basis(combinations, x_s): # relative coordinates
        coeff2amplitude = empty((x_s.shape[0], combinations.shape[0]))
        combinations_1 = (combinations > 0).astype(int)
        x_s_1 = (1 - x_s)
        for i in range(combinations.shape[0]): # All the coefficients
            p = combinations[i,:]
            coeff2amplitude[:,i] = np.prod(x_s**p, axis=1)
        return coeff2amplitude