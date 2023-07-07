import numpy as np
from numba import jit

# a polynomial is represented as a vector of coefficients in the order of increasing powers [1, x, x^2, ...]
@jit(nopython=True)
def integrate_polynomial(p):
    # integrate a polynomial
    # p is a vector of coefficients
    # return a vector of coefficients of the integral
    dim = len(p)
    
    q = np.zeros(dim+1)
    for i in range(dim):
        q[i+1] = p[i]/(i+1)
    return q

# A function that evaluates a polynomial vector at a point
@jit(nopython=True)
def eval_polynomial(p, x):
    # p is a vector of coefficients
    # x is a point
    # return the value of the polynomial at x
    dim = len(p)
    y = 0
    for i in range(dim):
        y += p[i]*x**i
    return y

# A function that integrates a polynomial vector from a to b
@jit(nopython=True)
def integrate_polynomial_interval(p, a, b):
    # p is a vector of coefficients
    # a, b are the integration limits
    # return the value of the integral
    dim = len(p)
    q = integrate_polynomial(p)
    return eval_polynomial(q, b) - eval_polynomial(q, a)

@jit(nopython=True)
def integrate_polynomial_0_to_x(p):
    # p is a vector of coefficients
    # x is the integration limit
    # return the value of the integral
    q = integrate_polynomial(p)
    # at 0 the integral is q[0]
    q[0] = 0
    return q

############################################################################################################################################################################


# A function that multiplies polynomial vectors
@jit(nopython=True)
def multiply_polynomial_vectors(p1, p2):
    # p1, p2 are vectors of coefficients
    # return the vector of coefficients of the product of the polynomials
    dim1 = len(p1)
    dim2 = len(p2)
    p = np.zeros(dim1+dim2-1)
    # multiply the polynomials elementwise
    for i in range(dim1):
        for j in range(dim2):
            p[i+j] += p1[i]*p2[j]
    return p

@jit(nopython=True)
def binom(n, k):
    # return the binomial coefficient n choose k
    if k > n:
        return 0
    if k == 0:
        return 1
    if k == 1:
        return n
    if k == n:
        return 1
    return binom(n-1, k-1) + binom(n-1, k)
# a function that creates a vector of coefficients of a polynomial of a variable y that corresponds to as input vector of coefficients of a polynomial of a variable x for which 1-y=x has been substituted

@jit(nopython=True)
def substitute_polynomial_1_x(p):
    # p is a vector of coefficients of a polynomial of a variable x
    # use the binoomial theorem to substitute 1-y=x
    # return a vector of coefficients of a polynomial of a variable y
    dim = len(p)
    q = np.zeros(dim)
    for i in range(dim):
        for k in range(i+1):
            q[k] += (-1)**k * p[i] * binom(i, k)
    return q

# Test substitute_polynomial_1_x
#p = np.array([0,1.0,0])
#q = substitute_polynomial_1_x(p)
#print(p)



############################################################################################################################################################################

# a function that constructs all the permutations of n numbers
@jit(nopython=True)
def gen_permutations(n):
    # n is the number of elements
    # return a list of all the permutations of the numbers from 0 to n-1
    if n == 1:
        return [[0]]
    else:
        perms = []
        curr_perms = gen_permutations(n-1)
        # Add n-1 to each permutation at each position from right to left
        for i in range(n):
            j = n-1-i
            for p in curr_perms:
                perms.append(p[:j] + [n-1] + p[j:]) 
        return perms

# Test
# Generate a list of all the permutations of 7 numbers
#perms = gen_permutations(7)
#print(len(perms))
#### Inititalize Vectors #################################################################################################################################################

@jit(nopython=True)
def gen_polynomial_vectors(combinations):
    # combinations is a vector of the orders of polynomials in each dimension
    # The unitary interpolation polynomial is a product of polynomials in each dimension
    # a polynomial for x_i of order n is given by: 1 if n = 0, x_i*(1-x_i)^n if n > 0
    # return a list of vectors of coefficients of polynomials of degree n from a unitary interpolation polynomial
    n = len(combinations)
    # generate a list of polynomials for each dimension
    polynomials = []
    for i, c in enumerate(combinations):
        curr_polynomial = np.zeros(c+2)
        if c == 0:
            curr_polynomial[0] = 1.0
        else:
            curr_polynomial[c] = 1.0
        polynomials.append(curr_polynomial)
    return polynomials

def combinate_polynomial_vectors(combinations, combinationsB):
    n = len(combinations)
    if combinations.dtype != int:
        print('combinations should be an integer vector')
        combinations = combinations.astype(int)
    # generate a list of polynomials for each dimension
    polynomials = gen_polynomial_vectors(combinations) 
    if combinationsB is not None:
        if combinationsB.dtype != int:
            print('combinationsB should be an integer vector')
            combinationsB = combinationsB.astype(int)
        if not len(combinations) == len(combinationsB):
            print('combinations and combinationsB should have the same length')
            return None
        polynomialsB = gen_polynomial_vectors(combinationsB)
        polynomials = [multiply_polynomial_vectors(polynomials[i], polynomialsB[i]) for i in range(n)]
    return polynomials

##### UI ###############################################################################################################################################################

# a function that generates a list of vectors of coefficients of polynomials of degree n from a unitary interpolation polynomial
# taking a combinations vector of the orders of polynomials in each dimension as an input

    # combinations is a vector of the orders of polynomials in each dimension
    # The unitary interpolation polynomial is a product of polynomials in each dimension
    # a polynomial for x_i of order n is given by: 1 if n = 0, x_i*(1-x_i)^n if n > 0
    # return the value of the integral over a ui integration hyper-volume
    
def integrate_ui_polynomial(polynomials):
    n = len(polynomials)
    if n == 1:
        # if there is only one dimension, the integral is just the integral of the polynomial
        return integrate_polynomial_interval(polynomials[0], 0, 1)
    else:
        # The limits ate given by x_i > x_j for all j =/= i, with x_i in [0, 1] and x_j in [0, min(x_i, 1-x_i)]
        # the integral is given by the sum of all integrals with x_i > x_j
        # Each integral can be split into x_i in [0, 1/2] and x_i in [1/2, 1] and the integrals are computed separately
        polynomial_integrals = []
        polynomial_integrals_1_x = []
        for i in range(n):
            curr_integral = integrate_polynomial_0_to_x(polynomials[i])
            polynomial_integrals.append(curr_integral)
            polynomial_integrals_1_x.append(substitute_polynomial_1_x(curr_integral))
        # Now add up the cases for every i (with x_i being the maximum of x)
        integral = 0
        for i in range(n):
            curr_polynomials_0_to_1_2 = polynomials[i].copy()
            curr_polynomials_1_2_to_1 = polynomials[i].copy()
            # integrate all j =/= i
            for j in range(n):
                if not i == j:
                    # integrate all other polynomials
                    curr_polynomials_0_to_1_2 = multiply_polynomial_vectors(curr_polynomials_0_to_1_2, polynomial_integrals[j])
                    curr_polynomials_1_2_to_1 = multiply_polynomial_vectors(curr_polynomials_1_2_to_1, polynomial_integrals_1_x[j])
                # integrate the polynomial for x_i
            curr_integral = integrate_polynomial_interval(curr_polynomials_0_to_1_2, 0, 0.5) + integrate_polynomial_interval(curr_polynomials_1_2_to_1, 0.5, 1) # add thw two integrals
            integral += curr_integral
        vol = 1.0/2.0**(n - 1)
        integral = integral/vol
        return integral


##### Trotter ###############################################################################################################################################################

def integrate_trotter_polynomial(polynomials):
    # combinations is a vector of the orders of polynomials in each dimension
    n = len(polynomials)
    integral = 1.0
    for i in range(n):
        # integrate the i'th polynomial from 0 to 1
        integral *= integrate_polynomial_interval(polynomials[i],0.0,1.0)
    return integral