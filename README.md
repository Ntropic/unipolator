# unipolator
[![Python package](https://github.com/Ntropic/unipolator/actions/workflows/python-package.yml/badge.svg)](https://github.com/Ntropic/unipolator/actions/workflows/python-package.yml)
![Python Version](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)

Unitary Interpolation, allows for the fast repeated exponentiation of parametric Hamiltonians [[1]](https://arxiv.org/abs/2402.01498). Construct propagators (and their derivatives) of time dependent quantum systems of the form $H(t) = H_0 + \sum_{i=1} c_i(t) H_i$ (for example in optimal control problems) or quantum circuits with parametric gates. We utilize a grid based interpolation scheme to calculate propagators from cached matrix decompositions. The computation of a propagator for a time step is as fast as a single Trotter step, but with the ability to achieve machine precision. 

## Install and Import
  Install via
  ```
  pip install unipolator
  ```
  and then simply import into a project via
  ```
  from unipolator import *
  ```

## Initialize the Unitary Interpolation Object:
Describe a system with a double complex array of Hamiltonians `H_s`, with `H_s[i,...] = H_i ∆t`, so that for $n$ control Hamiltonians `H_s` is a $(n+1) \times d \times d$ array for a $d$ dimensional Hilbert space. Define the bounds of the interpolation (hyper-) volume via $n$ dimensional double arrays `c_mins` and `c_maxs` and define the number of bins for every dimension via the $n$ dimensional int64 array `c_bins`, to initialize the unitary interpolation cache 
```
ui = UI(H_s, c_mins, c_maxs, c_bins)  
```
Equivalently if we wish to propagate only wavevectors $\ket{\psi(t)} = U(t) \ket{\psi(0)}$ we initialize the unitary interpolation cache via
```
ui_vector = UI_vector(H_s, c_mins, c_maxs, c_bins, m)  
```
where `m` is the number of wavevectors that are calculated in parallel.
The package contains further methods listed at the bottom of this document.

## Automatic Binning
The method `UI_bins` automatically calculates the optimal binning for a target infidelity (default `I_tar=1e-10`). Use via
```
bins = UI_bins(H_s, c_mins, c_maxs, I_tar=1e-10)
```
By calling 
```
ui = UI_auto(H_s, c_mins, c_maxs, I_tar=1e-10)
```
or
```
ui_vector = UI_vector_auto(H_s, c_mins, c_maxs, I_tar=1e-10, m)
```
this method is called automatically during the initialization of the unitary interpolation cache.

## Calculate:
We can now use `ui` to calculate matrix exponentials, their derivatives, pulse sequences, and their gradients via the following methods:
1. `expmH` calculates the unitary $U = \exp(-i H(c) \Delta t)$ for a given set of coefficients `c` (double array of length $n$), pass `U_ui` to the method to store the result (this avoids allocating new memory for every call, and allows reusing the same arrays)
    ```
    ui.expmH( c, U_ui)  
    ``` 
    Similarly we pass two $d \times m$ arrays `V_in` and `V_out`, with the $m$ input wavevectors and for the propagated wavevectors, via
    ```
    ui_vector.expmH( c, V_in, V_out)
    ```
2. `dexpmH` also outputs the derivatives of the unitaris (wavevectors) with respect to the control paracters `c`. This requires the additional passing of a $n \times d \times d$ array `dU` to store the derivatives in
    ```
    ui.dexpmH( c, U, dU)
    ``` 
    During the initalization we can also select which dervatives we wish to compute, via the additional argument `which_diffs` which requires an int64 array with the indexes of the control parameters for which we wish to compute the derivatives. 
    
    In the wavevector case, we replace the output variable `dU` with an additional $n \times m \times n$ arrays `dV_out`, so that
    ```
    ui_vector.dexpmH( c, V_in, V_out, dV_out)
    ```
3. `expmH_pulse` calculates the propagator of a piecewise constant pulse for a given set of coefficients `c_s`, now a 2d array of shape $N \times n$, where $N$ is the number of timesteps
    ```
    ui.expmH_pulse(cs, U)
    ``` 
4. `grape` calculates the infidelity of such a pulse with respect to a arget unitary `U_target` (using the indexes `target_indexes` of `U_target`), as well as the gradients of the control parameters along the pulse by using the GRAPE trick. We pass an array `dI_dj` of shape $n \times N$ to store the gradients at every time step for every control parameter
    ```
    ui.grape(cs, U_target, target_indexes, U, dU, dI_dj)
    ```

### Other Methods:
The package also contains classes for eigenvalue based exponentiations, Krylov based exponentiations and (symmetric-)  Trotterisations, namely 
- `Hamiltonian_System(H_s)`, 
- `Hamiltonian_System_vector(H_s, m)`, where `m` is the number of wavevectors that are calculated in parallel, 
- `Trotter_System=(H_s, m_times)`, where `m_times` is the number of doublings $2^\mathrm{m}$ Trotter steps, 
- `Symmetric_Trotter_System=(H_s, m_times)`,
- `Trotter_System_vector(H_s, n_times)` where `n_times` is the number of performed Trotter steps ,
- `Symmetric_Trotter_System_vector(H_s, n_times)`. 

- In the test Subdirectory we provide additional functions to generate Random Hamiltonians, construct infidelities and more. 

## Author: 
Michael Schilling

## References:
[1] Schilling, Michael, et al. Exponentiation of Parametric Hamiltonians via Unitary Interpolation [arxiv.org/abs/2402.01498](https://arxiv.org/abs/2402.01498)