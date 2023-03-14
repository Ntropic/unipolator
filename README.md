# unipolator
Unitary Interpolation, for the calculation of propagators of time dependent quantum systems. Speeds up your propagators. 

## Install 
  Via `pip install unipolator`

## Import  
  Via `from unipolator import *`

## How To:

Generate an array of Hamiltonians `H_s[0,...] = H_0`, `H_s[i,...] = H_i` for a system with Hamiltonian $H(t) = H_0 + \sum_{i=1} c_i(t) H_i$.
Initialize unitary interpolation 
```
ui = UI(H_s, c_mins, c_maxs, c_bins)  
```
Calculate unitary by passing a complex numpy array `U_ui` and coefficients `c` to, a return argument is not needed, the inputed array is simply modified
```
ui.expmH( c, U_ui)
``` 
Similarly, pass a 3d array `dU`and calculate the derivatives as well.
```
ui.dexpmH( c, U, dU)
``` 
A 2d array allows the calculation of a complete pulse via
```
ui.expmH_pulse(cs, U)
``` 
Finally the GRAPE method is supported via (pass an array `dI_dj.shape(n, cs.shape[0])` to store the gradients)
```
ui.grape(cs, U_target, target_indexes, U, dU, dI_dj)
```


 ## Author: 
 Michael Schilling
