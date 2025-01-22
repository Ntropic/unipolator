# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
cimport numpy as npc

cpdef Unitary_Grid(double complex[:,:,::1] H, double[::1] c_mins, double[::1] dcs, npc.intp_t[::1] c_bins)

cpdef Create_Interpolation_Cache(double complex[:,:,::1] U_grid, npc.intp_t[::1] grid_cum_prod, npc.intp_t[::1] c_bins)

cpdef Create_Sym_Interpolation_Cache(double complex[:,:,::1] U_grid2, npc.intp_t[::1] grid_cum_prod, npc.intp_t[::1] c_bins)

