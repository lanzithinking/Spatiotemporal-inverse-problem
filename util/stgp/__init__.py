"""
Utility functions
Shiwei Lan @ @ ASU
Created Nov 13, 2019
"""

# mpi4py
try:
    from mpi4py import MPI
except ImportError:
    print('mpi4py not installed! It may run slowly in serial...')
    pass

# # petsc4py
# try:
#     import sys, petsc4py
#     petsc4py.init(sys.argv)
#     from petsc4py import PETSc
# except ImportError:
#     print('petsc4py not installed! It may run slowly in serial...')
#     pass
# 
# # slepc4py
# try:
#     import sys, slepc4py
#     slepc4py.init(sys.argv)
#     from slepc4py import SLEPc
# except ImportError:
#     print('slepc4py not installed! It may run slowly in serial...')
#     pass