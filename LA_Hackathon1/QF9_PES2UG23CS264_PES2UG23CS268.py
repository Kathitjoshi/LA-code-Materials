# Conceptual Question:

# Title: Kernel and Range of a Linear Transformation

# Description:
# In linear algebra, the kernel (or null space) of a linear transformation 
# T is the set of all vectors that map to the zero vector, and the range (or image) is the set of all vectors that can be expressed as T(v) for some input vector v. Your task is to determine the kernel and range of a given linear transformation represented by a 2x2 matrix A.

# Input:
# -- 2x2 matrix A
#    For example, A = [1 2
# 		             2 4]


# -- The input will consist of two lines, each containing two space-separated integers or floating-point numbers representing the rows of the matrix.

# Output:
# -- Basis for the kernel of A.
# -- Basis for the range of A.

# The output will consist of:

# -- The basis for the kernel (if any), printed as space-separated values.

# -- The basis for the range, printed as space-separated values, with each basis vector on a new line.

# Test Cases 1:

# Input:
# 1 2
# 2 4

# Output:
# -2.0 1
# 1.0 2.0


# Test Case 2:
# Input:
# 1 1
# 1 1

# Output:
# -1.0 1.0
# 1.0 1.0

# Test Case 3:
# Input:
# 1 0
# 0 1

# Output:
# 1.0 0.0
# 0.0 1.0




import numpy as np
from sympy import Matrix

def compute_kernel_and_range(A):
    A = np.array(A, dtype=float)
    sympy_matrix = Matrix(A)
    
    # Compute the kernel (null space)
    kernel = sympy_matrix.nullspace()
    kernel_basis = [list(vec) for vec in kernel]
    
    # Compute the range (column space)
    col_space = sympy_matrix.columnspace()
    range_basis = [list(vec) for vec in col_space]
    
    return kernel_basis, range_basis

def main():
    # Read input
    # First row of the matrix
    row1 = list(map(float, input().strip().split()))
    # Second row of the matrix
    row2 = list(map(float, input().strip().split()))
    
    # Create the matrix A
    A = np.array([row1, row2])
    
    kernel_basis, range_basis = compute_kernel_and_range(A)
    
    # Print kernel basis
    for vec in kernel_basis:
        print(" ".join(map(str, vec)))
    
    # Print range basis
    for vec in range_basis:
        print(" ".join(map(str, vec)))

if __name__ == "__main__":
    main()