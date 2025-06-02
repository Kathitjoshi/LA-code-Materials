# The Adaptive Bridge Safety System

# Your coastal city now uses a smart sensor grid to monitor multiple bridges 
# with varying numbers of support joints. Each bridge's stability is governed by Ax = b, 
# where A is an n*n force distribution matrix and b is the dynamic load vector. 
# Sensors sometimes scramble equation orders (hint: organization matters) and safety 
# thresholds vary by bridge age. 

# Write a program to:
# Factorize A accounting for potential sensor errors
# Solve for force vector x

# Input:
# The first line contains an integer n, the size of the matrix A (number of joints).
# The next n lines contain n space-separated floating-point numbers, representing the rows of matrix A.
# The last line contains n space-separated floating-point numbers, representing the vector b.

# Output:
# Print the P matrix, rounded to 2 decimal places.
# Print the L matrix, rounded to 2 decimal places.
# Print the U matrix, rounded to 2 decimal places.
# Print the solution vector x, rounded to 2 decimal places.

# Sample Test Case 1
# Input:

# 3
# 1 2 3
# 4 5 6
# 7 8 9
# 10 11 12

# Expected Output:

# P (3x3):
#  [[0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
# L (3x3):
#  [[1.   0.   0.  ]
#  [0.14 1.   0.  ]
#  [0.57 0.5  1.  ]]
# U (3x3):
#  [[ 7.    8.    9.  ]
#  [ 0.    0.86  1.71]
#  [ 0.    0.   -0.  ]]
# Force magnitudes: [-25.33  41.67 -16.  ]

# Sample Test Case 2
# Input:

# 4
# 2  -1   0   0
# -1  2  -1   0
# 0  -1   2  -1
# 0   0  -1   2
# 1  0  1  0

# Expected Output:

# P (4x4):
#  [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]
# L (4x4):
#  [[ 1.    0.    0.    0.  ]
#  [-0.5   1.    0.    0.  ]
#  [ 0.   -0.67  1.    0.  ]
#  [ 0.    0.   -0.75  1.  ]]
# U (4x4):
#  [[ 2.   -1.    0.    0.  ]
#  [ 0.    1.5  -1.    0.  ]
#  [ 0.    0.    1.33 -1.  ]
#  [ 0.    0.    0.    1.25]]
# Force magnitudes: [1.2 1.4 1.6 0.8]


# Note: You should only write your logic in decompose_and_solve method and return the required answers. 
# Proper printing format will be taken care of by the boilerplate. 
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, norm


def decompose_and_solve(A, b):
    P, L, U = lu(A)
    lu_facs = lu_factor(A)
    x = lu_solve(lu_facs, b)
    return P, L, U, x


# Boilerplate (do not modify)
def main():
    n = int(input())
    A = []
    for _ in range(n):
        row = list(map(float, input().split()))
        A.append(row)
    b = list(map(float, input().split()))

    A = np.array(A)
    b = np.array(b)

    P, L, U, x = decompose_and_solve(A, b)

    print(f"P ({n}x{n}):\n", np.round(P, 2))
    print(f"L ({n}x{n}):\n", np.round(L, 2))
    print(f"U ({n}x{n}):\n", np.round(U, 2))
    print("Force magnitudes:", np.round(x, 2))


if __name__ == "__main__":
    main()
