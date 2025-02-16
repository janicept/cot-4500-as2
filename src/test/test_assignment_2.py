#Question 1
import numpy as np

def neville(x_points, y_points, x):
    n = len(x_points)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, 0] = y_points[i]
    
    for j in range(1, n):
        for i in range(n - j):
            Q[i, j] = ((x - x_points[i + j]) * Q[i, j - 1] - (x - x_points[i]) * Q[i + 1, j - 1]) / (x_points[i] - x_points[i + j])
    
    return Q[0, n - 1]

# Change values here
x_points = [3.6, 3.8, 3.9]
y_points = [1.675, 1.436, 1.318]
x = 3.7

result = neville(x_points, y_points, x)
print(result)
print()

#_____________________________
#Question 2
import numpy as np

def NewtonForward():
# Change values here    
    xi = [7.2, 7.4, 7.5, 7.6]  # x values
    fxi = [23.5492, 25.3913, 26.8224, 27.4589]  # f(x) values
    
    lim = len(xi)
    diffs = np.zeros((lim, lim))

    # Fill in the first column of the forward differences table
    for i in range(lim):
        diffs[i][0] = fxi[i]

    # Compute the forward differences
    for i in range(1, lim):
        for j in range(1, i + 1):
            diffs[i][j] = (diffs[i][j - 1] - diffs[i - 1][j - 1]) / (xi[i] - xi[i - j])

    for i in range(1, 4):
        if i < lim:
            print(f"{diffs[i][i]:.7f}")

    # Approximate f(7.3) using the Newton forward interpolation formula
    x = 7.3
    p = diffs[0][0]  # Start with f(x_0) which is f(7.2)
    h = xi[1] - xi[0]  # The step size (h)

    # Compute each term of the interpolation
    term = (x - xi[0]) 
    
    for i in range(1, lim):
        p += term * diffs[i][i]  # Add the current term to the polynomial approximation
        term *= (x - xi[i])  # Update the term for the next factor (x - xi[i])

#______________________________
#Question 3
    print()    
# Print the approximate value of f(7.3)
    print(f"{p:.7f}")

if __name__ == "__main__":
    NewtonForward()
print()
#____________________________
#Question 4
import numpy as np

def hermite_divided_difference(x, y, yp):
    n = len(x)
    
    # Create the table for divided differences
    table = np.zeros((2*n, 2*n - 1))  # We will have 2n rows because of duplicate points

    # Populate the first column (x values)
    for i in range(n):
        table[2*i][0] = x[i]    # x values (repeated points will appear twice)
        table[2*i+1][0] = x[i]  # duplicate x values for repeated points

    # Populate the second column (function values and derivatives)
    for i in range(n):
        table[2*i][1] = y[i]    # f(x) values
        table[2*i+1][1] = y[i]  # f(x) values for the repeated points
        table[2*i+1][2] = yp[i] # f'(x) values for the repeated points
    
    # Compute the divided differences for higher order
    for j in range(2, 2*n - 1):  # j is the degree of divided difference (from 2 upwards)
        for i in range(j - 1, 2 * n):  # i is the starting index
            # Handle the case where x[i] == x[i+1] (the points are duplicated)
            if table[i, j-1] == table[i - 1, j-1] and j < 3:
                table[i][j] = yp[i//2]  # For repeated points, use the derivative
            else:
                table[i][j] = (table[i, j-1] - table[i - 1, j-1]) / (x[i//2] - x[(i - (j - 1))//2])
    
    return table

def print_divided_difference_table(x_data, y_data, yp_data):
    
    # Generate the divided difference table
    table = hermite_divided_difference(x_data, y_data, yp_data)
    
    # Print the table in the required format
    for i in range(len(table)):
        print([f"{table[i][j]:.6f}" for j in range(len(table[i]))])

def hermite_polynomial(x_data, y_data, yp_data, x):
   
    n = len(x_data)
    table = hermite_divided_difference(x_data, y_data, yp_data)

    # Evaluate the Hermite polynomial at point x using the table
    result = table[0][0]  # Initialize with f(x_0)
    product_term = 1.0

    # Loop over all the terms and compute the value of the polynomial
    for i in range(1, 2*n - 1):  # Corrected loop to avoid out of bounds error
        product_term *= (x - x_data[(i-1)//2])  # Update the product term (x - x_i)
        result += table[0][i] * product_term  # Add the contribution of the current term

    return result

# Charge data here
x_values = [3.6, 3.8, 3.9]  # x values
f_values = [1.675, 1.436, 1.318]  # f(x) values
f_prime_values = [-1.195, -1.188, -1.182]  # f'(x) values

# Print the divided difference table

print_divided_difference_table(x_values, f_values, f_prime_values)


#____________________________
#Question 5
print()
import numpy as np

def cubic_spline_interpolation(x, y):
    n = len(x)
    
    # Step 1: Create matrix A (tridiagonal matrix)
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Step 2: Set up the A matrix and b vector
    # Natural spline boundary conditions (second derivative at the endpoints is zero)
    A[0, 0] = 1
    A[n-1, n-1] = 1
    b[0] = 0
    b[n-1] = 0
    
    # Internal equations (second derivative continuity at internal points)
    for i in range(1, n-1):
        A[i, i-1] = x[i] - x[i-1]
        A[i, i] = 2 * (x[i+1] - x[i-1])
        A[i, i+1] = x[i+1] - x[i]
        b[i] = 3 * ((y[i+1] - y[i]) / (x[i+1] - x[i]) - (y[i] - y[i-1]) / (x[i] - x[i-1]))

    # Step 3: Solve the system A * m = b, where m is the second derivatives
    m = np.linalg.solve(A, b)
    
    return A, b, m

# Change data here
x = np.array([2, 5, 8, 10])
y = np.array([3, 5, 7, 9])

# Solve cubic spline interpolation
A, b, m = cubic_spline_interpolation(x, y)

# Output results
print(A)
print(b)
print("[" + " ".join([f"{val:.8f}" if abs(val) > 1e-6 else "0" for val in m]) + "]")
