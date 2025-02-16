This repository contains Python code for solving various interpolation problems, as well as implementing numerical methods to estimate values and model data. Below is a breakdown of the methods included in this repository:

Question 1: Neville's Method
This function implements **Neville's Method** for polynomial interpolation. Given a set of x and y values, the function estimates the value of y at a given x.

Example:
x_points = [3.6, 3.8, 3.9]
y_points = [1.675, 1.436, 1.318]
x = 3.7
result = neville(x_points, y_points, x)
print(result)

This will return an estimated value for `x = 3.7` based on the provided data.

Question 2: Newton Forward Method
This function performs **Newton Forward Interpolation** for estimating the value of `f(x)` given data points. The example below estimates `f(7.3)`.

Example:
xi = [7.2, 7.4, 7.5, 7.6]  
fxi = [23.5492, 25.3913, 26.8224, 27.4589]
NewtonForward()

Question 3: Hermite Polynomial Interpolation
This method implements **Hermite Interpolation**, which works for cases where you not only have function values but also derivatives at some points. You can use this method when you have both `f(x)` and `f'(x)` values.

Example:
x_values = [3.6, 3.8, 3.9]
f_values = [1.675, 1.436, 1.318]
f_prime_values = [-1.195, -1.188, -1.182]
print_divided_difference_table(x_values, f_values, f_prime_values)

Question 4: Cubic Spline Interpolation
This method implements **Cubic Spline Interpolation**, which provides a smooth curve through the given data points by solving for second derivatives.

Example:
x = np.array([2, 5, 8, 10])
y = np.array([3, 5, 7, 9])
A, b, m = cubic_spline_interpolation(x, y)

How to Run the Code

1. Clone the repository:
   bash
   git clone https://github.com/your_username/repository_name.git
   
2. Install dependencies (if you haven't already):
   bash
   pip install numpy
   
3. Run the script:
   bash
   python script_name.py
   
Expected Output
The code outputs the results of interpolation for the given data points, such as:
- The estimated value using Neville's method.
- Forward differences and interpolated values using Newton's method.
- The divided difference table and Hermite polynomial results.
- The cubic spline matrix and second derivatives.
