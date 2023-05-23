from scipy.optimize import least_squares
import numpy as np


def fit_function(func, xdata, ydata, p0, bounds=None, minima=None):
    # Define residual function to be minimized
    def residual(params):
        return ydata - func(xdata, *params)

    # Check if any local minima/peaks are provided
    if minima is not None:
        for m in minima:
            # Shift parameter values to local minimum/peak
            p = least_squares(residual, p, bounds=bounds).x

    # Fit to data using least squares method
    fitted_params = least_squares(residual, p, bounds=bounds).x

    return fitted_params


# define the function to fit
def my_func(x, a, b, c):
    return a*np.sin(b*x) + c

# generate some data to fit to
xdata = np.linspace(0, 2*np.pi, 50)
ydata = my_func(xdata, 1, 2, 0.5) + 0.1*np.random.randn(50)

# specify the initial guess for the parameters
p0 = [1.5, 3, 0.3]

# specify the local minima of the function
minima = [1, 4]

# fit the function to the data
result = fit_function(my_func, xdata, ydata, p0, minima)

print(result)




