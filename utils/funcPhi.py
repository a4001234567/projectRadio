import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from numba import jit

# Define the integrand function
@jit(nopython=True)
def integrand(t, mu):
    '''
    The original integration function would be tanh(t / 2) * exp(-(t - mu)**2 / (4 * mu))
    Here for numeric precision, we shift by mu to make the center of the gaussian at 0
    This is consistent with scipy.integrate.quad which for infinite integration, centering at 0.
    '''
    return np.tanh(.5*(t+mu)) * np.exp(- np.pow(t,2) / (4 * mu))

# Define the function f(mu)
def f(mu):
    # Perform the numerical integration
    integral_value, _ = quad(integrand, -np.inf, np.inf, args=(mu,))
    return 1 - (1 / np.sqrt(4 * np.pi * mu)) * integral_value

# Generate an array of mu values
mu_values = np.logspace(-4, 4, 100)
f_values = np.clip(np.array([f(mu) for mu in mu_values]),a_min=0,a_max=np.inf)

def generate_func_from_points(x_values,fx_values):
    indices = np.argsort(x_values)
    x_sorted = x_values[indices]
    f_sorted = fx_values[indices]
    x_sorted,unique_indices = np.unique(x_sorted,return_index=True)
    f_sorted = f_sorted[unique_indices]
    return lambda x: np.interp(x, x_sorted, f_sorted)

_func_phi = generate_func_from_points(mu_values,f_values)
def func_phi(mu):
    return _func_phi(mu)
_func_phi_inv = generate_func_from_points(f_values,mu_values)
def func_phi_inv(mu):
    return _func_phi_inv(mu)

if __name__ == '__main__':
    #print(f(100000))
    #exit()
    #plt.plot(mu_values, f_values)
    #plt.show()
    xs = np.linspace(0,1,1000)
    plt.figure(figsize=(8, 6))
    plt.plot(xs, func_phi(xs), label=r'$f(\mu)$', color='b')
    plt.plot(xs, func_phi_inv(xs), label=r'$f^{-1}(\mu)$', color='r')
    plt.xlabel(r'$\mu$', fontsize=14)
    plt.ylabel(r'$f(\mu)$', fontsize=14)
    plt.title(r'Plot of $f(\mu)$', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()
