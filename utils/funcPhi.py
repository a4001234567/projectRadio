import numpy as np
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
    return np.tanh(.5*(t+mu)) * np.exp(- np.power(t,2) / (4 * mu))

# Define the function f(mu)
def f(mu):
    # Perform the numerical integration
    integral_value, _ = quad(integrand, -np.inf, np.inf, args=(mu,))
    return 1 - (1 / np.sqrt(4 * np.pi * mu)) * integral_value

# Generate an array of mu values
mu_values = np.logspace(-6, 6, 10000)
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

try:
    import torch

    indices = np.argsort(mu_values)
    x_sorted = mu_values[indices]
    f_sorted = f_values[indices]
    _X, unique_indices = np.unique(x_sorted, return_index=True)
    _Y = f_sorted[unique_indices]
    _dX = .5 * (_X[1:] + _X[:-1])
    _dY = (_Y[1:] - _Y[:-1]) / (_X[1:] - _X[:-1])

    inv_indices = np.argsort(f_values)
    x_sorted_inv = f_values[inv_indices]
    f_sorted_inv = mu_values[inv_indices]
    _X_inv, unique_indices_inv = np.unique(x_sorted_inv, return_index=True)
    _Y_inv = f_sorted_inv[unique_indices_inv]
    _dX_inv = .5 * (_X_inv[1:] + _X_inv[:-1])
    _dY_inv = (_Y_inv[1:] - _Y_inv[:-1]) / (_X_inv[1:] - _X_inv[:-1])

    class funcPhiTorch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return torch.tensor(np.interp(np.array(x), _X, _Y))
        @staticmethod
        def backward(ctx, grad_input):
            input_x, = ctx.saved_tensors
            return torch.tensor(np.interp(np.array(input_x), _dX, _dY)) * grad_input

    class funcPhiInvTorch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return torch.tensor(np.interp(np.array(x), _X_inv, _Y_inv))
        @staticmethod
        def backward(ctx, grad_input):
            input_x, = ctx.saved_tensors
            return torch.tensor(np.interp(np.array(input_x), _dX_inv, _dY_inv)) * grad_input

except ImportError:
    class funcPhiTorch:
        raise NotImplementedError("Torch is not available, funcPhiTorch cannot be used.")
    class funcPhiInvTorch:
        raise NotImplementedError("Torch is not available, funcPhiInvTorch cannot be used.")
    

if __name__ == '__main__':
    eps = 1e-5
    xs = np.abs(np.random.randn(5))
    assert np.all(np.abs(funcPhiTorch.apply(torch.tensor(xs)).numpy()-func_phi(xs)) <= eps)
    assert np.all(np.abs(funcPhiInvTorch.apply(torch.tensor(xs)).numpy()-func_phi_inv(xs)) <= eps)
    xs = torch.tensor(xs,dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(funcPhiTorch.apply, (xs,),rtol=1e-2,atol=1e-2)
    xs = torch.rand(5,dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(funcPhiInvTorch.apply, (xs,),rtol=1e-2,atol=1e-2)
    exit()
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
