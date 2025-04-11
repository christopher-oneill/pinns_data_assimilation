import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

def sawtooth_approximation(x, N):
    """Compute N-term Fourier approximation of sawtooth wave"""
    result = np.ones_like(x) * 0.5  # DC term
    for n in range(1, N+1):
        result -= np.sin(n*x)/(n*np.pi)
    return result

def true_sawtooth(x):
    """Compute true sawtooth wave (normalized to [0,2π])"""
    return (x % (2*np.pi))/(2*np.pi)

def compute_l2_error(N):
    """Compute L2 error for N-term approximation"""
    x = np.linspace(0, 2*np.pi, 1000)
    approx = sawtooth_approximation(x, N)
    true = true_sawtooth(x)
    squared_diff = (approx - true)**2
    return np.sqrt(trapezoid(squared_diff, x)/(2*np.pi))

# Compute errors for different numbers of terms
N_values = np.arange(1, 201)
errors = [compute_l2_error(N) for N in N_values]

# Compute theoretical error curve (C/N)
C = errors[0] * N_values[0]  # Scale to match at N=1
theoretical = C/N_values

# Create figure with two subplots
plt.figure(figsize=(15, 6))

# Plot 1: Show approximations
plt.subplot(1, 2, 1)
x = np.linspace(0, 2*np.pi, 1000)
true = true_sawtooth(x)
plt.plot(x, true, 'k-', label='True', linewidth=2)
for N in [1, 3, 5, 10]:
    approx = sawtooth_approximation(x, N)
    plt.plot(x, approx, '--', label=f'N={N}')
plt.grid(True)
plt.legend()
plt.title('Sawtooth Wave Approximations')
plt.xlabel('x')
plt.ylabel('f(x)')

# Plot 2: Show errors
plt.subplot(1, 2, 2)
plt.loglog(N_values, errors, 'bo-', label='Numerical')
plt.loglog(N_values, theoretical, 'r--', label='Theoretical O(1/N)')
plt.grid(True)
plt.legend()
plt.title('L2 Truncation Error vs N')
plt.xlabel('Number of terms (N)')
plt.ylabel('L2 Error')

plt.tight_layout()
plt.savefig('sawtooth_error.png')
plt.show()

# Print some numerical values
print("\nNumerical Error Values:")
for N in [1, 5, 10, 20, 50]:
    print(f"N={N:2d}: {compute_l2_error(N):.6f}")