import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 3*x - 5

def f_prime(x):
    return 2*x + 3

def g(x, y):
    return x**2 * y + 4*x * math.sin(y)

def dg_dx(x, y):
    return 2*x*y + 4*math.sin(y)

def dg_dy(x, y):
    return x**2 + 4*x*math.cos(y)

def f_prime_fd(x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2*h)

def g_x_fd(x, y, h=1e-6):
    return (g(x + h, y) - g(x - h, y)) / (2*h)

def g_y_fd(x, y, h=1e-6):
    return (g(x, y + h) - g(x, y - h)) / (2*h)

def tangent_line(slope):
    tangent = np.array([1, slope])
    normal = np.array([-slope, 1])
    normal_unit = normal / np.linalg.norm(normal)
    return tangent, normal, normal_unit

def tangent_plane(grad):
    gx, gy = grad
    normal_vec = np.array([-gx, -gy, 1])
    normal_unit = normal_vec / np.linalg.norm(normal_vec)
    return normal_vec, normal_unit

x = 1.2
y = 0.7

slope_exact = f_prime(x)
grad_exact = np.array([dg_dx(x,y), dg_dy(x,y)])

tangent, normal, normal_unit = tangent_line(slope_exact)
normal_plane, normal_plane_unit = tangent_plane(grad_exact)

data = [{
    "slope": slope_exact,
    "tangent": grad_exact,
    'h': None
}]

for h in np.logspace(-10, -1, 10):
    slope_fd = f_prime_fd(x, h)
    grad_fd = np.array([g_x_fd(x, y, h), g_y_fd(x, y, h)])
    data.append({
        "slope": slope_fd,
        "tangent": grad_fd,
        'h': h
    })

df = pd.DataFrame(data)

df_sorted = df[df['h'].notnull()].sort_values('h')

plt.figure(figsize=(8,5))
plt.loglog(df_sorted['h'], df_sorted['slope'], 'o-', label="Finite Difference f'(x)")
plt.axhline(y=slope_exact, color='r', linestyle='--', label="Exact f'(x)")
plt.xlabel("h")
plt.ylabel("f' approximation")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

error_slope = np.abs(df_sorted['slope'] - slope_exact)
plt.figure(figsize=(8,5))
plt.loglog(df_sorted['h'], error_slope, 'o-', color='orange')
plt.xlabel("h")
plt.ylabel("Error")
plt.title("Error in f'(x)")
plt.grid(True, which="both", ls="--")
plt.show()

gx_error = np.abs(df_sorted['tangent'].apply(lambda t: t[0]) - dg_dx(x,y))
gy_error = np.abs(df_sorted['tangent'].apply(lambda t: t[1]) - dg_dy(x,y))

plt.figure(figsize=(8,5))
plt.loglog(df_sorted['h'], gx_error, 'o-', label="Error in dg/dx")
plt.loglog(df_sorted['h'], gy_error, 'o-', label="Error in dg/dy")
plt.xlabel("h")
plt.ylabel("Error")
plt.title("Gradient Approximation Errors")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
