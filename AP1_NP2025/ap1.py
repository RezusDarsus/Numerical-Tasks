import numpy as np
import matplotlib.pyplot as plt

vec1 = np.random.rand(4)
vec2 = np.random.rand(4)

mat1 = np.random.rand(4).reshape(2, 2)
mat2 = np.random.rand(4).reshape(2, 2)

p = 5

dist = np.linalg.norm(vec2 - vec1, ord=p)
dist_euc = np.linalg.norm(vec2 - vec1)

mat_dist = np.linalg.norm((mat2 - mat1).ravel(), ord=p)
mat_dist_euc = np.linalg.norm(mat2 - mat1, ord='fro')

xr = vec1
mr = mat1

grid_n = 30

x = np.linspace(-1.5, 1.5, grid_n)
y = np.linspace(-1.5, 1.5, grid_n)
z = np.linspace(-1.5, 1.5, grid_n)
X, Y, Z = np.meshgrid(x, y, z)
X_flat = np.vstack([X.ravel(), Y.ravel(), Z.ravel(), np.zeros(X.size)])

def inside_unit_ball(points, xr, p):
    dists = np.linalg.norm(points.T - xr, ord=p, axis=1)
    return dists <= 1

norms = [2, p]
fig = plt.figure(figsize=(14, 10))

for i, p in enumerate(norms, 1):
    mask = inside_unit_ball(X_flat, xr, p)
    points_inside = X_flat[:, mask]

    ax = fig.add_subplot(2, 2, i, projection='3d')
    ax.scatter(points_inside[0], points_inside[1], points_inside[2], s=3)
    ax.set_title(f"Slice of 4D unit ball (p = {p})")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_zlabel("x₃")
    ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()

X_mat = np.stack([X.ravel(), Y.ravel(), Z.ravel(), np.zeros(X.size)], axis=1)
X_mat = X_mat.reshape(-1, 2, 2)

def inside_matrix_ball(points, mr, p):
    diffs = points - mr
    dists = np.sum(np.abs(diffs)**p, axis=(1, 2))**(1/p)
    return dists <= 1

norms = [2, p]
fig = plt.figure(figsize=(14, 10))

for i, p in enumerate(norms, 1):
    mask = inside_matrix_ball(X_mat, mr, p)
    inside = X_mat[mask]

    ax = fig.add_subplot(2, 2, i, projection='3d')
    ax.scatter(inside[:, 0, 0], inside[:, 0, 1], inside[:, 1, 0], s=3, color='orange')
    ax.set_title(f"Slice of 4D matrix ball(p = {p})")
    ax.set_xlabel("a₁₁"); ax.set_ylabel("a₁₂"); ax.set_zlabel("a₂₁")
    ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()
