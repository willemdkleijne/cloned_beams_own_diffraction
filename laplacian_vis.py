import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid size
N = 100
dx = 0.1
dt = 0.01
c = 1.0  # wave speed

# Initialize wave field (u), previous field (u_prev), and next field (u_next)
u = np.zeros((N, N))
u_prev = np.zeros((N, N))
u_next = np.zeros((N, N))

# Initial condition: a Gaussian bump
x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)
X, Y = np.meshgrid(x, y)
u = np.exp(-(X**2 + Y**2))
u_prev = np.copy(u)

# Compute Laplacian using finite differences
def laplacian(Z):
    return (
        -4 * Z
        + np.roll(Z,  1, axis=0) + np.roll(Z, -1, axis=0)
        + np.roll(Z,  1, axis=1) + np.roll(Z, -1, axis=1)
    ) / dx**2

# Set up figure
fig, ax = plt.subplots()
im = ax.imshow(u, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(im)

def update(frame):
    global u, u_prev, u_next

    # Apply the wave equation update rule:
    # u_next = 2u - u_prev + c^2 * dt^2 * Laplacian(u)
    u_next = 2*u - u_prev + c**2 * dt**2 * laplacian(u)

    # Rotate states
    u_prev, u = u, u_next

    im.set_data(u)
    return [im]

ani = FuncAnimation(fig, update, frames=300, interval=20)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
N = 300
L = 10.0
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]

# Wave number and parameters (example values)
lambda0 = 0.0037
U0 = 1.0
sigma = 1.0
dz = 0.01

# Initial wave (Gaussian)
psi = np.exp(-(X**2 + Y**2))

# Example potential (a circular phase object)
V = np.exp(-((X/2)**2 + (Y/2)**2))

# Fourier frequencies
kx = 2*np.pi*np.fft.fftfreq(N, d=dx)
ky = 2*np.pi*np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(kx, ky)

# Free-space propagator (your Laplacian exponential)
alpha = 4*np.pi*1j/(lambda0*U0) * dz
H = np.exp(alpha * (-(KX**2 + KY**2)))  # Fourier-domain Laplacian exponential

# --- One split-step propagation ---
# Phase shift (potential)
psi = np.exp(1j * sigma * V * dz) * psi

# Free-space propagation via FFT
psi_ft = np.fft.fft2(psi)
psi_ft *= H
psi = np.fft.ifft2(psi_ft)

# Plot result
plt.imshow(np.abs(psi)**2, cmap='inferno')
plt.title("Intensity after one propagation step")
plt.colorbar()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- Setup grid ---
N = 120
L = 10.0
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]

# Parameters
lambda0 = 0.0037
U0 = 1.0
sigma = 1.0
dz = 0.02
steps = 80

# Initial wave (Gaussian)
psi = np.exp(-(X**2 + Y**2))

# Potential (weak lens-like phase)
V = np.exp(-((X/3)**2 + (Y/3)**2))*0.2

# Fourier space
kx = 2*np.pi*np.fft.fftfreq(N, d=dx)
ky = 2*np.pi*np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(kx, ky)

alpha = 4*np.pi*1j/(lambda0*U0) * dz
H = np.exp(alpha * (-(KX**2 + KY**2)))

# --- Animation setup ---
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')

Z = np.abs(psi)**2
surf = ax.plot_surface(X, Y, Z, cmap="inferno", linewidth=0, antialiased=False)
ax.set_zlim(0, 1)

def update(frame):
    global psi, surf
    # potential step
    psi = np.exp(1j * sigma * V * dz) * psi
    # propagation step
    psi_ft = np.fft.fft2(psi)
    psi_ft *= H
    psi = np.fft.ifft2(psi_ft)

    ax.clear()
    Z = np.abs(psi)**2
    ax.plot_surface(X, Y, Z, cmap="inferno", linewidth=0, antialiased=False)
    ax.set_zlim(0, 1)
    ax.set_title(f"Propagation step {frame}")
    return []

ani = FuncAnimation(fig, update, frames=steps, interval=100)

#disp animation
plt.show()


