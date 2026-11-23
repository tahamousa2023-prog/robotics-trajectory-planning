import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(7, 5))
fig.patch.set_facecolor('#0D1B2A')

ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#0D1B2A')

# 12 viewpoints: 1 top + 11 ring at 50° elevation
n_ring = 11
angles = np.linspace(0, 2*np.pi, n_ring, endpoint=False)
elev = np.radians(50)
r = 0.35

# Ring positions
x_ring = r * np.cos(elev) * np.cos(angles)
y_ring = r * np.cos(elev) * np.sin(angles)
z_ring = np.full(n_ring, r * np.sin(elev))

# Top position
x_top = np.array([0])
y_top = np.array([0])
z_top = np.array([0.42])

# Combine
x = np.append(x_ring, x_top)
y = np.append(y_ring, y_top)
z = np.append(z_ring, z_top)

# Object
ax.scatter(0, 0, 0, color='#FF6D00', s=200, marker='s', zorder=5, label='Object')

# Camera positions
ax.scatter(x_ring, y_ring, z_ring, color='#00B4D8', s=80, zorder=5)
ax.scatter(x_top, y_top, z_top, color='#00B4D8', s=80, zorder=5)

# Lines from object to each viewpoint
for xi, yi, zi in zip(x, y, z):
    ax.plot([0, xi], [0, yi], [0, zi], color='#1565C0', alpha=0.4, linewidth=0.8)

# Ring circle
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(r*np.cos(elev)*np.cos(theta),
        r*np.cos(elev)*np.sin(theta),
        np.full(100, r*np.sin(elev)),
        color='#00B4D8', linestyle='--', linewidth=0.8, alpha=0.5)

# Elevation angle annotation
ax.plot([0, r*np.cos(elev)], [0, 0], [0, r*np.sin(elev)],
        color='#FF6D00', linewidth=1.2, linestyle='-', alpha=0.8)
ax.text(0.12, 0, 0.12, '50°', color='#FF6D00', fontsize=9)

ax.set_title('Hemispherical Orbit — 12 Viewpoints',
             color='white', fontsize=12, pad=10)

ax.tick_params(colors='gray', labelsize=7)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#1D3461')
ax.yaxis.pane.set_edgecolor('#1D3461')
ax.zaxis.pane.set_edgecolor('#1D3461')

plt.tight_layout()
plt.savefig('hemisphere_orbit.png', dpi=150, facecolor='#0D1B2A')
plt.show()