import matplotlib.pyplot as plt
import numpy as np

n_views = 12
angles = np.linspace(0, 2*np.pi, n_views, endpoint=False)
r = 0.35  # camera distance from object

x = r * np.cos(angles)
y = r * np.sin(angles)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor('#0D1B2A')
fig.patch.set_facecolor('#0D1B2A')

# Object
ax.scatter(0, 0, s=300, color='#FF6D00', zorder=5, label='Object')
ax.text(0, 0.04, 'object', color='white', ha='center', fontsize=9)

# Camera positions
ax.scatter(x, y, s=120, color='#00B4D8', zorder=5)
for i, (xi, yi) in enumerate(zip(x, y)):
    ax.annotate(f'{i+1}', (xi, yi), color='white', fontsize=7,
                ha='center', va='center')
    ax.plot([0, xi], [0, yi], color='#1565C0', alpha=0.3, linewidth=0.8)

# Orbit circle
circle = plt.Circle((0, 0), r, color='#00B4D8', fill=False,
                     linestyle='--', linewidth=1, alpha=0.5)
ax.add_patch(circle)

ax.set_xlim(-0.55, 0.55)
ax.set_ylim(-0.55, 0.55)
ax.set_aspect('equal')
ax.set_title('Viewpoint Orbit — 12 Positions (Top View)',
             color='white', fontsize=12)
ax.tick_params(colors='gray')
for spine in ax.spines.values():
    spine.set_edgecolor('#1D3461')
plt.tight_layout()
plt.savefig('viewpoints_topdown.png', dpi=150)
plt.show()