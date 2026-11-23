import matplotlib.pyplot as plt
import numpy as np

steps = 100
t = np.linspace(0, 1, steps)
linear = t
smooth = 3*t**2 - 2*t**3  # your cubic easing formula

plt.figure(figsize=(8, 4))
plt.plot(t, linear, '--', color='gray', label='Linear (jerky)')
plt.plot(t, smooth, color='#00B4D8', linewidth=2.5, label='Cubic Easing (smooth)')
plt.title('Joint Interpolation — Cubic Easing vs Linear', fontsize=13)
plt.xlabel('Step (normalized)')
plt.ylabel('Joint Position')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cubic_easing.png', dpi=150)
plt.show()