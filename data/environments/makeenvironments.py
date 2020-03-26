import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from tifffile import imsave

"""
Environments Have:
axis lims
water depth grid
water current x grid
water current y grid

"""


#
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
X, Y = np.meshgrid(x, y)


def f(x, y):
    s = np.hypot(x, y)
    phi = np.arctan2(y, x)
    tau = s + s * (1 - s) / 5 * np.sin(6 * phi)
    return 5 * (1 - tau) + tau


T = f(X, Y)
# Choose npts random point from the discrete domain of our model function
npts = 400
px, py = np.random.choice(x, npts), np.random.choice(y, npts)

fig, ax = plt.subplots(nrows=2, ncols=2)
# Plot the model function and the randomly selected sample points
ax[0, 0].contourf(X, Y, T)
ax[0, 0].scatter(px, py, c="k", alpha=0.2, marker=".")
ax[0, 0].set_title("Sample points on f(X,Y)")

# Interpolate using three different methods and plot
for i, method in enumerate(("nearest", "linear", "cubic")):
    Ti = griddata((px, py), f(px, py), (X, Y), method=method)
    r, c = (i + 1) // 2, (i + 1) % 2
    ax[r, c].contourf(X, Y, Ti)
    ax[r, c].set_title("method = '{}'".format(method))

plt.tight_layout()
plt.show()
#

myinterpolator = RegularGridInterpolator((x, y), T)

testvalx, testvaly = 0, 0
val = myinterpolator((testvalx, testvaly))
print(f"{val:.2f} == {f(testvalx, testvaly):.2f} ?")


# %% Make constant current field
SCENE_LIMS = [-100, 100, -100, 100]
NUM_DIM = [201, 201]

imsave("test.tiff", T)
