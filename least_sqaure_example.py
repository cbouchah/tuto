import numpy as np
from qpsolvers import solve_ls
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3
# R = np.identity(3)
# s = np.array([3., 2., 3.])
G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
# h = np.zeros(3)
# w = np.ones(9).reshape((3,3))
#
# plt.plot()
# x_sol = solve_ls(R, s, G, h)
# print(f"LS solution: x = {x_sol}")


# V = np.array([1,1])
# origin = np.array([[0, 0, 0],[0, 0, 0]]) # origin point
# R = np.identity(2)
# x_sol = solve_ls(R, V)
# plt.quiver(*origin, V[0], color=['r'], scale=21)
# plt.show()

# 100 linearly spaced numbers
x = np.linspace(-np.pi,np.pi,100)

# the functions, which are y = sin(x) and z = cos(x) here
y = np.sin(x)
z = np.cos(x)

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

#solving the approximation
g = -np.identity(100)
h = -np.ones(100)*0.25
solve_y = solve_ls(np.identity(100), y, g, h)
solve_z = solve_ls(np.identity(100), z, g, h)

# plot the functions
plt.plot(x,y, 'xc', label='y=sin(x)')
plt.plot(x,z, 'xm', label='y=cos(x)')
plt.plot(x,solve_y, 'xr', label='solve_y')
plt.plot(x,solve_z, 'xb', label='solve_z')

plt.legend(loc='upper left')
# show the plot
plt.show()
print()
