import crocoddyl
import numpy as np

model = crocoddyl.ActionModelUnicycle()
data  = model.createData()
model.costWeights = np.array([ 1, 1 ])
x0 = np.matrix([ -1., -1., 1. ]).T #x,y,theta
T  = 20
problem = crocoddyl.ShootingProblem(x0, [ model ] * T, model)

us = [ np.matrix([1., 1.]).T for t in range(T)]
xs = problem.rollout(us)

import matplotlib.pylab as plt
from tuto_5_utils import plotUnicycle
for x in xs: plotUnicycle(x)
plt.axis([-2,2.,-2.,2.])

ddp = crocoddyl.SolverDDP(problem)
done = ddp.solve()
assert(done)

plt.clf()
for x in ddp.xs: plotUnicycle(x)
plt.axis([-2,2,-2,2])
print(ddp.xs[-1])

print()