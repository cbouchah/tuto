import matplotlib.pyplot as plt
import numpy as np



def plotUnicycle(x):
    sc, delta = .1, .1
    a, b, th = np.asscalar(x[0]), np.asscalar(x[1]), np.asscalar(x[2])
    c, s = np.cos(th), np.sin(th)
    refs = [
        plt.arrow(a - sc / 2 * c - delta * s, b - sc / 2 * s + delta * c, c * sc, s * sc, head_width=.05),
        plt.arrow(a - sc / 2 * c + delta * s, b - sc / 2 * s - delta * c, c * sc, s * sc, head_width=.05)
    ]
    return refs


def displayTrajectory(viz, xs, dt=0.01, rate=-1):
    """  Display a robot trajectory xs using Gepetto-viewer gui.
    :param robot: Robot wrapper
    :param xs: state trajectory
    :param dt: step duration
    :param rate: visualization rate
    """
    import time
    S = 1 if rate <= 0 else max(int(1/dt/rate), 1)
    for i, x in enumerate(xs):
        if not i % S:
            viz.display(x[:viz.model.nq])
            time.sleep(dt*S)
    viz.display(xs[-1][:viz.model.nq])

def plotUnicycleSolution(xs, figIndex=1, show=True):
    import matplotlib.pylab as plt
    plt.figure(figIndex, figsize=(6.4, 6.4))
    for x in xs:
        plotUnicycle(x)
    plt.axis([-2, 2., -2., 2.])
    if show:
        plt.show()