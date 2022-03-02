from autograd import exp, log, sqrt, tanh, Variable as V
from autograd import Computational_tree as ct
from autograd import plt
from autograd import activation


def f(x):
    return (V(2)*x+V(7))


ct.auto_derivative_plotter(f,wrt_to="x",limit=[-10,10],dense=1000)


a=ct(f(V(5)))
a.auto_collection()
a.render(beautiful=True)
print(ct.find_area_undercurve(f,0,5))
plt.grid()
plt.show()