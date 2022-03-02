from numpy.lib.function_base import percentile
from autograd import Variable as t
import numpy as np
import matplotlib.pyplot as plt
from autograd import Computational_tree
import time
import pydot
s=time.perf_counter()

def f(x):
    return ((x+t(2))**t(2))
def g(xp):
    return (xp**2)+t(5)*(xp)

x=3
a=t(x,train_able=True)
bg=t.tanh((f(g(f(f(a))))))
#bg*=t(2)
bg+=f(bg+bg+bg)
#bg-=t(5)+a
s=time.perf_counter()
bg.backward(1)
a=time.perf_counter()

bc=Computational_tree(bg)
bc.auto_collection()

tree=bc.travel()[1]
bc.render(beautiful=True,parallel=True)

print(a-s)
'''
a=t(300)
def f(x):
    return (x**(-2))
b=f(a)
b.backward(1)

b_graph=Computational_tree(b)
b_graph.auto_collection()

b_graph.render(beautiful=True,parallel=False)
'''