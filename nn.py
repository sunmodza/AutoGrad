from autograd import *
import numpy as np

#y=mx+c

#x=1 y=5
#x=2 y=7

x1=Variable(1)
yt1=Variable(5)

x2=Variable(2)
yt2=Variable(7)

xa=[x1,x2]
yta=[yt1,yt2]

m=Variable(0,train_able=True,optimizer=SGD(10e-3))
c=Variable(0,train_able=True,optimizer=SGD(10e-3))

for i in range(30000):
    for x,yt in zip(xa,yta):
        y=m*x+c
        error=(yt-y)**2
        error.backward(1)

        m.update_with_optimizer()
        c.update_with_optimizer()

        print(error)

print(f'm={m},c={c}')


