from autograd import Variable,Tensor,losses,activation,optimizer
from autograd import Tensor as t
Adam=optimizer.RMSprop
import numpy as np
x1=Tensor(np.array([[1,1]]).reshape(-1,1))
x2=Tensor(np.array([[0,1]]).reshape(-1,1))
x3=Tensor(np.array([[1,0]]).reshape(-1,1))
x4=Tensor(np.array([[0,0]]).reshape(-1,1))

y1=Tensor(np.array([[1],[0]]))
y2=Tensor(np.array([[1],[0]]))
y3=Tensor(np.array([[1],[0]]))
y4=Tensor(np.array([[0],[1]]))




x=[x4,x1,x2,x3,x4]
y=[y4,y1,y2,y3,y4]

        
a=Tensor(np.random.rand(2,10).T,train_able=True,optimizer=Adam(lr=3e-4))
b=Tensor(np.random.rand(10,15).T,train_able=True,optimizer=Adam(lr=3e-4))
c=Tensor(np.random.rand(15,2).T,train_able=True,optimizer=Adam(lr=3e-4))
c1=Tensor(np.zeros((10,1)),train_able=True,optimizer=Adam(lr=3e-4))
c2=Tensor(np.zeros((15,1)),train_able=True,optimizer=Adam(lr=3e-4))
c3=Tensor(np.zeros((2,1)),train_able=True,optimizer=Adam(lr=3e-4))
        
train_variable=[a,b,c,c1,c2,c3]

for i in range(10000):
    tl=0
    for xi,yi in zip(x,y):
        l1=a.dot(xi)+c1
        l1=activation.sigmoid(l1)
        l2=b.dot(l1)+c2
        l2=activation.sigmoid(l2)
        l3=c.dot(l2)+c3
        l3=activation.Softmax(l3)
        #print(l3)
        #loss=(y-l3)**2
        loss=losses.Multiclass_CrossEntropy(yi,l3)
        loss.backward()
        tl+=loss.data[0,0].value
        for i in train_variable:
            i.apply_grad_with_optimizer()
    print(tl)


