# -*- coding: utf-8 -*-
"""
#AUTO_differential\n
Created on Mon Nov 16 09:50:52 2020\n
@author: sunmodza
"""
import numpy as np
import math
from PIL import Image
import threading
import _thread
from numpy.core.fromnumeric import var
from numpy.core.numeric import NaN
from numpy.lib.arraysetops import isin
import pydot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pygraphviz as pgv
import networkx as nx
from typing import *
from copy import copy
from graphviz import Source
from anytree import Node, RenderTree
from anytree.exporter import DotExporter,UniqueDotExporter
#from autograd import sin, sinh
eps=np.finfo(float).eps
class Optimizer:
    def __init__(self,lr):
        self.lr=lr
        self.obj=None
    def update_value(self):
        return 0
    def update(self):
        self.grad=self.obj.grad
        self.obj.value+=self.update_value()
        
class SGD(Optimizer):
    def __init__(self,lr):
        super().__init__(lr)
    def update_value(self):
        return -self.lr*self.grad
    def update(self):
        self.grad=self.obj.grad
        self.obj.value+=self.update_value()
        #print(self.obj.value,self.obj.grad)
        #print(self.obj.value)

class Variable:
    def __init__(self,value,sig=None,train_able=False,optimizer=SGD(10e-3),name=None):
        self.grad=None
        self.value=value
        self.name=name
        self.sig=sig
        self.integrated=None

        self.train_able=train_able
        self.optimizer=copy(optimizer)
        self.optimizer.obj=self
    @staticmethod
    def to_variable_if_not(other):
        if not isinstance(other,Variable):
            other=Variable(other)
        return other

    def as_variable(self):
       pass 

    def __add__(self,other):
        other=Variable.to_variable_if_not(other)
        #self=add(self, other).forward()
        return add(self, other).forward()
    
    def __eq__(self,other):
        return self.value==other.value
    
    def __lt__(self,other):
        return self.value<other.value
    
    def respect_to(self,variable):
        raise NotImplementedError
    
    def __mul__(self,other):
        other=Variable.to_variable_if_not(other)
        #self=mul(self, other).forward()
        return mul(self, other).forward()
    
    def __neg__(self):
        
        #self=mul(self, Variable(-1)).forward()
        return mul(self, Variable(-1)).forward()
    
    def __truediv__(self,other):
        other=Variable.to_variable_if_not(other)
        #if not isinstance(other,Variable):
            #other=Variable(other)
        #self=division(self, other).forward()
        return division(self, other).forward()
    
    def backward(self,grad):
        self.grad=grad
        #print(grad,type(self.sig))
        self.sig.backward(self.grad)

        
    def __call__(self):
        return self.value
    
    def __repr__(self):
        return f'{self.value}'
    
    def __pow__(self,other):
        #if not isinstance(other,Variable):
            #other=Variable(other)
        other=Variable.to_variable_if_not(other)
        #self=power(self, other).forward()
        return power(self, other).forward()
    
    def __sub__(self,v):
        other=Variable.to_variable_if_not(v)
        #self=minus(self,v).forward()
        return minus(self,v).forward()
    def update_with_optimizer(self):
        if self.train_able:
            self.optimizer.update()
        return self
    def set_train(self,train=False):
        self.train_able=train
        return self
    
    def backward_integrated(self,d):
        self.sig.backward_integrated(d)
    
    def plot_derivative(self):
        rm=np.linspace(-1000,1000,10000)
        plt.plot(rm,rm*self.grad)
        return rm*self.grad
    
    def sin(self):
        #self=sin(self).forward()
        return sin(self).forward()
    
    def cos(self):
        #self=cos(self).forward()
        return cos(self).forward()
    
    def tan(self):
        #self=tan(self).forward()
        return tan(self).forward()
    
    def tanh(self):
        #self=tanh(self).forward()
        return tanh(self).forward()
    
    def abs(self):
        #self=abs(self).forward()
        return abs(self).forward()

    def log(self):
        #self=log(self).forward()
        return log(self).forward()
    
    def log10(self):
        #self=log10(self).forward()
        return log10(self).forward()
    
    def exp(self):
        #self=exp(self).forward()
        return exp(self).forward()
    
    def log2(self):
        #self=log2(self).forward()
        return log2(self).forward()
    
    def sqrt(self):
        #self=sqrt(self).forward()
        return sqrt(self).forward()
    
    def inv(self):
        #self=inv(self).forward()
        return inv(self).forward()
    
    def csc(self):
        #self=inv(self.sin()).forward()
        return inv(self.sin()).forward()
    
    def sec(self):
        #self=inv(self.cos()).forward()
        return inv(self.cos()).forward()
    
    def cot(self):
        #self=inv(self.tan()).forward()
        return inv(self.tan()).forward()
    
    def sinh(self):
        #self=sinh(self).forward()
        return sinh(self).forward()
    
    def cosh(self):
        #self=cosh(self).forward()
        return cosh(self).forward()
    
class operater:
    def __init__(self,a,b,sign):
        self.a=a
        self.b=b
        self.sign=sign
    def forward(self):
        self.calculate_local_gradient()
        raise NotImplementedError
    
    def backward(self,p):
        self.grad=p
        self.calculate_total_gradient()
        self.deep_flow()

    def calculate_local_integrated(self):
        raise NotImplementedError
    
    def calculate_total_integrated(self):
        self.a.integrated*=self.integrated
        self.b.integrated*=self.integrated
    
    def backward_integrated(self,p):
        self.integrated=p
        self.calculate_local_integrated()
        self.calculate_total_integrated()
        self.deep_integrated()
    
    def calculate_local_gradient(self):
        raise NotImplementedError

    def deep_integrated(self):
        if self.a.sig is not None:
            self.a.backward_integrated(self.a.integrated)
        else:
            self.a.integrated=self.a.value**2
        if self.b.sig is not None:
            self.b.backward_integrated(self.b.integrated)
        else:
            self.b.integrated=self.b.value**2
            

    def deep_flow(self):
        if self.a.sig is not None:
            self.a.backward(self.a.grad)
        if self.b.sig is not None:
            self.b.backward(self.b.grad)
    def __repr__(self):
        return self.sign
    def __str__(self):
        return self.sign
    
    def calculate_total_gradient(self):
        self.a.grad*=self.grad
        self.b.grad*=self.grad

class add(operater):
    def __init__(self,a,b):
        self.a=a
        self.b=b
        super().__init__(a,b,"add")
    def forward(self):
        self.calculate_local_gradient()
        return Variable(self.a.value+self.b.value,sig=self)
    
    def calculate_local_gradient(self):
        self.a.grad=1
        self.b.grad=1
    
    def calculate_local_integrated(self):
        self.a.integrated=(self.a.value**2/2)+(self.a.value*self.b.value)
        self.b.integrated=(self.b.value**2/2)+(self.a.value*self.b.value)

class mul(operater):
    def __init__(self,a,b):
        super().__init__(a,b,"multiply")
    def forward(self):
        self.calculate_local_gradient()
        return Variable(self.a.value*self.b.value,sig=self)
    
    def calculate_local_gradient(self):
        self.a.grad=self.b.value
        self.b.grad=self.a.value
    
    def calculate_local_integrated(self):
        self.a.integrated=(self.b.value*self.a.value**2)/2
        self.b.integrated=(self.a.value*self.b.value**2)/2

    
class minus(operater):
    def __init__(self,a,b):
        super().__init__(a,b,"minus")
    def forward(self):
        self.calculate_local_gradient()
        return Variable(self.a.value-self.b.value,sig=self)
    def calculate_local_gradient(self):
        self.a.grad=1
        self.b.grad=-1
    def calculate_local_integrated(self):
        self.a.integrated=((self.a.value-(2*self.b.value))*self.a.value)/2
        self.b.integrated=(self.a.value*self.b.value)-(self.b.value**2/2)
    
class division(operater):
    def __init__(self,a,b):
        super().__init__(a,b,"divide")
    def forward(self):
        self.calculate_local_gradient()
        return Variable(self.a.value/self.b.value,sig=self)
    def calculate_local_gradient(self):
        self.a.grad=1/self.b.value
        self.b.grad=(-1*self.a.value)/(self.b.value**2)
    
    def calculate_local_integrated(self):
        self.a.integrated=self.a.value**2/(self.b.value*2)
        self.b.integrated=self.a.value*np.log(np.abs(self.b.value))
            
class power(operater):
    def __init__(self,a,b):
        super().__init__(a,b,"power")
    def forward(self):
        self.calculate_local_gradient()
        return Variable(self.a.value**self.b.value,sig=self)
    def calculate_local_gradient(self):
        self.a.grad=((self.b.value*(self.a.value**(self.b.value-1))))
        self.b.grad=((self.a.value**self.b.value)*(np.log(self.a.value)))
    
    def calculate_local_integrated(self):
        self.a.integrated=(self.a.value**(self.b.value+1))/(self.b.value+1)
        self.b.integrated=(self.a.value**self.b.value)/np.log(self.a.value)

class funct:
    def __init__(self,v,sign):
        self.v=v
        self.sign=sign
    def forward(self):
        raise NotImplementedError
    def backward(self,p):
        self.grad=p
        self.calculate_total_gradient()
        self.deep_flow()
    def deep_flow(self):
        if self.v.sig is not None:
            self.v.backward(self.v.grad)
    def calculate_local_integrated(self):
        raise NotImplementedError
    def calculate_total_integrated(self):
        self.v.integrated*=self.integrated
    def backward_integrated(self,p):
        self.integrated=p
        self.calculate_local_integrated()
        self.calculate_total_integrated()
        self.deep_integrated()
    def deep_integrated(self):
        if self.v.sig is not None:
            self.v.backward_integrated(self.v.integrated)
        else:
            self.v.integrated=self.v.value**2
    def calculate_local_gradient(self):
        raise NotImplementedError
    def calculate_total_gradient(self):
        self.v.grad*=self.grad
    def __repr__(self):
        return self.sign
    def __str__(self):
        return self.sign

class inv(funct):
    def __init__(self,v):
        super().__init__(v,"inv")
    def calculate_local_gradient(self):
        self.v.grad=-1/(self.v.value**2)
    def forward(self):
        self.calculate_local_gradient()
        return Variable((self.v.value)**-1,sig=self)
    def calculate_local_integrated(self):
        self.v.integrated=np.log(np.abs(self.v.value))

class sin(funct):
    def __init__(self,v):
        super().__init__(v,"sin")
    def calculate_local_gradient(self):
        self.v.grad=np.cos(self.v.value)
    def forward(self):
        self.calculate_local_gradient()
        return Variable(np.sin(self.v.value),sig=self)
    def calculate_local_integrated(self):
        self.v.integrated=-np.cos(self.v.value)

class sinh(funct):
    def __init__(self,v):
        super().__init__(v,"sinh")
    def calculate_local_gradient(self):
        self.v.grad=np.cosh(self.v.value)
    def forward(self):
        self.calculate_local_gradient()
        return Variable(np.sinh(self.v.value),sig=self)
    def calculate_local_integrated(self):
        self.v.integrated=np.cosh(self.v.value)

class cos(funct):
    def __init__(self,v):
        super().__init__(v,"cos")
    def forward(self):
        self.calculate_local_gradient()
        return Variable(np.cos(self.v.value),sig=self)
    def calculate_local_gradient(self):
        self.v.grad=-np.sin(self.v.value)
    def calculate_local_integrated(self):
        self.v.integrated=np.sin(self.v.value)

class cosh(funct):
    def __init__(self,v):
        super().__init__(v,"cosh")
    def calculate_local_gradient(self):
        self.v.grad=np.sinh(self.v.value)
    def forward(self):
        self.calculate_local_gradient()
        return Variable(np.cosh(self.v.value),sig=self)
    def calculate_local_integrated(self):
        self.v.integrated=np.sinh(self.v.value)

class tan(funct):
    def __init__(self,v):
        super().__init__(v,"tan")
    def calculate_local_gradient(self):
        self.v.grad=(np.cos(self.v.value)**-1)
    def forward(self):
        self.calculate_local_gradient()
        return Variable(np.tan(self.v.value),sig=self)
    def calculate_local_integrated(self):
        self.v.integrated=np.log(np.abs(np.sec(self.v.value)))

class tanh(funct):
    
    def __init__(self,v):
        super().__init__(v,"tanh")
    def calculate_local_gradient(self):
        self.v.grad=(np.cosh(self.v.value)**-2)
    def forward(self):
        self.calculate_local_gradient()
        return Variable(np.tanh(self.v.value),sig=self)
    
    def calculate_local_integrated(self):
        self.v.integrated=np.log(np.cosh(self.v.value))


class abs(funct):
    def __init__(self,v):
        super().__init__(v,"abs")
    def calculate_local_gradient(self):
        self.v.grad=((self.v.value/(np.fabs(self.v.value+eps))))
    def forward(self):
        self.calculate_local_gradient()
        return Variable(np.fabs(self.v.value),sig=self)
    def calculate_local_integrated(self):
        self.v.integrated=self.v.value*np.abs(self.v.value)/2

class log(funct):
    def __init__(self,v):
        super().__init__(v,"log")
    def calculate_local_gradient(self):
        self.v.grad=(1/self.v.value)
    def forward(self):
        self.calculate_local_gradient()
        return Variable(np.log(self.v.value),sig=self)
    def calculate_local_integrated(self):
        self.v.integrated=self.v.value*(np.log(self.v.value)-1)

class log10(funct):
    def __init__(self,v):
        super().__init__(v,"log10")
    def calculate_local_gradient(self):
        self.v.grad=(1/(np.log(10)*self.v.value))
    def forward(self):
        self.calculate_local_gradient()
        return Variable(np.log10(self.v.value),sig=self)
    def calculate_local_integrated(self):
        self.v.integrated=(self.v.value*(np.log(self.v.value)-1))/np.log(10)

class log2(funct):
    def __init__(self,v):
        super().__init__(v,"log2")
    def calculate_local_gradient(self):
        self.v.grad=(1/(np.log(2)*self.v.value))
    def forward(self):
        self.calculate_local_gradient()
        return Variable(np.log2(self.v.value),sig=self)
    def calculate_local_integrated(self):
        self.v.integrated=(self.v.value*(np.log(self.v.value)-1))/np.log(2)

class sqrt(funct):
    def __init__(self,v):
        super().__init__(v,"sqrt")
    def calculate_local_gradient(self):
        self.v.grad=(1/(2*np.sqrt(self.v.value)))
    def forward(self):
        self.calculate_local_gradient()
        return Variable(np.sqrt(self.v.value),sig=self)
    def calculate_local_integrated(self):
        self.v.integrated=(2*(self.v.value**(3/2)))/3
        
class exp(funct):
    def __init__(self,v):
        super().__init__(v,"exp")
    def calculate_local_gradient(self):
        self.v.grad=np.exp(self.v.value)
    def forward(self):
        self.calculate_local_gradient()
        return Variable(np.exp(self.v.value),sig=self)
    def calculate_local_integrated(self):
        self.v.integrated=self.v.value

class Tensor:
    def __init__(self,data,optimizer=SGD(10e-4),train_able=False):
        self.train_able=train_able
        self.optimizer=optimizer
        self.data=np.atleast_2d(data)
        if len(self.data.shape) > 1:
            if type(np.concatenate(self.data)[0]) is not Variable:
                self.to_array()
        if self.train_able:
            self.init_train()
        
    def to_array(self):
        newarr=[]
        for _,v in enumerate(np.concatenate(self.data)):
            newarr.append(Variable(v))
        self.data=np.array(newarr,dtype=Variable).reshape(self.data.shape)
    @property
    def return_array(self):
        newarr=[]
        for _,v in enumerate(np.concatenate(self.data)):
            newarr.append(v.value)
        return np.array(newarr).reshape(self.data.shape)

    def init_train(self):
        newarr=[]
        for _,v in enumerate(np.concatenate(self.data)):
            v.optimizer=copy(self.optimizer)
            v.optimizer.obj=v
            newarr.append(v.set_train(train=self.train_able))
        self.data=np.array(newarr,dtype=Variable).reshape(self.data.shape)

    def __mul__(self,other):
        return Tensor(self.data*other.data)
    
    def __neg__(self):
        return Tensor(-self.data)
    
    def __truediv__(self,other):
        return Tensor(self.data/other.data)
    
    def __add__(self,other):
        return Tensor(self.data+other.data)
    
    def __sub__(self,other):
        return Tensor(self.data-other.data)

    def __repr__(self):
        return f'{self.data}'
    
    def dot(self,other):
        return Tensor(self.data.dot(other.data))
    
    def __matmul__(self,other):
        return Tensor(self.data @ other.data)
    
    def backward(self,respect_to=1):
        self.al2d()
        arr=np.concatenate(self.data)
        '''
        for v in arr:
            v.backward(respect_to)
        '''
        #self.apply_function(Variable.backward,respect_to)

        for v in arr:
            _thread.start_new_thread(v.backward,tuple([respect_to]))
            
            #print(threading.active_count())
        self.data=arr.reshape(self.data.shape)
    def al2d(self):
        self.data=np.atleast_2d(self.data)
    @property
    def grad(self):
        self.al2d()
        arr=np.concatenate(self.data)
        for i,v in enumerate(arr):
            arr[i]=v.grad
        return arr.reshape(self.data.shape)
    
    def apply_grad_with_optimizer(self):
        for i,_ in np.ndenumerate(self.data):
            self.data[i].update_with_optimizer()

    def apply_function(self,func,*args):
        self.al2d()
        func=np.vectorize(func)
        newtensor=Tensor(func(self.data,*args))
        return newtensor

    def __pow__(self,v):
        if not isinstance(v,Variable):
            v=Variable(v)
        return Tensor(self.data**v)
    
    @property
    def T(self):
        self.data=self.data.T
        return self
    
    def abs(self):
        
        return self.apply_function(Variable.abs)
    def sin(self):
        
        return self.apply_function(Variable.sin)
    def tan(self):
        
        return self.apply_function(Variable.tan)
    def tanh(self):
        
        return self.apply_function(Variable.tanh)
    def cos(self):
        
        return self.apply_function(Variable.cos)
    def log(self):
        
        return self.apply_function(Variable.log)
    def log10(self):
        
        return self.apply_function(Variable.log10)
    def exp(self):
        
        return self.apply_function(Variable.exp)
    def log2(self):
        
        return self.apply_function(Variable.log2)
    def sqrt(self):
        
        return self.apply_function(Variable.sqrt)
    def csc(self):
        
        return self.apply_function(Variable.csc)
    def sec(self):
        
        return self.apply_function(Variable.sec)
    def cot(self):
        
        return self.apply_function(Variable.cot)
    def sinh(self):
        
        return self.apply_function(Variable.sinh)
    def cosh(self):
        
        return self.apply_function(Variable.cosh)
    
    def sum(self,axis=0):
        newTensor=Tensor(np.sum(self.data,axis=axis))
        #self=Tensor(np.sum(self.data,axis=axis))
        return newTensor
    
    def mean(self,axis=0):
        newTensor=Tensor(np.mean(self.data,axis=axis))
        return newTensor

class optimizer:
    class Adam(Optimizer):
        def __init__(self,lr=3e-4,b1=0.999,b2=0.9):
            self.b1=b1
            self.b2=b2
            self.w=0
            self.m=0
            super().__init__(lr)
        def update_value(self):
            self.w=self.b1*(self.w)+(1-self.b1)*(self.grad**2)
            self.m=self.b2*(self.w)+(1-self.b2)*(self.grad)
            
            wh=self.w/(1-self.b1)
            mh=self.m/(1-self.b2)
    
            update_param_value=-self.lr*(mh/np.sqrt(wh+np.finfo(float).eps))
            
            return update_param_value
        
    class RMSprop(Optimizer):
        def __init__(self,lr=10e-4,b=0.9):
            self.b=b
            self.dw=0
            super().__init__(lr)
        
        def update_value(self):
            self.dw=(self.b*self.dw)+((1-self.b)*(self.grad**2))
            update_param_value=-self.lr*(self.dw/np.sqrt(self.dw+np.finfo(float).eps))
            return update_param_value
    
    class SGD(Optimizer):
        def __init__(self,lr=10e-4):
            super().__init__(lr)
        def update_value(self):
            return -self.lr*self.grad
        def update(self):
            self.grad=self.obj.grad
            self.obj.value+=self.update_value()

class losses:
    @staticmethod
    def MSE(a,p):
        loss=((a-p)**2).mean()
        return loss
    @staticmethod
    def MAE(a,p):
        loss=((a-p).abs()).mean()
        return loss
    @staticmethod
    #fucking broken for some reason
    def Multiclass_CrossEntropy(a,p):
        loss=Tensor.sum(losses.binary_CrossEntropy(a,p))
        return loss
    
    @staticmethod
    def binary_CrossEntropy(a,p):
        loss=-(a*p.log()+((Tensor(1)-a)*Tensor.log(p)))
        return loss
    
class activation:
    @staticmethod
    def sigmoid(v):
        a=Tensor(1)/(Tensor(1)+Tensor.exp(-v))
        return a
    @staticmethod
    def tanh(v):
        a=Tensor.tanh(v)
        return a
    @staticmethod
    def linear(v):
        a=v
        return a
    @staticmethod
    def Softmax(v):
        return Tensor.exp(v)/(Tensor.sum(Tensor.exp(v),axis=0))
    @staticmethod
    def Relu(v):
        return np.where(v.data<Variable(0),Variable(0),v.data)
    @staticmethod
    def LeakyRelu(v,a=0.01):
        return np.where(v.data<Variable(0),v.data*Variable(a),v.data)
    @staticmethod
    def Swish(v):
        a=v/(Tensor(1)+Tensor.exp(-v))
        return a

#print(len(np.ones((1,1)).shape))
'''
t=Tensor(np.ones((3,1)))
b=Tensor(np.array([2]))

f=t.dot(b)

print(f)
'''

class Period_recorder:
    def __init__(self,is_rm=False) -> None:
        self.is_rm=is_rm
        super().__init__()
        self.period=[]
        self.curr_min = math.inf
        self.curr_max = -math.inf

        self.exception=[]
        self.all_receive_case_rm=[]
    
    def record(self,*vs,rmdm=3.14):
        for v in vs:
            if v<self.curr_min:
                self.curr_min=v
            elif v>=self.curr_max:
                self.curr_max=v
            if self.is_rm is False:
                if math.isnan(rmdm) or math.isnan(v):
                    self.exception.append(v)
                    #self.period.append((self.curr_min,self.curr_max))
            if self.is_rm:
                self.all_receive_case_rm.append(v)
        

    def end_record(self,U=None):
        self.period.append((self.curr_min,self.curr_max))
        if self.is_rm and len(self.exception)==0:
            U=set(U.flatten())
            self.exception=U.difference(set(self.all_receive_case_rm))
            exper=Period_recorder()
            exper.record(*list(self.exception))
            exper.end_record()
            self.exception=exper.exception

class Computational_tree:
    def __init__(self,variable:Variable,init=True,fcs=None):
        self.head=variable
        if init:
            self.head.backward(1)
            self.head.backward_integrated(1)
        self.right=None
        self.left=None
        self.signature=None
        self.fcs=fcs
        self.items=[]
    @staticmethod
    def exceptional_finder(data,limit):
        
        U=set(limit.tolist())
        if isinstance(data,np.ndarray):
            d=data.tolist()
        d=data[:]
        exclude=U.difference(set(d))
        
        data=np.array(data)
        minimum=data.min()
        maximum=data.max()

        text=f'[{minimum},{maximum}]'

        return text,exclude,[minimum,maximum]
    
    def auto_collection(self):
        focus_sig=self.head.sig
        if focus_sig is not None:
            self.signature=focus_sig.sign
            if isinstance(focus_sig,funct):
                self.left=Computational_tree(focus_sig.v,init=False,fcs=focus_sig)
                self.left.auto_collection()
            else:
                self.left=Computational_tree(focus_sig.a,init=False,fcs=focus_sig)
                self.right=Computational_tree(focus_sig.b,init=False,fcs=focus_sig)
                self.left.auto_collection()
                self.right.auto_collection()
    def auto_collect_search(self,name,robj=False):
        a=None
        if self.head.name == name:
            if robj:
                return self
            return self.head
        if self.left is not None:
            a = self.left.auto_collect_search(name=name,robj=robj)
        if a is not None:
            return a
        if self.right is not None:
            a = self.right.auto_collect_search(name=name,robj=robj)
        if a is not None:
            return a
    @staticmethod
    def get_total_integrated(f,value):
        v=Variable(value,name="x")
        fvalue=f(v)
        fvalue.backward_integrated(1)
        cfv=Computational_tree(fvalue)
        cfv.auto_collection()
        obj=cfv.auto_collect_search("x",robj=True)
        if obj.fcs.a.name == "x":
            db=obj.fcs.b.integrated
        elif obj.fcs.b.name == "x":
            db=obj.fcs.a.integrateds
        else:
            db=0
        
        total=obj.head.integrated+(db*value)
        return total
        
        
    @staticmethod
    def find_area_undercurve(f,x1,x2):
        upper=Computational_tree.get_total_integrated(f,x1)
        lower=Computational_tree.get_total_integrated(f,x2)
        area=lower-upper
        return area
    @staticmethod
    def auto_derivative_plotter(f,limit=[-100,100],wrt_to=None,dense=100):
        limit=np.linspace(limit[0],limit[1],limit[1]-limit[0]*dense)
        Rm=[]
        Dm=[]
        Rma=[]
        Dma=[]
        IntegralDm=[]
        IntegralRm=[]

        period_dm=Period_recorder()
        period_rm=Period_recorder(is_rm=True)
        slope=1
        for i in limit:
            y=f(Variable(i,name="x"))
            period_dm.record(i,rmdm=y.value)
            period_rm.record(y.value)
            if not math.isnan(y.value):
                Dm.append(i)
                Rm.append(y.value)
            y.backward(1)
            y.backward_integrated(1)

            y=Computational_tree(y)
            y.auto_collection()
            roj=y.auto_collect_search(name=wrt_to)
            Dma.append(i)
            Rma.append(roj.grad)
            IntegralDm.append(i)
            IntegralRm.append(roj.integrated)
        
        dml=Computational_tree.exceptional_finder(Dm,limit)
        rml=Computational_tree.exceptional_finder(Rm,limit)
        plt.plot(Dm,Rm,'b-',label="f")
        plt.plot(IntegralDm,IntegralRm,'g-',label=f'd{wrt_to}')
        plt.plot(Dma,Rma,'r-',label=f'df/d{wrt_to}')

        period_dm.end_record()
        period_rm.end_record(U=limit)

        plt.xlim(dml[2])
        plt.ylim(rml[2])

        print(period_dm.period)
        print(period_rm.period)

        excep=Period_recorder()
        excep.record(*period_dm.exception)
        excep.end_record()
        print(excep.period)
        print(period_dm.exception)
        #print(period_rm.exception)

        plt.legend()
        

        
        
                    
                
                
                
    def travel(self,t=0,collection=[],name='a',nextNone=True,ptr_node=None,parent=None):
        self.signature=(str(str(self.signature))).upper()
        name=f'{str(t)+name}'
        if self.head.name is not None:
            name=self.head.name
        plenty_data=f'name : {name} \nvalue : {self.head} \ngradient_wrt_0a : {self.head.grad} \nintegral_to_0a : {self.head.integrated} \n\n{self.signature}'
        if ptr_node is None:
            ptr_node=Node(plenty_data,parent=parent)
        collection.append(plenty_data)
        if nextNone:
            collection.append(None)
        nextNone=False
        if self.left is not None:
            if self.right is None:
                nextNone=True
            self.left.travel(t+1,collection,name='a',nextNone=nextNone,parent=ptr_node)
        if self.right is not None:
            self.right.travel(t+1,collection,name='b',parent=ptr_node)
        return collection,ptr_node
    
    def render(self,beautiful=False,show_now=True,parallel=False):
        data=self.travel()[1]
        if not beautiful:
            for pre, fill, node in RenderTree(data):
                print("%s%s" % (pre, node.name))
        else:
            if parallel:
                tool=DotExporter
            else:
                tool=UniqueDotExporter
            d=tool(data, edgeattrfunc = lambda node, child: "dir=back")
            d.to_dotfile('gg.dot')
            (graph,) = pydot.graph_from_dot_file('gg.dot')
            graph.write_png('gg.png')
            img=Image.open('gg.png')
            img.show()
            return d
        
    
            
        
            













