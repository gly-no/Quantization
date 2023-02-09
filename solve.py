import torch
import torch.nn as nn
from scipy.optimize import minimize
import numpy as np
import math
from model import LeNet
from model import FCnet
from numpy import *
from sympy import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

e = 1e-10  #非常接近0的值，保持计算稳定？

def solve_params(w):
    w = w.numpy()
    n = w.shape[0]*w.shape[1]

    #mu-law
    # fun = lambda x: np.sum(-np.log(x/((1+x*w)*np.log(1+x))))/n
    fun = lambda x: np.sum(np.log(1+x*w))/n - np.log(x/(np.log(1+x)))
    cons = ({'type': 'ineq', 'fun': lambda x: x - e})
    x0 = np.array((100))
    res = minimize(fun, x0, method='SLSQP', constraints=cons)
    # print('minimun value: ', res.fun)
    # print('Optimal solution', res.x)
    return res.x


    # #beta-law
    # fun = lambda x: np.sum(-(math.gamma(x[0]+x[1])*np.power(w,x[0]-1)*np.power(1-w,x[1]-1))/(math.gamma(x[0])*math.gamma(x[1])))/n
    # cons = ({'type': 'ineq', 'fun': lambda a: a - e},{'type': 'ineq', 'fun': lambda b: b - e})
    # x0 = np.array((0.5,0.5))
    # res = minimize(fun, x0, method='SLSQP', constraints=cons)

    # # print('minimun value: ', res.fun)
    # # print('Optimal solution', res.x)
    # return res.x


def W_Quan(model,M,t):
    #mapping to uniform
    ### 先归一化
    a1 = np.abs(model.F1.weight.data)
    a1 = a1.max()
    a2 = np.abs(model.F2.weight.data)
    a2 = a2.max()
    a3 = np.abs(model.OUT.weight.data)
    a3 = a3.max()
    w1_r = model.F1.weight.data
    w2_r = model.F2.weight.data
    w3_r = model.OUT.weight.data

    # w1_r = model.F1.weight.data/a1
    # n = len(w1_r)
    # m = len(w1_r[0])
    # for i in range(n):
    #     for j in range(m):
    #         if w1_r[i][j] < 0:
    #             w1_r[i][j] = -w1_r[i][j]

    w1 = (model.F1.weight.data/a1+1)/2  #移到(0,1)
    w2 = (model.F2.weight.data/a2+1)/2
    w3 = (model.OUT.weight.data/a3+1)/2

    ###解参数and mapping

    #mu-law
    mu1 = solve_params(w1)
    mu2 = solve_params(w2)
    mu3 = solve_params(w3)

    w1_u = np.log(1+mu1*w1.numpy())/np.log(1+mu1)
    w2_u = np.log(1+mu2*w2.numpy())/np.log(1+mu2)
    w3_u = np.log(1+mu3*w3.numpy())/np.log(1+mu3)
    
    # plot_hist(w1_r.numpy(), w1_u, M, t)
    # #beta-law
    # [al1,b1] = solve_params(w1)
    # [al2,b2] = solve_params(w2)
    # [al3,b3] = solve_params(w3)

    # tt = symbols('tt')
    # func1 = tt**(al1-1)*(1-tt)**(b1-1)
    # w1_u = integrate(func1, (tt, 0, w1))

    # func2 = math.pow(tt,al2-1)*math.pow(1-tt,b2-1)
    # w2_u = integrate(func2, (tt, 0, w2))

    # func3 = math.pow(tt,al3-1)*math.pow(1-tt,b3-1)
    # w3_u = integrate(func3, (tt, 0, w3))
    
    #uniform quantization
    w1_q = np.round((math.pow(2,M)-1)*w1_u)/(math.pow(2,M)-1)
    w2_q = np.round((math.pow(2,M)-1)*w2_u)/(math.pow(2,M)-1)
    w3_q = np.round((math.pow(2,M)-1)*w3_u)/(math.pow(2,M)-1)

    #inverse mapping

    # mu-law
    w1_Q = (2*(np.power(1+mu1,w1_q)-1)/mu1-1)*a1.numpy()
    w2_Q = (2*(np.power(1+mu2,w2_q)-1)/mu2-1)*a2.numpy()
    w3_Q = (2*(np.power(1+mu3,w3_q)-1)/mu3-1)*a3.numpy()


    #beta-law

    # if t % 10 == 9:
    plot_hist(w1.numpy(), w1_u, M, t)


    w1_Q = torch.from_numpy(w1_Q)
    w2_Q = torch.from_numpy(w2_Q)
    w3_Q = torch.from_numpy(w3_Q)

    model.F1.weight.data = w1_Q.to(torch.float)
    model.F2.weight.data = w2_Q.to(torch.float)
    model.OUT.weight.data = w3_Q.to(torch.float)

    # model.F1.weight.data = torch.zeros(100,784)
    # model.F2.weight.data = torch.zeros(20,100)
    # model.OUT.weight.data = torch.zeros(10,20)
    
    #grad

    w1_grad = np.power(1+mu1, w1_q)/(1+mu1*w1_q)
    w2_grad = np.power(1+mu2, w2_q)/(1+mu2*w2_q)
    w3_grad = np.power(1+mu3, w3_q)/(1+mu3*w3_q)

    # g1 = model.F1.weight.grad
    # g2 = model.F2.weight.grad
    # g3 = model.OUT.weight.grad
    
    # w1_grad = torch.from_numpy(w1_grad*g1)
    # w2_grad = torch.from_numpy(w2_grad*g2)
    # w3_grad = torch.from_numpy(w3_grad*g3)

    return model, w1_grad, w2_grad, w3_grad

def W_grad(model,grad):
    
    g1 = model.F1.weight.grad
    g2 = model.F2.weight.grad
    g3 = model.OUT.weight.grad
    
    gg1= torch.from_numpy(grad[0])*g1
    gg2 = torch.from_numpy(grad[1])*g2
    gg3 = torch.from_numpy(grad[2])*g3
    # gg1 = grad[0]*g1
    # gg2 = grad[1]*g2
    # gg3 = grad[2]*g3

    return  gg1, gg2, gg3

def plot_hist(x_r,x_Q,M,t):
    # x = x.numpy()
    # plt.ion()

    plt.figure(t+1)
    x_r = x_r.flatten()
    x_Q = x_Q.flatten()
    B = int(math.pow(2,M))

    plt.subplot(1,2,1)
    f,bins,_ = plt.hist(x_r, "auto", density=False)
    plt.title("Before")
    # print(f)

    plt.subplot(1,2,2)
    plt.hist(x_Q, "auto", density=False)
    plt.title("After")

    plt.show()


if __name__ == "__main__":
    # w1 = np.array([[0.2, 0.3, 0.8],[0.6,0.5,0.1]])
    # solve_para(w=w1)
    model = FCnet()
    out = W_Quan(model=model, M=2)
    # gg = W_grad(model=out[0],grad=out[1:3])
    print(out[0])
    # model.F1.weight.data = w1