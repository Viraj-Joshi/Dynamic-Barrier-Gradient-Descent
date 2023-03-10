import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
import numpy as np

"""
This file replicates the experiments in Section 8.1 of the Supplemental Material
f(\theta) = ||\theta - \theta^*||^2 and g(\theta)=||a^T \theta + b||^2
where \theta,\theta^*,a are in R^2 and b in R 
"""

def f1(theta1,theta2,theta_star):
    return torch.sqrt((theta1 - theta_star[0])**2 + (theta2-theta_star[1])**2)**2

def f(theta,theta_star):
    return torch.linalg.norm(theta-theta_star,ord=2)

def g(theta,a,b):
    return torch.linalg.norm(torch.dot(a,theta)+b,ord=2)
def Toy2DExperiment(theta_init,theta_star,a,b,c,gamma=.1):
    alphas = [.1,1,10]
    betas = [.01,1,10]
    
    def dynamic_barrier_gradient_descent(alpha,beta):
        thetas = [theta_init]
        theta = thetas[0]
        for i in range(200):
            grad_f = grad(f(theta,theta_star),theta)[0]
            grad_g = grad(g(theta,a,b),theta)[0]

            phi = min(alpha * (g(theta,a,b)-c),beta * torch.linalg.norm(grad_g,ord=2)**2)

            lam = max(0,(phi-torch.dot(grad_f,grad_g))/torch.linalg.norm(grad_g,ord=2)**2)

            delta = grad_f+lam*grad_g

            theta.grad = None

            theta = theta - gamma * delta
            thetas.append(theta)
        return thetas

    # Experiment 1a
    x = torch.linspace(-3, 3, 200)
    X1, X2 = torch.meshgrid(x, x)
    Z1 = f1(X1,X2,theta_star)

    # Compute the halfspace
    Z2 = a[0]*X1 + a[1]*X2 + b
    
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # Add the halfspace to the plot
    ax.fill_between(X1.flatten(), X2.flatten(), c, where=Z2.flatten() <= c, color='b', alpha=0.1)
    
    # Add the contour plot of f
    contours = plt.contour(X1, X2, Z1, 20)

    # Add labels and title
    plt.clabel(contours, inline=True, fontsize=10)
    plt.xlabel('x1', fontsize=11)
    plt.ylabel('x2', fontsize=11)
    plt.title(f'2D constrained optimization varying alpha with beta=1')

    for alpha in alphas:
        thetas = dynamic_barrier_gradient_descent(alpha,1)
        plt.plot([theta[0].item() for theta in thetas], [theta[1].item() for theta in thetas],'--',label=f"alpha={alpha}")
    plt.plot(theta_star[0],theta_star[1],'*')
    plt.colorbar()
    plt.legend(loc="lower left")
    plt.savefig('experiment_1a')

    ############################
    # Experiment 1b

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # Add the halfspace to the plot
    ax.fill_between(X1.flatten(), X2.flatten(), c, where=Z2.flatten() <= c, color='b', alpha=0.1)
    
    # Add the contour plot of f
    contours = plt.contour(X1, X2, Z1, 20)

    # Add labels and title
    plt.clabel(contours, inline=True, fontsize=10)
    plt.xlabel('x1', fontsize=11)
    plt.ylabel('x2', fontsize=11)
    plt.title(f'2D constrained optimization varying beta with alpha=20')

    for beta in betas:
        thetas = dynamic_barrier_gradient_descent(20,beta)
        plt.plot([theta[0].item() for theta in thetas], [theta[1].item() for theta in thetas],'--',label=f"beta={beta}")
    plt.plot(theta_star[0],theta_star[1],'*')
    plt.colorbar()
    plt.legend(loc="lower left")
    plt.savefig('experiment_1b')
    #######################
    # Experiment 1c - Lexicographic Optimization
    # the minimum of g(theta) is zero=> c = g* = 0

    c = 0    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # Add the halfspace to the plot
    ax.fill_between(X1.flatten(), X2.flatten(), c, where=Z2.flatten() <= c, color='b', alpha=0.1)
    
    # Add the contour plot of f
    contours = plt.contour(X1, X2, Z1, 20)

    # Add labels and title
    plt.clabel(contours, inline=True, fontsize=10)
    plt.xlabel('x1', fontsize=11)
    plt.ylabel('x2', fontsize=11)
    plt.title(f'2D lexicographic optimization varying beta with beta=1')

    for alpha in alphas:
        thetas = dynamic_barrier_gradient_descent(alpha,1)
        plt.plot([theta[0].item() for theta in thetas], [theta[1].item() for theta in thetas],'--',label=f"alpha={alpha}")
    plt.plot(theta_star[0],theta_star[1],'*')
    plt.colorbar()
    plt.legend(loc="lower left")
    plt.savefig('experiment_1c')

    




torch.manual_seed(993)

theta = torch.randn(2,requires_grad=True)-1


theta_star = torch.randn(2)

a = torch.randn(2)
b = torch.randn(1)

Toy2DExperiment(theta,theta_star,a,b,c=.01)
